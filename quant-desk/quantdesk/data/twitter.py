"""TwitterAPI.io source adapter for the Twitter/X sentiment slice (plan §1-4).

RESEARCH ONLY — this module has no path to the executor. It only fetches raw
tweets, writes them through the immutable :class:`~quantdesk.data.raw_store.RawStore`
landing zone, and returns lightly-parsed dicts plus a completeness record.
Normalization, dedup, scoring, and feature computation live elsewhere and are
out of scope here.

Provider: https://twitterapi.io — advanced search endpoint
``GET /twitter/tweet/advanced_search``, header ``x-api-key``. Advanced search
does **not** support reliable cursor pagination for point-in-time historical
completeness (plan §1); instead, when a single request appears to have hit
the provider's page limit (assumed ~20 tweets/page, see
``PROVIDER_PAGE_LIMIT_HINT`` below), this adapter recursively **splits the
requested time window in half** and re-fetches each half, until either a
sub-window fits in one page or the window has been split down to
``MIN_SPLIT_SECONDS`` (30s), at which point that sub-window is marked
incomplete with ``truncation_reason="provider_limit"``.

Timeline endpoints (curated authors, ``fetch_user_timeline``) are different:
the provider timeline endpoint *does* support cursor pagination and is used
here in the normal "keep following the cursor" way — no time-splitting.

API-shape assumptions (unverified against live traffic; documented per task
instructions since exact wire shape could not be confirmed at build time):

- Advanced search response: ``{"tweets": [...], "has_next_page": bool,
  "next_cursor": str}``. ``has_next_page``/``next_cursor`` are NOT used for
  correctness (see above) — only as a hint that a window may be truncated;
  the authoritative "did this window get capped" signal used here is
  "the page returned >= PROVIDER_PAGE_LIMIT_HINT tweets AND has_next_page is
  true", at which point we discard the pagination cursor entirely and
  time-split instead.
- Each tweet dict: ``{"id": str, "text": str, "createdAt": str (e.g.
  "Wed Oct 05 20:00:00 +0000 2022"), "retweetCount": int, "replyCount": int,
  "likeCount": int, "viewCount": int, "author": {"userName": str,
  "followers": int, "createdAt": str}}``. Any fields beyond these are passed
  through untouched in the raw payload (schema drift tolerance is a
  normalizer concern, not this module's).
- ``createdAt`` is parsed defensively: RFC-2822-ish Twitter format first,
  falling back to ISO-8601, and finally left as ``None`` (with the whole
  tweet treated as unparseable / dropped from ``tweets`` but still present
  verbatim in the raw payload written to RawStore) if neither parses.
- Rate-limit headers, if present, are assumed to be ``x-ratelimit-limit`` /
  ``x-ratelimit-remaining`` / ``x-ratelimit-reset`` (standard convention);
  ``preflight()`` returns whatever subset is actually present without
  assuming all three exist.

The provider key is read once from ``TWITTERAPI_IO_KEY`` and is never placed
in an exception message, a log line, or a fixture. Tests must not use a real
key.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal
from uuid import UUID, uuid4

import httpx
import yaml
from pydantic import BaseModel, Field

from quantdesk.data.raw_store import RawStore

SOURCE_ID = "twitterapi_io"
ADVANCED_SEARCH_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
USER_TIMELINE_URL = "https://api.twitterapi.io/twitter/user/last_tweets"

# Assumed provider page size for advanced search (undocumented precisely;
# treated as a conservative trigger for time-splitting rather than a hard
# fact — see module docstring).
PROVIDER_PAGE_LIMIT_HINT = 20
MIN_SPLIT_SECONDS = 30


class TermsNotReviewedError(Exception):
    """Raised when config/sentiment.yaml terms_review has unresolved fields.

    Fail-closed per plan §4: the source refuses to enable while any required
    terms-review item is null.
    """

    def __init__(self, missing_fields: list[str]):
        self.missing_fields = missing_fields
        super().__init__(
            "terms_review is not complete; unresolved required fields: "
            + ", ".join(missing_fields)
        )


REQUIRED_TERMS_REVIEW_FIELDS = (
    "reviewed_at",
    "reviewed_by",
    "permitted_storage",
    "raw_retention_days",
    "deletion_sync_required",
    "redistribution_allowed",
    "commercial_use_allowed",
    "model_training_allowed",
)


def require_terms_reviewed(sentiment_config_path: str = "config/sentiment.yaml") -> dict:
    """Load config/sentiment.yaml and raise if any required terms_review field is null.

    Returns the ``terms_review`` mapping on success.
    """
    with open(sentiment_config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    terms = cfg.get("terms_review") or {}
    missing = [f for f in REQUIRED_TERMS_REVIEW_FIELDS if terms.get(f) is None]
    if missing:
        raise TermsNotReviewedError(missing)
    return terms


class SourceContinuation(BaseModel):
    mode: Literal["cursor", "time_split", "none"]
    cursor: str | None = None
    next_start_time: datetime | None = None
    next_end_time: datetime | None = None


class FetchPage(BaseModel):
    tweets: list[dict]
    continuation: SourceContinuation | None = None
    complete: bool
    provider_latency_ms: float


class QueryWindowRecord(BaseModel):
    query_window_id: UUID = Field(default_factory=uuid4)
    asset: str
    source_class: Literal["broad", "curated"]
    requested_start: datetime
    requested_end: datetime
    coverage_start: datetime | None = None
    coverage_end: datetime | None = None
    fetched_count: int = 0
    complete: bool = True
    truncation_reason: (
        Literal[
            "provider_limit",
            "budget_cap",
            "rate_limit",
            "timeout",
            "schema_error",
            "partial_window",
        ]
        | None
    ) = None
    fetch_attempts: int = 0
    provider_latency_ms: float = 0.0
    quality_flags: list[str] = Field(default_factory=list)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _require_utc(dt: datetime, name: str) -> None:
    if dt.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")


def _fmt_query_time(dt: datetime) -> str:
    """Format a UTC datetime as ``YYYY-MM-DD_HH:MM:SS_UTC`` for the query string."""
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d_%H:%M:%S_UTC")


def _parse_created_at(value: Any) -> datetime | None:
    """Defensively parse a tweet's createdAt field (Twitter RFC-2822-ish, or ISO-8601)."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


class TwitterApiIoSource:
    """Adapter for TwitterAPI.io advanced search + user timeline endpoints.

    Every raw HTTP response body is written through :class:`RawStore` under
    ``source_id="twitterapi_io"`` before any parsing happens (plan §3/§4
    immutable landing zone).
    """

    def __init__(
        self,
        raw_store: RawStore,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
        sentiment_config_path: str = "config/sentiment.yaml",
        allow_unreviewed_terms: bool = False,
    ):
        if not allow_unreviewed_terms:
            require_terms_reviewed(sentiment_config_path)

        self.raw_store = raw_store
        self._api_key = api_key if api_key is not None else os.environ.get("TWITTERAPI_IO_KEY")
        if not self._api_key and not allow_unreviewed_terms:
            # Only enforce presence outside of explicit test overrides; tests
            # that bypass the terms gate typically also inject a fake key.
            raise ValueError("TWITTERAPI_IO_KEY is not set")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def __enter__(self) -> "TwitterApiIoSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._owns_client:
            self._client.close()

    def _headers(self) -> dict[str, str]:
        headers = {"x-api-key": self._api_key or ""}
        return headers

    def preflight(self) -> dict[str, Any]:
        """Validate the API key with a 1-result probe; return advertised rate limits.

        Never raises the key itself in any error; httpx status errors are
        re-raised with only status code / URL context.
        """
        params = {
            "query": "test lang:en",
            "queryType": "Latest",
        }
        try:
            resp = self._client.get(
                ADVANCED_SEARCH_URL, params=params, headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"preflight request failed: {type(exc).__name__}") from None
        if resp.status_code == 401 or resp.status_code == 403:
            raise RuntimeError(f"preflight failed: provider rejected credentials (status {resp.status_code})")
        resp.raise_for_status()
        rate_limit_info = {
            k: v
            for k, v in resp.headers.items()
            if k.lower().startswith("x-ratelimit")
        }
        return {"ok": True, "rate_limit": rate_limit_info}

    # -- advanced search (time-split, no cursor) ---------------------------

    def _advanced_search_request(
        self, query: str, start_time: datetime, end_time: datetime
    ) -> tuple[Any, float]:
        full_query = f"{query} since:{_fmt_query_time(start_time)} until:{_fmt_query_time(end_time)}"
        params = {"query": full_query, "queryType": "Latest"}
        t0 = _now_utc()
        try:
            resp = self._client.get(
                ADVANCED_SEARCH_URL, params=params, headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"advanced_search request failed: {type(exc).__name__}") from None
        latency_ms = (_now_utc() - t0).total_seconds() * 1000.0
        resp.raise_for_status()
        payload = resp.json()
        self.raw_store.write(
            SOURCE_ID,
            payload,
            request={"query": full_query, "since": start_time.isoformat(), "until": end_time.isoformat()},
        )
        return payload, latency_ms

    def fetch_window(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        continuation: SourceContinuation | None = None,
    ) -> FetchPage:
        """Fetch one time window of advanced-search results.

        Ignores any provider pagination cursor for advanced search — per
        plan §1, that is unreliable for point-in-time completeness. If the
        response looks page-capped, this returns ``complete=False`` with a
        ``time_split`` continuation describing the two sub-windows the
        caller (``fetch_asset_window``) should fetch next. Callers wanting
        the full window resolved recursively should use
        ``fetch_asset_window`` instead of calling this directly in a loop.
        """
        _require_utc(start_time, "start_time")
        _require_utc(end_time, "end_time")
        if continuation is not None and continuation.mode not in ("time_split", "none"):
            raise ValueError("advanced search does not support cursor continuation")

        payload, latency_ms = self._advanced_search_request(query, start_time, end_time)
        raw_tweets = payload.get("tweets", []) if isinstance(payload, dict) else []
        has_next = bool(payload.get("has_next_page")) if isinstance(payload, dict) else False

        page_capped = len(raw_tweets) >= PROVIDER_PAGE_LIMIT_HINT and has_next
        window_seconds = (end_time - start_time).total_seconds()

        if page_capped and window_seconds > MIN_SPLIT_SECONDS:
            midpoint = start_time + (end_time - start_time) / 2
            return FetchPage(
                tweets=raw_tweets,
                continuation=SourceContinuation(
                    mode="time_split",
                    next_start_time=start_time,
                    next_end_time=midpoint,
                ),
                complete=False,
                provider_latency_ms=latency_ms,
            )

        return FetchPage(
            tweets=raw_tweets,
            continuation=SourceContinuation(mode="none"),
            complete=not page_capped,
            provider_latency_ms=latency_ms,
        )

    def fetch_asset_window(
        self,
        asset: str,
        query: str,
        start_time: datetime,
        end_time: datetime,
        *,
        source_class: Literal["broad", "curated"] = "broad",
        max_tweets: int | None = None,
    ) -> tuple[list[dict], QueryWindowRecord]:
        """Recursively resolve a full time window via time-splitting.

        ``max_tweets`` is a hard spend bound: once collected tweets reach it,
        recursion stops, the record is marked incomplete with
        ``truncation_reason="budget_cap"``, and absolute-volume features for
        the window are therefore nulled downstream (plan §3).

        Returns all tweets found plus one :class:`QueryWindowRecord`
        summarizing completeness across the whole (possibly split) fetch.
        No tweet is fetched twice and no boundary timestamp is skipped: each
        split covers ``[start, mid)`` and ``[mid, end)`` — the two halves
        share the ``mid`` boundary with the left half exclusive of it.

        Since the provider's ``since:``/``until:`` operators are second
        granularity and inclusive on both ends in practice, we avoid
        duplicate boundary tweets by nudging the second half's start forward
        by one second past the midpoint.
        """
        _require_utc(start_time, "start_time")
        _require_utc(end_time, "end_time")

        record = QueryWindowRecord(
            asset=asset,
            source_class=source_class,
            requested_start=start_time,
            requested_end=end_time,
        )
        all_tweets: list[dict] = []
        seen_ids: set[str] = set()

        def _recurse(lo: datetime, hi: datetime) -> None:
            if max_tweets is not None and len(all_tweets) >= max_tweets:
                record.complete = False
                record.truncation_reason = "budget_cap"
                record.quality_flags.append(
                    f"budget_capped_at:{max_tweets}:unfetched:{lo.isoformat()}..{hi.isoformat()}")
                return
            record.fetch_attempts += 1
            page = self.fetch_window(query, lo, hi)
            record.provider_latency_ms += page.provider_latency_ms

            if page.complete:
                for t in page.tweets:
                    tid = t.get("id") if isinstance(t, dict) else None
                    if tid is not None and tid in seen_ids:
                        continue
                    if tid is not None:
                        seen_ids.add(tid)
                    all_tweets.append(t)
                if record.coverage_start is None or lo < record.coverage_start:
                    record.coverage_start = lo
                if record.coverage_end is None or hi > record.coverage_end:
                    record.coverage_end = hi
                return

            window_seconds = (hi - lo).total_seconds()
            if window_seconds <= MIN_SPLIT_SECONDS:
                # Truncated leaf: take what we got, mark incomplete.
                for t in page.tweets:
                    tid = t.get("id") if isinstance(t, dict) else None
                    if tid is not None and tid in seen_ids:
                        continue
                    if tid is not None:
                        seen_ids.add(tid)
                    all_tweets.append(t)
                record.complete = False
                record.truncation_reason = "provider_limit"
                record.quality_flags.append(f"truncated_window:{lo.isoformat()}..{hi.isoformat()}")
                return

            mid = lo + (hi - lo) / 2
            # Newer half first: if the max_tweets budget cap fires mid-fetch,
            # the sample retained is the freshest part of the window.
            _recurse(mid + timedelta(seconds=1), hi)
            _recurse(lo, mid)

        _recurse(start_time, end_time)

        record.fetched_count = len(all_tweets)
        if record.coverage_start is None:
            record.coverage_start = start_time
        if record.coverage_end is None:
            record.coverage_end = end_time
        return all_tweets, record

    # -- user timeline (cursor pagination allowed) --------------------------

    def fetch_user_timeline(
        self,
        user_name: str,
        *,
        since_time: datetime | None = None,
        cursor: str | None = None,
        max_pages: int = 50,
    ) -> tuple[list[dict], QueryWindowRecord]:
        """Fetch a curated author's timeline using provider cursor pagination.

        Unlike advanced search, timeline endpoints support cursors reliably
        per plan §1, so this follows ``next_cursor`` until the provider
        reports no more pages, ``since_time`` is passed, or ``max_pages`` is
        hit (a safety bound, not a correctness requirement).
        """
        if since_time is not None:
            _require_utc(since_time, "since_time")

        now = _now_utc()
        record = QueryWindowRecord(
            asset=user_name,
            source_class="curated",
            requested_start=since_time or now,
            requested_end=now,
        )
        tweets: list[dict] = []
        next_cursor = cursor
        pages = 0

        while pages < max_pages:
            params: dict[str, Any] = {"userName": user_name}
            if next_cursor:
                params["cursor"] = next_cursor
            t0 = _now_utc()
            try:
                resp = self._client.get(
                    USER_TIMELINE_URL, params=params, headers=self._headers()
                )
            except httpx.HTTPError as exc:
                record.complete = False
                record.truncation_reason = "timeout"
                record.quality_flags.append(f"timeline_error:{type(exc).__name__}")
                break
            latency_ms = (_now_utc() - t0).total_seconds() * 1000.0
            record.provider_latency_ms += latency_ms
            record.fetch_attempts += 1

            resp.raise_for_status()
            payload = resp.json()
            self.raw_store.write(
                SOURCE_ID, payload, request={"userName": user_name, "cursor": next_cursor}
            )

            page_tweets = payload.get("tweets", []) if isinstance(payload, dict) else []
            stop = False
            for t in page_tweets:
                created = _parse_created_at(t.get("createdAt")) if isinstance(t, dict) else None
                if since_time is not None and created is not None and created < since_time:
                    stop = True
                    continue
                tweets.append(t)

            pages += 1
            has_next = bool(payload.get("has_next_page")) if isinstance(payload, dict) else False
            next_cursor = payload.get("next_cursor") if isinstance(payload, dict) else None
            if stop or not has_next or not next_cursor:
                break

        record.fetched_count = len(tweets)
        record.coverage_start = since_time
        record.coverage_end = now
        if pages >= max_pages:
            record.complete = False
            record.truncation_reason = "partial_window"
            record.quality_flags.append("max_pages_reached")
        return tweets, record
