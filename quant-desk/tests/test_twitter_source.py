from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
import yaml

from quantdesk.data.raw_store import RawStore
from quantdesk.data.twitter import (
    PROVIDER_PAGE_LIMIT_HINT,
    QueryWindowRecord,
    TermsNotReviewedError,
    TwitterApiIoSource,
    require_terms_reviewed,
)


def _write_config(tmp_path, terms_review_overrides=None):
    cfg = {
        "terms_review": {
            "reviewed_at": None,
            "reviewed_by": None,
            "provider_terms_url": "https://twitterapi.io/terms",
            "platform_terms_url": "https://x.com/en/tos",
            "permitted_storage": None,
            "raw_retention_days": None,
            "deletion_sync_required": None,
            "redistribution_allowed": None,
            "commercial_use_allowed": None,
            "model_training_allowed": None,
            "notes": "pending",
        }
    }
    if terms_review_overrides:
        cfg["terms_review"].update(terms_review_overrides)
    path = tmp_path / "sentiment.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


REVIEWED_OVERRIDES = {
    "reviewed_at": "2026-07-01T00:00:00Z",
    "reviewed_by": "vivian",
    "permitted_storage": True,
    "raw_retention_days": 30,
    "deletion_sync_required": True,
    "redistribution_allowed": False,
    "commercial_use_allowed": False,
    "model_training_allowed": False,
}


def _tweet(tid: str, created_at: str = "Wed Jul 12 10:00:00 +0000 2026") -> dict:
    return {
        "id": tid,
        "text": f"tweet {tid} $BTC",
        "createdAt": created_at,
        "retweetCount": 0,
        "replyCount": 0,
        "likeCount": 0,
        "viewCount": 10,
        "author": {"userName": "someone", "followers": 500, "createdAt": "2018-01-01"},
    }


# --------------------------------------------------------------------------
# Terms gate
# --------------------------------------------------------------------------


def test_require_terms_reviewed_raises_on_unreviewed(tmp_path):
    cfg_path = _write_config(tmp_path)
    with pytest.raises(TermsNotReviewedError) as exc_info:
        require_terms_reviewed(str(cfg_path))
    assert "reviewed_at" in exc_info.value.missing_fields


def test_require_terms_reviewed_passes_when_complete(tmp_path):
    cfg_path = _write_config(tmp_path, REVIEWED_OVERRIDES)
    terms = require_terms_reviewed(str(cfg_path))
    assert terms["reviewed_by"] == "vivian"


def test_source_constructor_fails_closed_by_default(tmp_path):
    cfg_path = _write_config(tmp_path)
    raw_store = RawStore(tmp_path / "raw")
    with pytest.raises(TermsNotReviewedError):
        TwitterApiIoSource(
            raw_store,
            api_key="fake-key-not-real",
            sentiment_config_path=str(cfg_path),
        )


def test_source_constructor_allows_override_for_tests(tmp_path):
    raw_store = RawStore(tmp_path / "raw")
    src = TwitterApiIoSource(
        raw_store,
        api_key="fake-key-not-real",
        sentiment_config_path=str(tmp_path / "does_not_exist.yaml"),
        allow_unreviewed_terms=True,
    )
    assert src is not None


def test_no_secret_in_terms_not_reviewed_error(tmp_path):
    cfg_path = _write_config(tmp_path)
    try:
        require_terms_reviewed(str(cfg_path))
    except TermsNotReviewedError as exc:
        assert "fake-key-not-real" not in str(exc)


# --------------------------------------------------------------------------
# fetch_window / recursive time-splitting
# --------------------------------------------------------------------------


def _make_source(tmp_path, handler) -> TwitterApiIoSource:
    raw_store = RawStore(tmp_path / "raw")
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return TwitterApiIoSource(
        raw_store,
        api_key="fake-key-not-real",
        client=client,
        sentiment_config_path=str(tmp_path / "nope.yaml"),
        allow_unreviewed_terms=True,
    )


def test_fetch_window_single_page_complete(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tweets": [_tweet("1"), _tweet("2")], "has_next_page": False})

    src = _make_source(tmp_path, handler)
    start = datetime(2026, 7, 12, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=15)
    page = src.fetch_window("($BTC) lang:en", start, end)
    assert page.complete is True
    assert len(page.tweets) == 2


def test_fetch_window_page_capped_returns_incomplete_with_split(tmp_path):
    full_page = [_tweet(str(i)) for i in range(PROVIDER_PAGE_LIMIT_HINT)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tweets": full_page, "has_next_page": True})

    src = _make_source(tmp_path, handler)
    start = datetime(2026, 7, 12, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=15)
    page = src.fetch_window("($BTC) lang:en", start, end)
    assert page.complete is False
    assert page.continuation.mode == "time_split"
    assert page.continuation.next_start_time == start
    assert page.continuation.next_end_time == start + (end - start) / 2


def test_fetch_window_never_sends_api_key_in_url(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"tweets": [], "has_next_page": False})

    src = _make_source(tmp_path, handler)
    start = datetime(2026, 7, 12, 10, 0, tzinfo=timezone.utc)
    src.fetch_window("($BTC) lang:en", start, start + timedelta(minutes=1))
    assert "fake-key-not-real" not in seen["url"]
    assert seen["headers"]["x-api-key"] == "fake-key-not-real"


# --------------------------------------------------------------------------
# fetch_asset_window: acceptance test — 15 minute window exceeding one page
# --------------------------------------------------------------------------


def test_fetch_asset_window_recursive_split_no_gaps_no_dupes(tmp_path):
    """A 15-minute window whose volume exceeds one page is retrieved
    completely via recursive splitting, with no duplicated and no omitted
    boundary timestamps (plan §1 acceptance test)."""
    start = datetime(2026, 7, 12, 10, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=15)

    # Simulate real volume: 45 unique tweets spread evenly across the window,
    # each page-of-20 capped until sub-windows are small enough to fit.
    total_tweets = 45
    all_ids = [f"t{i}" for i in range(total_tweets)]

    def handler(request: httpx.Request) -> httpx.Response:
        query = str(request.url.params["query"])
        # crude parse of since/until embedded in the query string
        since_str = query.split("since:")[1].split(" until:")[0]
        until_str = query.split("until:")[1].split(" ")[0]

        def parse(s: str) -> datetime:
            date_part, time_part, _ = s.split("_")
            return datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )

        lo = parse(since_str)
        hi = parse(until_str)

        # Deterministically assign tweets to their window based on evenly
        # spaced synthetic timestamps across the full 15-minute span.
        span = (end - start).total_seconds()
        matched = []
        for i, tid in enumerate(all_ids):
            frac = i / total_tweets
            ts = start + timedelta(seconds=frac * span)
            if lo <= ts <= hi:
                matched.append(tid)

        tweets = [_tweet(tid) for tid in matched]
        has_next = len(tweets) >= PROVIDER_PAGE_LIMIT_HINT
        if has_next:
            tweets = tweets[:PROVIDER_PAGE_LIMIT_HINT]
        return httpx.Response(200, json={"tweets": tweets, "has_next_page": has_next})

    src = _make_source(tmp_path, handler)
    tweets, record = src.fetch_asset_window("BTC", "($BTC) lang:en", start, end)

    ids = [t["id"] for t in tweets]
    assert len(ids) == len(set(ids)), "no duplicated tweets across split boundaries"
    assert set(ids) == set(all_ids), "no omitted tweets across split boundaries"
    assert record.complete is True
    assert record.fetched_count == total_tweets
    assert record.fetch_attempts > 1
    assert isinstance(record, QueryWindowRecord)


def test_fetch_asset_window_marks_incomplete_when_min_split_reached(tmp_path):
    """If volume is so dense even a 30s window is page-capped, the leaf is
    marked incomplete with truncation_reason=provider_limit."""
    full_page = [_tweet(str(i)) for i in range(PROVIDER_PAGE_LIMIT_HINT)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tweets": full_page, "has_next_page": True})

    src = _make_source(tmp_path, handler)
    start = datetime(2026, 7, 12, 10, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=1)
    tweets, record = src.fetch_asset_window("BTC", "($BTC) lang:en", start, end)

    assert record.complete is False
    assert record.truncation_reason == "provider_limit"
    assert len(record.quality_flags) > 0


def test_fetch_asset_window_writes_through_raw_store(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tweets": [_tweet("1")], "has_next_page": False})

    raw_store = RawStore(tmp_path / "raw")
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    src = TwitterApiIoSource(
        raw_store,
        api_key="fake-key-not-real",
        client=client,
        sentiment_config_path=str(tmp_path / "nope.yaml"),
        allow_unreviewed_terms=True,
    )
    start = datetime(2026, 7, 12, 10, 0, tzinfo=timezone.utc)
    src.fetch_asset_window("BTC", "($BTC) lang:en", start, start + timedelta(minutes=1))

    records = list(raw_store.iter_records("twitterapi_io", "2026-07-12"))
    assert len(records) == 1
    assert records[0].payload["tweets"][0]["id"] == "1"


def test_fetch_window_rejects_naive_datetime(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tweets": [], "has_next_page": False})

    src = _make_source(tmp_path, handler)
    with pytest.raises(ValueError):
        src.fetch_window("($BTC)", datetime(2026, 7, 12), datetime(2026, 7, 12, 0, 15))


# --------------------------------------------------------------------------
# fetch_user_timeline: cursor pagination
# --------------------------------------------------------------------------


def test_fetch_user_timeline_follows_cursor(tmp_path):
    pages = {
        None: {"tweets": [_tweet("a"), _tweet("b")], "has_next_page": True, "next_cursor": "cur2"},
        "cur2": {"tweets": [_tweet("c")], "has_next_page": False, "next_cursor": None},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        return httpx.Response(200, json=pages[cursor])

    src = _make_source(tmp_path, handler)
    tweets, record = src.fetch_user_timeline("some_analyst")
    ids = [t["id"] for t in tweets]
    assert ids == ["a", "b", "c"]
    assert record.source_class == "curated"
    assert record.fetch_attempts == 2
    assert record.complete is True


def test_preflight_returns_rate_limit_headers(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"tweets": [], "has_next_page": False},
            headers={"x-ratelimit-remaining": "100", "x-ratelimit-limit": "1000"},
        )

    src = _make_source(tmp_path, handler)
    result = src.preflight()
    assert result["ok"] is True
    assert result["rate_limit"]["x-ratelimit-remaining"] == "100"


def test_preflight_raises_on_auth_failure_without_leaking_key(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    src = _make_source(tmp_path, handler)
    with pytest.raises(RuntimeError) as exc_info:
        src.preflight()
    assert "fake-key-not-real" not in str(exc_info.value)
