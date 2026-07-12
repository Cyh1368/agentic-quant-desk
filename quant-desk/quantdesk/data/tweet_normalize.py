"""Tweet normalizer: raw tweet dict -> normalized record (plan §4).

Produces the four text representations mandated by the plan — never one
destructively normalized string:

* ``raw_text_restricted``       verbatim tweet text, retention-governed.
* ``normalized_text_for_dedup`` lowercase, handles/URLs/whitespace stripped,
  used only to group near-duplicates.
* ``model_input_text``          versioned transform fed to the sentiment
  scorer: URLs and mentions replaced with stable tokens, cashtags/hashtags/
  emojis/negation/punctuation preserved, NFC-normalized, truncated to a
  fixed character budget (truncation recorded in ``quality_flags``).
* ``display_text_for_llm``      links/markdown/HTML/control characters
  stripped, truncated, wrapped in an untrusted envelope before an LLM ever
  sees it.

This module never touches storage or the network; it is a pure function of
its inputs plus wall-clock-free timestamps supplied by the caller.
"""
from __future__ import annotations

import html
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

NORMALIZER_VERSION = "tweet_normalize_v1"

MODEL_INPUT_MAX_CHARS = 512
DISPLAY_MAX_CHARS = 512

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"(?<![\w@])@\w+")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MARKDOWN_CHARS_RE = re.compile(r"[*_`\[\]()#>~]")
HTML_TAG_RE = re.compile(r"<[^>]*>")
WHITESPACE_RE = re.compile(r"\s+")

ENVELOPE_OPEN = "<untrusted_tweet>"
ENVELOPE_CLOSE = "</untrusted_tweet>"
# Anything that looks like the envelope tags (open/close, with any casing or
# internal whitespace variations) is escaped so hostile tweet text can never
# fake an envelope boundary.
_ENVELOPE_LOOKALIKE_RE = re.compile(
    r"</?\s*untrusted_tweet\s*>", re.IGNORECASE
)


def _escape_envelope_lookalikes(text: str) -> str:
    """Escape any substring that looks like an envelope tag inside ``text``."""

    def _escape(match: re.Match) -> str:
        return match.group(0).replace("<", "\\<").replace(">", "\\>")

    return _ENVELOPE_LOOKALIKE_RE.sub(_escape, text)


def make_untrusted_envelope(text: str) -> str:
    """Wrap ``text`` in an untrusted-content envelope, escaping lookalikes.

    Any sequence inside ``text`` that resembles ``<untrusted_tweet>`` or
    ``</untrusted_tweet>`` (any casing/whitespace) is escaped first so the
    tweet body can never forge an envelope boundary.
    """
    escaped = _escape_envelope_lookalikes(text)
    return f"{ENVELOPE_OPEN}{escaped}{ENVELOPE_CLOSE}"


def _strip_urls_and_handles_for_dedup(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.lower()
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_model_input_text(raw_text: str) -> tuple[str, bool]:
    """Versioned transform for the sentiment scorer.

    Returns ``(text, truncated)``.
    """
    text = unicodedata.normalize("NFC", raw_text)
    text = URL_RE.sub("<URL>", text)
    text = MENTION_RE.sub("<USER>", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    truncated = len(text) > MODEL_INPUT_MAX_CHARS
    if truncated:
        text = text[:MODEL_INPUT_MAX_CHARS]
    return text, truncated


def build_display_text_for_llm(raw_text: str) -> tuple[str, bool]:
    """Strip links/markdown/HTML/control chars, truncate, return (text, truncated)."""
    text = html.unescape(raw_text)
    text = URL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = MARKDOWN_CHARS_RE.sub(" ", text)
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    truncated = len(text) > DISPLAY_MAX_CHARS
    if truncated:
        text = text[:DISPLAY_MAX_CHARS]
    return text, truncated


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            # Twitter/X legacy form: "Sun Jul 12 12:21:07 +0000 2026"
            dt = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    else:
        raise ValueError(f"cannot parse time value: {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _author_age_days(author_created_at: datetime | None, event_time: datetime) -> float | None:
    if author_created_at is None:
        return None
    delta = event_time - author_created_at
    return delta.total_seconds() / 86400.0


def _asset_mentions(text: str, config_assets: dict[str, Any]) -> list[str]:
    """Which configured assets are mentioned (cashtag or name), case-insensitive."""
    lowered = text.lower()
    mentions: list[str] = []
    for symbol in config_assets:
        symbol_l = symbol.lower()
        patterns = [rf"\${re.escape(symbol_l)}\b", rf"\b{re.escape(symbol_l)}\b"]
        names = config_assets[symbol].get("names", []) if isinstance(config_assets[symbol], dict) else []
        for name in names:
            patterns.append(rf"\b{re.escape(name.lower())}\b")
        if any(re.search(p, lowered) for p in patterns):
            mentions.append(symbol)
    return mentions


def normalize_tweet(
    raw: dict,
    ingested_time: datetime,
    raw_payload_hash: str,
    source_class: str,
    asset_query_matches: list[str],
    *,
    assets_config: dict[str, Any] | None = None,
) -> dict:
    """Normalize one raw tweet dict into the storage-ready record.

    ``asset_query_matches`` is the list of asset symbols whose fetch query
    matched this tweet (from the fetch adapter, out of scope here).
    ``assets_config`` is optionally the ``assets`` section of
    ``config/sentiment.yaml`` (symbol -> {names: [...]}) used to additionally
    detect co-mentions of *other* configured assets inside the tweet text
    itself, so ``target_ambiguity`` can be computed even for cashtag-only
    query matches. If omitted, only ``asset_query_matches`` is used.
    """
    if ingested_time.tzinfo is None:
        raise ValueError("ingested_time must be timezone-aware UTC")
    ingested_time = ingested_time.astimezone(timezone.utc)

    quality_flags: list[str] = []

    raw_text = raw.get("text", "") or ""
    event_time = _parse_time(raw["createdAt"])

    author = raw.get("author") or {}
    author_id = author.get("id")
    author_handle = author.get("userName")
    author_followers = author.get("followers")
    author_created_raw = author.get("createdAt")
    author_created_at = _parse_time(author_created_raw) if author_created_raw else None
    author_age_days = _author_age_days(author_created_at, event_time)

    normalized_text_for_dedup = _strip_urls_and_handles_for_dedup(raw_text)
    model_input_text, model_truncated = build_model_input_text(raw_text)
    if model_truncated:
        quality_flags.append("model_input_truncated")
    display_text_stripped, display_truncated = build_display_text_for_llm(raw_text)
    if display_truncated:
        quality_flags.append("display_text_truncated")
    display_text_for_llm = make_untrusted_envelope(display_text_stripped)

    asset_mentions = list(dict.fromkeys(asset_query_matches or []))
    if assets_config:
        for symbol in _asset_mentions(raw_text, assets_config):
            if symbol not in asset_mentions:
                asset_mentions.append(symbol)

    if len(asset_mentions) == 0:
        target_ambiguity = "none"
    elif len(asset_mentions) == 1:
        target_ambiguity = "single"
    else:
        target_ambiguity = "multi_asset"

    is_retweet = bool(raw.get("isRetweet") or raw.get("retweeted") is not None and raw.get("retweeted"))
    is_reply = bool(raw.get("inReplyToId") or raw.get("isReply"))
    lang = raw.get("lang")

    if lang is not None:
        quality_flags.append(f"lang:{lang}")
    if is_retweet:
        quality_flags.append("is_retweet")
    if is_reply:
        quality_flags.append("is_reply")

    return {
        # lineage
        "event_time": event_time,
        "ingested_time": ingested_time,
        "available_to_strategy_time": ingested_time,
        "source_id": str(raw.get("id")),
        "raw_payload_hash": raw_payload_hash,
        "normalizer_version": NORMALIZER_VERSION,
        "quality_flags": quality_flags,
        "source_class": source_class,
        # text representations
        "raw_text_restricted": raw_text,
        "normalized_text_for_dedup": normalized_text_for_dedup,
        "model_input_text": model_input_text,
        "display_text_for_llm": display_text_for_llm,
        # author
        "author_id": str(author_id) if author_id is not None else None,
        "author_handle": author_handle,
        "author_followers": author_followers,
        "author_created_at": author_created_at,
        "author_age_days": author_age_days,
        # asset targeting
        "asset_mentions": asset_mentions,
        "target_ambiguity": target_ambiguity,
        # client-side filters, flagged not deleted
        "is_retweet": is_retweet,
        "is_reply": is_reply,
        "lang": lang,
        # raw engagement counters as seen at fetch time (caller may also
        # record these via TweetStore.add_engagement_observation)
        "retweet_count": raw.get("retweetCount"),
        "reply_count": raw.get("replyCount"),
        "like_count": raw.get("likeCount"),
        "view_count": raw.get("viewCount"),
        "tweet_id": str(raw.get("id")),
    }
