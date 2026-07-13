"""Deterministic, versioned sentiment feature family (plan §7).

Pure functions only: no I/O, no network. Inputs are plain lists of dict
"rows" (tweet-level scored records) and "query_windows" (fetch completeness
records), matching the boundary contract of the (out-of-scope) tweet store
and fetch adapter. Every feature id follows the ``compute.py`` convention:
``feature_id(instrument_id, name, version)`` -> ``"btc_sent_mean_1h@v1"``.

Row boundary contract (plain dicts, keys):
    tweet_id, author_id, sentiment, p_negative, p_neutral, p_positive,
    scorer_confidence, event_time, available_to_strategy_time,
    author_followers, author_age_days, exclusions (list[str]),
    target_ambiguity (bool), source_class ("broad" | "curated"),
    canonical_tweet_id, duplicate_count

Rows are assumed pre-filtered to the instrument/asset in question upstream
(mirroring ``compute_feature_set``'s per-instrument candle input).

Query-window boundary contract (plain dicts, keys):
    window_start, window_end (datetime, UTC), complete (bool),
    and optional data-health components in [0, 1]:
    time_coverage, fetch_success_rate, completeness, scoring_success_rate,
    schema_valid_rate. Missing components default to 1.0 (assume healthy);
    a window record with ``complete=False`` is never interpreted as low
    social activity -- it nulls absolute volume/author/spike features for
    any feature window it overlaps (plan §3).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import median, pstdev
from typing import Any, Sequence

from quantdesk.features.compute import feature_id
from quantdesk.features.sentiment_baseline import BaselineMeta, bucket_key, zscore

FEATURES_VERSION = "v1"

LOW_CONFIDENCE_THRESHOLD = 0.5
NEW_AUTHOR_AGE_DAYS = 30
TOP_N_AUTHORS = 5

_HEALTH_COMPONENTS = (
    "time_coverage",
    "fetch_success_rate",
    "completeness",
    "scoring_success_rate",
    "schema_valid_rate",
)


# --------------------------------------------------------------------------
# Row filtering / helpers
# --------------------------------------------------------------------------

def _is_eligible(row: dict, cutoff: datetime) -> bool:
    """available_to_strategy_time <= cutoff, no exclusions, canonical only."""
    ats = row.get("available_to_strategy_time")
    if ats is None or ats > cutoff:
        return False
    if row.get("exclusions"):
        return False
    canonical_id = row.get("canonical_tweet_id")
    if canonical_id is not None and canonical_id != row.get("tweet_id"):
        # Non-canonical duplicate: excluded from sentiment/volume features,
        # used only by (out-of-scope) propagation/campaign features.
        return False
    return True


def _in_window(row: dict, start: datetime, end: datetime) -> bool:
    et = row.get("event_time")
    return et is not None and start <= et <= end


def _window_rows(rows: Sequence[dict], cutoff: datetime, start: datetime, end: datetime) -> list[dict]:
    return [r for r in rows if _is_eligible(r, cutoff) and _in_window(r, start, end)]


def _non_ambiguous(rows: Sequence[dict]) -> list[dict]:
    """Exclude multi-asset-ambiguous tweets from asset-specific means (plan §6).

    target_ambiguity is "single" | "multi_asset" | "none" (tweet_normalize);
    only "multi_asset" is excluded here.
    """
    return [r for r in rows if r.get("target_ambiguity") != "multi_asset"]


def _scored(rows: Sequence[dict]) -> list[dict]:
    """Rows with a usable sentiment score (excludes unscored failures)."""
    return [r for r in rows if r.get("sentiment") is not None]


def _author_median_sentiment(rows: Sequence[dict]) -> dict[str, float]:
    """Per-author median sentiment (author-first aggregation)."""
    by_author: dict[str, list[float]] = {}
    for r in rows:
        by_author.setdefault(r["author_id"], []).append(r["sentiment"])
    return {author: median(vals) for author, vals in by_author.items()}


def _population_mean(rows: Sequence[dict]) -> float | None:
    """median of per-author medians -- author-first aggregation."""
    author_medians = _author_median_sentiment(_scored(_non_ambiguous(rows)))
    if not author_medians:
        return None
    return median(author_medians.values())


def _author_log_follower_weights(rows: Sequence[dict]) -> dict[str, float]:
    """Per-author weight = 1 + log1p(max followers observed in window).

    The +1 floor keeps unknown/zero-follower authors in the aggregate; log
    compression keeps a 1M-follower account at ~3x a 1k-follower one rather
    than 1000x, limiting how much reach can buy.
    """
    followers: dict[str, float] = {}
    for r in rows:
        f = r.get("author_followers")
        if f is None:
            f = 0
        prev = followers.get(r["author_id"], 0.0)
        followers[r["author_id"]] = max(prev, float(f))
    return {a: 1.0 + math.log1p(max(f, 0.0)) for a, f in followers.items()}


def _population_weighted_mean(rows: Sequence[dict]) -> float | None:
    """Shadow variant of _population_mean: per-author median (same flood
    resistance), then follower-log-weighted mean across authors instead of
    the median. Research-only until prospective scoring says otherwise."""
    eligible = _scored(_non_ambiguous(rows))
    author_medians = _author_median_sentiment(eligible)
    if not author_medians:
        return None
    weights = _author_log_follower_weights(eligible)
    total = sum(weights[a] for a in author_medians)
    if total <= 0:
        return None
    return sum(m * weights[a] for a, m in author_medians.items()) / total


def _meets_minimums(rows: Sequence[dict], config: dict) -> bool:
    features_cfg = config.get("features", {})
    min_tweets = features_cfg.get("min_tweets_per_window", 1)
    min_authors = features_cfg.get("min_authors_per_window", 1)
    eligible = _scored(_non_ambiguous(rows))
    n_authors = len({r["author_id"] for r in eligible})
    return len(eligible) >= min_tweets and n_authors >= min_authors


# --------------------------------------------------------------------------
# Query-window completeness
# --------------------------------------------------------------------------

def _overlapping_windows(query_windows: Sequence[dict], start: datetime, end: datetime) -> list[dict]:
    out = []
    for qw in query_windows:
        qs, qe = qw.get("window_start"), qw.get("window_end")
        if qs is None or qe is None:
            continue
        if qs < end and qe > start:
            out.append(qw)
    return out


def _window_complete(query_windows: Sequence[dict], start: datetime, end: datetime) -> bool:
    overlapping = _overlapping_windows(query_windows, start, end)
    if not overlapping:
        # No record at all: never interpreted as low activity, but we also
        # cannot claim completeness -- treat as incomplete (fail closed).
        return False
    return all(qw.get("complete", False) for qw in overlapping)


def _health_component(query_windows: Sequence[dict], start: datetime, end: datetime, key: str) -> float:
    overlapping = _overlapping_windows(query_windows, start, end)
    if not overlapping:
        return 0.0
    values = [qw.get(key, 1.0) for qw in overlapping]
    return sum(values) / len(values)


def compute_data_health(query_windows: Sequence[dict], start: datetime, end: datetime) -> dict[str, float]:
    """data_health = min(components); all components returned for diagnosis."""
    components = {
        key: _health_component(query_windows, start, end, key) for key in _HEALTH_COMPONENTS
    }
    components["data_health"] = min(components.values()) if components else 0.0
    return components


# --------------------------------------------------------------------------
# Information-quality helpers
# --------------------------------------------------------------------------

def _entropy(p_neg: float, p_neu: float, p_pos: float) -> float:
    total = 0.0
    for p in (p_neg, p_neu, p_pos):
        if p and p > 0:
            total -= p * math.log(p)
    return total


def _label(row: dict) -> str:
    probs = {
        "negative": row.get("p_negative") or 0.0,
        "neutral": row.get("p_neutral") or 0.0,
        "positive": row.get("p_positive") or 0.0,
    }
    return max(probs, key=probs.get)


def _hhi_and_effective_authors(rows: Sequence[dict]) -> tuple[float | None, float | None]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["author_id"]] = counts.get(r["author_id"], 0) + 1
    total = sum(counts.values())
    if total == 0:
        return None, None
    hhi = sum((c / total) ** 2 for c in counts.values())
    effective_authors = 1.0 / hhi if hhi > 0 else None
    return hhi, effective_authors


def _top5_weight_share(rows: Sequence[dict]) -> float | None:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["author_id"]] = counts.get(r["author_id"], 0) + 1
    total = sum(counts.values())
    if total == 0:
        return None
    top = sorted(counts.values(), reverse=True)[:TOP_N_AUTHORS]
    return sum(top) / total


def _new_author_fraction(rows: Sequence[dict]) -> float | None:
    ages: dict[str, Any] = {}
    for r in rows:
        ages.setdefault(r["author_id"], r.get("author_age_days"))
    known = {a: v for a, v in ages.items() if v is not None}
    if not known:
        return None
    new_count = sum(1 for v in known.values() if v < NEW_AUTHOR_AGE_DAYS)
    return new_count / len(known)


def _median_author_age(rows: Sequence[dict]) -> float | None:
    ages: dict[str, Any] = {}
    for r in rows:
        ages.setdefault(r["author_id"], r.get("author_age_days"))
    known = [v for v in ages.values() if v is not None]
    if not known:
        return None
    return median(known)


# --------------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------------

def compute_sentiment_features(
    asset: str,
    rows: Sequence[dict],
    query_windows: Sequence[dict],
    now: datetime,
    config: dict,
    baseline: dict[str, dict[int, BaselineMeta]] | None = None,
) -> dict[str, Any]:
    """Compute the full versioned sentiment feature set for one asset.

    ``baseline`` maps ``{"volume": {bucket: BaselineMeta}, "authors": {...}}``
    (see ``quantdesk.features.sentiment_baseline.build_seasonal_baseline``).
    """
    now = now.astimezone(timezone.utc)
    cutoff = now
    baseline = baseline or {}
    features_cfg = config.get("features", {})
    health_threshold = features_cfg.get("data_health_null_threshold", 0.8)

    out: dict[str, Any] = {}

    def fid(name: str) -> str:
        return feature_id(asset, name, FEATURES_VERSION)

    # ---- Windows -----------------------------------------------------
    w1h_start, w1h_end = now - timedelta(hours=1), now
    w24h_start, w24h_end = now - timedelta(hours=24), now
    delta_recent_start, delta_recent_end = now - timedelta(hours=1), now
    delta_prior_start, delta_prior_end = now - timedelta(hours=7), now - timedelta(hours=6)

    # ---- Data health (computed over the 24h window, most conservative) --
    health = compute_data_health(query_windows, w24h_start, w24h_end)
    for key, value in health.items():
        out[fid(key)] = value

    if health["data_health"] < health_threshold:
        out[fid("quality_flags")] = ["low_data_health"]
        # All sentiment features null; health components above still emitted.
        for name in _ALL_SENTIMENT_FEATURE_NAMES:
            out[fid(name)] = None
        return out

    quality_flags: list[str] = []

    # ---- Rows per window ----------------------------------------------
    rows_1h = _window_rows(rows, cutoff, w1h_start, w1h_end)
    rows_24h = _window_rows(rows, cutoff, w24h_start, w24h_end)
    rows_delta_recent = _window_rows(rows, cutoff, delta_recent_start, delta_recent_end)
    rows_delta_prior = _window_rows(rows, cutoff, delta_prior_start, delta_prior_end)

    complete_1h = _window_complete(query_windows, w1h_start, w1h_end)
    complete_24h = _window_complete(query_windows, w24h_start, w24h_end)

    # ---- Volume / spike (null when contributing window incomplete) ----
    if complete_1h:
        out[fid("tweet_volume_1h")] = len(rows_1h)
    else:
        out[fid("tweet_volume_1h")] = None
        quality_flags.append("incomplete_window_1h")

    if complete_24h:
        out[fid("tweet_volume_24h")] = len(rows_24h)
    else:
        out[fid("tweet_volume_24h")] = None
        quality_flags.append("incomplete_window_24h")

    n_authors_1h = len({r["author_id"] for r in rows_1h})
    bucket = bucket_key(now)
    volume_baseline = (baseline.get("volume") or {}).get(bucket)
    authors_baseline = (baseline.get("authors") or {}).get(bucket)

    if complete_1h:
        vol_z, vol_meta = zscore(float(len(rows_1h)), volume_baseline)
        authors_z, authors_meta = zscore(float(n_authors_1h), authors_baseline)
    else:
        vol_z, vol_meta = None, {}
        authors_z, authors_meta = None, {}

    out[fid("volume_zscore")] = vol_z
    out[fid("volume_zscore_baseline")] = vol_meta
    out[fid("authors_zscore")] = authors_z
    out[fid("authors_zscore_baseline")] = authors_meta

    spike_cfg = features_cfg
    spike_vol_thresh = spike_cfg.get("spike_volume_z", 3.0)
    spike_authors_thresh = spike_cfg.get("spike_authors_z", 2.0)
    if complete_1h and vol_z is not None and authors_z is not None:
        out[fid("spike_flag")] = bool(vol_z > spike_vol_thresh and authors_z > spike_authors_thresh)
    else:
        out[fid("spike_flag")] = None

    # ---- Sentiment means (per population), independent of completeness -
    for population in ("broad", "curated"):
        pop_rows_1h = [r for r in rows_1h if r.get("source_class") == population]
        pop_rows_24h = [r for r in rows_24h if r.get("source_class") == population]
        out[fid(f"sent_mean_1h_{population}")] = _population_mean(pop_rows_1h)
        out[fid(f"sent_mean_24h_{population}")] = _population_mean(pop_rows_24h)
        # Shadow follower-weighted variants (not consumed by any advisor yet).
        out[fid(f"sent_mean_1h_{population}_weighted")] = _population_weighted_mean(pop_rows_1h)
        out[fid(f"sent_mean_24h_{population}_weighted")] = _population_weighted_mean(pop_rows_24h)

    # sent_dispersion_1h: population-combined dispersion of per-author medians
    combined_author_medians = _author_median_sentiment(_scored(_non_ambiguous(rows_1h)))
    if len(combined_author_medians) >= 2:
        out[fid("sent_dispersion_1h")] = pstdev(combined_author_medians.values())
    else:
        out[fid("sent_dispersion_1h")] = None

    # sent_delta_6h = mean([t-1h,t]) - mean([t-7h,t-6h]) if both windows
    # meet minimum tweet/author counts.
    if _meets_minimums(rows_delta_recent, config) and _meets_minimums(rows_delta_prior, config):
        recent_mean = _population_mean(rows_delta_recent)
        prior_mean = _population_mean(rows_delta_prior)
        if recent_mean is not None and prior_mean is not None:
            out[fid("sent_delta_6h")] = recent_mean - prior_mean
        else:
            out[fid("sent_delta_6h")] = None
    else:
        out[fid("sent_delta_6h")] = None

    # ---- Information-quality (1h window, all populations combined) -----
    eligible_1h = _non_ambiguous(rows_1h)
    scored_1h = _scored(eligible_1h)

    if scored_1h:
        labels = [_label(r) for r in scored_1h]
        out[fid("positive_fraction")] = labels.count("positive") / len(labels)
        out[fid("negative_fraction")] = labels.count("negative") / len(labels)
        out[fid("neutral_probability_mean")] = sum(r.get("p_neutral") or 0.0 for r in scored_1h) / len(scored_1h)
        out[fid("sentiment_entropy_mean")] = sum(
            _entropy(r.get("p_negative") or 0.0, r.get("p_neutral") or 0.0, r.get("p_positive") or 0.0)
            for r in scored_1h
        ) / len(scored_1h)
    else:
        out[fid("positive_fraction")] = None
        out[fid("negative_fraction")] = None
        out[fid("neutral_probability_mean")] = None
        out[fid("sentiment_entropy_mean")] = None

    if eligible_1h:
        with_confidence = [r for r in eligible_1h if r.get("scorer_confidence") is not None]
        if with_confidence:
            low_conf = sum(1 for r in with_confidence if r["scorer_confidence"] < LOW_CONFIDENCE_THRESHOLD)
            out[fid("low_confidence_fraction")] = low_conf / len(with_confidence)
        else:
            out[fid("low_confidence_fraction")] = None
    else:
        out[fid("low_confidence_fraction")] = None

    hhi, effective_authors = _hhi_and_effective_authors(eligible_1h)
    out[fid("author_concentration_1h")] = hhi
    out[fid("effective_authors_1h")] = effective_authors
    out[fid("new_author_fraction_1h")] = _new_author_fraction(eligible_1h)
    out[fid("median_author_age_days")] = _median_author_age(eligible_1h)
    out[fid("top5_author_weight_share_1h")] = _top5_weight_share(eligible_1h)

    out[fid("quality_flags")] = quality_flags
    return out


_ALL_SENTIMENT_FEATURE_NAMES = (
    "tweet_volume_1h",
    "tweet_volume_24h",
    "volume_zscore",
    "volume_zscore_baseline",
    "authors_zscore",
    "authors_zscore_baseline",
    "spike_flag",
    "sent_mean_1h_broad",
    "sent_mean_1h_curated",
    "sent_mean_24h_broad",
    "sent_mean_24h_curated",
    "sent_mean_1h_broad_weighted",
    "sent_mean_1h_curated_weighted",
    "sent_mean_24h_broad_weighted",
    "sent_mean_24h_curated_weighted",
    "sent_dispersion_1h",
    "sent_delta_6h",
    "positive_fraction",
    "negative_fraction",
    "neutral_probability_mean",
    "low_confidence_fraction",
    "sentiment_entropy_mean",
    "author_concentration_1h",
    "effective_authors_1h",
    "new_author_fraction_1h",
    "median_author_age_days",
    "top5_author_weight_share_1h",
)


# --------------------------------------------------------------------------
# Snapshot block: compact features + up to 5 sample tweets for the advisor
# --------------------------------------------------------------------------

def build_sentiment_snapshot_block(
    features: dict[str, Any],
    sample_rows: Sequence[dict],
    config: dict,
) -> dict[str, Any]:
    """Build the compact snapshot block: features + up to 5 sample tweets.

    Selection rules (plan §7): max one per author, quotas across broad
    (3) / curated (2), canonical only, no duplicate campaigns, stratified by
    sentiment tercile (not top-engagement). Must work with zero samples.
    """
    snapshot_cfg = config.get("snapshot_samples", {})
    max_samples = snapshot_cfg.get("max_samples", 5)
    include_samples = snapshot_cfg.get("include_samples", True)
    quotas = {"broad": 3, "curated": 2}

    # Canonical only, scored, no exclusions.
    candidates = [
        r
        for r in sample_rows
        if r.get("sentiment") is not None
        and not r.get("exclusions")
        and (r.get("canonical_tweet_id") in (None, r.get("tweet_id")))
    ]

    def tercile(sentiment: float) -> str:
        if sentiment <= -1 / 3:
            return "negative"
        if sentiment >= 1 / 3:
            return "positive"
        return "neutral"

    samples: list[dict] = []
    used_authors: set[str] = set()

    if include_samples:
        for population, quota in quotas.items():
            pop_candidates = [r for r in candidates if r.get("source_class") == population]
            by_tercile: dict[str, list[dict]] = {"negative": [], "neutral": [], "positive": []}
            for r in pop_candidates:
                by_tercile[tercile(r["sentiment"])].append(r)

            selected_for_pop = 0
            tercile_order = ["negative", "neutral", "positive"]
            idx = 0
            attempts = 0
            # Round-robin across terciles, skipping authors already used.
            while selected_for_pop < quota and attempts < quota * 10:
                tname = tercile_order[idx % len(tercile_order)]
                idx += 1
                attempts += 1
                bucket_list = by_tercile[tname]
                chosen = None
                for r in bucket_list:
                    if r["author_id"] not in used_authors:
                        chosen = r
                        break
                if chosen is not None:
                    bucket_list.remove(chosen)
                    used_authors.add(chosen["author_id"])
                    samples.append(
                        {
                            "tweet_id": chosen["tweet_id"],
                            "author_id": chosen["author_id"],
                            "source_class": population,
                            "sentiment": chosen["sentiment"],
                            "display_text_for_llm": chosen.get("display_text_for_llm", ""),
                        }
                    )
                    selected_for_pop += 1
                if not any(by_tercile.values()):
                    break

    samples = samples[:max_samples]

    return {
        "features": features,
        "samples": samples,
        "include_samples": include_samples and len(samples) > 0,
    }
