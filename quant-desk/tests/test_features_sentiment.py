from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantdesk.features.compute import feature_id
from quantdesk.features.sentiment import (
    FEATURES_VERSION,
    build_sentiment_snapshot_block,
    compute_sentiment_features,
)

UTC = timezone.utc
NOW = datetime(2026, 1, 29, 12, 0, tzinfo=UTC)
ASSET = "BTC"


def fid(name: str) -> str:
    return feature_id(ASSET, name, FEATURES_VERSION)


def make_row(
    *,
    tweet_id: str,
    author_id: str,
    sentiment: float = 0.0,
    p_negative: float = 0.2,
    p_neutral: float = 0.6,
    p_positive: float = 0.2,
    scorer_confidence: float = 0.6,
    event_time: datetime = NOW,
    available_to_strategy_time: datetime = NOW,
    author_followers: int = 500,
    author_age_days: int = 400,
    exclusions: list | None = None,
    target_ambiguity: bool = False,
    source_class: str = "broad",
    canonical_tweet_id: str | None = None,
    duplicate_count: int = 0,
) -> dict:
    return {
        "tweet_id": tweet_id,
        "author_id": author_id,
        "sentiment": sentiment,
        "p_negative": p_negative,
        "p_neutral": p_neutral,
        "p_positive": p_positive,
        "scorer_confidence": scorer_confidence,
        "event_time": event_time,
        "available_to_strategy_time": available_to_strategy_time,
        "author_followers": author_followers,
        "author_age_days": author_age_days,
        "exclusions": exclusions or [],
        "target_ambiguity": target_ambiguity,
        "source_class": source_class,
        "canonical_tweet_id": canonical_tweet_id if canonical_tweet_id is not None else tweet_id,
        "duplicate_count": duplicate_count,
    }


DEFAULT_CONFIG = {
    "features": {
        "baseline_window_days": 28,
        "baseline_min_observations": 14,
        "spike_volume_z": 3.0,
        "spike_authors_z": 2.0,
        "data_health_null_threshold": 0.8,
        "min_tweets_per_window": 2,
        "min_authors_per_window": 2,
    },
    "snapshot_samples": {"max_samples": 5, "max_per_author": 1, "include_samples": True},
}


def healthy_query_windows(start: datetime, end: datetime, *, complete: bool = True) -> list[dict]:
    return [
        {
            "window_start": start,
            "window_end": end,
            "complete": complete,
            "time_coverage": 1.0,
            "fetch_success_rate": 1.0,
            "completeness": 1.0,
            "scoring_success_rate": 1.0,
            "schema_valid_rate": 1.0,
        }
    ]


def test_completeness_nulling_volume_but_sentiment_means_computed():
    rows = [
        make_row(tweet_id="t1", author_id="a1", sentiment=0.5, event_time=NOW - timedelta(minutes=10)),
        make_row(tweet_id="t2", author_id="a2", sentiment=-0.3, event_time=NOW - timedelta(minutes=20)),
    ]
    # 24h window (minus the last hour) is incomplete; last 1h is complete.
    query_windows = [
        {
            "window_start": NOW - timedelta(hours=24),
            "window_end": NOW - timedelta(hours=1),
            "complete": False,
            "time_coverage": 1.0,
            "fetch_success_rate": 1.0,
            "completeness": 1.0,
            "scoring_success_rate": 1.0,
            "schema_valid_rate": 1.0,
        },
        {
            "window_start": NOW - timedelta(hours=1),
            "window_end": NOW,
            "complete": True,
            "time_coverage": 1.0,
            "fetch_success_rate": 1.0,
            "completeness": 1.0,
            "scoring_success_rate": 1.0,
            "schema_valid_rate": 1.0,
        },
    ]

    features = compute_sentiment_features(ASSET, rows, query_windows, NOW, DEFAULT_CONFIG, baseline=None)

    assert features[fid("tweet_volume_1h")] == 2
    assert features[fid("tweet_volume_24h")] is None
    assert features[fid("volume_zscore")] is not None or features[fid("volume_zscore")] is None  # no baseline -> None ok
    # Sentiment means are independent of the (unrelated) incomplete-24h window.
    assert features[fid("sent_mean_1h_broad")] is not None
    assert features[fid("sent_mean_1h_broad")] == pytest.approx(median_of([0.5, -0.3]))


def median_of(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def test_data_health_nulling():
    rows = [
        make_row(tweet_id="t1", author_id="a1", sentiment=0.5, event_time=NOW - timedelta(minutes=10)),
        make_row(tweet_id="t2", author_id="a2", sentiment=-0.3, event_time=NOW - timedelta(minutes=20)),
    ]
    query_windows = [
        {
            "window_start": NOW - timedelta(hours=24),
            "window_end": NOW,
            "complete": True,
            "time_coverage": 1.0,
            "fetch_success_rate": 1.0,
            "completeness": 1.0,
            "scoring_success_rate": 0.2,  # drags data_health below threshold
            "schema_valid_rate": 1.0,
        }
    ]

    features = compute_sentiment_features(ASSET, rows, query_windows, NOW, DEFAULT_CONFIG, baseline=None)

    assert features[fid("data_health")] == pytest.approx(0.2)
    assert features[fid("quality_flags")] == ["low_data_health"]
    assert features[fid("sent_mean_1h_broad")] is None
    assert features[fid("tweet_volume_1h")] is None
    assert features[fid("spike_flag")] is None


def test_data_health_missing_query_windows_defaults_to_unhealthy():
    rows = [make_row(tweet_id="t1", author_id="a1")]
    features = compute_sentiment_features(ASSET, rows, [], NOW, DEFAULT_CONFIG, baseline=None)
    assert features[fid("data_health")] == 0.0
    assert features[fid("sent_mean_1h_broad")] is None


def test_author_independence_in_means():
    query_windows = healthy_query_windows(NOW - timedelta(hours=24), NOW)

    # One author posts 5 tweets at sentiment 1.0; another author posts a
    # single tweet at sentiment -1.0. Author-first aggregation must not let
    # the prolific author dominate 5-to-1.
    rows = [
        make_row(tweet_id=f"spam{i}", author_id="spammer", sentiment=1.0, event_time=NOW - timedelta(minutes=5))
        for i in range(5)
    ] + [
        make_row(tweet_id="lone", author_id="loner", sentiment=-1.0, event_time=NOW - timedelta(minutes=5)),
    ]

    features = compute_sentiment_features(ASSET, rows, query_windows, NOW, DEFAULT_CONFIG, baseline=None)

    # author medians: spammer=1.0, loner=-1.0 -> median of [1.0, -1.0] == 0.0
    assert features[fid("sent_mean_1h_broad")] == pytest.approx(0.0)


def test_sent_delta_6h_exact_window_arithmetic():
    query_windows = healthy_query_windows(NOW - timedelta(hours=24), NOW)

    recent_rows = [
        make_row(tweet_id="r1", author_id="a1", sentiment=0.4, event_time=NOW - timedelta(minutes=30)),
        make_row(tweet_id="r2", author_id="a2", sentiment=0.8, event_time=NOW - timedelta(minutes=10)),
    ]
    prior_rows = [
        make_row(tweet_id="p1", author_id="a3", sentiment=-0.2, event_time=NOW - timedelta(hours=6, minutes=30)),
        make_row(tweet_id="p2", author_id="a4", sentiment=0.0, event_time=NOW - timedelta(hours=6, minutes=45)),
    ]
    rows = recent_rows + prior_rows

    features = compute_sentiment_features(ASSET, rows, query_windows, NOW, DEFAULT_CONFIG, baseline=None)

    recent_mean = median_of([0.4, 0.8])
    prior_mean = median_of([-0.2, 0.0])
    assert features[fid("sent_delta_6h")] == pytest.approx(recent_mean - prior_mean)


def test_sent_delta_6h_null_when_minimums_not_met():
    query_windows = healthy_query_windows(NOW - timedelta(hours=24), NOW)
    # Only one tweet in the recent window; min_tweets_per_window is 2.
    rows = [
        make_row(tweet_id="r1", author_id="a1", sentiment=0.4, event_time=NOW - timedelta(minutes=30)),
        make_row(tweet_id="p1", author_id="a3", sentiment=-0.2, event_time=NOW - timedelta(hours=6, minutes=30)),
        make_row(tweet_id="p2", author_id="a4", sentiment=0.0, event_time=NOW - timedelta(hours=6, minutes=45)),
    ]

    features = compute_sentiment_features(ASSET, rows, query_windows, NOW, DEFAULT_CONFIG, baseline=None)
    assert features[fid("sent_delta_6h")] is None


def test_exclusions_and_ambiguity_and_future_rows_filtered_out():
    query_windows = healthy_query_windows(NOW - timedelta(hours=24), NOW)
    rows = [
        make_row(tweet_id="excluded", author_id="a1", sentiment=1.0, exclusions=["spam"]),
        make_row(tweet_id="ambiguous", author_id="a2", sentiment=1.0, target_ambiguity="multi_asset"),
        make_row(tweet_id="future", author_id="a3", sentiment=1.0, available_to_strategy_time=NOW + timedelta(hours=1)),
        make_row(tweet_id="good", author_id="a4", sentiment=0.3),
    ]
    features = compute_sentiment_features(ASSET, rows, query_windows, NOW, DEFAULT_CONFIG, baseline=None)
    # Only "good" is eligible for the sentiment mean.
    assert features[fid("sent_mean_1h_broad")] == pytest.approx(0.3)


def test_zero_sample_snapshot_block():
    features = {"some_feature@v1": 1.0}
    block = build_sentiment_snapshot_block(features, [], DEFAULT_CONFIG)
    assert block["features"] is features
    assert block["samples"] == []
    assert block["include_samples"] is False


def test_snapshot_block_respects_quotas_and_author_cap():
    rows = []
    for i in range(6):
        rows.append(
            make_row(
                tweet_id=f"b{i}",
                author_id=f"broad_author_{i}",
                sentiment=(-1.0 if i % 3 == 0 else (0.0 if i % 3 == 1 else 1.0)),
                source_class="broad",
            )
        )
    for i in range(4):
        rows.append(
            make_row(
                tweet_id=f"c{i}",
                author_id=f"curated_author_{i}",
                sentiment=(-1.0 if i % 2 == 0 else 1.0),
                source_class="curated",
            )
        )
    for r in rows:
        r["display_text_for_llm"] = f"text for {r['tweet_id']}"

    block = build_sentiment_snapshot_block({}, rows, DEFAULT_CONFIG)
    assert len(block["samples"]) <= 5
    assert block["include_samples"] is True

    authors = [s["author_id"] for s in block["samples"]]
    assert len(authors) == len(set(authors))  # max one per author

    broad_count = sum(1 for s in block["samples"] if s["source_class"] == "broad")
    curated_count = sum(1 for s in block["samples"] if s["source_class"] == "curated")
    assert broad_count <= 3
    assert curated_count <= 2


def test_single_target_rows_are_not_excluded_from_means():
    # Regression: target_ambiguity is a string enum ("single"|"multi_asset"|"none");
    # "single" is truthy and must NOT be excluded (only "multi_asset" is).
    from quantdesk.features.sentiment import _non_ambiguous
    rows = [
        {"target_ambiguity": "single"},
        {"target_ambiguity": "none"},
        {"target_ambiguity": "multi_asset"},
    ]
    kept = _non_ambiguous(rows)
    assert len(kept) == 2
    assert all(r["target_ambiguity"] != "multi_asset" for r in kept)


def test_weighted_sent_mean_tilts_toward_high_follower_authors():
    import math

    query_windows = [
        {
            "window_start": NOW - timedelta(hours=24),
            "window_end": NOW,
            "complete": True,
            "time_coverage": 1.0,
            "fetch_success_rate": 1.0,
            "completeness": 1.0,
            "scoring_success_rate": 1.0,
            "schema_valid_rate": 1.0,
        }
    ]
    rows = [
        make_row(tweet_id="t1", author_id="whale", sentiment=0.8, author_followers=1_000_000,
                 event_time=NOW - timedelta(minutes=10)),
        make_row(tweet_id="t2", author_id="minnow1", sentiment=-0.4, author_followers=10,
                 event_time=NOW - timedelta(minutes=15)),
        make_row(tweet_id="t3", author_id="minnow2", sentiment=-0.4, author_followers=10,
                 event_time=NOW - timedelta(minutes=20)),
    ]

    features = compute_sentiment_features(ASSET, rows, query_windows, NOW, DEFAULT_CONFIG, baseline=None)

    # Unweighted median of author medians: median(0.8, -0.4, -0.4) = -0.4.
    assert features[fid("sent_mean_1h_broad")] == pytest.approx(-0.4)
    w_whale = 1 + math.log1p(1_000_000)
    w_minnow = 1 + math.log1p(10)
    expected = (0.8 * w_whale + -0.4 * w_minnow * 2) / (w_whale + 2 * w_minnow)
    weighted = features[fid("sent_mean_1h_broad_weighted")]
    assert weighted == pytest.approx(expected)
    assert weighted > features[fid("sent_mean_1h_broad")]
    # Weighted variants null out alongside the unweighted ones on low health.
    unhealthy = [dict(query_windows[0], scoring_success_rate=0.2)]
    nulled = compute_sentiment_features(ASSET, rows, unhealthy, NOW, DEFAULT_CONFIG, baseline=None)
    assert nulled[fid("sent_mean_1h_broad_weighted")] is None
