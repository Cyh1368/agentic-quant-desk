from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantdesk.features.sentiment_baseline import (
    BASELINE_VERSION,
    SCALE_EPSILON,
    bucket_key,
    build_seasonal_baseline,
    zscore,
)

UTC = timezone.utc


def dt(days_ago: float, hour: int = 12, minute: int = 0) -> datetime:
    base = datetime(2026, 1, 29, hour, minute, tzinfo=UTC)
    return base - timedelta(days=days_ago)


def test_bucket_key_15_minutes():
    assert bucket_key(datetime(2026, 1, 1, 0, 0, tzinfo=UTC)) == 0
    assert bucket_key(datetime(2026, 1, 1, 0, 14, tzinfo=UTC)) == 0
    assert bucket_key(datetime(2026, 1, 1, 0, 15, tzinfo=UTC)) == 1
    assert bucket_key(datetime(2026, 1, 1, 12, 0, tzinfo=UTC)) == 48
    assert bucket_key(datetime(2026, 1, 1, 23, 45, tzinfo=UTC)) == 95


def test_seasonal_baseline_median_mad_hand_verified():
    now = datetime(2026, 1, 29, 12, 0, tzinfo=UTC)
    # 14 observations at the same 12:00 bucket, one per day for 14 days.
    values = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 100]
    history = [(now - timedelta(days=i + 1), v) for i, v in enumerate(values)]

    baselines = build_seasonal_baseline(history, now, min_observations=14)
    bucket = bucket_key(now)
    meta = baselines[bucket]

    assert meta.n == 14
    assert meta.source == "seasonal"
    # median of 13x10 + 1x100, sorted -> position 7 (0-indexed) among 14 = 10
    assert meta.center == 10
    # MAD: |10-10|*13, |100-10|=90 -> median of deviations = 0
    # scale floored to epsilon since MAD is 0.
    assert meta.scale == pytest.approx(SCALE_EPSILON)
    assert meta.version == BASELINE_VERSION


def test_seasonal_baseline_insufficient_observations_falls_back_to_ewma():
    now = datetime(2026, 1, 29, 12, 0, tzinfo=UTC)
    values = [5, 6, 7]  # fewer than min_observations
    history = [(now - timedelta(days=i + 1), v) for i, v in enumerate(values)]

    baselines = build_seasonal_baseline(history, now, min_observations=14)
    bucket = bucket_key(now)
    meta = baselines[bucket]

    assert meta.n == 3
    assert meta.source == "ewma_fallback"


def test_baseline_excludes_observations_outside_window():
    now = datetime(2026, 1, 29, 12, 0, tzinfo=UTC)
    in_window = [(now - timedelta(days=i + 1), 10.0) for i in range(14)]
    out_of_window = [(now - timedelta(days=40), 999.0)]
    baselines = build_seasonal_baseline(in_window + out_of_window, now, min_observations=14)
    bucket = bucket_key(now)
    assert baselines[bucket].n == 14


def test_zscore_with_valid_baseline():
    now = datetime(2026, 1, 29, 12, 0, tzinfo=UTC)
    values = list(range(1, 29))  # 1..28, median-ish center
    history = [(now - timedelta(days=i + 1), float(v)) for i, v in enumerate(values)]
    baselines = build_seasonal_baseline(history, now, min_observations=14)
    bucket = bucket_key(now)
    meta = baselines[bucket]

    z, persisted_meta = zscore(meta.center + meta.scale * 2, meta)
    assert z == pytest.approx(2.0)
    assert persisted_meta["version"] == BASELINE_VERSION
    assert persisted_meta["n"] == meta.n


def test_zscore_none_when_no_baseline():
    z, meta = zscore(5.0, None)
    assert z is None
    assert meta["n"] == 0


def test_zscore_none_when_value_is_none():
    now = datetime(2026, 1, 29, 12, 0, tzinfo=UTC)
    values = [10.0] * 14
    history = [(now - timedelta(days=i + 1), v) for i, v in enumerate(values)]
    baselines = build_seasonal_baseline(history, now, min_observations=14)
    bucket = bucket_key(now)
    z, meta = zscore(None, baselines[bucket])
    assert z is None
