"""Versioned robust seasonal baseline for sentiment volume/author z-scores
(plan §7).

Pure functions only -- no I/O. UTC 15-minute buckets, keyed by
``(hour, minute_bucket)`` within a day (i.e. time-of-day-of-week is *not*
distinguished; the bucket key is minute-of-day // 15, matching the plan's
"UTC 15-minute buckets, prior 28 calendar days" description). Robust center
= median, robust scale = 1.4826 * MAD (median absolute deviation), with a
tiny epsilon floor to avoid division by zero. Buckets with fewer than
``min_observations`` valid observations fall back to an exponentially
weighted intraday baseline computed from the same history.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import NamedTuple, Sequence

BASELINE_VERSION = "v1"

BUCKET_MINUTES = 15
DEFAULT_WINDOW_DAYS = 28
DEFAULT_MIN_OBSERVATIONS = 14
SCALE_EPSILON = 1e-6
EWMA_ALPHA = 0.3  # fallback intraday exponential weight


class BaselineMeta(NamedTuple):
    version: str
    window_days: int
    n: int
    center: float
    scale: float
    source: str  # "seasonal" | "ewma_fallback"


def bucket_key(ts: datetime) -> int:
    """Minute-of-day // 15, in UTC. Public so callers can index a baseline
    dict for a given timestamp (e.g. "now") without recomputing buckets."""
    ts = ts.astimezone(timezone.utc)
    minute_of_day = ts.hour * 60 + ts.minute
    return minute_of_day // BUCKET_MINUTES


_bucket_key = bucket_key  # backwards-compat alias for internal callers


def _mad(values: Sequence[float], center: float) -> float:
    deviations = [abs(v - center) for v in values]
    return median(deviations)


def build_seasonal_baseline(
    history: Sequence[tuple[datetime, float]],
    now: datetime,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> dict[int, BaselineMeta]:
    """Build a per-bucket robust baseline from ``history``.

    ``history`` is a sequence of ``(bucket_start_utc, value)`` pairs. Only
    observations within the prior ``window_days`` calendar days (relative to
    ``now``) are used. Returns a dict keyed by bucket index (0..95) ->
    ``BaselineMeta``. Buckets with fewer than ``min_observations`` valid
    observations get an exponentially weighted intraday fallback baseline
    computed from all in-window observations for that bucket (or, if there
    are none at all, an empty/degenerate baseline with n=0).
    """
    now = now.astimezone(timezone.utc)
    window_start = now - timedelta(days=window_days)

    by_bucket: dict[int, list[float]] = {}
    for ts, value in history:
        ts = ts.astimezone(timezone.utc)
        if ts < window_start or ts > now:
            continue
        if value is None:
            continue
        bucket = _bucket_key(ts)
        by_bucket.setdefault(bucket, []).append(float(value))

    result: dict[int, BaselineMeta] = {}
    for bucket, values in by_bucket.items():
        n = len(values)
        if n >= min_observations:
            center = median(values)
            scale = max(1.4826 * _mad(values, center), SCALE_EPSILON)
            result[bucket] = BaselineMeta(
                version=BASELINE_VERSION,
                window_days=window_days,
                n=n,
                center=center,
                scale=scale,
                source="seasonal",
            )
        else:
            center, scale = _ewma_baseline(values)
            result[bucket] = BaselineMeta(
                version=BASELINE_VERSION,
                window_days=window_days,
                n=n,
                center=center,
                scale=scale,
                source="ewma_fallback",
            )
    return result


def _ewma_baseline(values: Sequence[float]) -> tuple[float, float]:
    """Exponentially weighted mean/std fallback for sparse buckets.

    ``values`` should be in chronological order (oldest first) as produced
    by iterating ``history`` in order upstream; callers are responsible for
    ordering. If empty, returns (0.0, SCALE_EPSILON) as a degenerate
    baseline (z-score callers should treat n=0 as "no baseline").
    """
    if not values:
        return 0.0, SCALE_EPSILON
    mean = values[0]
    var = 0.0
    for v in values[1:]:
        delta = v - mean
        mean += EWMA_ALPHA * delta
        var = (1 - EWMA_ALPHA) * (var + EWMA_ALPHA * delta * delta)
    scale = max(math.sqrt(var), SCALE_EPSILON)
    return mean, scale


def zscore(
    value: float,
    bucket_meta: BaselineMeta | None,
) -> tuple[float | None, dict]:
    """Compute a robust z-score for ``value`` against a bucket baseline.

    Returns ``(z, baseline_meta_dict)``. ``z`` is None when there is no
    baseline for the bucket (n == 0) or value is None; ``baseline_meta_dict``
    is always returned (with n=0 / center=0 / scale=epsilon when degenerate)
    so it can be persisted alongside the (possibly null) z-score.
    """
    if bucket_meta is None:
        meta = {
            "version": BASELINE_VERSION,
            "window_days": DEFAULT_WINDOW_DAYS,
            "n": 0,
            "center": 0.0,
            "scale": SCALE_EPSILON,
            "source": "none",
        }
        return None, meta

    meta = bucket_meta._asdict()
    if value is None or bucket_meta.n == 0:
        return None, meta

    z = (value - bucket_meta.center) / bucket_meta.scale
    return z, meta
