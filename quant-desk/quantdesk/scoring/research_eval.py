"""Overlap-aware evaluation harness for research-only forecasts (plan §8-9).

Pure-function module: no ledger/store imports here. Callers pass in
already-materialized ``EvalRecord`` lists (e.g. joined forecast + outcome
rows read from ``quantdesk.ledger.research_store.ResearchStore``) and, where
persistence is needed, they own that separately.

Primary metric is Spearman IC (expected_excess_return_bps vs realized
excess log return) with a stationary block bootstrap for uncertainty and a
Newey-West HAC standard error, per plan §9's research contract. Secondary
metrics (brier_score, log_loss, calibration_slope/intercept,
balanced_accuracy, coverage) and a non-overlapping daily-cohort robustness
slice are also computed.

``by_source_class`` is intentionally left empty here: grouping by source
class (e.g. tweet vs news vs on-chain) requires evidence-feature-id
provenance plumbing that is out of scope for this task slice. TODO: wire up
once evidence_feature_ids carry a source-class tag.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

import numpy as np

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
except ImportError:  # pragma: no cover - scipy not available in this env
    _scipy_spearmanr = None


@dataclass(frozen=True)
class EvalRecord:
    """One resolved research forecast joined with its realized outcome."""

    instrument_id: str
    generated_at: datetime
    forecast_bps: float | None
    probability_positive: float | None
    direction: str
    abstain: bool
    realized_excess_log_return: float


# --------------------------------------------------------------------------
# Rank correlation (Spearman IC)
# --------------------------------------------------------------------------

def _rank(values: Sequence[float]) -> np.ndarray:
    """Average-tie ranks, 0-indexed float ranks."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    sorted_vals = arr[order]
    i = 0
    n = len(arr)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _non_abstain_pairs(records: Sequence[EvalRecord]) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for r in records:
        if r.abstain or r.forecast_bps is None:
            continue
        xs.append(r.forecast_bps)
        ys.append(r.realized_excess_log_return)
    return xs, ys


def spearman_ic(records: Sequence[EvalRecord]) -> float:
    """Spearman rank correlation between non-abstain forecast_bps and
    realized_excess_log_return. Returns 0.0 if fewer than 2 usable pairs or
    if either series is constant."""
    xs, ys = _non_abstain_pairs(records)
    if len(xs) < 2:
        return 0.0
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0
    if _scipy_spearmanr is not None:
        rho, _ = _scipy_spearmanr(xs, ys)
        return float(rho) if rho == rho else 0.0  # guard NaN
    rx = _rank(xs)
    ry = _rank(ys)
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------
# Stationary block bootstrap (Politis-Romano)
# --------------------------------------------------------------------------

def stationary_block_bootstrap_ci(
    records: Sequence[EvalRecord],
    *,
    statistic_fn: Callable[[Sequence[EvalRecord]], float],
    n_boot: int = 1000,
    seed: int,
    expected_block_length: float | None = None,
) -> tuple[float, float, float]:
    """Politis-Romano stationary bootstrap CI for ``statistic_fn`` over
    ``records``. Returns (point_estimate, ci_lo, ci_hi) at the 2.5/97.5
    percentiles. Deterministic given the same seed."""
    n = len(records)
    point = statistic_fn(records)
    if n < 2:
        return point, point, point

    if expected_block_length is None:
        expected_block_length = max(1.0, n ** (1.0 / 3.0))
    p = 1.0 / expected_block_length

    rng = np.random.default_rng(seed)
    items = list(records)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = 0
        resample_idx: list[int] = []
        start = int(rng.integers(0, n))
        resample_idx.append(start)
        idx = start
        while len(resample_idx) < n:
            if rng.random() < p:
                idx = int(rng.integers(0, n))
            else:
                idx = (idx + 1) % n
            resample_idx.append(idx)
        resample = [items[i] for i in resample_idx]
        stats.append(statistic_fn(resample))

    stats_arr = np.asarray(stats, dtype=float)
    ci_lo = float(np.percentile(stats_arr, 2.5))
    ci_hi = float(np.percentile(stats_arr, 97.5))
    return point, ci_lo, ci_hi


# --------------------------------------------------------------------------
# HAC / Newey-West standard error
# --------------------------------------------------------------------------

def hac_standard_error(
    x: Sequence[float], y: Sequence[float], *, max_lag: int | None = None
) -> float:
    """Newey-West HAC standard error for the mean of the elementwise
    product series ``product_i = x_i * y_i`` (a simple proxy for
    forecast/outcome co-movement), using a Bartlett kernel."""
    n = len(x)
    if n == 0:
        return 0.0
    product = np.asarray(x, dtype=float) * np.asarray(y, dtype=float)
    mean = product.mean()
    centered = product - mean
    if max_lag is None:
        max_lag = max(1, int(4 * (n / 100.0) ** (2.0 / 9.0)))
    max_lag = min(max_lag, n - 1)

    gamma0 = float(np.dot(centered, centered) / n)
    var = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma_lag = float(np.dot(centered[lag:], centered[:-lag]) / n)
        var += 2.0 * weight * gamma_lag
    var = max(var, 0.0)
    return math.sqrt(var / n)


def effective_sample_size(n_raw: int, avg_overlap: float = 1.0) -> float:
    """Overlap-adjusted effective sample size. avg_overlap is e.g.
    horizon/decision_interval; 1.0 (no adjustment) if unknown."""
    if avg_overlap <= 0:
        return float(n_raw)
    return n_raw / avg_overlap


def non_overlapping_daily_cohort(records: Sequence[EvalRecord]) -> list[EvalRecord]:
    """At most one record per (instrument_id, calendar day of generated_at),
    keeping the first record encountered per group (input order)."""
    seen: set[tuple[str, Any]] = set()
    out: list[EvalRecord] = []
    for r in records:
        key = (r.instrument_id, r.generated_at.date())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# --------------------------------------------------------------------------
# Calibration / classification metrics
# --------------------------------------------------------------------------

def _binary_outcome(r: EvalRecord) -> int:
    return 1 if r.realized_excess_log_return > 0 else 0


def brier_score(records: Sequence[EvalRecord]) -> float | None:
    scored = [r for r in records if r.probability_positive is not None]
    if not scored:
        return None
    total = sum((r.probability_positive - _binary_outcome(r)) ** 2 for r in scored)
    return total / len(scored)


def log_loss(records: Sequence[EvalRecord]) -> float | None:
    scored = [r for r in records if r.probability_positive is not None]
    if not scored:
        return None
    eps = 1e-12
    total = 0.0
    for r in scored:
        p = min(max(r.probability_positive, eps), 1 - eps)
        y = _binary_outcome(r)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(scored)


def calibration_slope_intercept(records: Sequence[EvalRecord]) -> tuple[float, float]:
    """Logistic regression of binary outcome on logit(probability_positive):
    outcome ~ sigmoid(intercept + slope * logit(p)). Implemented manually
    with Newton's method (no sklearn dependency)."""
    scored = [r for r in records if r.probability_positive is not None]
    if len(scored) < 2:
        return 1.0, 0.0

    eps = 1e-6
    logits = np.array(
        [
            math.log(min(max(r.probability_positive, eps), 1 - eps) / (1 - min(max(r.probability_positive, eps), 1 - eps)))
            for r in scored
        ]
    )
    y = np.array([_binary_outcome(r) for r in scored], dtype=float)

    if np.all(y == y[0]):
        return 1.0, 0.0

    intercept, slope = 0.0, 1.0
    for _ in range(50):
        z = intercept + slope * logits
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1 - p)
        w = np.clip(w, 1e-8, None)
        grad_intercept = np.sum(y - p)
        grad_slope = np.sum((y - p) * logits)
        h_ii = -np.sum(w)
        h_is = -np.sum(w * logits)
        h_ss = -np.sum(w * logits * logits)
        det = h_ii * h_ss - h_is * h_is
        if abs(det) < 1e-10:
            break
        d_intercept = -(h_ss * grad_intercept - h_is * grad_slope) / det
        d_slope = -(-h_is * grad_intercept + h_ii * grad_slope) / det
        intercept += d_intercept
        slope += d_slope
        if abs(d_intercept) < 1e-8 and abs(d_slope) < 1e-8:
            break
    return float(slope), float(intercept)


def balanced_accuracy(records: Sequence[EvalRecord]) -> float | None:
    """Directional balanced accuracy vs. sign of realized return, over
    non-abstain, non-flat directional calls. ``flat`` calls are excluded
    (treated as an abstention from the directional classification task,
    since ``abstain`` already captures "no view" and ``flat`` direction on
    a non-abstained record has no realized-sign to be scored against)."""
    active = [r for r in records if not r.abstain and r.direction in ("long", "short")]
    if not active:
        return None
    tp = fn = tn = fp = 0
    for r in active:
        predicted_positive = r.direction == "long"
        actual_positive = r.realized_excess_log_return > 0
        if actual_positive and predicted_positive:
            tp += 1
        elif actual_positive and not predicted_positive:
            fn += 1
        elif not actual_positive and not predicted_positive:
            tn += 1
        else:
            fp += 1
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    parts = [v for v in (sensitivity, specificity) if v is not None]
    if not parts:
        return None
    return sum(parts) / len(parts)


def coverage(records: Sequence[EvalRecord]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if not r.abstain) / len(records)


# --------------------------------------------------------------------------
# Top-level report
# --------------------------------------------------------------------------

def _core_metrics(records: Sequence[EvalRecord], *, seed: int) -> dict[str, Any]:
    xs, ys = _non_abstain_pairs(records)
    point, ci_lo, ci_hi = stationary_block_bootstrap_ci(
        records, statistic_fn=spearman_ic, seed=seed
    )
    return {
        "spearman_ic": point,
        "spearman_ic_ci": (ci_lo, ci_hi),
        "hac_se": hac_standard_error(xs, ys) if xs else 0.0,
        "coverage": coverage(records),
        "brier_score": brier_score(records),
        "log_loss": log_loss(records),
        "calibration_slope": calibration_slope_intercept(records)[0],
        "calibration_intercept": calibration_slope_intercept(records)[1],
        "balanced_accuracy": balanced_accuracy(records),
        "n_decisions": len(records),
    }


def evaluate(
    records: Sequence[EvalRecord], *, seed: int, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Top-level research-forecast evaluation report (plan §9's registered
    metrics). Always invokes ``stationary_block_bootstrap_ci`` with
    ``spearman_ic`` as the primary uncertainty-adjusted metric."""
    config = config or {}
    avg_overlap = config.get("avg_overlap", 1.0)

    core = _core_metrics(records, seed=seed)
    core["effective_sample_size"] = effective_sample_size(len(records), avg_overlap)

    non_overlapping = non_overlapping_daily_cohort(records)
    core["n_non_overlapping"] = len(non_overlapping)
    core["non_overlapping_robustness"] = _core_metrics(non_overlapping, seed=seed)

    by_asset: dict[str, Any] = {}
    grouped: dict[str, list[EvalRecord]] = defaultdict(list)
    for r in records:
        grouped[r.instrument_id].append(r)
    for instrument_id, group in grouped.items():
        by_asset[instrument_id] = _core_metrics(group, seed=seed)

    core["by_asset"] = by_asset
    # by_source_class deferred: requires evidence-feature-id source-class
    # provenance not plumbed in this task slice. See module docstring.
    core["by_source_class"] = {}
    return core
