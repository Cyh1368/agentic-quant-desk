"""Prospective forecast scoring (plan §6).

Scores *stored* forecasts against resolved outcomes. This module never
touches the ledger or any storage layer directly — callers pass in already
materialized records and, where persistence is needed, inject callables.

Only prospective shadow forecasting is scored here (plan §6, method 2):
every record must already carry its realized outcome. Historical LLM
replay is out of scope by design and has no representation in this module.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

Action = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class ScoredRecord:
    """One resolved forecast: a stored ForecastSignal joined with its outcome.

    realized_excess_return: the actual next-24h excess return for
    (instrument_id, decision time), sign of which is the forecast target.
    """
    instrument_id: str
    advisor_id: str
    action: Action
    raw_score: float
    realized_excess_return: float
    calibrated_probability_positive: float | None = None


# --------------------------------------------------------------------------
# Probability mapping
# --------------------------------------------------------------------------

def implied_probability_positive(record: ScoredRecord) -> float:
    """Probability the advisor implicitly assigns to a positive outcome.

    Uses calibrated_probability_positive when present. Otherwise maps
    action + raw_score through a logistic squash: flat -> 0.5 (no
    directional view), long/short -> sigmoid(sign * |raw_score|).
    """
    if record.calibrated_probability_positive is not None:
        return record.calibrated_probability_positive
    if record.action == "flat":
        return 0.5
    sign = 1.0 if record.action == "long" else -1.0
    x = sign * abs(record.raw_score)
    # logistic squash, numerically stable
    if x >= 0:
        ez = _exp_neg(x)
        return 1.0 / (1.0 + ez)
    else:
        ez = _exp_neg(-x)
        return ez / (1.0 + ez)


def _exp_neg(x: float) -> float:
    import math
    return math.exp(-x)


def outcome_label(record: ScoredRecord) -> int:
    """1 if realized excess return positive, else 0. Exact zero counts as 0."""
    return 1 if record.realized_excess_return > 0 else 0


def _non_abstained(records: list[ScoredRecord]) -> list[ScoredRecord]:
    return [r for r in records if r.action != "flat"]


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------

def coverage(records: list[ScoredRecord]) -> float:
    """Fraction of forecasts that were non-flat (i.e. not abstained)."""
    if not records:
        return 0.0
    return len(_non_abstained(records)) / len(records)


def abstention_rate(records: list[ScoredRecord]) -> float:
    return 1.0 - coverage(records)


def hit_rate(records: list[ScoredRecord]) -> float | None:
    """Directional hit rate over non-flat forecasts only. None if no non-flat records."""
    active = _non_abstained(records)
    if not active:
        return None
    hits = 0
    for r in active:
        predicted_positive = r.action == "long"
        hits += int(predicted_positive == bool(outcome_label(r)))
    return hits / len(active)


def brier_score(records: list[ScoredRecord]) -> float | None:
    """Mean squared error between implied probability and realized outcome,
    over non-flat forecasts only. None if no non-flat records."""
    active = _non_abstained(records)
    if not active:
        return None
    total = 0.0
    for r in active:
        p = implied_probability_positive(r)
        y = outcome_label(r)
        total += (p - y) ** 2
    return total / len(active)


def calibration_table(
    records: list[ScoredRecord],
    bucket_edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> list[dict[str, Any]]:
    """Reliability table: bucket non-flat forecasts by implied probability,
    report predicted-vs-empirical positive rate per bucket."""
    active = _non_abstained(records)
    buckets: list[dict[str, Any]] = []
    last_hi = bucket_edges[-1]
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        is_last = hi == last_hi
        members = [
            r for r in active
            if lo <= implied_probability_positive(r) < hi
            or (is_last and implied_probability_positive(r) == hi)
        ]
        if not members:
            buckets.append(
                {
                    "bucket": f"[{lo:.1f}, {hi:.1f}]",
                    "n": 0,
                    "avg_predicted_probability": None,
                    "empirical_positive_rate": None,
                }
            )
            continue
        preds = [implied_probability_positive(r) for r in members]
        empirical = [outcome_label(r) for r in members]
        buckets.append(
            {
                "bucket": f"[{lo:.1f}, {hi:.1f}]",
                "n": len(members),
                "avg_predicted_probability": sum(preds) / len(preds),
                "empirical_positive_rate": sum(empirical) / len(empirical),
            }
        )
    return buckets


def calibration_error(records: list[ScoredRecord], bucket_edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> float | None:
    """Expected calibration error: n-weighted mean |predicted - empirical| across buckets."""
    table = calibration_table(records, bucket_edges)
    populated = [b for b in table if b["n"] > 0]
    if not populated:
        return None
    total_n = sum(b["n"] for b in populated)
    weighted = sum(
        b["n"] * abs(b["avg_predicted_probability"] - b["empirical_positive_rate"])
        for b in populated
    )
    return weighted / total_n


def _signed_score(record: ScoredRecord) -> float:
    if record.action == "flat":
        return 0.0
    sign = 1.0 if record.action == "long" else -1.0
    return sign * abs(record.raw_score)


def net_information_coefficient(records: list[ScoredRecord]) -> float | None:
    """Pearson correlation between the advisor's signed score and the
    realized excess return, over non-flat forecasts only.

    "net" here follows the contract's primary_metric name (net of the
    advisor's own abstentions, i.e. computed on the decisions it actually
    made); no explicit transaction-cost adjustment is applied in this
    scoring layer since costs are advisor-agnostic and belong to the
    portfolio/execution layer, not to advisor validation.
    """
    active = _non_abstained(records)
    if len(active) < 2:
        return None
    scores = [_signed_score(r) for r in active]
    returns = [r.realized_excess_return for r in active]
    if len(set(scores)) < 2 or len(set(returns)) < 2:
        return 0.0
    return _pearson(scores, returns)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    if denom == 0:
        return 0.0
    return cov / denom


# --------------------------------------------------------------------------
# Baseline comparisons
# --------------------------------------------------------------------------

def always_flat_records(records: list[ScoredRecord]) -> list[ScoredRecord]:
    return [
        ScoredRecord(
            instrument_id=r.instrument_id,
            advisor_id="always_flat",
            action="flat",
            raw_score=0.0,
            realized_excess_return=r.realized_excess_return,
        )
        for r in records
    ]


def buy_and_hold_records(records: list[ScoredRecord]) -> list[ScoredRecord]:
    return [
        ScoredRecord(
            instrument_id=r.instrument_id,
            advisor_id="buy_and_hold",
            action="long",
            raw_score=1.0,
            realized_excess_return=r.realized_excess_return,
            calibrated_probability_positive=1.0,
        )
        for r in records
    ]


def random_same_frequency_records(records: list[ScoredRecord], seed: int = 42) -> list[ScoredRecord]:
    """Random forecaster with the same non-flat frequency and long/short mix
    as the input, seeded for reproducibility."""
    rng = random.Random(seed)
    active_actions = [r.action for r in records if r.action != "flat"]
    n_active = len(active_actions)
    n_long = sum(1 for a in active_actions if a == "long")
    p_active = n_active / len(records) if records else 0.0
    p_long_given_active = (n_long / n_active) if n_active else 0.5

    out = []
    for r in records:
        if rng.random() < p_active:
            action: Action = "long" if rng.random() < p_long_given_active else "short"
            raw_score = rng.uniform(0.25, 1.5)
        else:
            action = "flat"
            raw_score = 0.0
        out.append(
            ScoredRecord(
                instrument_id=r.instrument_id,
                advisor_id="random_same_frequency",
                action=action,
                raw_score=raw_score,
                realized_excess_return=r.realized_excess_return,
            )
        )
    return out


def compare_vs_baselines(
    records: list[ScoredRecord],
    baseline_advisor_records: list[ScoredRecord] | None = None,
    seed: int = 42,
) -> dict[str, dict[str, float | None]]:
    """Score the advisor's own records plus always-flat, buy-and-hold,
    a seeded random-same-frequency forecaster, and (if supplied) the
    deterministic baseline advisor's records over the same decisions."""
    comparisons: dict[str, dict[str, float | None]] = {
        "advisor": _summary(records),
        "always_flat": _summary(always_flat_records(records)),
        "buy_and_hold": _summary(buy_and_hold_records(records)),
        "random_same_frequency": _summary(random_same_frequency_records(records, seed=seed)),
    }
    if baseline_advisor_records is not None:
        comparisons["ts_momentum_baseline"] = _summary(baseline_advisor_records)
    return comparisons


def _summary(records: list[ScoredRecord]) -> dict[str, float | None]:
    return {
        "coverage": coverage(records),
        "hit_rate": hit_rate(records),
        "brier_score": brier_score(records),
        "net_information_coefficient": net_information_coefficient(records),
    }


# --------------------------------------------------------------------------
# Calibration summary for prompt injection (plan §7)
# --------------------------------------------------------------------------

def build_calibration_summary(
    records: list[ScoredRecord],
    *,
    window: int,
    status: str = "shadow",
    size_multiplier: float = 1.0,
) -> dict[str, Any]:
    """CalibrationSummary dict, injected verbatim (as JSON) into the advisor
    prompt's CALIBRATION SUMMARY block (plan §7)."""
    recent = records[-window:] if window else records
    return {
        "window": window,
        "n_scored": len(recent),
        "coverage": coverage(recent),
        "abstention_rate": abstention_rate(recent),
        "hit_rate": hit_rate(recent),
        "brier_score": brier_score(recent),
        "calibration_error": calibration_error(recent),
        "calibration_by_bucket": calibration_table(recent),
        "net_information_coefficient": net_information_coefficient(recent),
        "status": status,
        "size_multiplier": size_multiplier,
    }


# --------------------------------------------------------------------------
# Contract status / demotion (plan §6, research contract YAML)
# --------------------------------------------------------------------------

def contract_status(
    records: list[ScoredRecord],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered demotion thresholds from the research
    contract YAML (config/contracts/crypto_trend_llm_v1.yaml) to a rolling
    window of the most recent scored records.

    Only demotion is evaluated here (a purely mechanical, deterministic
    breach check per plan §6/§7: "no LLM decides this"). Promotion is a
    separate, higher-bar decision gated on minimum_sample_size and is left
    to human/process review, not asserted automatically by this function.
    """
    demotion = contract.get("demotion_thresholds", {})
    rolling_window = demotion.get("rolling_window", 100)
    brier_max = demotion.get("brier_score_max")
    action_on_breach = demotion.get("action_on_breach", "shadow")

    window_records = records[-rolling_window:] if rolling_window else records
    n_scored_in_window = len(window_records)
    total_scored = len(records)
    minimum_sample_size = contract.get("minimum_sample_size", 200)

    b = brier_score(window_records)
    breached = brier_max is not None and b is not None and b > brier_max

    # minimum_sample_size (plan §6) gates on the total prospective sample
    # collected so far, independent of the rolling window used for the
    # (narrower) demotion breach check.
    if total_scored < minimum_sample_size:
        status = "insufficient_sample"
    elif breached:
        status = action_on_breach
    else:
        status = "active"

    return {
        "advisor_id": contract.get("advisor_id"),
        "n_scored_in_window": n_scored_in_window,
        "total_scored": total_scored,
        "rolling_window": rolling_window,
        "minimum_sample_size": minimum_sample_size,
        "brier_score": b,
        "brier_score_max": brier_max,
        "breached": breached,
        "status": status,
    }
