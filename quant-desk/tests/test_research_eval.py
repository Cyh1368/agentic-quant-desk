"""Tests for quantdesk.scoring.research_eval (overlap-aware evaluation)."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from quantdesk.scoring import research_eval
from quantdesk.scoring.research_eval import (
    EvalRecord,
    balanced_accuracy,
    brier_score,
    coverage,
    effective_sample_size,
    evaluate,
    log_loss,
    non_overlapping_daily_cohort,
    spearman_ic,
    stationary_block_bootstrap_ci,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _rec(
    i: int,
    forecast_bps: float | None = 0.0,
    realized: float = 0.0,
    prob: float | None = None,
    direction: str = "long",
    abstain: bool = False,
    instrument_id: str = "BTC",
    hours: int = 4,
) -> EvalRecord:
    return EvalRecord(
        instrument_id=instrument_id,
        generated_at=T0 + timedelta(hours=hours * i),
        forecast_bps=forecast_bps,
        probability_positive=prob,
        direction=direction,
        abstain=abstain,
        realized_excess_log_return=realized,
    )


def _ar1_null_records(seed: int, n: int = 400) -> list[EvalRecord]:
    """Forecasts independent of realized outcomes (true IC = 0), each
    series AR(1)-autocorrelated to mimic overlapping windows."""
    rng = random.Random(seed)
    records = []
    f = 0.0
    y = 0.0
    for i in range(n):
        f = 0.6 * f + rng.gauss(0, 1)
        y = 0.6 * y + rng.gauss(0, 1)
        records.append(_rec(i, forecast_bps=f * 10, realized=y * 0.01))
    return records


# --------------------------------------------------------------------------
# Bootstrap CI sanity
# --------------------------------------------------------------------------

def test_bootstrap_ci_contains_zero_under_null():
    # True IC is 0 by construction; over 20 seeds the 95% CI should contain
    # 0 in the large majority of runs. Tolerance: >= 17/20 (allows for the
    # nominal ~5% miss rate plus block-bootstrap approximation error).
    hits = 0
    for seed in range(20):
        records = _ar1_null_records(seed)
        report = evaluate(records, seed=seed)
        lo, hi = report["spearman_ic_ci"]
        if lo <= 0.0 <= hi:
            hits += 1
    assert hits >= 17


def test_bootstrap_deterministic_given_seed():
    records = _ar1_null_records(seed=7, n=100)
    a = stationary_block_bootstrap_ci(records, statistic_fn=spearman_ic, seed=123)
    b = stationary_block_bootstrap_ci(records, statistic_fn=spearman_ic, seed=123)
    assert a == b
    c = stationary_block_bootstrap_ci(records, statistic_fn=spearman_ic, seed=124)
    assert a != c


def test_evaluate_invokes_bootstrap(monkeypatch):
    calls = []
    real = research_eval.stationary_block_bootstrap_ci

    def spy(records, **kwargs):
        calls.append(kwargs)
        return real(records, **kwargs)

    monkeypatch.setattr(research_eval, "stationary_block_bootstrap_ci", spy)
    evaluate(_ar1_null_records(seed=1, n=50), seed=42)
    assert calls, "evaluate() must invoke stationary_block_bootstrap_ci"
    assert any(kw.get("statistic_fn") is spearman_ic for kw in calls)


# --------------------------------------------------------------------------
# Spearman IC
# --------------------------------------------------------------------------

def test_spearman_ic_perfect_monotone():
    records = [_rec(i, forecast_bps=float(i), realized=float(i) ** 3) for i in range(10)]
    assert spearman_ic(records) == pytest.approx(1.0)


def test_spearman_ic_perfect_inverse():
    records = [_rec(i, forecast_bps=float(i), realized=-float(i)) for i in range(10)]
    assert spearman_ic(records) == pytest.approx(-1.0)


def test_spearman_ic_ignores_abstain_and_none():
    good = [_rec(i, forecast_bps=float(i), realized=float(i)) for i in range(5)]
    noise = [
        _rec(10, forecast_bps=None, realized=-99.0),
        _rec(11, forecast_bps=5.0, realized=-99.0, abstain=True),
    ]
    assert spearman_ic(good + noise) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Coverage / calibration / classification sanity
# --------------------------------------------------------------------------

def test_coverage():
    records = [_rec(0), _rec(1, abstain=True), _rec(2), _rec(3, abstain=True)]
    assert coverage(records) == pytest.approx(0.5)
    assert coverage([]) == 0.0


def test_brier_score_known_values():
    records = [
        _rec(0, prob=1.0, realized=0.01),   # perfect
        _rec(1, prob=0.0, realized=-0.01),  # perfect
        _rec(2, prob=0.5, realized=0.01),   # 0.25
        _rec(3, prob=None, realized=0.01),  # skipped
    ]
    assert brier_score(records) == pytest.approx(0.25 / 3)
    assert brier_score([_rec(0, prob=None)]) is None


def test_log_loss_better_for_sharper_correct_probs():
    good = [_rec(0, prob=0.9, realized=0.01), _rec(1, prob=0.1, realized=-0.01)]
    bad = [_rec(0, prob=0.6, realized=0.01), _rec(1, prob=0.4, realized=-0.01)]
    assert log_loss(good) < log_loss(bad)
    assert log_loss([_rec(0, prob=None)]) is None


def test_balanced_accuracy_directional():
    records = [
        _rec(0, direction="long", realized=0.01),    # correct positive
        _rec(1, direction="long", realized=-0.01),   # wrong
        _rec(2, direction="short", realized=-0.01),  # correct negative
        _rec(3, direction="short", realized=-0.01),  # correct negative
        _rec(4, direction="flat", realized=0.05),    # excluded
        _rec(5, direction="long", realized=0.05, abstain=True),  # excluded
    ]
    # sensitivity = 1/1, specificity = 2/3 -> (1 + 2/3)/2
    assert balanced_accuracy(records) == pytest.approx((1.0 + 2.0 / 3.0) / 2)
    assert balanced_accuracy([_rec(0, direction="flat")]) is None


def test_calibration_slope_intercept_well_calibrated():
    rng = random.Random(0)
    records = []
    for i in range(2000):
        p = rng.uniform(0.05, 0.95)
        realized = 0.01 if rng.random() < p else -0.01
        records.append(_rec(i, prob=p, realized=realized))
    slope, intercept = research_eval.calibration_slope_intercept(records)
    assert slope == pytest.approx(1.0, abs=0.2)
    assert intercept == pytest.approx(0.0, abs=0.2)


# --------------------------------------------------------------------------
# Non-overlapping cohort / ESS
# --------------------------------------------------------------------------

def test_non_overlapping_daily_cohort():
    records = [_rec(i, hours=4) for i in range(12)]  # 2 calendar days of BTC
    records += [_rec(i, instrument_id="ETH", hours=4) for i in range(6)]  # 1 day ETH
    cohort = non_overlapping_daily_cohort(records)
    assert len(cohort) == 3
    keys = {(r.instrument_id, r.generated_at.date()) for r in cohort}
    assert len(keys) == 3


def test_effective_sample_size():
    assert effective_sample_size(600, 6.0) == pytest.approx(100.0)
    assert effective_sample_size(100) == pytest.approx(100.0)
    assert effective_sample_size(100, 0.0) == 100.0


# --------------------------------------------------------------------------
# evaluate() report shape
# --------------------------------------------------------------------------

def test_evaluate_report_keys_and_by_asset():
    records = _ar1_null_records(seed=3, n=60)
    records += [
        _rec(i, instrument_id="ETH", forecast_bps=float(i), realized=float(i), prob=0.6)
        for i in range(30)
    ]
    report = evaluate(records, seed=99, config={"avg_overlap": 6.0})
    for key in (
        "spearman_ic", "spearman_ic_ci", "hac_se", "effective_sample_size",
        "coverage", "brier_score", "log_loss", "calibration_slope",
        "calibration_intercept", "balanced_accuracy", "n_decisions",
        "n_non_overlapping", "non_overlapping_robustness", "by_asset",
        "by_source_class",
    ):
        assert key in report, key
    assert report["n_decisions"] == 90
    assert report["effective_sample_size"] == pytest.approx(15.0)
    assert set(report["by_asset"]) == {"BTC", "ETH"}
    assert report["by_asset"]["ETH"]["spearman_ic"] == pytest.approx(1.0)
    assert report["by_source_class"] == {}
    assert report["n_non_overlapping"] == report["non_overlapping_robustness"]["n_decisions"]
