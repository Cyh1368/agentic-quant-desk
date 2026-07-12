"""Tests for quantdesk.advisors.sent_zscore_baseline.run_sent_zscore_contrarian_baseline_v1."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from quantdesk.advisors.sent_zscore_baseline import (
    PROB_OFFSET,
    run_sent_zscore_contrarian_baseline_v1,
)

GENERATED_AT = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


def _run(features: dict, data_health: float = 0.95, instrument_ids=None):
    instrument_ids = instrument_ids or ["BTC"]
    snapshot = {"features": features, "samples": [], "data_health": data_health}
    return run_sent_zscore_contrarian_baseline_v1(
        snapshot=snapshot,
        instrument_ids=instrument_ids,
        snapshot_id=uuid4(),
        generated_at=GENERATED_AT,
        data_cutoff_at=GENERATED_AT,
    )


def test_exact_boundary_z_minus_2_triggers_long():
    forecasts = _run({"BTC:sent_mean_1h_zscore": -2.0, "BTC:spike_flag": True})
    f = forecasts[0]
    assert f.direction == "long"
    assert f.abstain is False
    assert f.probability_positive == 0.56
    assert f.model_run_id is None


def test_just_below_boundary_does_not_trigger():
    forecasts = _run({"BTC:sent_mean_1h_zscore": -1.999, "BTC:spike_flag": True})
    f = forecasts[0]
    assert f.direction == "flat"
    assert f.abstain is False
    assert f.probability_positive == 0.5
    assert f.model_run_id is None


def test_exact_boundary_z_plus_2_triggers_short():
    forecasts = _run({"BTC:sent_mean_1h_zscore": 2.0, "BTC:spike_flag": True})
    f = forecasts[0]
    assert f.direction == "short"
    assert f.abstain is False
    assert f.probability_positive == 0.44
    assert f.model_run_id is None


def test_spike_flag_required_for_trigger():
    forecasts = _run({"BTC:sent_mean_1h_zscore": -3.0, "BTC:spike_flag": False})
    f = forecasts[0]
    assert f.direction == "flat"
    assert f.abstain is False
    assert f.probability_positive == 0.5
    assert f.model_run_id is None


def test_spike_flag_falsy_value_does_not_trigger():
    forecasts = _run({"BTC:sent_mean_1h_zscore": 3.0, "BTC:spike_flag": 0})
    f = forecasts[0]
    assert f.direction == "flat"
    assert f.model_run_id is None


def test_degraded_data_health_abstains():
    forecasts = _run({"BTC:sent_mean_1h_zscore": -3.0, "BTC:spike_flag": True}, data_health=0.5)
    f = forecasts[0]
    assert f.abstain is True
    assert f.direction == "flat"
    assert f.probability_positive is None
    assert f.model_run_id is None


def test_missing_features_abstain():
    forecasts = _run({})
    f = forecasts[0]
    assert f.abstain is True
    assert f.direction == "flat"
    assert f.model_run_id is None


def test_null_z_score_abstains():
    forecasts = _run({"BTC:sent_mean_1h_zscore": None, "BTC:spike_flag": True})
    f = forecasts[0]
    assert f.abstain is True
    assert f.model_run_id is None


def test_null_spike_flag_abstains():
    forecasts = _run({"BTC:sent_mean_1h_zscore": -3.0, "BTC:spike_flag": None})
    f = forecasts[0]
    assert f.abstain is True
    assert f.model_run_id is None


def test_single_instrument_falls_back_to_unprefixed_keys():
    forecasts = _run({"sent_mean_1h_zscore": -2.5, "spike_flag": True}, instrument_ids=["BTC"])
    f = forecasts[0]
    assert f.direction == "long"
    assert f.probability_positive == 0.5 + PROB_OFFSET
    assert f.model_run_id is None


def test_all_model_run_ids_none_across_all_cases():
    cases = [
        _run({"BTC:sent_mean_1h_zscore": -2.0, "BTC:spike_flag": True}),
        _run({"BTC:sent_mean_1h_zscore": 2.0, "BTC:spike_flag": True}),
        _run({"BTC:sent_mean_1h_zscore": 0.1, "BTC:spike_flag": False}),
        _run({}),
        _run({"BTC:sent_mean_1h_zscore": -3.0, "BTC:spike_flag": True}, data_health=0.1),
    ]
    for forecasts in cases:
        for f in forecasts:
            assert f.model_run_id is None
