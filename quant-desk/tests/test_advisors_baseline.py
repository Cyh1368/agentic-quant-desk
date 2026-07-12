from datetime import datetime, timezone
from uuid import uuid4

import pytest

from quantdesk.advisors.baseline import (
    ADVISOR_ID,
    ADVISOR_VERSION,
    DEAD_ZONE_Z,
    ts_momentum_baseline,
)


def _base_kwargs(**overrides):
    kwargs = dict(
        instrument_id="BTC",
        venue="hyperliquid",
        features={"return_24h": 0.01, "return_7d": 0.10, "realized_vol_7d": 0.20},
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        data_cutoff_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        snapshot_id=uuid4(),
    )
    kwargs.update(overrides)
    return kwargs


def test_long_signal_above_dead_zone():
    signal = ts_momentum_baseline(**_base_kwargs())
    assert signal.action == "long"
    assert signal.raw_score == pytest.approx(0.5)
    assert signal.advisor_id == ADVISOR_ID
    assert signal.advisor_version == ADVISOR_VERSION
    assert signal.model_run_id is None
    assert signal.invalidation_condition is not None


def test_short_signal_below_negative_dead_zone():
    kwargs = _base_kwargs(features={"return_24h": -0.01, "return_7d": -0.10, "realized_vol_7d": 0.20})
    signal = ts_momentum_baseline(**kwargs)
    assert signal.action == "short"
    assert signal.raw_score == pytest.approx(-0.5)


def test_flat_within_dead_zone():
    # z = 0.01 / 0.20 = 0.05, well within +/- DEAD_ZONE_Z
    kwargs = _base_kwargs(features={"return_24h": 0.001, "return_7d": 0.01, "realized_vol_7d": 0.20})
    signal = ts_momentum_baseline(**kwargs)
    assert signal.action == "flat"
    assert signal.invalidation_condition is None


def test_flat_at_dead_zone_boundary_is_exclusive():
    return_7d = DEAD_ZONE_Z * 0.20  # exactly at the boundary -> not > dead zone -> flat
    kwargs = _base_kwargs(features={"return_24h": 0.0, "return_7d": return_7d, "realized_vol_7d": 0.20})
    signal = ts_momentum_baseline(**kwargs)
    assert signal.action == "flat"


def test_zero_vol_is_flat_not_error():
    kwargs = _base_kwargs(features={"return_24h": 0.01, "return_7d": 0.10, "realized_vol_7d": 0.0})
    signal = ts_momentum_baseline(**kwargs)
    assert signal.action == "flat"
    assert signal.raw_score == 0.0


def test_deterministic_same_input_same_output():
    kwargs = _base_kwargs(signal_id=uuid4())
    s1 = ts_momentum_baseline(**kwargs)
    s2 = ts_momentum_baseline(**kwargs)
    assert s1.action == s2.action
    assert s1.raw_score == s2.raw_score
    assert s1.thesis == s2.thesis
    assert s1.signal_id == s2.signal_id  # explicit signal_id passed through unchanged


def test_evidence_feature_ids_includes_optional_trend_state():
    kwargs = _base_kwargs(
        features={
            "return_24h": 0.01,
            "return_7d": 0.10,
            "realized_vol_7d": 0.20,
            "trend_state": "trending",
        }
    )
    signal = ts_momentum_baseline(**kwargs)
    assert "trend_state" in signal.evidence_feature_ids
    assert "return_7d" in signal.evidence_feature_ids
    assert "realized_vol_7d" in signal.evidence_feature_ids
