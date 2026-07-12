"""Tests for quantdesk.portfolio.engine (deterministic portfolio construction)."""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quantdesk.common.schemas import ForecastSignal
from quantdesk.portfolio.engine import (
    AdvisorTrackRecord,
    CalibratedForecast,
    InstrumentMarketData,
    PortfolioCaps,
    apply_reliability_shrinkage,
    build_order_intents,
    calibrate,
    crash_override_covariance,
    dedup_cluster,
    reapply_caps_after_rounding,
    round_to_tick_lot,
    shrunk_covariance,
    size_positions,
    suppress_turnover,
)

UTC = timezone.utc


def _now():
    return datetime(2026, 7, 12, 0, 0, tzinfo=UTC)


def make_forecast(
    advisor_id="crypto_trend_llm_v1",
    instrument_id="BTC",
    action="long",
    raw_score=0.5,
    calibrated_probability_positive=None,
    expected_excess_return_bps=None,
    evidence_feature_ids=None,
    signal_id=None,
) -> ForecastSignal:
    return ForecastSignal(
        signal_id=signal_id or uuid.uuid4(),
        advisor_id=advisor_id,
        advisor_version="v1",
        generated_at=_now(),
        data_cutoff_at=_now(),
        expires_at=_now() + timedelta(hours=4),
        venue="hyperliquid",
        instrument_id=instrument_id,
        forecast_target="next_24h_excess_return_sign",
        horizon=timedelta(hours=24),
        action=action,
        raw_score=raw_score,
        calibrated_probability_positive=calibrated_probability_positive,
        expected_excess_return_bps=expected_excess_return_bps,
        evidence_feature_ids=evidence_feature_ids or [],
        snapshot_id=uuid.uuid4(),
    )


# ---------------------------------------------------------------------------
# calibrate()
# ---------------------------------------------------------------------------


def test_calibrate_below_minimum_sample_size_gives_zero_weight():
    tr = {
        "crypto_trend_llm_v1": AdvisorTrackRecord(
            advisor_id="crypto_trend_llm_v1",
            prospective_sample_count=50,
            contract_minimum_sample_size=200,
            reliability=0.8,
        )
    }
    fc = [make_forecast()]
    out = calibrate(fc, tr)
    assert len(out) == 1
    assert out[0].weight == 0.0


def test_calibrate_above_minimum_gives_nonzero_weight():
    tr = {
        "crypto_trend_llm_v1": AdvisorTrackRecord(
            advisor_id="crypto_trend_llm_v1",
            prospective_sample_count=500,
            contract_minimum_sample_size=200,
            reliability=0.8,
        )
    }
    fc = [make_forecast()]
    out = calibrate(fc, tr)
    assert out[0].weight == pytest.approx(0.8)


def test_calibrate_bucket_mapping():
    tr = {
        "crypto_trend_llm_v1": AdvisorTrackRecord(
            advisor_id="crypto_trend_llm_v1",
            prospective_sample_count=500,
            contract_minimum_sample_size=200,
            calibration_buckets=[(0.0, 0.5, 0.4), (0.5, 1.0, 0.7)],
            reliability=0.9,
        )
    }
    fc = [make_forecast(raw_score=0.6, action="long", expected_excess_return_bps=100.0)]
    out = calibrate(fc, tr)
    assert out[0].calibrated_probability_positive == pytest.approx(0.7)
    # expected_excess_return_bps = (0.7-0.5)*2*100 = 40, sign-adjusted long -> positive
    assert out[0].expected_excess_return_bps == pytest.approx(40.0)


def test_calibrate_flat_action_always_zero_excess():
    tr = {
        "crypto_trend_llm_v1": AdvisorTrackRecord(
            advisor_id="crypto_trend_llm_v1",
            prospective_sample_count=500,
            contract_minimum_sample_size=200,
            calibration_buckets=[(0.0, 1.0, 0.9)],
            reliability=0.9,
        )
    }
    fc = [make_forecast(raw_score=0.6, action="flat")]
    out = calibrate(fc, tr)
    assert out[0].expected_excess_return_bps == 0.0


def test_calibrate_passthrough_when_explicit_values_present():
    tr = {}
    fc = [
        make_forecast(
            calibrated_probability_positive=0.65,
            expected_excess_return_bps=55.0,
        )
    ]
    out = calibrate(fc, tr)
    assert out[0].calibrated_probability_positive == 0.65
    assert out[0].expected_excess_return_bps == 55.0
    # missing track record -> weight 0 regardless of passthrough
    assert out[0].weight == 0.0


# ---------------------------------------------------------------------------
# apply_reliability_shrinkage()
# ---------------------------------------------------------------------------


def test_reliability_shrinkage_monotonic():
    tr = {
        "adv": AdvisorTrackRecord(
            advisor_id="adv", prospective_sample_count=500, contract_minimum_sample_size=200, reliability=0.8
        )
    }
    cf = [
        CalibratedForecast(
            advisor_id="adv",
            instrument_id="BTC",
            signal_id=uuid.uuid4(),
            evidence_feature_ids=[],
            calibrated_probability_positive=0.6,
            expected_excess_return_bps=30.0,
            weight=0.8,
        )
    ]
    w0 = apply_reliability_shrinkage(cf, tr, 0.0)[0].weight
    w1 = apply_reliability_shrinkage(cf, tr, 0.5)[0].weight
    w2 = apply_reliability_shrinkage(cf, tr, 1.0)[0].weight
    assert w0 == pytest.approx(0.8)
    assert w1 == pytest.approx(0.4)
    assert w2 == pytest.approx(0.0)
    assert w0 > w1 > w2 or (w0 >= w1 >= w2)


def test_reliability_shrinkage_leaves_already_zero_weight_at_zero():
    tr = {
        "adv": AdvisorTrackRecord(
            advisor_id="adv", prospective_sample_count=1, contract_minimum_sample_size=200, reliability=0.9
        )
    }
    cf = [
        CalibratedForecast(
            advisor_id="adv",
            instrument_id="BTC",
            signal_id=uuid.uuid4(),
            evidence_feature_ids=[],
            calibrated_probability_positive=0.5,
            expected_excess_return_bps=0.0,
            weight=0.0,
        )
    ]
    out = apply_reliability_shrinkage(cf, tr, 0.5)
    assert out[0].weight == 0.0


# ---------------------------------------------------------------------------
# dedup_cluster()
# ---------------------------------------------------------------------------


def _cf(advisor_id, instrument_id, evidence, prob, excess, weight):
    return CalibratedForecast(
        advisor_id=advisor_id,
        instrument_id=instrument_id,
        signal_id=uuid.uuid4(),
        evidence_feature_ids=evidence,
        calibrated_probability_positive=prob,
        expected_excess_return_bps=excess,
        weight=weight,
    )


def test_dedup_cluster_merges_overlapping_evidence():
    a = _cf("adv1", "BTC", ["f1", "f2", "f3"], 0.6, 30.0, 0.5)
    b = _cf("adv2", "BTC", ["f1", "f2"], 0.8, 60.0, 0.5)
    # Jaccard = |{f1,f2}| / |{f1,f2,f3}| = 2/3 > 0.5 -> merge
    out = dedup_cluster([a, b])
    assert len(out) == 1
    expected_prob = (0.6 * 0.5 + 0.8 * 0.5) / 1.0
    expected_excess = (30.0 * 0.5 + 60.0 * 0.5) / 1.0
    assert out[0].calibrated_probability_positive == pytest.approx(expected_prob)
    assert out[0].expected_excess_return_bps == pytest.approx(expected_excess)
    assert set(out[0].evidence_feature_ids) == {"f1", "f2", "f3"}


def test_dedup_cluster_keeps_separate_when_low_overlap():
    a = _cf("adv1", "BTC", ["f1", "f2"], 0.6, 30.0, 0.5)
    b = _cf("adv2", "BTC", ["f3", "f4"], 0.8, 60.0, 0.5)
    out = dedup_cluster([a, b])
    assert len(out) == 2


def test_dedup_cluster_empty_vs_empty_not_a_match():
    a = _cf("adv1", "BTC", [], 0.6, 30.0, 0.5)
    b = _cf("adv2", "BTC", [], 0.8, 60.0, 0.5)
    out = dedup_cluster([a, b])
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


def test_shrunk_covariance_shrinks_offdiagonal_toward_zero():
    instruments = ["BTC", "ETH"]
    vols = {"BTC": 0.6, "ETH": 0.8}
    correlation = 0.5
    unshrunk_offdiag = correlation * vols["BTC"] * vols["ETH"]
    cov = shrunk_covariance(instruments, vols, correlation, shrinkage=0.2)
    assert cov[("BTC", "ETH")] == pytest.approx(0.8 * unshrunk_offdiag)
    assert cov[("ETH", "BTC")] == pytest.approx(0.8 * unshrunk_offdiag)
    # diagonal untouched
    assert cov[("BTC", "BTC")] == pytest.approx(vols["BTC"] ** 2)
    assert cov[("ETH", "ETH")] == pytest.approx(vols["ETH"] ** 2)
    # shrunk toward zero: closer to 0 than unshrunk
    assert abs(cov[("BTC", "ETH")]) < abs(unshrunk_offdiag)


def test_crash_override_covariance_forces_correlation():
    instruments = ["BTC", "ETH"]
    vols = {"BTC": 0.6, "ETH": 0.8}
    cov = shrunk_covariance(instruments, vols, correlation=0.1, shrinkage=0.2)
    crash_cov = crash_override_covariance(cov, instruments, crash_correlation_override=0.95)
    expected_offdiag = 0.95 * vols["BTC"] * vols["ETH"]
    assert crash_cov[("BTC", "ETH")] == pytest.approx(expected_offdiag)
    assert crash_cov[("ETH", "BTC")] == pytest.approx(expected_offdiag)
    # diagonal (vols) unchanged regardless of original correlation
    assert crash_cov[("BTC", "BTC")] == pytest.approx(vols["BTC"] ** 2)


# ---------------------------------------------------------------------------
# size_positions(): hand-checked ERC fixture
# ---------------------------------------------------------------------------


def _caps(single=Decimal("100"), gross=Decimal("100"), net=Decimal("100"), order=Decimal("100")):
    return PortfolioCaps(
        max_single_position_pct=single,
        max_gross_exposure_pct=gross,
        max_net_directional_pct=net,
        max_order_size_pct_equity=order,
    )


def test_size_positions_hand_checked_erc_fixture():
    # BTC vol=0.6, ETH vol=0.8.
    # raw_weight_BTC = (1/0.6) / (1/0.6 + 1/0.8) = 1.6667 / 2.9167 = 0.5714
    # raw_weight_ETH = (1/0.8) / (1/0.6 + 1/0.8) = 1.25 / 2.9167 = 0.4286
    # Both long -> signed_weight == raw_weight (before vol-target scaling).
    # cov (shrinkage=0.2, correlation=0.5):
    #   cov_BTC_BTC = 0.36, cov_ETH_ETH = 0.64
    #   cov_BTC_ETH = 0.8 * 0.5 * 0.6 * 0.8 = 0.192
    # port_var = wB^2*0.36 + wE^2*0.64 + 2*wB*wE*0.192
    wB, wE = 0.571428571, 0.428571429
    cov_bb, cov_ee, cov_be = 0.36, 0.64, 0.8 * 0.5 * 0.6 * 0.8
    port_var = wB * wB * cov_bb + wE * wE * cov_ee + 2 * wB * wE * cov_be
    port_vol = math.sqrt(port_var)
    target_vol = 0.10  # target_annual_vol_pct = 10
    scale = target_vol / port_vol
    expected_btc_weight = wB * scale
    expected_eth_weight = wE * scale
    equity = Decimal("1000")
    expected_btc_notional = Decimal(str(expected_btc_weight)) * equity
    expected_eth_notional = Decimal(str(expected_eth_weight)) * equity

    market_data = {
        "BTC": InstrumentMarketData("BTC", Decimal("50000"), 0.6, Decimal("500"), Decimal("1"), Decimal("0.0001")),
        "ETH": InstrumentMarketData("ETH", Decimal("3000"), 0.8, Decimal("50"), Decimal("0.01"), Decimal("0.001")),
    }
    clustered = [
        _cf("adv1", "BTC", ["f1"], 0.7, 50.0, 0.8),
        _cf("adv1", "ETH", ["f2"], 0.7, 50.0, 0.8),
    ]
    cov = shrunk_covariance(["BTC", "ETH"], {"BTC": 0.6, "ETH": 0.8}, 0.5, 0.2)
    crash_cov = crash_override_covariance(cov, ["BTC", "ETH"], 0.95)
    caps = _caps()

    result = size_positions(clustered, market_data, cov, crash_cov, 10.0, equity, caps)

    tol = Decimal("1")
    assert abs(result["BTC"] - expected_btc_notional) < tol
    assert abs(result["ETH"] - expected_eth_notional) < tol


def test_size_positions_zero_net_edge_gives_zero_notional():
    market_data = {
        "BTC": InstrumentMarketData("BTC", Decimal("50000"), 0.6, Decimal("500"), Decimal("1"), Decimal("0.0001")),
        "ETH": InstrumentMarketData("ETH", Decimal("3000"), 0.8, Decimal("50"), Decimal("0.01"), Decimal("0.001")),
    }
    clustered = [_cf("adv1", "BTC", ["f1"], 0.5, 0.0, 0.8)]  # BTC net edge 0, ETH missing
    cov = shrunk_covariance(["BTC", "ETH"], {"BTC": 0.6, "ETH": 0.8}, 0.5, 0.2)
    crash_cov = crash_override_covariance(cov, ["BTC", "ETH"], 0.95)
    result = size_positions(clustered, market_data, cov, crash_cov, 10.0, Decimal("1000"), _caps())
    assert result["BTC"] == Decimal("0")
    assert result["ETH"] == Decimal("0")


def test_size_positions_single_position_cap_enforced():
    market_data = {
        "BTC": InstrumentMarketData("BTC", Decimal("50000"), 0.6, Decimal("500"), Decimal("1"), Decimal("0.0001")),
        "ETH": InstrumentMarketData("ETH", Decimal("3000"), 0.8, Decimal("50"), Decimal("0.01"), Decimal("0.001")),
    }
    clustered = [
        _cf("adv1", "BTC", ["f1"], 0.9, 500.0, 1.0),
        _cf("adv1", "ETH", ["f2"], 0.9, 500.0, 1.0),
    ]
    cov = shrunk_covariance(["BTC", "ETH"], {"BTC": 0.6, "ETH": 0.8}, 0.5, 0.2)
    crash_cov = crash_override_covariance(cov, ["BTC", "ETH"], 0.95)
    equity = Decimal("1000")
    caps = _caps(single=Decimal("5"))  # 5% of 1000 = 50
    result = size_positions(clustered, market_data, cov, crash_cov, 50.0, equity, caps)
    cap_amount = equity * Decimal("5") / Decimal("100")
    for v in result.values():
        assert abs(v) <= cap_amount + Decimal("0.01")


# ---------------------------------------------------------------------------
# suppress_turnover()
# ---------------------------------------------------------------------------


def test_suppress_turnover_drops_small_change():
    # Cost of a change is proportional to the change itself
    # (cost_usd = change * bps / 10000), so suppression triggers exactly
    # when bps * multiple > 10000 — a ratio rule, not an absolute one.
    current = {"BTC": Decimal("1000")}
    target_tiny = {"BTC": Decimal("1000.001")}
    round_trip_cost_bps = {"BTC": Decimal("500")}
    result = suppress_turnover(
        target_tiny, current, round_trip_cost_bps, min_trade_cost_multiple=1000.0
    )
    assert result["BTC"] == current["BTC"]


def test_suppress_turnover_passes_large_change():
    target = {"BTC": Decimal("2000")}
    current = {"BTC": Decimal("1000")}
    round_trip_cost_bps = {"BTC": Decimal("10")}
    result = suppress_turnover(target, current, round_trip_cost_bps, min_trade_cost_multiple=1.0)
    assert result["BTC"] == Decimal("2000")


# ---------------------------------------------------------------------------
# round_to_tick_lot() / reapply_caps_after_rounding()
# ---------------------------------------------------------------------------


def test_round_to_tick_lot_floors_down():
    notional = Decimal("1234")
    price = Decimal("100")
    lot = Decimal("0.1")
    qty = round_to_tick_lot(notional, price, Decimal("0.01"), lot)
    # raw qty = 12.34 -> floor to nearest 0.1 -> 12.3
    assert qty == Decimal("12.3")


def test_round_to_tick_lot_negative_notional():
    notional = Decimal("-1234")
    price = Decimal("100")
    lot = Decimal("0.1")
    qty = round_to_tick_lot(notional, price, Decimal("0.01"), lot)
    assert qty == Decimal("-12.3")


def test_caps_reapplied_after_rounding():
    # Construct a case where lot-size rounding could push a position over cap.
    # single_cap = equity * 10% = 100. price=100, lot_size=3 -> only qty
    # multiples of 3 allowed: 0,3,6,9(=900)... Choose raw notional just under
    # cap boundary such that naive floor-round still respects cap, then verify
    # reapply_caps trims any case where it wouldn't.
    equity = Decimal("1000")
    caps = _caps(single=Decimal("10"), order=Decimal("10"))  # cap = 100
    price = Decimal("100")
    lot_size = Decimal("1.5")
    # raw qty targeting notional 99 -> 0.99 -> floor to lot multiple of 1.5 -> 0
    # Instead pick a qty that floors UP relative to cap: e.g target notional
    # slightly above cap but rounds down to something still above cap.
    raw_qty = Decimal("1.05")  # notional = 105, over cap of 100
    trimmed = reapply_caps_after_rounding(raw_qty, price, lot_size, equity, caps)
    trimmed_notional = abs(trimmed) * price
    cap_amount = equity * caps.max_single_position_pct / Decimal("100")
    assert trimmed_notional <= cap_amount


def test_caps_reapplied_after_rounding_noop_when_within_cap():
    equity = Decimal("1000")
    caps = _caps(single=Decimal("10"), order=Decimal("10"))  # cap = 100
    price = Decimal("100")
    lot_size = Decimal("0.1")
    raw_qty = Decimal("0.5")  # notional 50, within cap
    result = reapply_caps_after_rounding(raw_qty, price, lot_size, equity, caps)
    assert result == raw_qty


# ---------------------------------------------------------------------------
# build_order_intents()
# ---------------------------------------------------------------------------


def _market_data():
    return {
        "BTC": InstrumentMarketData("BTC", Decimal("50000"), 0.6, Decimal("500"), Decimal("1"), Decimal("0.0001")),
        "ETH": InstrumentMarketData("ETH", Decimal("3000"), 0.8, Decimal("50"), Decimal("0.01"), Decimal("0.001")),
    }


def test_build_order_intents_deterministic_intent_id():
    decision_id = uuid.uuid4()
    snapshot_id = uuid.uuid4()
    created_at = _now()
    final_targets = {"BTC": Decimal("5000")}
    current_notional = {"BTC": Decimal("0")}
    caps = _caps()

    out1 = build_order_intents(
        final_targets, current_notional, _market_data(), caps, "hyperliquid", "acct1",
        decision_id, snapshot_id, created_at, 25, 10, Decimal("1000"),
    )
    out2 = build_order_intents(
        final_targets, current_notional, _market_data(), caps, "hyperliquid", "acct1",
        decision_id, snapshot_id, created_at, 25, 10, Decimal("1000"),
    )
    assert len(out1) == 1 and len(out2) == 1
    assert out1[0].intent_id == out2[0].intent_id


def test_build_order_intents_expires_at_plus_4h():
    decision_id = uuid.uuid4()
    snapshot_id = uuid.uuid4()
    created_at = _now()
    final_targets = {"BTC": Decimal("5000")}
    current_notional = {"BTC": Decimal("0")}
    out = build_order_intents(
        final_targets, current_notional, _market_data(), _caps(), "hyperliquid", "acct1",
        decision_id, snapshot_id, created_at, 25, 10, Decimal("1000"),
    )
    assert out[0].expires_at == created_at + timedelta(hours=4)


def test_build_order_intents_stop_price_set_on_open():
    decision_id = uuid.uuid4()
    snapshot_id = uuid.uuid4()
    created_at = _now()
    final_targets = {"BTC": Decimal("5000"), "ETH": Decimal("3000")}
    current_notional = {"BTC": Decimal("0"), "ETH": Decimal("0")}
    out = build_order_intents(
        final_targets, current_notional, _market_data(), _caps(), "hyperliquid", "acct1",
        decision_id, snapshot_id, created_at, 25, 10, Decimal("1000"),
    )
    opens = [i for i in out if i.effect == "open"]
    assert len(opens) > 0
    for intent in opens:
        assert intent.stop_price is not None


def test_build_order_intents_explicit_close_emitted():
    decision_id = uuid.uuid4()
    snapshot_id = uuid.uuid4()
    created_at = _now()
    final_targets = {"BTC": Decimal("0")}
    current_notional = {"BTC": Decimal("5000")}
    out = build_order_intents(
        final_targets, current_notional, _market_data(), _caps(), "hyperliquid", "acct1",
        decision_id, snapshot_id, created_at, 25, 10, Decimal("1000"),
    )
    assert len(out) == 1
    assert out[0].effect == "close"
    assert out[0].reduce_only is True


def test_build_order_intents_quantity_is_trade_delta_not_target():
    """Close/reduce quantities must be the trade delta, not the target size.

    Close of a 5000-notional BTC position at price 50000, lot 0.0001:
    delta qty = 5000/50000 = 0.1 BTC exactly. Reduce from 5000 -> 2500:
    delta qty = 2500/50000 = 0.05 BTC.
    """
    decision_id = uuid.uuid4()
    snapshot_id = uuid.uuid4()
    created_at = _now()

    out = build_order_intents(
        {"BTC": Decimal("0")}, {"BTC": Decimal("5000")}, _market_data(), _caps(),
        "hyperliquid", "acct1", decision_id, snapshot_id, created_at, 25, 10, Decimal("100000"),
    )
    assert len(out) == 1
    assert out[0].effect == "close"
    assert out[0].quantity == Decimal("0.1")

    out2 = build_order_intents(
        {"BTC": Decimal("2500")}, {"BTC": Decimal("5000")}, _market_data(), _caps(),
        "hyperliquid", "acct1", decision_id, snapshot_id, created_at, 25, 10, Decimal("100000"),
    )
    assert len(out2) == 1
    assert out2[0].effect == "reduce"
    assert out2[0].side == "sell"
    assert out2[0].quantity == Decimal("0.05")
