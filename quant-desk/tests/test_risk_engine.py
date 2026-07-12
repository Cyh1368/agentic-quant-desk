from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from quantdesk.ledger.store import LedgerStore
from quantdesk.risk.engine import PortfolioState, evaluate
from quantdesk.common.schemas import OrderIntent

CONFIG = {
    "account": {
        "max_gross_exposure_pct": 30,
        "max_net_directional_pct": 20,
        "max_single_position_pct": 10,
        "max_leverage_per_position": 1,
        "daily_loss_soft_halt_pct": 0.75,
        "daily_loss_hard_halt_pct": 1.25,
        "weekly_loss_hard_halt_pct": 2.5,
        "max_drawdown_kill_pct": 4,
        "human_restart_required_after_hard_halt": True,
    },
    "orders": {
        "max_order_size_pct_equity": 5,
        "max_orders_per_hour": 4,
        "min_stop_distance_atr": 0.5,
        "max_stop_distance_atr": 3.0,
    },
}


def make_intent(**overrides) -> OrderIntent:
    now = datetime.now(timezone.utc)
    defaults = dict(
        intent_id=uuid4(),
        decision_id=uuid4(),
        created_at=now,
        expires_at=now + timedelta(minutes=30),
        venue="hyperliquid",
        account_id="acct-1",
        instrument_id="BTC",
        instrument_type="perp",
        side="buy",
        effect="open",
        quantity=Decimal("0.01"),
        quantity_unit="BTC",
        order_type="limit",
        limit_price=Decimal("50000"),
        stop_price=None,
        time_in_force="GTC",
        reduce_only=False,
        max_slippage_bps=25,
        max_fee_bps=10,
        snapshot_id=uuid4(),
        risk_verdict_id=None,
    )
    defaults.update(overrides)
    return OrderIntent(**defaults)


def make_state(**overrides) -> PortfolioState:
    defaults = dict(
        equity=Decimal("1000"),
        mark_price=Decimal("50000"),
        atr=Decimal("1000"),
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


@pytest.fixture
def store(tmp_path):
    s = LedgerStore(tmp_path / "ledger.sqlite")
    yield s
    s.close()


def test_small_order_approved(store):
    # 0.01 BTC * 50000 = 500 USD notional on 1000 equity = 50% -> too big, use smaller
    intent = make_intent(quantity=Decimal("0.0005"))  # 25 USD = 2.5% equity
    state = make_state()
    verdict = evaluate(intent, state, CONFIG, store=store)
    assert verdict.verdict == "approve"
    assert verdict.approved_quantity == intent.quantity


def test_order_exceeding_order_size_cap_reduced(store):
    # 0.002 BTC * 50000 = 100 USD = 10% equity, cap is 5% -> approve_reduced
    intent = make_intent(quantity=Decimal("0.002"))
    state = make_state()
    verdict = evaluate(intent, state, CONFIG, store=store)
    assert verdict.verdict == "approve_reduced"
    assert verdict.approved_quantity < intent.quantity
    assert verdict.approved_quantity > 0
    # Reduced quantity notional should be <= 5% of equity (50 USD @ 50000/BTC = 0.001 BTC)
    reduced_notional = verdict.approved_quantity * state.mark_price
    assert reduced_notional <= Decimal("50.01")


def test_gross_exposure_cap_blocks_new_exposure(store):
    intent = make_intent(quantity=Decimal("0.0001"))  # 5 USD = 0.5% equity, small order
    state = make_state(current_gross_exposure_pct=Decimal("29.9"))
    verdict = evaluate(intent, state, CONFIG, store=store)
    # 29.9 + 0.5 = 30.4 > 30 cap -> reduced or rejected, never a plain approve of full qty beyond cap
    assert verdict.verdict in ("approve_reduced", "reject")


def test_reduce_effect_bypasses_exposure_caps_even_when_over_cap(store):
    # Huge order that would blow every exposure cap, but effect=reduce.
    intent = make_intent(quantity=Decimal("1.0"), effect="reduce", side="sell", reduce_only=True)
    state = make_state(
        current_gross_exposure_pct=Decimal("29"),
        current_net_directional_pct=Decimal("19"),
        current_position_pct=Decimal("9"),
    )
    verdict = evaluate(intent, state, CONFIG, store=store)
    assert verdict.verdict == "approve"
    assert verdict.approved_quantity == intent.quantity
    assert "reduce_close_bypasses_exposure_caps" in verdict.reason_codes


def test_close_effect_bypasses_exposure_caps(store):
    intent = make_intent(quantity=Decimal("5.0"), effect="close", side="sell", reduce_only=True)
    state = make_state(current_gross_exposure_pct=Decimal("29"))
    verdict = evaluate(intent, state, CONFIG, store=store)
    assert verdict.verdict == "approve"


def test_reduce_effect_never_blocked_even_in_hard_halt(store):
    store.set_halt_state("HARD_HALT", reason="test", human_restart_required=True)
    intent = make_intent(effect="reduce", side="sell", reduce_only=True, quantity=Decimal("0.01"))
    state = make_state()
    verdict = evaluate(intent, state, CONFIG, store=store)
    assert verdict.verdict == "approve"


def test_open_effect_blocked_in_hard_halt(store):
    store.set_halt_state("HARD_HALT", reason="test", human_restart_required=True)
    intent = make_intent(effect="open", quantity=Decimal("0.0001"))
    state = make_state()
    verdict = evaluate(intent, state, CONFIG, store=store)
    assert verdict.verdict == "reject"
    assert any(c.check_id == "halt_gate" and not c.passed for c in verdict.hard_checks)


def test_expiry_passed_rejected(store):
    now = datetime.now(timezone.utc)
    intent = make_intent(expires_at=now - timedelta(minutes=1), quantity=Decimal("0.0001"))
    state = make_state()
    verdict = evaluate(intent, state, CONFIG, store=store)
    assert verdict.verdict == "reject"
    assert "expiry_not_passed" in verdict.reason_codes


def test_stop_distance_within_bounds_passes(store):
    intent = make_intent(
        quantity=Decimal("0.0001"), stop_price=Decimal("49500"), limit_price=Decimal("50000")
    )
    state = make_state(atr=Decimal("1000"))  # distance = 500 / 1000 = 0.5 ATR, min bound
    verdict = evaluate(intent, state, CONFIG, store=store)
    assert verdict.verdict == "approve"


def test_stop_distance_too_close_rejected(store):
    intent = make_intent(
        quantity=Decimal("0.0001"), stop_price=Decimal("49900"), limit_price=Decimal("50000")
    )
    state = make_state(atr=Decimal("1000"))  # distance = 100/1000 = 0.1 ATR < 0.5 min
    verdict = evaluate(intent, state, CONFIG, store=store)
    assert verdict.verdict == "reject"
    assert "stop_distance_atr" in verdict.reason_codes


def test_stop_distance_too_far_rejected(store):
    intent = make_intent(
        quantity=Decimal("0.0001"), stop_price=Decimal("46000"), limit_price=Decimal("50000")
    )
    state = make_state(atr=Decimal("1000"))  # distance = 4000/1000 = 4.0 ATR > 3.0 max
    verdict = evaluate(intent, state, CONFIG, store=store)
    assert verdict.verdict == "reject"


def test_max_orders_per_hour_rejects(store):
    now = datetime.now(timezone.utc)
    for _ in range(4):
        existing_intent = make_intent()
        store.insert_intent(existing_intent)
        store.insert_order(
            str(uuid4()), existing_intent.intent_id, "BTC", "FILLED", {"created_at": now.isoformat()}
        )
    intent = make_intent(quantity=Decimal("0.0001"))
    state = make_state()
    verdict = evaluate(intent, state, CONFIG, store=store)
    assert verdict.verdict == "reject"
    assert "max_orders_per_hour" in verdict.reason_codes


def test_leverage_check_uses_config_threshold(store):
    config = {**CONFIG, "account": {**CONFIG["account"], "max_leverage_per_position": 1}}
    intent = make_intent(quantity=Decimal("0.0001"))
    state = make_state()
    verdict = evaluate(intent, state, config, store=store)
    leverage_check = next(c for c in verdict.hard_checks if c.check_id == "leverage")
    assert leverage_check.passed
    assert leverage_check.threshold == Decimal("1")


def test_decimal_cap_arithmetic_exact(store):
    intent = make_intent(quantity=Decimal("0.001"))  # 50 USD notional
    state = make_state(equity=Decimal("1000"))  # exactly 5% -> at threshold, passes
    verdict = evaluate(intent, state, CONFIG, store=store)
    order_check = next(c for c in verdict.hard_checks if c.check_id == "max_order_size_pct_equity")
    assert order_check.observed == Decimal("5")
    assert order_check.passed is True
