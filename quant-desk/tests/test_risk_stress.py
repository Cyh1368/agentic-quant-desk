from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from quantdesk.risk.stress import StressInputs, funding_spike, gap_down, run_all, venue_outage
from quantdesk.common.schemas import OrderIntent


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


def test_gap_down_exact_decimal():
    intent = make_intent(quantity=Decimal("0.01"))  # notional = 500
    inputs = StressInputs(mark_price=Decimal("50000"), daily_vol_pct=Decimal("5"))
    result = gap_down(intent, inputs)
    assert result.executable_loss == Decimal("75.00")  # 500 * 0.15
    assert result.scenario_id == "gap_down"


def test_funding_spike_exact_decimal():
    intent = make_intent(quantity=Decimal("0.01"))
    inputs = StressInputs(
        mark_price=Decimal("50000"), daily_vol_pct=Decimal("5"), funding_rate_shock_pct=Decimal("2")
    )
    result = funding_spike(intent, inputs)
    assert result.executable_loss == Decimal("10.000")  # 500 * 0.02


def test_venue_outage_exact_decimal():
    intent = make_intent(quantity=Decimal("0.01"))
    inputs = StressInputs(mark_price=Decimal("50000"), daily_vol_pct=Decimal("5"))
    result = venue_outage(intent, inputs)
    # 500 * (5/100 * 2) = 500 * 0.10
    assert result.executable_loss == Decimal("50.000")


def test_usd_notional_quantity_unit():
    intent = make_intent(quantity=Decimal("500"), quantity_unit="USD-notional")
    inputs = StressInputs(mark_price=Decimal("50000"), daily_vol_pct=Decimal("5"))
    result = gap_down(intent, inputs)
    assert result.executable_loss == Decimal("75.00")


def test_within_limit_flag():
    intent = make_intent(quantity=Decimal("0.01"))
    inputs = StressInputs(
        mark_price=Decimal("50000"), daily_vol_pct=Decimal("5"), stress_loss_limit=Decimal("10")
    )
    result = gap_down(intent, inputs)
    assert result.within_limit is False


def test_run_all_returns_three_scenarios():
    intent = make_intent()
    inputs = StressInputs(mark_price=Decimal("50000"), daily_vol_pct=Decimal("5"))
    results = run_all(intent, inputs)
    ids = {r.scenario_id for r in results}
    assert ids == {"gap_down", "funding_spike", "venue_outage"}
    for r in results:
        assert isinstance(r.executable_loss, Decimal)
