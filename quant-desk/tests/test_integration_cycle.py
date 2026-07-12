"""End-to-end seam test: a non-flat forecast flows through portfolio ->
risk -> intent queue -> shadow fill -> ledger, exercising the exact wiring
used by quantdesk/__main__.py."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from quantdesk.common.schemas import ForecastSignal
from quantdesk.execution.shadow import OrderBookSnapshot, simulate_fill
from quantdesk.execution.state import client_order_id
from quantdesk.ledger.store import LedgerStore
from quantdesk.portfolio.engine import (
    AdvisorTrackRecord,
    InstrumentMarketData,
    PortfolioCaps,
    run_portfolio_engine,
)
from quantdesk.risk.engine import PortfolioState, evaluate

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


def _forecast(instrument: str, snapshot_id) -> ForecastSignal:
    return ForecastSignal(
        signal_id=uuid4(),
        advisor_id="ts_momentum_baseline",
        advisor_version="test",
        generated_at=NOW,
        data_cutoff_at=NOW,
        expires_at=NOW + timedelta(hours=4),
        venue="hyperliquid",
        instrument_id=instrument,
        forecast_target="next_24h_excess_return_sign",
        horizon=timedelta(hours=24),
        action="long",
        raw_score=1.5,
        snapshot_id=snapshot_id,
    )


def test_full_cycle_long_signal_produces_ledgered_shadow_fill(tmp_path):
    store = LedgerStore(tmp_path / "ledger.sqlite")
    snapshot_id = uuid4()
    price = Decimal("100000")
    md = {
        "BTC": InstrumentMarketData(
            instrument_id="BTC", price=price, realized_vol_annual=0.5,
            atr=Decimal("800"), tick_size=Decimal("0.1"), lot_size=Decimal("0.0001"),
        )
    }
    track = {
        "ts_momentum_baseline": AdvisorTrackRecord(
            advisor_id="ts_momentum_baseline",
            prospective_sample_count=200, contract_minimum_sample_size=200,
            calibration_buckets=[(0.75, 10.0, 0.58)], reliability=1.0,
        )
    }
    equity = Decimal("1000")
    intents = run_portfolio_engine(
        forecasts=[_forecast("BTC", snapshot_id)], track_records=track,
        current_notional={"BTC": Decimal("0")}, market_data=md,
        caps=PortfolioCaps(
            max_single_position_pct=Decimal("10"),
            max_gross_exposure_pct=Decimal("30"),
            max_net_directional_pct=Decimal("20"),
            max_order_size_pct_equity=Decimal("5"),
        ),
        target_annual_vol_pct=10.0, crash_correlation_override=0.95,
        min_trade_cost_multiple=1.0, reliability_shrinkage=0.5,
        round_trip_cost_bps={"BTC": Decimal("12")},
        decision_id=uuid4(), snapshot_id=snapshot_id, created_at=NOW,
        venue="hyperliquid", account_id="shadow",
        max_slippage_bps=25, max_fee_bps=10, equity_usd=equity,
    )
    assert len(intents) == 1
    intent = intents[0]
    assert intent.side == "buy"
    assert intent.stop_price is not None  # mandatory protective exit
    assert intent.quantity > 0

    state = PortfolioState(equity=equity, mark_price=price, atr=Decimal("800"), now=NOW)
    verdict = evaluate(intent, state, {
        "account": {
            "max_gross_exposure_pct": 30, "max_net_directional_pct": 20,
            "max_single_position_pct": 10, "max_leverage_per_position": 1,
        },
        "orders": {
            "max_order_size_pct_equity": 5, "max_orders_per_hour": 4,
            "min_stop_distance_atr": 0.5, "max_stop_distance_atr": 3.0,
        },
    }, store=store)
    assert verdict.verdict in ("approve", "approve_reduced")

    store.insert_intent(intent)   # durable queue: insert == enqueue
    store.record_verdict(verdict)
    consumed = store.consume_intent(intent.intent_id)
    assert consumed is not None
    assert store.consume_intent(intent.intent_id) is None  # at-most-once

    book = OrderBookSnapshot(
        bids=[(Decimal("99995"), Decimal("5"))],
        asks=[(Decimal("100005"), Decimal("5"))], as_of=NOW,
    )
    fill = simulate_fill(intent, book, NOW)
    assert fill is not None and not fill.partial
    coid = client_order_id(intent.intent_id)
    store.insert_order(coid, intent.intent_id, "BTC", "SUBMITTING",
                       intent.model_dump(mode="json"))
    store.update_order_state(coid, "FILLED")
    store.record_fill(str(fill.fill_id), coid, "BTC", fill.quantity, fill.price,
                      fill.model_dump(mode="json"))
    store.upsert_shadow_position("BTC", fill.quantity, fill.price)
    pos = store.get_shadow_position("BTC")
    assert Decimal(str(pos["quantity"])) == fill.quantity
    assert store.integrity_check()
    store.close()
