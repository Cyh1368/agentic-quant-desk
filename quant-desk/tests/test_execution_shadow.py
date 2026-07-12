from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quantdesk.common.schemas import OrderIntent
from quantdesk.execution.shadow import (
    DEFAULT_TAKER_FEE_BPS,
    OrderBookSnapshot,
    simulate_fill,
)
from quantdesk.execution.state import client_order_id

NOW = datetime(2026, 7, 12, 12, 0, 0, tzinfo=timezone.utc)


def make_intent(
    quantity: Decimal,
    side: str = "buy",
    max_slippage_bps: int = 100,
    max_fee_bps: int = 10,
) -> OrderIntent:
    return OrderIntent(
        intent_id=uuid.uuid4(),
        decision_id=uuid.uuid4(),
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
        venue="hyperliquid",
        account_id="acct-1",
        instrument_id="BTC-PERP",
        instrument_type="perp",
        side=side,
        effect="open",
        quantity=quantity,
        quantity_unit="BTC",
        order_type="market",
        limit_price=None,
        stop_price=None,
        time_in_force="IOC",
        reduce_only=False,
        max_slippage_bps=max_slippage_bps,
        max_fee_bps=max_fee_bps,
        snapshot_id=uuid.uuid4(),
        risk_verdict_id=None,
    )


def make_book() -> OrderBookSnapshot:
    return OrderBookSnapshot(
        bids=[
            (Decimal("99.90"), Decimal("1")),
            (Decimal("99.80"), Decimal("1")),
            (Decimal("99.60"), Decimal("2")),
        ],
        asks=[
            (Decimal("100.00"), Decimal("1")),
            (Decimal("100.10"), Decimal("1")),
            (Decimal("100.30"), Decimal("2")),
        ],
        as_of=NOW,
    )


def test_vwap_and_fee_exact_computation():
    # walks 1@100.00 + 1@100.10 + 0.5@100.30 = 2.5 total
    # total_cost = 100.00 + 100.10 + 50.15 = 250.25
    # vwap = 250.25 / 2.5 = 100.10
    # reference_price = 100.00 (best ask)
    # slippage_bps = abs(100.10 - 100.00) / 100.00 * 10000 = 10 bps
    # fee = 2.5 * 100.10 * 4.5 / 10000 = 0.1126125
    intent = make_intent(Decimal("2.5"), side="buy", max_slippage_bps=100)
    book = make_book()
    fill = simulate_fill(intent, book, NOW)

    assert fill is not None
    expected_vwap = Decimal("100.10")
    expected_fee = Decimal("2.5") * expected_vwap * DEFAULT_TAKER_FEE_BPS / Decimal(10000)
    assert fill.price == expected_vwap
    assert fill.quantity == Decimal("2.5")
    assert fill.fee == expected_fee
    assert fill.fee_bps == DEFAULT_TAKER_FEE_BPS
    assert fill.slippage_bps == Decimal("10")
    assert fill.partial is False
    assert fill.client_order_id == client_order_id(intent.intent_id)
    assert fill.intent_id == intent.intent_id


def test_slippage_bound_rejection():
    intent = make_intent(Decimal("2.5"), side="buy", max_slippage_bps=1)
    book = make_book()
    fill = simulate_fill(intent, book, NOW)
    assert fill is None


def test_partial_fill_when_book_exhausted():
    intent = make_intent(Decimal("10"), side="buy", max_slippage_bps=1000)
    book = make_book()
    fill = simulate_fill(intent, book, NOW)

    assert fill is not None
    assert fill.partial is True
    # total book depth on asks = 1 + 1 + 2 = 4
    assert fill.quantity == Decimal("4")
    total_cost = (
        Decimal("1") * Decimal("100.00")
        + Decimal("1") * Decimal("100.10")
        + Decimal("2") * Decimal("100.30")
    )
    expected_vwap = total_cost / Decimal("4")
    assert fill.price == expected_vwap
    expected_fee = Decimal("4") * expected_vwap * DEFAULT_TAKER_FEE_BPS / Decimal(10000)
    assert fill.fee == expected_fee


def test_empty_book_returns_none():
    intent = make_intent(Decimal("1"), side="sell", max_slippage_bps=1000)
    book = OrderBookSnapshot(bids=[], asks=[(Decimal("100"), Decimal("1"))], as_of=NOW)
    fill = simulate_fill(intent, book, NOW)
    assert fill is None


def test_sell_walks_bids():
    intent = make_intent(Decimal("1.5"), side="sell", max_slippage_bps=1000)
    book = make_book()
    fill = simulate_fill(intent, book, NOW)
    assert fill is not None
    # walks 1@99.90 + 0.5@99.80 = 149.80; vwap = 149.80/1.5
    expected_vwap = (Decimal("99.90") + Decimal("0.5") * Decimal("99.80")) / Decimal("1.5")
    assert fill.price == expected_vwap
    assert fill.quantity == Decimal("1.5")


def test_zero_or_negative_quantity_raises():
    intent = make_intent(Decimal("0"), side="buy")
    book = make_book()
    with pytest.raises(ValueError):
        simulate_fill(intent, book, NOW)


def test_fee_cap_exceeded_rejects():
    intent = make_intent(Decimal("1"), side="buy", max_slippage_bps=1000, max_fee_bps=1)
    book = make_book()
    fill = simulate_fill(intent, book, NOW)
    assert fill is None
