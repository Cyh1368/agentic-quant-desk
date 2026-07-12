"""Shadow execution engine: conservative taker fill simulation (plan §11).

Deterministic given its inputs: no randomness, no wall-clock reads except
via the explicit `now` parameter.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from quantdesk.common.schemas import OrderIntent
from quantdesk.execution.state import client_order_id as _client_order_id

FEE_SCHEDULE_VERSION = "hyperliquid-v1"
DEFAULT_TAKER_FEE_BPS = Decimal("4.5")

FEE_SCHEDULES: dict[str, dict[str, Decimal]] = {
    "hyperliquid-v1": {"taker_bps": DEFAULT_TAKER_FEE_BPS},
}


def get_fee_bps(schedule_version: str = FEE_SCHEDULE_VERSION) -> Decimal:
    return FEE_SCHEDULES[schedule_version]["taker_bps"]


class OrderBookSnapshot(BaseModel):
    bids: list[tuple[Decimal, Decimal]]  # (price, size), best-first (highest first)
    asks: list[tuple[Decimal, Decimal]]  # (price, size), best-first (lowest first)
    as_of: datetime


class ShadowFill(BaseModel):
    fill_id: UUID
    intent_id: UUID
    client_order_id: str
    price: Decimal  # volume-weighted average fill price
    quantity: Decimal  # filled quantity, may be < intent.quantity if partial
    fee: Decimal  # USD
    fee_bps: Decimal
    fee_schedule_version: str
    slippage_bps: Decimal  # abs value, >= 0, vs top-of-book reference price
    filled_at: datetime
    partial: bool


def simulate_fill(
    intent: OrderIntent,
    book: OrderBookSnapshot,
    now: datetime,
    fee_schedule_version: str = FEE_SCHEDULE_VERSION,
) -> ShadowFill | None:
    """Conservative taker simulation: walk the book, VWAP the consumed depth.

    Returns None (no fill) when:
      - the relevant side of the book is empty,
      - the resulting slippage (vs top-of-book reference price) exceeds
        intent.max_slippage_bps, computed on the actually-consumed VWAP
        (whether the fill ended up full or partial), or
      - the fee schedule's fee_bps exceeds intent.max_fee_bps.

    Raises ValueError if intent.quantity <= 0.
    """
    if intent.quantity <= 0:
        raise ValueError("intent.quantity must be > 0")

    if intent.side == "buy":
        levels = book.asks
    else:
        levels = book.bids

    if not levels:
        return None

    reference_price = levels[0][0]

    remaining = intent.quantity
    total_cost = Decimal("0")
    total_qty = Decimal("0")
    for price, size in levels:
        if remaining <= 0:
            break
        take = size if size <= remaining else remaining
        total_cost += take * price
        total_qty += take
        remaining -= take

    if total_qty <= 0:
        return None

    vwap = total_cost / total_qty
    partial = total_qty < intent.quantity

    slippage_bps = abs(vwap - reference_price) / reference_price * Decimal(10000)
    if slippage_bps > intent.max_slippage_bps:
        return None

    fee_bps = get_fee_bps(fee_schedule_version)
    if fee_bps > intent.max_fee_bps:
        return None

    fee = total_qty * vwap * fee_bps / Decimal(10000)

    fill_id = uuid.uuid4()

    return ShadowFill(
        fill_id=fill_id,
        intent_id=intent.intent_id,
        client_order_id=_client_order_id(intent.intent_id),
        price=vwap,
        quantity=total_qty,
        fee=fee,
        fee_bps=fee_bps,
        fee_schedule_version=fee_schedule_version,
        slippage_bps=slippage_bps,
        filled_at=now,
        partial=partial,
    )
