"""Protective-stop coverage checker (plan §9/§10): protection is verified,
not assumed. Pure and deterministic; no ledger/portfolio imports.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel


class Position(BaseModel):
    instrument_id: str
    quantity: Decimal  # signed: positive=long, negative=short
    entry_price: Decimal


class ProtectiveOrder(BaseModel):
    instrument_id: str
    order_type: Literal["stop_market", "stop_limit"]
    side: Literal["buy", "sell"]
    quantity: Decimal
    stop_price: Decimal
    status: Literal["open", "filled", "cancelled"] = "open"


def find_unprotected_positions(
    positions: list[Position], protective_orders: list[ProtectiveOrder]
) -> list[Position]:
    """Return positions lacking full protective coverage, input order preserved.

    A position is protected when open protective orders on the same
    instrument, on the closing side, sum to >= abs(position.quantity).
    Any coverage gap (partial or none) counts as unprotected. Flat
    positions (quantity == 0) are always excluded.
    """
    unprotected: list[Position] = []
    for position in positions:
        if position.quantity == 0:
            continue
        closing_side: Literal["buy", "sell"] = "sell" if position.quantity > 0 else "buy"
        covered = sum(
            (
                order.quantity
                for order in protective_orders
                if order.instrument_id == position.instrument_id
                and order.side == closing_side
                and order.status == "open"
            ),
            Decimal("0"),
        )
        if covered < abs(position.quantity):
            unprotected.append(position)
    return unprotected


def resize_protection_for_partial_fill(
    original_protective_order: ProtectiveOrder, filled_quantity: Decimal
) -> ProtectiveOrder:
    """Return a new ProtectiveOrder resized to match an actual partial fill.

    Never mutates the input. The resized quantity must be strictly between
    0 (exclusive) and the original order's quantity (inclusive) — a stop
    should only ever shrink to match the actual fill, never grow.
    """
    if filled_quantity <= 0:
        raise ValueError("filled_quantity must be > 0")
    if filled_quantity > original_protective_order.quantity:
        raise ValueError(
            "filled_quantity cannot exceed original protective order quantity"
        )
    return ProtectiveOrder(
        instrument_id=original_protective_order.instrument_id,
        order_type=original_protective_order.order_type,
        side=original_protective_order.side,
        quantity=filled_quantity,
        stop_price=original_protective_order.stop_price,
        status=original_protective_order.status,
    )
