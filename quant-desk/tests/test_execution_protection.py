from __future__ import annotations

from decimal import Decimal

import pytest

from quantdesk.execution.protection import (
    Position,
    ProtectiveOrder,
    find_unprotected_positions,
    resize_protection_for_partial_fill,
)


def test_fully_covered_long_position_not_unprotected():
    positions = [Position(instrument_id="BTC-PERP", quantity=Decimal("1"), entry_price=Decimal("100"))]
    orders = [
        ProtectiveOrder(
            instrument_id="BTC-PERP",
            order_type="stop_market",
            side="sell",
            quantity=Decimal("1"),
            stop_price=Decimal("90"),
        )
    ]
    assert find_unprotected_positions(positions, orders) == []


def test_uncovered_position_is_unprotected():
    positions = [Position(instrument_id="ETH-PERP", quantity=Decimal("-2"), entry_price=Decimal("50"))]
    orders: list[ProtectiveOrder] = []
    result = find_unprotected_positions(positions, orders)
    assert result == positions


def test_wrong_side_stop_does_not_cover():
    positions = [Position(instrument_id="BTC-PERP", quantity=Decimal("1"), entry_price=Decimal("100"))]
    orders = [
        ProtectiveOrder(
            instrument_id="BTC-PERP",
            order_type="stop_market",
            side="buy",  # wrong side for a long position's protection
            quantity=Decimal("5"),
            stop_price=Decimal("90"),
        )
    ]
    assert find_unprotected_positions(positions, orders) == positions


def test_summed_partial_stops_cover_position():
    positions = [Position(instrument_id="BTC-PERP", quantity=Decimal("1"), entry_price=Decimal("100"))]
    orders = [
        ProtectiveOrder(
            instrument_id="BTC-PERP",
            order_type="stop_market",
            side="sell",
            quantity=Decimal("0.4"),
            stop_price=Decimal("90"),
        ),
        ProtectiveOrder(
            instrument_id="BTC-PERP",
            order_type="stop_market",
            side="sell",
            quantity=Decimal("0.6"),
            stop_price=Decimal("91"),
        ),
    ]
    assert find_unprotected_positions(positions, orders) == []


def test_partial_coverage_still_unprotected():
    positions = [Position(instrument_id="BTC-PERP", quantity=Decimal("1"), entry_price=Decimal("100"))]
    orders = [
        ProtectiveOrder(
            instrument_id="BTC-PERP",
            order_type="stop_market",
            side="sell",
            quantity=Decimal("0.5"),
            stop_price=Decimal("90"),
        )
    ]
    assert find_unprotected_positions(positions, orders) == positions


def test_flat_position_excluded():
    positions = [Position(instrument_id="BTC-PERP", quantity=Decimal("0"), entry_price=Decimal("100"))]
    assert find_unprotected_positions(positions, []) == []


def test_cancelled_order_does_not_count_as_coverage():
    positions = [Position(instrument_id="BTC-PERP", quantity=Decimal("1"), entry_price=Decimal("100"))]
    orders = [
        ProtectiveOrder(
            instrument_id="BTC-PERP",
            order_type="stop_market",
            side="sell",
            quantity=Decimal("1"),
            stop_price=Decimal("90"),
            status="cancelled",
        )
    ]
    assert find_unprotected_positions(positions, orders) == positions


def test_resize_shrinks_quantity():
    original = ProtectiveOrder(
        instrument_id="BTC-PERP",
        order_type="stop_market",
        side="sell",
        quantity=Decimal("1"),
        stop_price=Decimal("90"),
    )
    resized = resize_protection_for_partial_fill(original, Decimal("0.4"))
    assert resized.quantity == Decimal("0.4")
    assert resized.instrument_id == original.instrument_id
    assert resized.order_type == original.order_type
    assert resized.side == original.side
    assert resized.stop_price == original.stop_price
    # original unchanged
    assert original.quantity == Decimal("1")


def test_resize_raises_on_overgrow():
    original = ProtectiveOrder(
        instrument_id="BTC-PERP",
        order_type="stop_market",
        side="sell",
        quantity=Decimal("1"),
        stop_price=Decimal("90"),
    )
    with pytest.raises(ValueError):
        resize_protection_for_partial_fill(original, Decimal("1.5"))


def test_resize_raises_on_nonpositive():
    original = ProtectiveOrder(
        instrument_id="BTC-PERP",
        order_type="stop_market",
        side="sell",
        quantity=Decimal("1"),
        stop_price=Decimal("90"),
    )
    with pytest.raises(ValueError):
        resize_protection_for_partial_fill(original, Decimal("0"))
    with pytest.raises(ValueError):
        resize_protection_for_partial_fill(original, Decimal("-1"))
