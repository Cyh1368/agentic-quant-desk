"""Stress scenarios producing StressResult (plan §9).

All arithmetic is Decimal. Each scenario estimates the *executable loss*
if the scenario occurred right now against the intent under evaluation,
not merely stop distance -- a stop is an order instruction, not a
guaranteed fill.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quantdesk.common.schemas import OrderIntent, StressResult

GAP_DOWN_PCT = Decimal("0.15")          # price gaps 15% through the stop
VENUE_OUTAGE_HOURS = Decimal("24")      # cannot exit for 24h
VENUE_OUTAGE_VOL_MULT = Decimal("2")    # at 2x daily vol move


@dataclass(frozen=True)
class StressInputs:
    """Market/position context a stress scenario needs.

    All monetary/quantity fields are Decimal. ``mark_price`` is the
    current reference price for the instrument; ``daily_vol_pct`` is the
    instrument's estimated daily volatility as a percent (e.g. Decimal("5")
    for 5%); ``funding_rate_shock_pct`` is the assumed adverse funding-rate
    jump used by the funding_spike scenario, expressed as percent of
    notional over the stress horizon.
    """
    mark_price: Decimal
    daily_vol_pct: Decimal
    funding_rate_shock_pct: Decimal = Decimal("1.0")
    stress_loss_limit: Decimal = Decimal("1e18")  # effectively "no separate cap" by default
    loss_unit: str = "USD"


def _notional(intent: OrderIntent, price: Decimal) -> Decimal:
    if intent.quantity_unit == "USD-notional":
        return intent.quantity
    return intent.quantity * price


def gap_down(intent: OrderIntent, inputs: StressInputs) -> StressResult:
    """Price gaps GAP_DOWN_PCT through any stop; loss realized on the full position."""
    notional = _notional(intent, inputs.mark_price)
    loss = notional * GAP_DOWN_PCT
    within = loss <= inputs.stress_loss_limit
    return StressResult(
        scenario_id="gap_down",
        executable_loss=loss,
        loss_unit=inputs.loss_unit,
        within_limit=within,
        detail=f"gap of {GAP_DOWN_PCT * 100}% through stop on notional {notional}",
    )


def funding_spike(intent: OrderIntent, inputs: StressInputs) -> StressResult:
    """An adverse funding-rate jump applied to the position's notional."""
    notional = _notional(intent, inputs.mark_price)
    loss = notional * (inputs.funding_rate_shock_pct / Decimal("100"))
    within = loss <= inputs.stress_loss_limit
    return StressResult(
        scenario_id="funding_spike",
        executable_loss=loss,
        loss_unit=inputs.loss_unit,
        within_limit=within,
        detail=f"funding shock {inputs.funding_rate_shock_pct}% on notional {notional}",
    )


def venue_outage(intent: OrderIntent, inputs: StressInputs) -> StressResult:
    """Cannot exit for 24h; assume a 2x daily-vol adverse move over that window."""
    notional = _notional(intent, inputs.mark_price)
    move_pct = (inputs.daily_vol_pct / Decimal("100")) * VENUE_OUTAGE_VOL_MULT
    loss = notional * move_pct
    within = loss <= inputs.stress_loss_limit
    return StressResult(
        scenario_id="venue_outage",
        executable_loss=loss,
        loss_unit=inputs.loss_unit,
        within_limit=within,
        detail=(
            f"unable to exit for {VENUE_OUTAGE_HOURS}h, "
            f"{VENUE_OUTAGE_VOL_MULT}x daily vol ({inputs.daily_vol_pct}%) adverse move"
        ),
    )


def run_all(intent: OrderIntent, inputs: StressInputs) -> list[StressResult]:
    return [gap_down(intent, inputs), funding_spike(intent, inputs), venue_outage(intent, inputs)]
