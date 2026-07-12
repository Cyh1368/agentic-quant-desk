"""Deterministic hard risk engine (plan §9).

Authority rule, explicit and tested: intents with effect ``reduce`` or
``close`` bypass exposure caps entirely and are always allowed when
mechanically valid (correct instrument, positive quantity, not expired,
leverage <= 1, stop distance sane if a stop is attached) -- they never
wait on the halt state or on any of the exposure/order-size/order-rate
caps below. This module contains no LLM imports and never will.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from quantdesk.common.schemas import OrderIntent, RiskCheckResult, RiskVerdict, StressResult
from quantdesk.ledger.store import LedgerStore
from quantdesk.risk import halts, stress as stress_mod

HARD_CHECK_VERSION = "hard_risk_engine_v1"

_REDUCING_EFFECTS = {"reduce", "close"}


@dataclass(frozen=True)
class PortfolioState:
    """Minimal portfolio context needed to evaluate an intent's risk checks.

    All percentages are expressed as plain Decimal percent values (e.g.
    Decimal("12.5") means 12.5%), consistent with config/desk.yaml.
    ``current_position_pct`` and ``current_net_directional_pct`` /
    ``current_gross_exposure_pct`` reflect the book *before* this intent
    is applied; this module adds the intent's contribution to compute the
    post-trade observed values it checks against thresholds.
    """
    equity: Decimal
    mark_price: Decimal
    atr: Decimal
    current_gross_exposure_pct: Decimal = Decimal("0")
    current_net_directional_pct: Decimal = Decimal("0")
    current_position_pct: Decimal = Decimal("0")
    daily_vol_pct: Decimal = Decimal("5")
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _notional(intent: OrderIntent, mark_price: Decimal) -> Decimal:
    if intent.quantity_unit == "USD-notional":
        return intent.quantity
    return intent.quantity * mark_price


def _order_pct_equity(intent: OrderIntent, state: PortfolioState) -> Decimal:
    if state.equity == 0:
        return Decimal("0")
    return (_notional(intent, state.mark_price) / state.equity) * Decimal("100")


def _direction_sign(intent: OrderIntent) -> Decimal:
    return Decimal("1") if intent.side == "buy" else Decimal("-1")


def _post_trade_gross_pct(intent: OrderIntent, state: PortfolioState) -> Decimal:
    return state.current_gross_exposure_pct + _order_pct_equity(intent, state)


def _post_trade_net_pct(intent: OrderIntent, state: PortfolioState) -> Decimal:
    return state.current_net_directional_pct + _direction_sign(intent) * _order_pct_equity(intent, state)


def _post_trade_position_pct(intent: OrderIntent, state: PortfolioState) -> Decimal:
    return state.current_position_pct + _order_pct_equity(intent, state)


def _check(check_id: str, passed: bool, observed: Decimal, threshold: Decimal, unit: str) -> RiskCheckResult:
    return RiskCheckResult(
        check_id=check_id,
        passed=passed,
        observed=observed,
        threshold=threshold,
        unit=unit,
        check_code_version=HARD_CHECK_VERSION,
    )


def _leverage_check(intent: OrderIntent, config: dict) -> RiskCheckResult:
    max_lev = Decimal(str(config["account"]["max_leverage_per_position"]))
    # v1: instrument_type spot/perp both constrained to <=1x effective leverage in this engine.
    observed = Decimal("1")
    return _check("leverage", observed <= max_lev, observed, max_lev, "x")


def _stop_distance_check(intent: OrderIntent, state: PortfolioState, config: dict) -> RiskCheckResult | None:
    if intent.stop_price is None:
        return None
    min_atr = Decimal(str(config["orders"]["min_stop_distance_atr"]))
    max_atr = Decimal(str(config["orders"]["max_stop_distance_atr"]))
    if state.atr == 0:
        distance_atr = Decimal("0")
    else:
        reference_price = intent.limit_price if intent.limit_price is not None else state.mark_price
        distance_atr = abs(reference_price - intent.stop_price) / state.atr
    passed = min_atr <= distance_atr <= max_atr
    return _check("stop_distance_atr", passed, distance_atr, max_atr, "atr_multiple")


def _expiry_check(intent: OrderIntent, state: PortfolioState) -> RiskCheckResult:
    passed = intent.expires_at > state.now
    observed = Decimal("1") if passed else Decimal("0")
    return _check("expiry_not_passed", passed, observed, Decimal("1"), "bool")


def _orders_per_hour_check(intent: OrderIntent, store: LedgerStore | None, config: dict) -> RiskCheckResult:
    max_per_hour = Decimal(str(config["orders"]["max_orders_per_hour"]))
    if store is None:
        return _check("max_orders_per_hour", True, Decimal("0"), max_per_hour, "orders")
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    count = store.count_orders_since(intent.instrument_id, since)
    observed = Decimal(count)
    return _check("max_orders_per_hour", observed < max_per_hour, observed, max_per_hour, "orders")


def _halt_gate_check(intent: OrderIntent, store: LedgerStore | None) -> RiskCheckResult:
    if store is None:
        return _check("halt_gate", True, Decimal("0"), Decimal("0"), "bool")
    allowed = halts.allows(store, intent.effect)
    observed = Decimal("0") if allowed else Decimal("1")
    return _check("halt_gate", allowed, observed, Decimal("0"), "bool")


def _exposure_checks(intent: OrderIntent, state: PortfolioState, config: dict) -> list[RiskCheckResult]:
    gross_thresh = Decimal(str(config["account"]["max_gross_exposure_pct"]))
    net_thresh = Decimal(str(config["account"]["max_net_directional_pct"]))
    single_thresh = Decimal(str(config["account"]["max_single_position_pct"]))
    order_thresh = Decimal(str(config["orders"]["max_order_size_pct_equity"]))

    gross = _post_trade_gross_pct(intent, state)
    net = abs(_post_trade_net_pct(intent, state))
    single = _post_trade_position_pct(intent, state)
    order_pct = _order_pct_equity(intent, state)

    return [
        _check("max_gross_exposure_pct", gross <= gross_thresh, gross, gross_thresh, "pct_equity"),
        _check("max_net_directional_pct", net <= net_thresh, net, net_thresh, "pct_equity"),
        _check("max_single_position_pct", single <= single_thresh, single, single_thresh, "pct_equity"),
        _check("max_order_size_pct_equity", order_pct <= order_thresh, order_pct, order_thresh, "pct_equity"),
    ]


def _max_quantity_for_caps(intent: OrderIntent, state: PortfolioState, config: dict) -> Decimal:
    """Largest quantity (same unit as intent.quantity) that passes every
    quantity-dependent cap, holding everything else about the intent fixed.
    Used by approve_reduced. Returns 0 if no positive quantity can pass."""
    if state.equity == 0 or state.mark_price == 0:
        return Decimal("0")

    gross_thresh = Decimal(str(config["account"]["max_gross_exposure_pct"]))
    net_thresh = Decimal(str(config["account"]["max_net_directional_pct"]))
    single_thresh = Decimal(str(config["account"]["max_single_position_pct"]))
    order_thresh = Decimal(str(config["orders"]["max_order_size_pct_equity"]))

    # order_pct_equity per unit quantity, in percent-of-equity terms
    per_unit_pct = (state.mark_price / state.equity) * Decimal("100")
    if intent.quantity_unit == "USD-notional":
        per_unit_pct = Decimal("100") / state.equity

    if per_unit_pct <= 0:
        return Decimal("0")

    room_gross = gross_thresh - state.current_gross_exposure_pct
    room_single = single_thresh - state.current_position_pct
    room_order = order_thresh

    sign = _direction_sign(intent)
    # net cap: |current_net + sign*x| <= net_thresh
    if sign > 0:
        room_net = net_thresh - state.current_net_directional_pct
    else:
        room_net = net_thresh + state.current_net_directional_pct

    max_pct = min(room_gross, room_single, room_order, room_net)
    if max_pct <= 0:
        return Decimal("0")
    return max_pct / per_unit_pct


def evaluate(
    intent: OrderIntent,
    portfolio_state: PortfolioState,
    config: dict,
    store: LedgerStore | None = None,
    stress_inputs: "stress_mod.StressInputs | None" = None,
) -> RiskVerdict:
    """Evaluate an OrderIntent against the deterministic hard risk engine.

    Returns a RiskVerdict with verdict in {approve, reject, approve_reduced}.
    Reducing/closing intents bypass all exposure-related caps (gross, net,
    single-position, order-size-pct-equity, orders-per-hour, halt gate) and
    are approved as long as they are mechanically valid (not expired,
    leverage within bound, stop distance sane if present).
    """
    now = datetime.now(timezone.utc)
    is_reducing = intent.effect in _REDUCING_EFFECTS

    mechanical_checks: list[RiskCheckResult] = [
        _leverage_check(intent, config),
        _expiry_check(intent, portfolio_state),
    ]
    stop_check = _stop_distance_check(intent, portfolio_state, config)
    if stop_check is not None:
        mechanical_checks.append(stop_check)

    if is_reducing:
        checks = list(mechanical_checks)
        reason_codes: list[str] = ["reduce_close_bypasses_exposure_caps"]
        approved = all(c.passed for c in checks)
        approved_quantity = intent.quantity if approved else Decimal("0")
        verdict_str = "approve" if approved else "reject"
        if not approved:
            reason_codes.extend(c.check_id for c in checks if not c.passed)
    else:
        halt_check = _halt_gate_check(intent, store)
        rate_check = _orders_per_hour_check(intent, store, config)
        exposure_checks = _exposure_checks(intent, portfolio_state, config)
        checks = mechanical_checks + [halt_check, rate_check] + exposure_checks
        reason_codes = []
        failed = [c for c in checks if not c.passed]
        if not failed:
            approved = True
            approved_quantity = intent.quantity
            verdict_str = "approve"
        else:
            # Mechanical / halt / rate failures cannot be fixed by shrinking size -> reject.
            hard_fail_ids = {"leverage", "expiry_not_passed", "stop_distance_atr", "halt_gate", "max_orders_per_hour"}
            if any(c.check_id in hard_fail_ids for c in failed):
                approved = False
                approved_quantity = Decimal("0")
                verdict_str = "reject"
                reason_codes = [c.check_id for c in failed]
            else:
                max_qty = _max_quantity_for_caps(intent, portfolio_state, config)
                max_qty = min(max_qty, intent.quantity)
                if max_qty <= 0:
                    approved_quantity = Decimal("0")
                    verdict_str = "reject"
                    reason_codes = [c.check_id for c in failed]
                else:
                    approved_quantity = max_qty
                    verdict_str = "approve_reduced"
                    reason_codes = [c.check_id for c in failed] + ["quantity_shrunk_to_pass_caps"]

    stress_results: list[StressResult] = []
    if stress_inputs is not None and approved_quantity > 0:
        shrunk_intent = intent.model_copy(update={"quantity": approved_quantity})
        stress_results = stress_mod.run_all(shrunk_intent, stress_inputs)

    return RiskVerdict(
        verdict_id=uuid4(),
        intent_id=intent.intent_id,
        evaluated_at=now,
        hard_check_version=HARD_CHECK_VERSION,
        portfolio_snapshot_id=intent.snapshot_id,
        verdict=verdict_str,
        approved_quantity=approved_quantity,
        hard_checks=checks,
        stress_results=stress_results,
        llm_critic_result_id=None,
        reason_codes=reason_codes,
    )
