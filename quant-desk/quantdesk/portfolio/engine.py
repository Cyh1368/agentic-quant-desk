"""Deterministic portfolio construction engine (plan §8).

Pure, deterministic pipeline: calibration -> reliability shrinkage ->
de-duplication -> covariance -> ERC-approx sizing -> turnover suppression ->
tick/lot rounding -> re-capping -> OrderIntent construction.

No network/LLM imports, no imports from quantdesk/ledger/. All money and
quantity values are Decimal; all datetimes are UTC-aware.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from typing import Literal
from uuid import UUID

from quantdesk.common.schemas import ForecastSignal, OrderIntent

# Fixed namespace used for deterministic intent_id generation (uuid5).
# Arbitrary but stable constant, documented here rather than reused from
# elsewhere so this module has no cross-package coupling.
INTENT_ID_NAMESPACE = uuid.UUID("6f1e2a2e-6b8b-4a2a-9d2e-3a2b6c1d9e10")


# ---------------------------------------------------------------------------
# 1. Local models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdvisorTrackRecord:
    advisor_id: str
    prospective_sample_count: int
    contract_minimum_sample_size: int
    calibration_buckets: list[tuple[float, float, float]] = field(default_factory=list)
    reliability: float = 0.0  # 0.0 if unknown


@dataclass(frozen=True)
class CalibratedForecast:
    advisor_id: str
    instrument_id: str
    signal_id: UUID
    evidence_feature_ids: list[str]
    calibrated_probability_positive: float
    expected_excess_return_bps: float
    weight: float


@dataclass(frozen=True)
class PortfolioCaps:
    max_single_position_pct: Decimal
    max_gross_exposure_pct: Decimal
    max_net_directional_pct: Decimal
    max_order_size_pct_equity: Decimal


@dataclass(frozen=True)
class InstrumentMarketData:
    instrument_id: str
    price: Decimal
    realized_vol_annual: float
    atr: Decimal
    tick_size: Decimal
    lot_size: Decimal


# ---------------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------------


def _sign_adjust(action: str, magnitude: float) -> float:
    if action == "long":
        return abs(magnitude)
    if action == "short":
        return -abs(magnitude)
    return 0.0


def calibrate(
    forecasts: list[ForecastSignal],
    track_records: dict[str, AdvisorTrackRecord],
) -> list[CalibratedForecast]:
    out: list[CalibratedForecast] = []
    for fc in forecasts:
        tr = track_records.get(fc.advisor_id)
        gate_passed = tr is not None and tr.prospective_sample_count >= tr.contract_minimum_sample_size

        if fc.calibrated_probability_positive is not None and fc.expected_excess_return_bps is not None:
            prob = fc.calibrated_probability_positive
            excess = fc.expected_excess_return_bps
            if fc.action == "flat":
                excess = 0.0
        elif fc.action == "flat":
            prob = 0.5
            excess = 0.0
        elif tr is not None and tr.calibration_buckets:
            prob = None
            for lo, hi, rate in tr.calibration_buckets:
                if lo <= fc.raw_score < hi:
                    prob = rate
                    break
            if prob is None:
                prob = 0.5
                excess = 0.0
            else:
                base_mag = abs(fc.expected_excess_return_bps) if fc.expected_excess_return_bps is not None else 100.0
                raw_excess = (prob - 0.5) * 2 * base_mag
                excess = _sign_adjust(fc.action, raw_excess)
        else:
            prob = 0.5
            excess = 0.0

        weight = 0.0
        if gate_passed:
            weight = tr.reliability

        out.append(
            CalibratedForecast(
                advisor_id=fc.advisor_id,
                instrument_id=fc.instrument_id,
                signal_id=fc.signal_id,
                evidence_feature_ids=list(fc.evidence_feature_ids),
                calibrated_probability_positive=prob,
                expected_excess_return_bps=excess,
                weight=weight,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 3. Reliability shrinkage
# ---------------------------------------------------------------------------


def apply_reliability_shrinkage(
    calibrated: list[CalibratedForecast],
    track_records: dict[str, AdvisorTrackRecord],
    shrinkage: float,
) -> list[CalibratedForecast]:
    out: list[CalibratedForecast] = []
    for cf in calibrated:
        if cf.weight == 0.0:
            out.append(cf)
            continue
        tr = track_records.get(cf.advisor_id)
        new_weight = (tr.reliability if tr is not None else 0.0) * (1 - shrinkage)
        out.append(replace(cf, weight=new_weight))
    return out


# ---------------------------------------------------------------------------
# 4. De-duplication / clustering
# ---------------------------------------------------------------------------


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        # Empty vs empty is NOT a match by spec -- treat as zero similarity.
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def dedup_cluster(calibrated: list[CalibratedForecast]) -> list[CalibratedForecast]:
    by_instrument: dict[str, list[CalibratedForecast]] = {}
    for cf in calibrated:
        by_instrument.setdefault(cf.instrument_id, []).append(cf)

    result: list[CalibratedForecast] = []
    for instrument_id, members in by_instrument.items():
        n = len(members)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n):
            for j in range(i + 1, n):
                if _jaccard(members[i].evidence_feature_ids, members[j].evidence_feature_ids) > 0.5:
                    union(i, j)

        clusters: dict[int, list[CalibratedForecast]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(members[i])

        for cluster_members in clusters.values():
            weights = [m.weight for m in cluster_members]
            total_w = sum(weights)
            if total_w > 0:
                prob = sum(m.calibrated_probability_positive * m.weight for m in cluster_members) / total_w
                excess = sum(m.expected_excess_return_bps * m.weight for m in cluster_members) / total_w
                cluster_weight = sum(weights) / len(weights)
            else:
                prob = sum(m.calibrated_probability_positive for m in cluster_members) / len(cluster_members)
                excess = sum(m.expected_excess_return_bps for m in cluster_members) / len(cluster_members)
                cluster_weight = 0.0

            evidence_union: list[str] = sorted({fid for m in cluster_members for fid in m.evidence_feature_ids})
            rep = sorted(cluster_members, key=lambda m: (m.advisor_id, str(m.signal_id)))[0]

            result.append(
                CalibratedForecast(
                    advisor_id=rep.advisor_id,
                    instrument_id=instrument_id,
                    signal_id=rep.signal_id,
                    evidence_feature_ids=evidence_union,
                    calibrated_probability_positive=prob,
                    expected_excess_return_bps=excess,
                    weight=cluster_weight,
                )
            )
    return result


# ---------------------------------------------------------------------------
# 5. Covariance
# ---------------------------------------------------------------------------


def shrunk_covariance(
    instruments: list[str],
    vols: dict[str, float],
    correlation: float,
    shrinkage: float = 0.2,
) -> dict[tuple[str, str], float]:
    cov: dict[tuple[str, str], float] = {}
    for a in instruments:
        for b in instruments:
            if a == b:
                cov[(a, b)] = vols[a] * vols[a]
            else:
                cov[(a, b)] = correlation * vols[a] * vols[b]

    shrunk: dict[tuple[str, str], float] = {}
    for a in instruments:
        for b in instruments:
            if a == b:
                shrunk[(a, b)] = cov[(a, b)]
            else:
                shrunk[(a, b)] = (1 - shrinkage) * cov[(a, b)]
    return shrunk


def crash_override_covariance(
    cov: dict[tuple[str, str], float],
    instruments: list[str],
    crash_correlation_override: float,
) -> dict[tuple[str, str], float]:
    """Separate covariance dict with all pairwise correlations forced to
    ``crash_correlation_override``; per-asset vols (diagonal) unchanged.

    Used only for exposure-cap / stress checks -- the shrunk covariance
    from :func:`shrunk_covariance` remains the one used for ERC sizing math.
    """
    vols = {a: math.sqrt(cov[(a, a)]) for a in instruments}
    out: dict[tuple[str, str], float] = {}
    for a in instruments:
        for b in instruments:
            if a == b:
                out[(a, b)] = vols[a] * vols[a]
            else:
                out[(a, b)] = crash_correlation_override * vols[a] * vols[b]
    return out


# ---------------------------------------------------------------------------
# 6. Sizing
# ---------------------------------------------------------------------------


def _net_edge_by_instrument(clustered: list[CalibratedForecast]) -> dict[str, float]:
    net: dict[str, float] = {}
    for cf in clustered:
        net[cf.instrument_id] = net.get(cf.instrument_id, 0.0) + cf.weight * cf.expected_excess_return_bps
    return net


def size_positions(
    clustered: list[CalibratedForecast],
    market_data: dict[str, InstrumentMarketData],
    cov: dict[tuple[str, str], float],
    crash_cov: dict[tuple[str, str], float],
    target_annual_vol_pct: float,
    equity_usd: Decimal,
    caps: PortfolioCaps,
) -> dict[str, Decimal]:
    instruments = sorted(market_data.keys())
    net_edge = _net_edge_by_instrument(clustered)

    inv_vol = {i: (1.0 / market_data[i].realized_vol_annual if market_data[i].realized_vol_annual > 0 else 0.0) for i in instruments}
    total_inv_vol = sum(inv_vol.values())
    if total_inv_vol == 0:
        return {i: Decimal("0") for i in instruments}

    raw_weight = {i: inv_vol[i] / total_inv_vol for i in instruments}

    # Signed raw weight by direction of net calibrated edge.
    signed_weight: dict[str, float] = {}
    for i in instruments:
        edge = net_edge.get(i, 0.0)
        if edge == 0.0:
            signed_weight[i] = 0.0
        else:
            signed_weight[i] = raw_weight[i] * (1.0 if edge > 0 else -1.0)

    # Portfolio vol of unit-scaled signed weights, using shrunk covariance.
    port_var = 0.0
    for a in instruments:
        for b in instruments:
            port_var += signed_weight[a] * signed_weight[b] * cov[(a, b)]
    port_vol = math.sqrt(port_var) if port_var > 0 else 0.0

    target_vol = target_annual_vol_pct / 100.0
    if port_vol == 0:
        scale = 0.0
    else:
        scale = target_vol / port_vol

    notional: dict[str, Decimal] = {}
    equity_f = float(equity_usd)
    for i in instruments:
        w = signed_weight[i] * scale
        notional[i] = Decimal(str(w * equity_f))

    # --- Re-apply caps ---
    # 1. max_single_position_pct per instrument.
    single_cap = equity_usd * caps.max_single_position_pct / Decimal("100")
    for i in instruments:
        if abs(notional[i]) > single_cap:
            notional[i] = single_cap if notional[i] > 0 else -single_cap

    # 2. max_gross_exposure_pct: scale ALL positions down proportionally.
    gross = sum(abs(v) for v in notional.values())
    gross_cap = equity_usd * caps.max_gross_exposure_pct / Decimal("100")
    if gross > gross_cap and gross > 0:
        factor = gross_cap / gross
        notional = {i: v * factor for i, v in notional.items()}

    # 3. max_net_directional_pct: cap plain signed sum of notional (at least
    #    as strict as plain signed-sum capping -- documented judgment call).
    net_cap = equity_usd * caps.max_net_directional_pct / Decimal("100")
    net_sum = sum(notional.values())
    if abs(net_sum) > net_cap and net_sum != 0:
        factor = net_cap / abs(net_sum)
        notional = {i: v * factor for i, v in notional.items()}

    return notional


# ---------------------------------------------------------------------------
# 7. Turnover suppression
# ---------------------------------------------------------------------------


def suppress_turnover(
    target_notional: dict[str, Decimal],
    current_notional: dict[str, Decimal],
    round_trip_cost_bps: dict[str, Decimal],
    min_trade_cost_multiple: float,
) -> dict[str, Decimal]:
    result: dict[str, Decimal] = {}
    for i, target in target_notional.items():
        current = current_notional.get(i, Decimal("0"))
        proposed_change = target - current
        cost_bps = round_trip_cost_bps.get(i, Decimal("0"))
        cost_usd = abs(proposed_change) * cost_bps / Decimal("10000")
        threshold = cost_usd * Decimal(str(min_trade_cost_multiple))
        if abs(proposed_change) < threshold:
            result[i] = current
        else:
            result[i] = target
    return result


# ---------------------------------------------------------------------------
# 8. Rounding + re-capping
# ---------------------------------------------------------------------------


def round_to_tick_lot(notional: Decimal, price: Decimal, tick_size: Decimal, lot_size: Decimal) -> Decimal:
    if price == 0 or lot_size == 0:
        return Decimal("0")
    sign = 1 if notional >= 0 else -1
    qty = abs(notional) / price
    lots = (qty / lot_size).to_integral_value(rounding=ROUND_DOWN)
    rounded_qty = lots * lot_size
    return sign * rounded_qty


def reapply_caps_after_rounding(
    rounded_qty: Decimal,
    price: Decimal,
    lot_size: Decimal,
    equity_usd: Decimal,
    caps: PortfolioCaps,
) -> Decimal:
    """Re-check single-position and order-size caps against the rounded
    notional (rounded_qty * price); trim down (never up) to fit if the
    lot-size rounding pushed it over a cap boundary."""
    if rounded_qty == 0 or price == 0 or lot_size == 0:
        return rounded_qty

    sign = 1 if rounded_qty >= 0 else -1
    notional = abs(rounded_qty) * price

    single_cap = equity_usd * caps.max_single_position_pct / Decimal("100")
    order_cap = equity_usd * caps.max_order_size_pct_equity / Decimal("100")
    # order_cap governs the size of THIS order; single_cap governs final
    # position size. Both apply to a from-flat order; use the tighter one.
    tightest_cap = min(single_cap, order_cap)

    if notional <= tightest_cap:
        return rounded_qty

    max_qty = tightest_cap / price
    max_lots = (max_qty / lot_size).to_integral_value(rounding=ROUND_DOWN)
    trimmed_qty = max_lots * lot_size
    return sign * trimmed_qty


# ---------------------------------------------------------------------------
# 9. Order intent construction
# ---------------------------------------------------------------------------


def _effect_for(current: Decimal, target: Decimal) -> Literal["open", "increase", "reduce", "close"]:
    if target == 0 and current != 0:
        return "close"
    if current == 0:
        return "open"
    same_sign = (current > 0) == (target > 0)
    if same_sign and abs(target) > abs(current):
        return "increase"
    if same_sign and abs(target) < abs(current):
        return "reduce"
    # sign flip: treat as open of new direction (exposure-increasing risk).
    return "open"


def build_order_intents(
    final_targets: dict[str, Decimal],
    current_notional: dict[str, Decimal],
    market_data: dict[str, InstrumentMarketData],
    caps: PortfolioCaps,
    venue: str,
    account_id: str,
    decision_id: UUID,
    snapshot_id: UUID,
    created_at: datetime,
    max_slippage_bps: int,
    max_fee_bps: int,
    equity_usd: Decimal,
    atr_stop_multiple: float = 1.0,
) -> list[OrderIntent]:
    intents: list[OrderIntent] = []
    expires_at = created_at + timedelta(hours=4)

    for instrument_id, target_notional in final_targets.items():
        current = current_notional.get(instrument_id, Decimal("0"))
        md = market_data[instrument_id]

        rounded_qty = round_to_tick_lot(target_notional, md.price, md.tick_size, md.lot_size)
        rounded_qty = reapply_caps_after_rounding(rounded_qty, md.price, md.lot_size, equity_usd, caps)

        # Determine net change in signed notional terms using rounded qty vs current.
        rounded_notional = rounded_qty * md.price

        net_change = rounded_notional - current
        if net_change == 0:
            # No change (includes flat -> flat). An explicit close (target 0,
            # current nonzero) has net_change == -current != 0 and proceeds.
            continue

        # The order quantity is the TRADE DELTA (|net_change|), not the target
        # position size: a close trades the full current position, an
        # increase/reduce trades only the difference. Rounded down to lot.
        trade_qty = abs(round_to_tick_lot(net_change, md.price, md.tick_size, md.lot_size))
        if trade_qty == 0:
            # Delta smaller than one lot -> not representable, skip.
            continue

        side: Literal["buy", "sell"] = "buy" if net_change > 0 else "sell"
        effect = _effect_for(current, rounded_notional)

        base_symbol = instrument_id.split("-")[0].split("/")[0]

        stop_price: Decimal | None = None
        if effect == "open":
            if side == "buy":
                stop_price = md.price - md.atr * Decimal(str(atr_stop_multiple))
            else:
                stop_price = md.price + md.atr * Decimal(str(atr_stop_multiple))

        intent_id = uuid.uuid5(INTENT_ID_NAMESPACE, f"{decision_id}:{instrument_id}:{side}")

        intent = OrderIntent(
            intent_id=intent_id,
            decision_id=decision_id,
            created_at=created_at,
            expires_at=expires_at,
            venue=venue,
            account_id=account_id,
            instrument_id=instrument_id,
            instrument_type="spot",
            side=side,
            effect=effect,
            quantity=trade_qty,
            quantity_unit=base_symbol,
            order_type="limit",
            limit_price=md.price,
            stop_price=stop_price,
            time_in_force="ALO",
            reduce_only=effect in ("reduce", "close"),
            max_slippage_bps=max_slippage_bps,
            max_fee_bps=max_fee_bps,
            snapshot_id=snapshot_id,
            risk_verdict_id=None,
        )
        intents.append(intent)

    return intents


# ---------------------------------------------------------------------------
# 10. Top-level orchestrator
# ---------------------------------------------------------------------------


def run_portfolio_engine(
    forecasts: list[ForecastSignal],
    track_records: dict[str, AdvisorTrackRecord],
    current_notional: dict[str, Decimal],
    market_data: dict[str, InstrumentMarketData],
    caps: PortfolioCaps,
    target_annual_vol_pct: float,
    crash_correlation_override: float,
    min_trade_cost_multiple: float,
    reliability_shrinkage: float,
    round_trip_cost_bps: dict[str, Decimal],
    decision_id: UUID,
    snapshot_id: UUID,
    created_at: datetime,
    venue: str,
    account_id: str,
    max_slippage_bps: int,
    max_fee_bps: int,
    equity_usd: Decimal,
    correlation: float = 0.5,
    covariance_shrinkage: float = 0.2,
    atr_stop_multiple: float = 1.0,
) -> list[OrderIntent]:
    calibrated = calibrate(forecasts, track_records)
    shrunk = apply_reliability_shrinkage(calibrated, track_records, reliability_shrinkage)
    clustered = dedup_cluster(shrunk)

    instruments = sorted(market_data.keys())
    vols = {i: market_data[i].realized_vol_annual for i in instruments}
    cov = shrunk_covariance(instruments, vols, correlation, covariance_shrinkage)
    crash_cov = crash_override_covariance(cov, instruments, crash_correlation_override)

    target_notional = size_positions(
        clustered, market_data, cov, crash_cov, target_annual_vol_pct, equity_usd, caps
    )

    final_targets = suppress_turnover(
        target_notional, current_notional, round_trip_cost_bps, min_trade_cost_multiple
    )

    return build_order_intents(
        final_targets,
        current_notional,
        market_data,
        caps,
        venue,
        account_id,
        decision_id,
        snapshot_id,
        created_at,
        max_slippage_bps,
        max_fee_bps,
        equity_usd,
        atr_stop_multiple,
    )
