"""One full shadow decision cycle: data -> advisors -> portfolio -> risk ->
shadow execution -> ledger.  Run with ``python -m quantdesk``.

Authority rule (plan v2): no LLM sits in the mandatory path.  The LLM
advisor runs shadow-only (reliability weight 0 in the portfolio engine);
only the deterministic baseline can move the shadow book.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID, uuid4

from quantdesk.advisors.baseline import ts_momentum_baseline
from quantdesk.common.alerts import DiscordNotifier
from quantdesk.advisors.llm_trend import LlmTrendAdvisorConfig, run_crypto_trend_llm_v1
from quantdesk.common.config import REPO_ROOT, load_config
from quantdesk.data.hyperliquid import HyperliquidClient
from quantdesk.data.raw_store import RawStore
from quantdesk.data.reference import ReferencePriceClient, cross_source_check
from quantdesk.data.snapshot import build_snapshot
from quantdesk.execution.protection import Position, ProtectiveOrder, find_unprotected_positions
from quantdesk.execution.shadow import OrderBookSnapshot, simulate_fill
from quantdesk.execution.state import client_order_id
from quantdesk.features.compute import Candle, compute_feature_set, feature_id
from quantdesk.ledger.store import LedgerStore
from quantdesk.portfolio.engine import (
    AdvisorTrackRecord,
    InstrumentMarketData,
    PortfolioCaps,
    run_portfolio_engine,
)
from quantdesk.risk import halts
from quantdesk.risk.engine import PortfolioState, evaluate

# Placeholder calibration for the deterministic baseline, standing in for the
# classical-backtest mapping (plan §6: backtests are valid for deterministic
# rules). Buckets: (raw_score_lo, raw_score_hi, empirical p_positive).
BASELINE_CALIBRATION_BUCKETS = [
    (-10.0, -0.75, 0.44), (-0.75, -0.25, 0.47), (-0.25, 0.25, 0.50),
    (0.25, 0.75, 0.53), (0.75, 10.0, 0.56),
]


def _baseline_track_record() -> AdvisorTrackRecord:
    return AdvisorTrackRecord(
        advisor_id="ts_momentum_baseline",
        prospective_sample_count=200,
        contract_minimum_sample_size=200,
        calibration_buckets=BASELINE_CALIBRATION_BUCKETS,
        reliability=1.0,
    )


async def _fetch_market_data(cfg: dict, raw_store: RawStore, universe: list[str]):
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 200 * 3600 * 1000  # 200h of 1h candles
    async with HyperliquidClient(raw_store) as hl:
        candles = {}
        books = {}
        for coin in universe:
            _, rows = await hl.candle_snapshot(coin, "1h", start_ms, now_ms)
            candles[coin] = rows
            _, books[coin] = await hl.l2_book(coin)
        _, ctx_rows = await hl.meta_and_asset_ctxs()
    ctx = {r["instrument_id"]: r for r in ctx_rows if r.get("instrument_id") in universe}
    async with ReferencePriceClient(raw_store) as ref_client:
        refs = {coin: (await ref_client.spot_price(coin))[1] for coin in universe}
    return candles, books, ctx, refs


def main() -> int:
    cfg = load_config()
    universe: list[str] = cfg["desk"]["universe"]
    venue: str = cfg["desk"]["venue"]
    now = datetime.now(timezone.utc)

    discord = DiscordNotifier()
    trail: list[str] = []

    def log(msg: str, severity: str = "info") -> None:
        """Print to stdout, collect into the cycle trail, and page Discord
        immediately for P0s (the end-of-cycle summary carries the rest)."""
        print(msg)
        trail.append(msg)
        if severity == "p0":
            discord.send(msg, severity="p0")

    # --- dry/live switch: config desk.mode, overridable via DESK_MODE env.
    mode = (os.environ.get("DESK_MODE") or cfg["desk"]["mode"]).strip().lower()
    if mode in ("dry", "shadow"):
        mode = "dry"
    elif mode == "live":
        network = cfg["desk"].get("network", "testnet")
        if network != "testnet":
            msg = ("desk.mode=live network=mainnet REFUSED: mainnet unlock "
                   "is a reviewed code change after Gate C sign-off "
                   "(eligibility call + kill drill + 30d clean record).")
            print(msg)
            discord.send(msg, severity="p1")
            return 2
    else:
        print(f"unknown desk.mode {mode!r}; expected dry|live")
        return 2
    executor = None
    if mode == "live":
        from quantdesk.execution.hyperliquid_exec import HyperliquidTestnetExecutor
        executor = HyperliquidTestnetExecutor()
        log(f"decision cycle start {now:%Y-%m-%d %H:%M} UTC — mode=LIVE "
            f"(network=testnet, account {executor.address[:8]}…)")
    else:
        log(f"decision cycle start {now:%Y-%m-%d %H:%M} UTC — mode={mode}")

    data_dir = REPO_ROOT / cfg["storage"]["ledger_path"]
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    store = LedgerStore(data_dir)
    raw_store = RawStore(REPO_ROOT / cfg["storage"]["raw_landing_dir"])

    candles_raw, books_raw, ctx, refs = asyncio.run(
        _fetch_market_data(cfg, raw_store, universe)
    )

    # --- cross-source sanity: divergence => degraded mode (no new exposure)
    degraded = False
    for coin in universe:
        mark = Decimal(str(ctx[coin]["mark_px"]))
        check = cross_source_check(
            mark, refs[coin]["price"], cfg["process"]["cross_source_divergence_bps"]
        )
        if not check.ok:
            degraded = True
            log(f"DEGRADED: {coin} mark {mark} vs ref {refs[coin]['price']} "
                f"({check.divergence_bps} bps)", severity="p1")

    # --- features + snapshot
    features: dict = {}
    market_data: dict[str, InstrumentMarketData] = {}
    newest_close = None
    for coin in universe:
        cds = [Candle.from_row(r) for r in candles_raw[coin]]
        features.update(compute_feature_set(coin, cds))
        newest = datetime.fromisoformat(cds[-1].close_time_iso)
        newest_close = newest if newest_close is None or newest > newest_close else newest_close
        market_data[coin] = InstrumentMarketData(
            instrument_id=coin,
            price=Decimal(str(ctx[coin]["mark_px"])),
            realized_vol_annual=features[feature_id(coin, "rvol_7d")] or 0.5,
            atr=Decimal(str(features[feature_id(coin, "atr_14")])),
            tick_size=Decimal("0.1") if coin == "BTC" else Decimal("0.01"),
            lot_size=Decimal("0.0001") if coin == "BTC" else Decimal("0.001"),
        )

    positions = []
    current_notional: dict[str, Decimal] = {}
    for coin in universe:
        pos = store.get_shadow_position(coin)
        qty = Decimal(str(pos["quantity"])) if pos else Decimal("0")
        positions.append({"instrument_id": coin, "quantity": str(qty)})
        current_notional[coin] = qty * market_data[coin].price

    snapshot = build_snapshot(
        [ctx[c] for c in universe], features, positions, now,
        snapshot_dir=REPO_ROOT / cfg["storage"]["snapshot_dir"],
        newest_candle_close_time=newest_close,
        stale_data_max_minutes=90,
        extra_quality_flags=(["cross_source_divergent"] if degraded else None),
    )
    snapshot_id = UUID(str(snapshot["snapshot_id"]))

    # --- advisors
    forecasts = []
    for coin in universe:
        forecasts.append(ts_momentum_baseline(
            instrument_id=coin, venue=venue,
            features={
                "return_24h": features[feature_id(coin, "ret_24h")],
                "return_7d": features[feature_id(coin, "ret_7d")],
                "realized_vol_7d": features[feature_id(coin, "rvol_7d")],
                "trend_state": features[feature_id(coin, "trend_state")],
            },
            generated_at=now, data_cutoff_at=now, snapshot_id=snapshot_id,
        ))

    llm_cfg = cfg["llm"]
    key_env = "OPENROUTER_API_KEY" if llm_cfg["provider"] == "openrouter" else "ANTHROPIC_API_KEY"
    if os.environ.get(key_env):
        forecasts += run_crypto_trend_llm_v1(
            snapshot=snapshot, instrument_ids=universe,
            calibration_summary={"status": "shadow", "note": "no prospective record yet"},
            config=LlmTrendAdvisorConfig(
                advisor_model=llm_cfg["advisor_model"],
                monthly_budget_usd=Decimal(str(llm_cfg["monthly_budget_usd"])),
                max_cost_per_decision_usd=Decimal(str(llm_cfg["max_cost_per_decision_usd"])),
                provider=llm_cfg["provider"],
            ),
            snapshot_id=snapshot_id, generated_at=now, data_cutoff_at=now,
            monthly_spend=lambda: store.monthly_llm_spend(now.year, now.month),
            persist=lambda prov: (
                store.insert_model_provenance(prov),
                store.log_llm_cost(prov.cost_usd, advisor_id="crypto_trend_llm_v1"),
            ),
        )
    else:
        log(f"NOTE: {key_env} not set — LLM advisor skipped this cycle.")

    for fc in forecasts:
        store.insert_forecast(fc)

    # --- portfolio engine (LLM advisor has no track record -> weight 0)
    track_records = {"ts_momentum_baseline": _baseline_track_record()}
    acct, orders_cfg, pf = cfg["account"], cfg["orders"], cfg["portfolio"]
    equity = Decimal(str(acct["shadow_equity_usd"]))
    decision_id = uuid4()

    intents = run_portfolio_engine(
        forecasts=forecasts, track_records=track_records,
        current_notional=current_notional, market_data=market_data,
        caps=PortfolioCaps(
            max_single_position_pct=Decimal(str(acct["max_single_position_pct"])),
            max_gross_exposure_pct=Decimal(str(acct["max_gross_exposure_pct"])),
            max_net_directional_pct=Decimal(str(acct["max_net_directional_pct"])),
            max_order_size_pct_equity=Decimal(str(orders_cfg["max_order_size_pct_equity"])),
        ),
        target_annual_vol_pct=pf["target_annual_vol_pct"],
        crash_correlation_override=pf["crash_correlation_override"],
        min_trade_cost_multiple=pf["min_trade_cost_multiple"],
        reliability_shrinkage=pf["reliability_shrinkage"],
        round_trip_cost_bps={c: Decimal("12") for c in universe},
        decision_id=decision_id, snapshot_id=snapshot_id, created_at=now,
        venue=venue, account_id="shadow",
        max_slippage_bps=orders_cfg["default_max_slippage_bps"],
        max_fee_bps=orders_cfg["default_max_fee_bps"],
        equity_usd=equity,
    )

    # --- risk gate, then shadow execution
    gross = sum(abs(v) for v in current_notional.values())
    net = sum(current_notional.values())
    fills = []
    for intent in intents:
        if degraded and intent.effect in ("open", "increase"):
            log(f"degraded mode: skipping new-exposure intent {intent.instrument_id}")
            continue
        md = market_data[intent.instrument_id]
        state = PortfolioState(
            equity=equity, mark_price=md.price, atr=md.atr,
            current_gross_exposure_pct=gross / equity * 100,
            current_net_directional_pct=net / equity * 100,
            current_position_pct=abs(current_notional[intent.instrument_id]) / equity * 100,
            now=now,
        )
        verdict = evaluate(intent, state, cfg, store=store)
        store.insert_intent(intent)   # durable queue: insert == enqueue
        store.record_verdict(verdict)
        if verdict.verdict == "reject":
            log(f"risk REJECT {intent.instrument_id} {intent.side}: {verdict.reason_codes}")
            continue

        consumed = store.consume_intent(intent.intent_id)
        if consumed is None:
            continue
        book = OrderBookSnapshot(
            bids=books_raw[intent.instrument_id]["bids"],
            asks=books_raw[intent.instrument_id]["asks"], as_of=now,
        )
        exec_intent = intent if verdict.verdict == "approve" else intent.model_copy(
            update={"quantity": verdict.approved_quantity}
        )
        coid = client_order_id(intent.intent_id)
        store.insert_order(coid, intent.intent_id, intent.instrument_id, "SUBMITTING", intent.model_dump(mode="json"))
        if executor is not None:
            # LIVE path: idempotent venue submission + mandatory stop attach.
            res = executor.market_order(
                intent.instrument_id, intent.side == "buy",
                float(exec_intent.quantity), intent.intent_id,
            )
            if not res.ok or res.filled_size == 0:
                store.update_order_state(coid, "REJECTED")
                log(f"LIVE order not filled ({res.status}) {intent.instrument_id}", severity="p1")
                continue
            store.update_order_state(coid, "FILLED")
            if intent.stop_price is not None and intent.effect in ("open", "increase"):
                from hyperliquid.utils.types import Cloid
                from quantdesk.execution.hyperliquid_exec import _cloid_hex
                # Venue px rule: 5 sig figs / szDecimals-bounded decimals
                # (drill finding: violations rejected as Invalid TP/SL price).
                from quantdesk.execution.hyperliquid_exec import round_price_for_venue
                stop_px = round_price_for_venue(
                    float(intent.stop_price),
                    executor.sz_decimals[intent.instrument_id])
                stop_resp = executor.exchange.order(
                    intent.instrument_id, intent.side != "buy",
                    float(res.filled_size), stop_px,
                    {"trigger": {"triggerPx": stop_px,
                                 "isMarket": True, "tpsl": "sl"}},
                    reduce_only=True, cloid=Cloid(_cloid_hex(uuid4())),
                )
                stop_statuses = (stop_resp.get("response", {})
                                 .get("data", {}).get("statuses", []))
                stop_failed = (stop_resp.get("status") != "ok"
                               or any("error" in st for st in stop_statuses))
                if stop_failed:
                    log(f"P0: stop attach FAILED for {intent.instrument_id}: "
                        f"{stop_statuses or stop_resp}; flattening", severity="p0")
                    executor.close_position(intent.instrument_id, uuid4())
                    continue
            signed = res.filled_size if intent.side == "buy" else -res.filled_size
            prev = store.get_shadow_position(intent.instrument_id)
            prev_qty = Decimal(str(prev["quantity"])) if prev else Decimal("0")
            store.record_fill(res.cloid, coid, intent.instrument_id,
                              res.filled_size, res.avg_price, {"live": True})
            store.upsert_shadow_position(intent.instrument_id, prev_qty + signed, res.avg_price)
            fills.append(res)
            log(f"LIVE fill: {intent.side} {res.filled_size} {intent.instrument_id} "
                f"@ {res.avg_price} (stop {intent.stop_price})")
            continue
        fill = simulate_fill(exec_intent, book, now)
        if fill is None:
            store.update_order_state(coid, "REJECTED")
            log(f"shadow fill rejected (slippage/fee bound) {intent.instrument_id}")
            continue
        store.update_order_state(coid, "FILLED" if not fill.partial else "PARTIALLY_FILLED")
        signed = fill.quantity if intent.side == "buy" else -fill.quantity
        prev = store.get_shadow_position(intent.instrument_id)
        prev_qty = Decimal(str(prev["quantity"])) if prev else Decimal("0")
        store.record_fill(str(fill.fill_id), coid, intent.instrument_id,
                          fill.quantity, fill.price, fill.model_dump(mode="json"))
        store.upsert_shadow_position(intent.instrument_id, prev_qty + signed, fill.price)
        fills.append(fill)
        log(f"shadow fill: {intent.side} {fill.quantity} {intent.instrument_id} "
            f"@ {fill.price} (slip {fill.slippage_bps}bps, fee ${fill.fee})")

    # --- protection watchdog pass (shadow: stops are ledger-recorded intents)
    open_positions = []
    for coin in universe:
        pos = store.get_shadow_position(coin)
        if pos and Decimal(str(pos["quantity"])) != 0:
            open_positions.append(Position(
                instrument_id=coin, quantity=Decimal(str(pos["quantity"])),
                entry_price=Decimal(str(pos.get("avg_price") or pos.get("avg_entry_price") or 0)),
            ))
    protective = [
        ProtectiveOrder(
            instrument_id=i.instrument_id, order_type="stop_market",
            side="sell" if i.side == "buy" else "buy",
            quantity=i.quantity, stop_price=i.stop_price,
        )
        for i in intents if i.stop_price is not None
    ]
    if executor is not None:
        # LIVE: reconcile ledger vs venue truth; protection from venue orders.
        venue = executor.balances()
        for vp in venue["positions"]:
            ledg = store.get_shadow_position(vp["coin"])
            lq = Decimal(str(ledg["quantity"])) if ledg else Decimal("0")
            if lq != vp["size"]:
                log(f"P0 RECONCILIATION: {vp['coin']} ledger={lq} venue={vp['size']}", severity="p0")
        triggers = [o for o in executor.info.frontend_open_orders(executor.address)
                    if o.get("isTrigger")]
        protective = [
            ProtectiveOrder(instrument_id=o["coin"], order_type="stop_market",
                            side=o["side"] if o.get("side") in ("buy", "sell")
                            else ("sell" if o.get("side") == "A" else "buy"),
                            quantity=Decimal(str(o["sz"])),
                            stop_price=Decimal(str(o.get("triggerPx", 0))))
            for o in triggers
        ]
        open_positions = [
            Position(instrument_id=vp["coin"], quantity=vp["size"],
                     entry_price=Decimal(str(vp.get("entry_px") or 0)))
            for vp in venue["positions"] if vp["size"] != 0
        ]
    unprotected = find_unprotected_positions(open_positions, protective)
    for p in unprotected:
        log(f"P0 WATCHDOG: unprotected position {p.instrument_id} qty {p.quantity}", severity="p0")

    halt = halts.current_state(store)

    # --- sentiment research summary (report-only: research forecasts are
    # structurally isolated from orders; this is informational context) ----
    try:
        import sqlite3 as _sq
        _rc = _sq.connect(REPO_ROOT / "data" / "research.sqlite")
        _rc.row_factory = _sq.Row
        import json as _json
        for _r in _rc.execute(
            "SELECT payload FROM research_forecasts ORDER BY rowid DESC LIMIT 4"
        ):
            _d = _json.loads(_r["payload"])
            trail.append(
                f"sentiment research {_d['advisor_id']} {_d['instrument_id']}: "
                f"{'abstain' if _d['abstain'] else _d['direction']}"
                + (f" (p+={_d['probability_positive']})" if _d.get('probability_positive') else "")
                + " [research-only, no order influence]"
            )
        _rc.close()
    except Exception:
        trail.append("sentiment research: unavailable this cycle")

    # Explicit per-asset position decision, every cycle.
    intents_by_asset = {}
    for i in intents:
        intents_by_asset.setdefault(i.instrument_id, []).append(i)
    for coin in universe:
        pos_notional = current_notional[coin]
        coin_intents = intents_by_asset.get(coin, [])
        if coin_intents:
            desc = "; ".join(f"{i.side} {i.quantity} ({i.effect})" for i in coin_intents)
            trail.append(f"POSITION DECISION {coin}: {desc} | current ${pos_notional:.2f}")
        else:
            actions = {fc.advisor_id: fc.action for fc in forecasts if fc.instrument_id == coin}
            trail.append(
                f"POSITION DECISION {coin}: hold (target flat, no trade) | "
                f"current ${pos_notional:.2f} | advisor views: {actions}"
            )
    for fc in forecasts:
        trail.append(
            f"forecast {fc.advisor_id} {fc.instrument_id}: {fc.action} "
            f"(score {fc.raw_score:+.2f}) — {fc.thesis[:120]}"
        )
    log(f"cycle complete: {len(forecasts)} forecasts, {len(intents)} intents, "
        f"{len(fills)} shadow fills, halt={halt['state']}, "
        f"llm spend this month=${store.monthly_llm_spend(now.year, now.month)}")
    if discord.enabled:
        delivered = discord.send(
            "**quant-desk decision trail**\n" + "\n".join(f"- {t}" for t in trail)
        )
        if not delivered:
            print("WARN: decision trail failed to deliver to Discord.")
    else:
        print("NOTE: DISCORD_WEBHOOK_URL not set — decision trail not posted.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
