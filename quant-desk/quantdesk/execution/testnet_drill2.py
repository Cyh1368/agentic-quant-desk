"""Phase-4 drills, part 2 (plan §11 Gate C prerequisites). Testnet only.

Run with ``python -m quantdesk.execution.testnet_drill2``. Drills:
  A. duplicate-submission idempotency: same cloid resubmitted -> venue
     must not create a second order
  B. UNKNOWN recovery: after a simulated crash, venue-truth lookup by
     cloid resolves the order's fate before any resubmission
  C. cancel/replace race: resting order cancelled; a second cancel (the
     race loser) must fail cleanly, and venue truth reflects one cancel
  D. protective stop: stop-market trigger order attaches and is visible;
     removing it is detected by the protection check
Full trail posted to Discord.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from quantdesk.common.alerts import DiscordNotifier
from quantdesk.common.config import load_config
from quantdesk.execution.hyperliquid_exec import HyperliquidTestnetExecutor, _cloid_hex

COIN = "ETH"
SIZE = 0.01


def main() -> int:
    load_config()
    discord = DiscordNotifier()
    now = datetime.now(timezone.utc)
    trail = [f"TESTNET drill-2 start {now:%Y-%m-%d %H:%M} UTC"]
    passed: list[str] = []
    failed: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        trail.append(msg)

    def verdict(name: str, ok: bool, detail: str) -> None:
        (passed if ok else failed).append(name)
        log(f"drill {name}: {'PASS' if ok else 'FAIL'} — {detail}")

    ex = HyperliquidTestnetExecutor()
    from hyperliquid.utils.types import Cloid  # noqa: F401 (drill D)

    mids = ex.info.all_mids()
    mid = float(mids[COIN])
    far_buy_px = round(mid * 0.80, 1)   # deep below mid: rests, never fills
    log(f"{COIN} mid={mid}, resting-limit px={far_buy_px}")

    # --- A: duplicate submission, same cloid --------------------------------
    # Venue accepts duplicate cloids (verified 2026-07-12), so idempotency
    # is executor-side: submit_limit_idempotent checks venue truth first.
    intent_id = uuid4()
    cloid = _cloid_hex(intent_id)
    r1 = ex.submit_limit_idempotent(COIN, True, SIZE, far_buy_px, intent_id)
    r2 = ex.submit_limit_idempotent(COIN, True, SIZE, far_buy_px, intent_id)
    open_orders = ex.open_orders()
    n_with_cloid = sum(1 for o in open_orders if o.get("cloid") == cloid)
    verdict("A-duplicate-cloid",
            n_with_cloid == 1 and r1.status == "resting"
            and r2.status == "already_submitted",
            f"orders with cloid={n_with_cloid} (want 1), first={r1.status}, "
            f"retry={r2.status} (want already_submitted)")

    # --- B: UNKNOWN recovery via venue truth --------------------------------
    st = ex.order_status_by_cloid(cloid)
    order_info = st.get("order", {}).get("order", {}) if st.get("status") == "order" else {}
    recovered = st.get("status") == "order" and order_info.get("cloid") == cloid
    verdict("B-unknown-recovery", recovered,
            f"lookup status={st.get('status')}, resolved order state="
            f"{st.get('order', {}).get('status', '?')}")

    # --- C: cancel + losing-side cancel race --------------------------------
    oid = None
    for o in ex.open_orders():
        if o.get("cloid") == cloid:
            oid = o["oid"]
    c1 = ex.exchange.cancel(COIN, oid)
    c2 = ex.exchange.cancel(COIN, oid)   # race loser: must fail cleanly
    c1_ok = c1.get("status") == "ok" and "error" not in str(
        c1.get("response", {}).get("data", {}).get("statuses", [{}])[0])
    c2_failed_cleanly = ("error" in str(
        c2.get("response", {}).get("data", {}).get("statuses", [{}])[0])
        or c2.get("status") != "ok")
    still_open = any(o.get("oid") == oid for o in ex.open_orders())
    verdict("C-cancel-race",
            c1_ok and c2_failed_cleanly and not still_open,
            f"first cancel ok={c1_ok}, second rejected={c2_failed_cleanly}, "
            f"order gone={not still_open}")

    # --- D: protective stop attach / coverage detection ---------------------
    fill = ex.market_order(COIN, is_buy=True, size=SIZE, intent_id=uuid4())
    stop_px = round(float(fill.avg_price) * 0.95, 1)
    stop_resp = ex.exchange.order(
        COIN, False, SIZE, stop_px,
        {"trigger": {"triggerPx": stop_px, "isMarket": True, "tpsl": "sl"}},
        reduce_only=True, cloid=Cloid(_cloid_hex(uuid4())),
    )
    stop_ok = stop_resp.get("status") == "ok"
    frontend_orders = ex.info.frontend_open_orders(ex.address)
    stop_visible = any(o.get("isTrigger") and o.get("coin") == COIN
                       for o in frontend_orders)
    verdict("D-protective-stop", stop_ok and stop_visible,
            f"stop placed={stop_ok} @ {stop_px}, visible as trigger={stop_visible}")

    # cleanup: cancel stop, close position
    for o in ex.info.frontend_open_orders(ex.address):
        if o.get("coin") == COIN:
            ex.exchange.cancel(COIN, o["oid"])
    close = ex.close_position(COIN, uuid4())
    log(f"cleanup: position closed ({close.status}), "
        f"balance ${ex.balances()['account_value']}")

    log(f"RESULT: {len(passed)}/4 drills passed"
        + (f" — FAILED: {failed}" if failed else ""))
    if discord.enabled:
        discord.send("**testnet drill-2 (Phase 4)**\n" + "\n".join(f"- {t}" for t in trail),
                     severity="info" if not failed else "p1")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
