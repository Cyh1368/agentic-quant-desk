"""Testnet execution drill (plan Phase 4, first pass).

Run with ``python -m quantdesk.execution.testnet_drill``. Verifies against
Hyperliquid TESTNET (no real capital):
  1. preflight: account reachable, balance readable
  2. tiny market order with deterministic cloid
  3. venue-truth order lookup by cloid (UNKNOWN-recovery path)
  4. position visible in user state
  5. close position (reduce-only path never blocked)
  6. final balance; full trail posted to Discord
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from quantdesk.common.alerts import DiscordNotifier
from quantdesk.common.config import load_config
from quantdesk.execution.hyperliquid_exec import HyperliquidTestnetExecutor

DRILL_COIN = "ETH"
DRILL_SIZE = 0.01   # ~ $30 notional on testnet mock USDC


def main() -> int:
    load_config()
    discord = DiscordNotifier()
    now = datetime.now(timezone.utc)
    trail = [f"TESTNET drill start {now:%Y-%m-%d %H:%M} UTC (no real capital)"]

    def log(msg: str) -> None:
        print(msg)
        trail.append(msg)

    def finish(code: int) -> int:
        if discord.enabled:
            discord.send("**testnet execution drill**\n" + "\n".join(f"- {t}" for t in trail))
        return code

    ex = HyperliquidTestnetExecutor()
    log(f"preflight: signing address {ex.address[:8]}…{ex.address[-4:]}")

    bal0 = ex.balances()
    log(f"balance before: account_value=${bal0['account_value']} "
        f"withdrawable=${bal0['withdrawable']} positions={len(bal0['positions'])}")
    if bal0["account_value"] == 0:
        log("HALT: testnet account has no mock USDC. Visit "
            "https://app.hyperliquid-testnet.xyz (connect this wallet) and "
            "claim the faucet, then rerun.")
        return finish(3)

    intent_id = uuid4()
    log(f"placing tiny market buy: {DRILL_SIZE} {DRILL_COIN} (intent {str(intent_id)[:8]})")
    res = ex.market_order(DRILL_COIN, is_buy=True, size=DRILL_SIZE, intent_id=intent_id)
    log(f"order result: {res.status} filled={res.filled_size} avg_px={res.avg_price}")
    if not res.ok:
        log(f"HALT: order rejected: {res.raw}")
        return finish(4)

    status = ex.order_status_by_cloid(res.cloid)
    log(f"venue-truth lookup by cloid: {status.get('status', status)}")

    bal_mid = ex.balances()
    pos = [p for p in bal_mid["positions"] if p["coin"] == DRILL_COIN]
    log(f"position after fill: {pos or 'NONE (check!)'}")

    close_res = ex.close_position(DRILL_COIN, uuid4())
    log(f"close result: {close_res.status} filled={close_res.filled_size} "
        f"avg_px={close_res.avg_price}")

    bal1 = ex.balances()
    pnl = bal1["account_value"] - bal0["account_value"]
    log(f"balance after: account_value=${bal1['account_value']} "
        f"(round-trip cost/pnl: ${pnl})")
    log("drill PASSED" if close_res.ok else "drill FAILED at close")
    return finish(0 if close_res.ok else 5)


if __name__ == "__main__":
    raise SystemExit(main())
