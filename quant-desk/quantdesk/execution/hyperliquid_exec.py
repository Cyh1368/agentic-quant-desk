"""Hyperliquid execution adapter — TESTNET ONLY in this version.

Deterministic code, no LLM imports (plan §10). The sole holder of the
trading credential. Mainnet is refused at construction until Gate C
(venue admission checklist, 30 days clean shadow, drills) is passed —
that unlock is a deliberate human code change, not a config flip.

Uses the official hyperliquid-python-sdk. Client order IDs (cloid) are
derived deterministically from intent_id (see execution.state), so any
retry of the same intent is venue-side idempotent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from quantdesk.execution.state import client_order_id

TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"
MAINNET_BLOCK_MESSAGE = (
    "Mainnet execution is not unlocked (Gate C). This adapter runs on "
    "testnet only; unlocking mainnet is a reviewed code change."
)


class MainnetNotUnlockedError(RuntimeError):
    pass


@dataclass
class ExecResult:
    ok: bool
    status: str                 # "filled" | "resting" | "rejected" | "error"
    cloid: str
    filled_size: Decimal
    avg_price: Decimal | None
    raw: Any


def round_price_for_venue(px: float, sz_decimals: int) -> float:
    """Hyperliquid px rule: max 5 significant figures AND max
    (6 - szDecimals) decimal places. DRILL FINDING (2026-07-12): venue
    rejects TP/SL prices violating this with 'Invalid TP/SL price'."""
    return round(float(f"{px:.5g}"), 6 - sz_decimals)


def _cloid_hex(intent_id) -> str:
    """Hyperliquid cloid must be a 16-byte hex string (0x + 32 hex chars)."""
    digest = client_order_id(intent_id).removeprefix("qd-")  # 20 hex chars
    return "0x" + digest.ljust(32, "0")


class HyperliquidTestnetExecutor:
    def __init__(self, private_key: str | None = None, network: str = "testnet"):
        if network != "testnet":
            raise MainnetNotUnlockedError(MAINNET_BLOCK_MESSAGE)
        key = (private_key or os.environ.get("HYPERLIQUID_AGENT_PRIVATE_KEY", "")).strip()
        if not key:
            raise RuntimeError("HYPERLIQUID_AGENT_PRIVATE_KEY not set")
        from eth_account import Account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info

        self._wallet = Account.from_key(key)
        # Agent/API wallets sign on behalf of the main account; balances and
        # positions live at the main address.
        main = os.environ.get("HYPERLIQUID_MAIN_WALLET_ADDRESS", "").strip()
        self.address = main or self._wallet.address
        self.info = Info(TESTNET_API_URL, skip_ws=True)
        meta = self.info.meta()
        self.sz_decimals = {a["name"]: a["szDecimals"] for a in meta["universe"]}
        self.exchange = Exchange(
            self._wallet, TESTNET_API_URL,
            account_address=self.address,
        )

    # -- account state ----------------------------------------------------
    def balances(self) -> dict:
        """Perp account summary: withdrawable, margin, positions."""
        state = self.info.user_state(self.address)
        summary = state.get("marginSummary", {})
        return {
            "account_value": Decimal(str(summary.get("accountValue", "0"))),
            "withdrawable": Decimal(str(state.get("withdrawable", "0"))),
            "total_margin_used": Decimal(str(summary.get("totalMarginUsed", "0"))),
            "positions": [
                {
                    "coin": p["position"]["coin"],
                    "size": Decimal(str(p["position"]["szi"])),
                    "entry_px": p["position"].get("entryPx"),
                    "unrealized_pnl": p["position"].get("unrealizedPnl"),
                }
                for p in state.get("assetPositions", [])
            ],
        }

    def open_orders(self) -> list[dict]:
        return self.info.open_orders(self.address)

    # -- orders -----------------------------------------------------------
    def cloid_already_submitted(self, cloid: str) -> dict | None:
        """Venue-truth pre-submission check.

        DRILL FINDING (2026-07-12, testnet): Hyperliquid does NOT enforce
        cloid uniqueness — the venue accepts two orders with the same
        cloid. Idempotency therefore lives HERE: every submission path
        must call this first and skip the submit when the cloid is
        already known to the venue (open, filled, or canceled).
        Returns the venue's order record, or None if unknown.
        """
        st = self.order_status_by_cloid(cloid)
        if st.get("status") == "order":
            return st.get("order")
        return None

    def submit_limit_idempotent(self, coin: str, is_buy: bool, size: float,
                                px: float, intent_id,
                                tif: str = "Gtc") -> ExecResult:
        """GTC limit order; resubmitting the same intent is a no-op."""
        from hyperliquid.utils.types import Cloid
        cloid = _cloid_hex(intent_id)
        existing = self.cloid_already_submitted(cloid)
        if existing is not None:
            return ExecResult(True, "already_submitted", cloid,
                              Decimal("0"), None, existing)
        resp = self.exchange.order(coin, is_buy, size, px,
                                   {"limit": {"tif": tif}}, cloid=Cloid(cloid))
        return self._parse_response(resp, cloid)

    def market_order(self, coin: str, is_buy: bool, size: float,
                     intent_id, slippage: float = 0.01) -> ExecResult:
        """IOC market order with deterministic cloid; idempotent on retry."""
        from hyperliquid.utils.types import Cloid
        cloid = _cloid_hex(intent_id)
        existing = self.cloid_already_submitted(cloid)
        if existing is not None:
            return ExecResult(True, "already_submitted", cloid,
                              Decimal("0"), None, existing)
        resp = self.exchange.market_open(
            coin, is_buy, size, None, slippage, cloid=Cloid(cloid)
        )
        return self._parse_response(resp, cloid)

    def close_position(self, coin: str, intent_id) -> ExecResult:
        from hyperliquid.utils.types import Cloid
        cloid = _cloid_hex(intent_id)
        resp = self.exchange.market_close(coin, cloid=Cloid(cloid))
        return self._parse_response(resp, cloid)

    def order_status_by_cloid(self, cloid: str) -> dict:
        """Venue-truth lookup for UNKNOWN recovery: query by client order id."""
        from hyperliquid.utils.types import Cloid
        return self.info.query_order_by_cloid(self.address, Cloid(cloid))

    @staticmethod
    def _parse_response(resp: Any, cloid: str) -> ExecResult:
        try:
            if resp.get("status") != "ok":
                return ExecResult(False, "rejected", cloid, Decimal("0"), None, resp)
            statuses = resp["response"]["data"]["statuses"]
            filled = Decimal("0")
            avg_px = None
            status = "resting"
            for st in statuses:
                if "filled" in st:
                    filled += Decimal(str(st["filled"]["totalSz"]))
                    avg_px = Decimal(str(st["filled"]["avgPx"]))
                    status = "filled"
                elif "error" in st:
                    return ExecResult(False, "rejected", cloid, Decimal("0"), None, resp)
            return ExecResult(True, status, cloid, filled, avg_px, resp)
        except (KeyError, TypeError, IndexError):
            return ExecResult(False, "error", cloid, Decimal("0"), None, resp)
