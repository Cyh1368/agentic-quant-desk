"""Async client for the Hyperliquid public info API (no auth required).

Docs: POST JSON to https://api.hyperliquid.xyz/info

Every response is written to the immutable raw landing zone first via
:class:`~quantdesk.data.raw_store.RawStore`, then normalized into rows
carrying full :class:`~quantdesk.common.schemas.Lineage`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx

from quantdesk.common.schemas import Lineage
from quantdesk.data.raw_store import RawStore

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
SOURCE_ID = "hyperliquid_info"
NORMALIZER_VERSION = "hyperliquid_normalizer@v1"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class HyperliquidClient:
    """Thin async wrapper around the Hyperliquid public info endpoint."""

    def __init__(
        self,
        raw_store: RawStore,
        *,
        base_url: str = HYPERLIQUID_INFO_URL,
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.raw_store = raw_store
        self.base_url = base_url
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> "HyperliquidClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, body: dict) -> Any:
        resp = await self._client.post(self.base_url, json=body)
        resp.raise_for_status()
        return resp.json()

    async def candle_snapshot(
        self, coin: str, interval: str, start_time_ms: int, end_time_ms: int
    ) -> tuple[Any, list[dict]]:
        """Fetch 1h (or other interval) candles for ``coin``.

        Returns ``(raw_payload, normalized_rows)``. Rows carry full Lineage.
        """
        body = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
            },
        }
        payload = await self._post(body)
        ingested_time = _now_utc()
        record = self.raw_store.write(
            SOURCE_ID, payload, request=body, ingested_time=ingested_time
        )
        rows = normalize_candles(
            payload,
            coin=coin,
            interval=interval,
            ingested_time=ingested_time,
            raw_payload_hash=record.raw_payload_hash,
        )
        return payload, rows

    async def meta_and_asset_ctxs(self) -> tuple[Any, list[dict]]:
        """Fetch mark price / funding / open interest for all listed assets."""
        body = {"type": "metaAndAssetCtxs"}
        payload = await self._post(body)
        ingested_time = _now_utc()
        record = self.raw_store.write(
            SOURCE_ID, payload, request=body, ingested_time=ingested_time
        )
        rows = normalize_meta_and_asset_ctxs(
            payload,
            ingested_time=ingested_time,
            raw_payload_hash=record.raw_payload_hash,
        )
        return payload, rows

    async def l2_book(self, coin: str) -> tuple[Any, dict]:
        """Fetch the L2 order book for ``coin``.

        Returns ``(raw_payload, {"bids": [(px, sz), ...], "asks": [...]})``
        with Decimal prices/sizes, best-first on both sides.
        """
        body = {"type": "l2Book", "coin": coin}
        payload = await self._post(body)
        ingested_time = _now_utc()
        self.raw_store.write(SOURCE_ID, payload, request=body, ingested_time=ingested_time)
        levels = payload.get("levels", [[], []])
        book = {
            side: [(Decimal(str(lv["px"])), Decimal(str(lv["sz"]))) for lv in levels[i]]
            for i, side in enumerate(("bids", "asks"))
        }
        return payload, book

    async def all_mids(self) -> tuple[Any, dict[str, Decimal]]:
        """Fetch mid prices for all listed assets, keyed by coin symbol."""
        body = {"type": "allMids"}
        payload = await self._post(body)
        ingested_time = _now_utc()
        self.raw_store.write(SOURCE_ID, payload, request=body, ingested_time=ingested_time)
        mids = {
            k: Decimal(str(v))
            for k, v in payload.items()
            if not k.startswith("#") and not k.startswith("@")
        }
        return payload, mids


def normalize_candles(
    payload: Any,
    *,
    coin: str,
    interval: str,
    ingested_time: datetime,
    raw_payload_hash: str,
) -> list[dict]:
    """Turn a raw candleSnapshot payload into lineage-tagged rows.

    Hyperliquid candle shape (list of objects):
    ``{"t": open_ms, "T": close_ms, "s": coin, "i": interval,
       "o": open, "h": high, "l": low, "c": close, "v": volume, "n": trades}``
    """
    rows: list[dict] = []
    for c in payload or []:
        open_time = datetime.fromtimestamp(c["t"] / 1000, tz=timezone.utc)
        close_time = datetime.fromtimestamp(c["T"] / 1000, tz=timezone.utc)
        lineage = Lineage(
            event_time=close_time,
            published_time=None,
            provider_time=close_time,
            ingested_time=ingested_time,
            available_to_strategy_time=ingested_time,
            source_id=SOURCE_ID,
            source_revision=None,
            raw_payload_hash=raw_payload_hash,
            normalizer_version=NORMALIZER_VERSION,
            quality_flags=[],
        )
        rows.append(
            {
                "instrument_id": coin,
                "interval": interval,
                "open_time": open_time.isoformat(),
                "close_time": close_time.isoformat(),
                "open": str(c["o"]),
                "high": str(c["h"]),
                "low": str(c["l"]),
                "close": str(c["c"]),
                "volume": str(c["v"]),
                "n_trades": c.get("n"),
                "lineage": lineage.model_dump(mode="json"),
            }
        )
    return rows


def normalize_meta_and_asset_ctxs(
    payload: Any,
    *,
    ingested_time: datetime,
    raw_payload_hash: str,
    coins: tuple[str, ...] = ("BTC", "ETH"),
) -> list[dict]:
    """Turn a raw metaAndAssetCtxs payload into lineage-tagged rows.

    Shape: ``[{"universe": [{"name": "BTC", ...}, ...]}, [{"funding": ...,
    "openInterest": ..., "markPx": ..., ...}, ...]]`` where the asset ctx
    list is positionally aligned with ``universe``.
    """
    universe = payload[0]["universe"]
    asset_ctxs = payload[1]
    rows: list[dict] = []
    for idx, asset in enumerate(universe):
        name = asset.get("name")
        if name not in coins:
            continue
        ctx = asset_ctxs[idx]
        lineage = Lineage(
            event_time=ingested_time,
            published_time=None,
            provider_time=None,
            ingested_time=ingested_time,
            available_to_strategy_time=ingested_time,
            source_id=SOURCE_ID,
            source_revision=None,
            raw_payload_hash=raw_payload_hash,
            normalizer_version=NORMALIZER_VERSION,
            quality_flags=[],
        )
        rows.append(
            {
                "instrument_id": name,
                "mark_px": str(ctx.get("markPx")),
                "mid_px": str(ctx.get("midPx")),
                "oracle_px": str(ctx.get("oraclePx")),
                "funding": str(ctx.get("funding")),
                "open_interest": str(ctx.get("openInterest")),
                "prev_day_px": str(ctx.get("prevDayPx")),
                "day_ntl_vlm": str(ctx.get("dayNtlVlm")),
                "lineage": lineage.model_dump(mode="json"),
            }
        )
    return rows
