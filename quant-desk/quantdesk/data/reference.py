"""Independent reference price feed (Coinbase public spot API).

Used purely as a cross-check against venue mark price (plan §3): "cross-source
price sanity checks... run before every trading decision; divergence beyond
a threshold puts the desk in degraded mode."
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx

from quantdesk.common.schemas import Lineage
from quantdesk.data.raw_store import RawStore

COINBASE_SPOT_URL = "https://api.coinbase.com/v2/prices/{pair}/spot"
SOURCE_ID = "coinbase_spot"
NORMALIZER_VERSION = "coinbase_reference_normalizer@v1"

# Hyperliquid coin symbol -> Coinbase trading pair.
COINBASE_PAIR_FOR = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ReferencePriceClient:
    """Fetches independent spot reference prices from Coinbase."""

    def __init__(
        self,
        raw_store: RawStore,
        *,
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.raw_store = raw_store
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> "ReferencePriceClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def spot_price(self, coin: str) -> tuple[Any, dict]:
        """Fetch the current Coinbase spot price for ``coin`` (e.g. "BTC").

        Returns ``(raw_payload, normalized_row)``.
        """
        pair = COINBASE_PAIR_FOR.get(coin, f"{coin}-USD")
        url = COINBASE_SPOT_URL.format(pair=pair)
        resp = await self._client.get(url)
        resp.raise_for_status()
        payload = resp.json()
        ingested_time = _now_utc()
        record = self.raw_store.write(
            SOURCE_ID, payload, request={"pair": pair}, ingested_time=ingested_time
        )
        lineage = Lineage(
            event_time=ingested_time,
            published_time=None,
            provider_time=None,
            ingested_time=ingested_time,
            available_to_strategy_time=ingested_time,
            source_id=SOURCE_ID,
            source_revision=None,
            raw_payload_hash=record.raw_payload_hash,
            normalizer_version=NORMALIZER_VERSION,
            quality_flags=[],
        )
        data = payload["data"]
        row = {
            "instrument_id": coin,
            "pair": pair,
            "price": str(Decimal(data["amount"])),
            "currency": data.get("currency"),
            "lineage": lineage.model_dump(mode="json"),
        }
        return payload, row


@dataclass(frozen=True)
class CrossSourceCheck:
    mark: Decimal
    reference: Decimal
    divergence_bps: Decimal
    threshold_bps: Decimal
    status: str  # "ok" | "divergent"

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def cross_source_check(
    mark: Decimal | str | float,
    reference: Decimal | str | float,
    threshold_bps: Decimal | str | float,
) -> CrossSourceCheck:
    """Compare venue mark price to an independent reference price.

    ``divergence_bps`` is computed relative to the reference price:
    ``abs(mark - reference) / reference * 10_000``.
    Status is "divergent" when divergence exceeds ``threshold_bps``.
    """
    mark_d = Decimal(str(mark))
    ref_d = Decimal(str(reference))
    threshold_d = Decimal(str(threshold_bps))
    if ref_d == 0:
        raise ValueError("reference price must be non-zero")

    divergence_bps = abs(mark_d - ref_d) / ref_d * Decimal(10_000)
    status = "divergent" if divergence_bps > threshold_d else "ok"
    return CrossSourceCheck(
        mark=mark_d,
        reference=ref_d,
        divergence_bps=divergence_bps,
        threshold_bps=threshold_d,
        status=status,
    )
