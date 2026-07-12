from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from quantdesk.data.hyperliquid import (
    HyperliquidClient,
    normalize_candles,
    normalize_meta_and_asset_ctxs,
)
from quantdesk.data.raw_store import RawStore

CANDLE_PAYLOAD = [
    {
        "t": 1765836000000,
        "T": 1765839599999,
        "s": "BTC",
        "i": "1h",
        "o": "86226.0",
        "c": "86255.0",
        "h": "86255.0",
        "l": "85817.0",
        "v": "486.20977",
        "n": 14220,
    }
]

META_PAYLOAD = [
    {
        "universe": [
            {"name": "BTC", "szDecimals": 5},
            {"name": "ETH", "szDecimals": 4},
            {"name": "SOL", "szDecimals": 2},
        ]
    },
    [
        {"markPx": "63911.0", "midPx": "63911.5", "oraclePx": "63914.0", "funding": "0.0000125", "openInterest": "37530.76798", "prevDayPx": "64183.0", "dayNtlVlm": "737762641.06"},
        {"markPx": "1801.4", "midPx": "1801.55", "oraclePx": "1801.8", "funding": "0.0000125", "openInterest": "786126.4024", "prevDayPx": "1798.7", "dayNtlVlm": "401840690.78"},
        {"markPx": "150.0", "midPx": "150.1", "oraclePx": "150.2", "funding": "0.0001", "openInterest": "100.0", "prevDayPx": "149.0", "dayNtlVlm": "1000.0"},
    ],
]

ALL_MIDS_PAYLOAD = {"BTC": "63900.0", "ETH": "1800.5", "#1710": "0.5", "@1": "14.42"}


def test_normalize_candles_shape_and_lineage():
    ingested = datetime(2026, 7, 12, tzinfo=timezone.utc)
    rows = normalize_candles(
        CANDLE_PAYLOAD, coin="BTC", interval="1h", ingested_time=ingested, raw_payload_hash="deadbeef"
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["instrument_id"] == "BTC"
    assert row["open"] == "86226.0"
    assert row["close"] == "86255.0"
    assert row["open_time"] == datetime.fromtimestamp(1765836000000 / 1000, tz=timezone.utc).isoformat()
    lineage = row["lineage"]
    assert lineage["source_id"] == "hyperliquid_info"
    assert lineage["raw_payload_hash"] == "deadbeef"
    assert lineage["ingested_time"] is not None


def test_normalize_meta_and_asset_ctxs_filters_universe():
    ingested = datetime(2026, 7, 12, tzinfo=timezone.utc)
    rows = normalize_meta_and_asset_ctxs(
        META_PAYLOAD, ingested_time=ingested, raw_payload_hash="abc123", coins=("BTC", "ETH")
    )
    ids = {r["instrument_id"] for r in rows}
    assert ids == {"BTC", "ETH"}
    btc = next(r for r in rows if r["instrument_id"] == "BTC")
    assert btc["mark_px"] == "63911.0"
    assert btc["funding"] == "0.0000125"
    assert btc["open_interest"] == "37530.76798"
    assert btc["lineage"]["raw_payload_hash"] == "abc123"


@pytest.mark.asyncio
async def test_candle_snapshot_writes_raw_and_normalizes(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CANDLE_PAYLOAD)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        raw_store = RawStore(tmp_path)
        client = HyperliquidClient(raw_store, client=http_client)
        payload, rows = await client.candle_snapshot("BTC", "1h", 0, 1)

    assert payload == CANDLE_PAYLOAD
    assert len(rows) == 1
    assert rows[0]["instrument_id"] == "BTC"

    verify_results = raw_store.verify_all("hyperliquid_info")
    assert all(ok for day_results in verify_results.values() for _, ok in day_results)


@pytest.mark.asyncio
async def test_meta_and_asset_ctxs_writes_raw_and_normalizes(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=META_PAYLOAD)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        raw_store = RawStore(tmp_path)
        client = HyperliquidClient(raw_store, client=http_client)
        payload, rows = await client.meta_and_asset_ctxs()

    assert payload == META_PAYLOAD
    assert {r["instrument_id"] for r in rows} == {"BTC", "ETH"}


@pytest.mark.asyncio
async def test_all_mids_filters_non_coin_keys(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ALL_MIDS_PAYLOAD)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        raw_store = RawStore(tmp_path)
        client = HyperliquidClient(raw_store, client=http_client)
        payload, mids = await client.all_mids()

    assert payload == ALL_MIDS_PAYLOAD
    assert set(mids.keys()) == {"BTC", "ETH"}
    assert str(mids["BTC"]) == "63900.0"


@pytest.mark.asyncio
async def test_candle_snapshot_raises_on_http_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        raw_store = RawStore(tmp_path)
        client = HyperliquidClient(raw_store, client=http_client)
        with pytest.raises(httpx.HTTPStatusError):
            await client.candle_snapshot("BTC", "1h", 0, 1)
