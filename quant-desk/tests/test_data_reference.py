from __future__ import annotations

from decimal import Decimal

import httpx
import pytest

from quantdesk.data.raw_store import RawStore
from quantdesk.data.reference import ReferencePriceClient, cross_source_check

COINBASE_PAYLOAD = {"data": {"amount": "63879.535", "base": "BTC", "currency": "USD"}}


def test_cross_source_check_ok_within_threshold():
    result = cross_source_check(Decimal("100.0"), Decimal("100.05"), Decimal("10"))
    assert result.status == "ok"
    assert result.ok is True


def test_cross_source_check_divergent_beyond_threshold():
    result = cross_source_check(Decimal("101.0"), Decimal("100.0"), Decimal("50"))
    # (101-100)/100 * 10000 = 100 bps > 50 bps threshold
    assert result.divergence_bps == Decimal("100")
    assert result.status == "divergent"
    assert result.ok is False


def test_cross_source_check_boundary_is_ok():
    result = cross_source_check(Decimal("100.5"), Decimal("100.0"), Decimal("50"))
    # exactly 50 bps: not strictly greater, so ok
    assert result.divergence_bps == Decimal("50")
    assert result.status == "ok"


def test_cross_source_check_rejects_zero_reference():
    with pytest.raises(ValueError):
        cross_source_check(Decimal("1"), Decimal("0"), Decimal("10"))


@pytest.mark.asyncio
async def test_spot_price_writes_raw_and_normalizes(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "BTC-USD" in str(request.url)
        return httpx.Response(200, json=COINBASE_PAYLOAD)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        raw_store = RawStore(tmp_path)
        client = ReferencePriceClient(raw_store, client=http_client)
        payload, row = await client.spot_price("BTC")

    assert payload == COINBASE_PAYLOAD
    assert row["instrument_id"] == "BTC"
    assert row["price"] == "63879.535"
    assert row["lineage"]["source_id"] == "coinbase_spot"

    verify_results = raw_store.verify_all("coinbase_spot")
    assert all(ok for day_results in verify_results.values() for _, ok in day_results)
