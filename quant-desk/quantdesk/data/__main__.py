"""Integration-style fetch cycle: python -m quantdesk.data

Performs one real fetch cycle against Hyperliquid (public info API) and
Coinbase (public spot reference), runs the cross-source sanity check,
computes features, builds a snapshot, and prints a summary. Network calls
are allowed here (this is the one place in the data slice that isn't
unit-tested against mocks).
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal

import yaml

from quantdesk.data.hyperliquid import HyperliquidClient
from quantdesk.data.raw_store import RawStore
from quantdesk.data.reference import cross_source_check, ReferencePriceClient
from quantdesk.data.snapshot import build_snapshot
from quantdesk.features.compute import Candle, compute_feature_set

CONFIG_PATH = "config/desk.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


async def run() -> None:
    config = _load_config()
    universe = config["desk"]["universe"]
    storage = config["storage"]
    divergence_threshold_bps = config["process"]["cross_source_divergence_bps"]

    raw_store = RawStore(storage["raw_landing_dir"])

    print(f"=== quantdesk data fetch cycle @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Universe: {universe}")

    async with HyperliquidClient(raw_store) as hl, ReferencePriceClient(raw_store) as ref:
        now_ms = int(time.time() * 1000)
        lookback_ms = 24 * 7 * 60 * 60 * 1000  # 7 days of 1h candles
        start_ms = now_ms - lookback_ms

        _, meta_rows = await hl.meta_and_asset_ctxs()
        meta_by_coin = {r["instrument_id"]: r for r in meta_rows}

        feature_sets = {}
        newest_close_times = []
        instruments_summary = []

        for coin in universe:
            _, candle_rows = await hl.candle_snapshot(coin, "1h", start_ms, now_ms)
            candles = [Candle.from_row(r) for r in candle_rows]
            print(f"  {coin}: fetched {len(candles)} 1h candles")

            if candles:
                newest_close_times.append(
                    datetime.fromisoformat(candle_rows[-1]["close_time"])
                )

            features = compute_feature_set(coin, candles)
            feature_sets.update(features)

            mark_row = meta_by_coin.get(coin)
            if mark_row is None:
                print(f"  {coin}: WARNING no mark price row from metaAndAssetCtxs")
                continue

            _, ref_row = await ref.spot_price(coin)
            check = cross_source_check(
                Decimal(mark_row["mark_px"]), Decimal(ref_row["price"]), divergence_threshold_bps
            )
            print(
                f"  {coin}: mark={check.mark} ref={check.reference} "
                f"divergence_bps={check.divergence_bps:.2f} status={check.status}"
            )

            instruments_summary.append(
                {
                    "instrument_id": coin,
                    "mark_px": mark_row["mark_px"],
                    "reference_px": ref_row["price"],
                    "funding": mark_row["funding"],
                    "open_interest": mark_row["open_interest"],
                    "cross_source_status": check.status,
                    "cross_source_divergence_bps": str(check.divergence_bps),
                }
            )

            for fid, value in features.items():
                print(f"    {fid} = {value}")

        data_cutoff_at = datetime.now(timezone.utc)
        newest_candle_close_time = min(newest_close_times) if newest_close_times else None

        try:
            snapshot = build_snapshot(
                instruments_summary,
                feature_sets,
                positions=[],
                data_cutoff_at=data_cutoff_at,
                snapshot_dir=storage["snapshot_dir"],
                newest_candle_close_time=newest_candle_close_time,
                reject_on_stale=False,
            )
            print(f"Snapshot written: {snapshot['snapshot_id']}")
            print(f"  quality_flags={snapshot['quality_flags']}")
        except Exception as exc:  # pragma: no cover - integration script
            print(f"Snapshot build failed: {exc}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
