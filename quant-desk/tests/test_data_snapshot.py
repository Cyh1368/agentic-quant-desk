from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import pytest

from quantdesk.data.snapshot import (
    DEFAULT_STALE_DATA_MAX_MINUTES,
    SnapshotStalenessError,
    build_snapshot,
    check_staleness,
    load_snapshot,
)


def test_check_staleness_fresh_data_no_flags():
    cutoff = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    newest = cutoff - timedelta(minutes=10)
    assert check_staleness(newest, cutoff) == []


def test_check_staleness_stale_data_flags():
    cutoff = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    newest = cutoff - timedelta(minutes=91)
    assert check_staleness(newest, cutoff, stale_data_max_minutes=90) == ["stale_data"]


def test_check_staleness_boundary_is_not_flagged():
    cutoff = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    newest = cutoff - timedelta(minutes=90)
    assert check_staleness(newest, cutoff, stale_data_max_minutes=90) == []


def test_build_snapshot_writes_file_and_returns_dict(tmp_path):
    cutoff = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    newest_candle = cutoff - timedelta(minutes=5)

    snapshot = build_snapshot(
        instruments=[{"instrument_id": "BTC", "mark_px": Decimal("63000.5")}],
        features={"btc_ret_24h@v1": 0.01},
        positions=[{"instrument_id": "BTC", "qty": Decimal("0.1")}],
        data_cutoff_at=cutoff,
        snapshot_dir=tmp_path,
        newest_candle_close_time=newest_candle,
    )

    # snapshot_id is a valid UUID
    UUID(snapshot["snapshot_id"])
    assert snapshot["data_cutoff_at"] == cutoff.isoformat()
    assert snapshot["quality_flags"] == []
    assert snapshot["instruments"][0]["mark_px"] == "63000.5"
    assert snapshot["positions"][0]["qty"] == "0.1"

    written_path = tmp_path / f"{snapshot['snapshot_id']}.json"
    assert written_path.exists()
    on_disk = json.loads(written_path.read_text())
    assert on_disk == snapshot


def test_build_snapshot_rejects_naive_data_cutoff(tmp_path):
    with pytest.raises(ValueError):
        build_snapshot(
            instruments=[],
            features={},
            positions=[],
            data_cutoff_at=datetime(2026, 7, 12),
            snapshot_dir=tmp_path,
        )


def test_build_snapshot_raises_on_stale_data_by_default(tmp_path):
    cutoff = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    newest_candle = cutoff - timedelta(minutes=200)

    with pytest.raises(SnapshotStalenessError):
        build_snapshot(
            instruments=[],
            features={},
            positions=[],
            data_cutoff_at=cutoff,
            snapshot_dir=tmp_path,
            newest_candle_close_time=newest_candle,
        )

    # nothing should have been written
    assert list(tmp_path.glob("*.json")) == []


def test_build_snapshot_flags_stale_when_not_rejecting(tmp_path):
    cutoff = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    newest_candle = cutoff - timedelta(minutes=200)

    snapshot = build_snapshot(
        instruments=[],
        features={},
        positions=[],
        data_cutoff_at=cutoff,
        snapshot_dir=tmp_path,
        newest_candle_close_time=newest_candle,
        reject_on_stale=False,
    )
    assert snapshot["quality_flags"] == ["stale_data"]


def test_load_snapshot_roundtrip(tmp_path):
    cutoff = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    snapshot = build_snapshot(
        instruments=[],
        features={},
        positions=[],
        data_cutoff_at=cutoff,
        snapshot_dir=tmp_path,
    )
    loaded = load_snapshot(tmp_path, snapshot["snapshot_id"])
    assert loaded == snapshot


def test_default_stale_minutes_is_90():
    assert DEFAULT_STALE_DATA_MAX_MINUTES == 90
