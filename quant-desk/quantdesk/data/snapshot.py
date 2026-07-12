"""Point-in-time decision snapshots (plan §3).

Each decision cycle assembles a compact, immutable JSON snapshot with an
explicit ``data_cutoff_at``: "nothing after this informed the forecast."
The snapshot is the unit of point-in-time evaluation, so once written it
must never be mutated.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

DEFAULT_STALE_DATA_MAX_MINUTES = 90  # 1h candles: allow one bar + buffer


class SnapshotStalenessError(ValueError):
    """Raised when the newest candle data is older than the staleness budget."""


def _to_decimal_str(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_decimal_str(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_decimal_str(v) for v in value]
    return value


def check_staleness(
    newest_candle_close_time: datetime,
    data_cutoff_at: datetime,
    *,
    stale_data_max_minutes: int = DEFAULT_STALE_DATA_MAX_MINUTES,
) -> list[str]:
    """Return quality flags; includes "stale_data" if the newest candle is
    older than ``stale_data_max_minutes`` relative to ``data_cutoff_at``.
    """
    flags: list[str] = []
    age = data_cutoff_at - newest_candle_close_time
    if age > timedelta(minutes=stale_data_max_minutes):
        flags.append("stale_data")
    return flags


def build_snapshot(
    instruments: list[dict[str, Any]],
    features: dict[str, Any],
    positions: list[dict[str, Any]],
    data_cutoff_at: datetime,
    *,
    snapshot_dir: str | Path,
    newest_candle_close_time: datetime | None = None,
    stale_data_max_minutes: int = DEFAULT_STALE_DATA_MAX_MINUTES,
    reject_on_stale: bool = True,
    extra_quality_flags: list[str] | None = None,
) -> dict[str, Any]:
    """Build and persist an immutable point-in-time snapshot.

    Parameters
    ----------
    instruments: list of instrument dicts (e.g. mark price, funding, oi rows)
    features: dict of versioned feature_id -> value (see quantdesk.features.compute)
    positions: current book positions, as plain dicts
    data_cutoff_at: nothing observed after this time may have informed this snapshot
    snapshot_dir: directory to write ``{snapshot_id}.json`` into
    newest_candle_close_time: close time of the freshest 1h candle used; if
        provided and older than ``stale_data_max_minutes`` relative to
        ``data_cutoff_at``, staleness is flagged (and, if ``reject_on_stale``,
        raises instead of writing).

    Returns the snapshot dict that was written to disk.
    """
    if data_cutoff_at.tzinfo is None:
        raise ValueError("data_cutoff_at must be timezone-aware UTC")

    quality_flags: list[str] = list(extra_quality_flags or [])

    if newest_candle_close_time is not None:
        if newest_candle_close_time.tzinfo is None:
            raise ValueError("newest_candle_close_time must be timezone-aware UTC")
        staleness_flags = check_staleness(
            newest_candle_close_time,
            data_cutoff_at,
            stale_data_max_minutes=stale_data_max_minutes,
        )
        if staleness_flags and reject_on_stale:
            raise SnapshotStalenessError(
                f"newest candle close_time={newest_candle_close_time.isoformat()} is "
                f"older than {stale_data_max_minutes} minutes relative to "
                f"data_cutoff_at={data_cutoff_at.isoformat()}"
            )
        quality_flags.extend(staleness_flags)

    snapshot_id = uuid4()
    generated_at = datetime.now(timezone.utc)

    snapshot = {
        "snapshot_id": str(snapshot_id),
        "generated_at": generated_at.isoformat(),
        "data_cutoff_at": data_cutoff_at.isoformat(),
        "instruments": _to_decimal_str(instruments),
        "features": _to_decimal_str(features),
        "positions": _to_decimal_str(positions),
        "quality_flags": sorted(set(quality_flags)),
    }

    out_dir = Path(snapshot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{snapshot_id}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, sort_keys=True, indent=2)

    return snapshot


def load_snapshot(snapshot_dir: str | Path, snapshot_id: UUID | str) -> dict[str, Any]:
    """Load a previously written, immutable snapshot by id."""
    path = Path(snapshot_dir) / f"{snapshot_id}.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
