"""Immutable, append-only raw landing zone (plan §3).

Every payload fetched from any external source is written here, verbatim,
before any normalization happens. Records are never mutated or deleted.
Each line is written as JSONL under ``data/raw/{source_id}/{YYYY-MM-DD}.jsonl``.

Normalizers read from this store; nothing ever writes back into it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def sha256_hex(payload: bytes) -> str:
    """Return the sha256 hex digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """Deterministic JSON encoding used for hashing and storage.

    Keys are sorted and separators are compact so the same logical payload
    always hashes to the same digest.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )


@dataclass(frozen=True)
class RawRecord:
    """One immutable landing-zone entry as read back off disk."""

    source_id: str
    ingested_time: datetime
    raw_payload_hash: str
    payload: Any
    request: Any | None = None

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "ingested_time": self.ingested_time.isoformat(),
            "raw_payload_hash": self.raw_payload_hash,
            "payload": self.payload,
            "request": self.request,
        }


class RawStore:
    """Append-only JSONL landing zone rooted at ``raw_landing_dir``."""

    def __init__(self, raw_landing_dir: str | Path):
        self.root = Path(raw_landing_dir)

    def _path_for(self, source_id: str, when: datetime) -> Path:
        day = when.astimezone(timezone.utc).strftime("%Y-%m-%d")
        return self.root / source_id / f"{day}.jsonl"

    def write(
        self,
        source_id: str,
        payload: Any,
        *,
        request: Any | None = None,
        ingested_time: datetime | None = None,
    ) -> RawRecord:
        """Append one immutable record and return it (with its hash).

        ``payload`` is the exact (json-serializable) response body from the
        source. ``request`` optionally records what was asked for (e.g. the
        POST body sent to Hyperliquid), for forensic reconstruction.
        """
        if ingested_time is None:
            ingested_time = datetime.now(timezone.utc)
        if ingested_time.tzinfo is None:
            raise ValueError("ingested_time must be timezone-aware UTC")

        payload_hash = sha256_hex(canonical_json_bytes(payload))
        record = RawRecord(
            source_id=source_id,
            ingested_time=ingested_time,
            raw_payload_hash=payload_hash,
            payload=payload,
            request=request,
        )

        path = self._path_for(source_id, ingested_time)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), sort_keys=True, default=str))
            fh.write("\n")
        return record

    def iter_records(self, source_id: str, day: str) -> Iterator[RawRecord]:
        """Iterate all records for ``source_id`` on a given YYYY-MM-DD day."""
        path = self.root / source_id / f"{day}.jsonl"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                yield RawRecord(
                    source_id=raw["source_id"],
                    ingested_time=datetime.fromisoformat(raw["ingested_time"]),
                    raw_payload_hash=raw["raw_payload_hash"],
                    payload=raw["payload"],
                    request=raw.get("request"),
                )

    def verify_file(self, source_id: str, day: str) -> list[tuple[int, bool]]:
        """Recompute each record's hash and compare to the stored one.

        Returns a list of ``(line_number, ok)`` tuples, 0-indexed by line.
        An empty list means the file does not exist (nothing to verify).
        """
        results: list[tuple[int, bool]] = []
        for idx, record in enumerate(self.iter_records(source_id, day)):
            recomputed = sha256_hex(canonical_json_bytes(record.payload))
            results.append((idx, recomputed == record.raw_payload_hash))
        return results

    def verify_all(self, source_id: str) -> dict[str, list[tuple[int, bool]]]:
        """Verify every day-file stored for a source_id."""
        source_dir = self.root / source_id
        out: dict[str, list[tuple[int, bool]]] = {}
        if not source_dir.exists():
            return out
        for path in sorted(source_dir.glob("*.jsonl")):
            day = path.stem
            out[day] = self.verify_file(source_id, day)
        return out
