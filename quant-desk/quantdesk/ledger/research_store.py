"""Separate SQLite store for research-only sentiment advisor forecasts.

INVARIANT: this store lives at a separate database file (data/research.sqlite
by default) from the operational ledger (quantdesk/ledger/store.py). It is a
single-writer store mirroring the operational ledger's connection setup
(WAL mode, foreign keys enforced, busy timeout) but persists ONLY
``ResearchForecast`` rows and their later-scored outcomes.

This module must never import from quantdesk.portfolio, quantdesk.risk, or
quantdesk.execution, and must never construct an ``OrderIntent``. Forecasts
stored here are research artifacts only; nothing in this module can promote
one into a tradable signal (see quantdesk/common/research_schemas.py).

Outcome timing: a forecast may only be scored (``record_outcome``) once its
full outcome window (generated_at + horizon) has closed. Calling
``record_outcome`` before the window closes raises ``OutcomeWindowOpen``.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from quantdesk.common.research_schemas import ResearchForecast

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_forecasts (
    forecast_id TEXT PRIMARY KEY,
    advisor_id TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    horizon_seconds INTEGER NOT NULL,
    research_only INTEGER NOT NULL DEFAULT 1,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_forecasts_advisor ON research_forecasts(advisor_id);
CREATE INDEX IF NOT EXISTS idx_research_forecasts_instrument ON research_forecasts(instrument_id);
CREATE INDEX IF NOT EXISTS idx_research_forecasts_generated_at ON research_forecasts(generated_at);

CREATE TABLE IF NOT EXISTS research_outcomes (
    forecast_id TEXT PRIMARY KEY REFERENCES research_forecasts(forecast_id),
    realized_excess_log_return REAL NOT NULL,
    outcome_window_close_at TEXT NOT NULL,
    scored_at TEXT NOT NULL
);
"""


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OutcomeWindowOpen(Exception):
    """Raised when recording an outcome before the forecast's horizon has closed."""


class ResearchStore:
    """Single-writer SQLite store for research-only forecasts and outcomes.

    Mirrors quantdesk.ledger.store.LedgerStore's connection setup: WAL
    journal mode, foreign keys enforced, a busy timeout so readers can open
    the same file concurrently, and an explicit ``transaction()`` context
    manager wrapping all writes. Only one instance should hold a writable
    connection to a given path at a time.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ---------------------------------------------------------------
    # Forecasts
    # ---------------------------------------------------------------
    def insert_forecast(self, forecast: ResearchForecast) -> None:
        payload = json.dumps(forecast.model_dump(mode="json"), default=_json_default)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO research_forecasts (forecast_id, advisor_id, "
                "instrument_id, generated_at, horizon_seconds, research_only, "
                "payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(forecast.forecast_id),
                    forecast.advisor_id,
                    forecast.instrument_id,
                    forecast.generated_at.isoformat(),
                    int(forecast.horizon.total_seconds()),
                    1 if forecast.research_only else 0,
                    payload,
                ),
            )

    # ---------------------------------------------------------------
    # Outcomes
    # ---------------------------------------------------------------
    def record_outcome(
        self,
        forecast_id: UUID | str,
        realized_excess_log_return: float,
        now: datetime | None = None,
    ) -> None:
        now = now or _utcnow()
        row = self._conn.execute(
            "SELECT generated_at, horizon_seconds FROM research_forecasts "
            "WHERE forecast_id = ?",
            (str(forecast_id),),
        ).fetchone()
        if row is None:
            raise ValueError(f"forecast_id {forecast_id} not found")
        generated_at = datetime.fromisoformat(row[0])
        close_at = generated_at + timedelta(seconds=row[1])
        if now < close_at:
            raise OutcomeWindowOpen(
                f"forecast_id {forecast_id} outcome window closes at "
                f"{close_at.isoformat()}, now is {now.isoformat()}"
            )
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO research_outcomes (forecast_id, "
                "realized_excess_log_return, outcome_window_close_at, scored_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    str(forecast_id),
                    float(realized_excess_log_return),
                    close_at.isoformat(),
                    now.isoformat(),
                ),
            )

    def pending_outcomes(self, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or _utcnow()
        rows = self._conn.execute(
            "SELECT f.forecast_id, f.instrument_id, f.advisor_id, "
            "f.generated_at, f.horizon_seconds FROM research_forecasts f "
            "LEFT JOIN research_outcomes o ON o.forecast_id = f.forecast_id "
            "WHERE o.forecast_id IS NULL"
        ).fetchall()
        pending: list[dict[str, Any]] = []
        for forecast_id, instrument_id, advisor_id, generated_at, horizon_seconds in rows:
            close_at = datetime.fromisoformat(generated_at) + timedelta(seconds=horizon_seconds)
            if close_at <= now:
                pending.append(
                    {
                        "forecast_id": forecast_id,
                        "instrument_id": instrument_id,
                        "advisor_id": advisor_id,
                        "generated_at": generated_at,
                        "outcome_window_close_at": close_at.isoformat(),
                    }
                )
        return pending
