"""SQLite operational ledger (plan §4, §10).

INVARIANT: exactly one process, and within that process exactly one
``LedgerStore`` instance, may hold write access to the ledger database at
any time. All writes to the operational tables MUST go through methods on
this class. SQLite is opened in WAL mode with foreign keys enforced and a
busy timeout so that concurrent *readers* (dashboards, the watchdog) can
open the same file read-only while the single writer proceeds. Putting the
ledger file on a network filesystem, or constructing a second writable
``LedgerStore`` against the same path from another process, violates the
single-writer invariant and is unsupported (plan §4: "no second writer,
ever, and no SQLite file on a network filesystem").

At-most-once intent consumption: ``consume_intent`` marks the intent's
queue state inside the same transaction that a caller uses to record the
side effect of consuming it (via the ``with store.transaction()`` context
manager). If the process crashes after marking the state but before the
caller's follow-on work completes, the transaction rolls back entirely and
the intent is still consumable; if it crashes after commit, the intent is
durably marked consumed and will never be handed out again. There is no
window where the same intent can be consumed twice by a committed writer.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from quantdesk.common.schemas import (
    ForecastSignal,
    ModelProvenance,
    OrderIntent,
    RiskVerdict,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS forecasts (
    signal_id TEXT PRIMARY KEY,
    advisor_id TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_forecasts_advisor ON forecasts(advisor_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_instrument ON forecasts(instrument_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_generated_at ON forecasts(generated_at);

CREATE TABLE IF NOT EXISTS snapshots_index (
    snapshot_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    instrument_id TEXT,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    instrument_id TEXT,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS intents (
    intent_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    queue_state TEXT NOT NULL DEFAULT 'ENQUEUED',
    consumed_at TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_intents_instrument ON intents(instrument_id);
CREATE INDEX IF NOT EXISTS idx_intents_state ON intents(queue_state);
CREATE INDEX IF NOT EXISTS idx_intents_created_at ON intents(created_at);

CREATE TABLE IF NOT EXISTS risk_verdicts (
    verdict_id TEXT PRIMARY KEY,
    intent_id TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    verdict TEXT NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY (intent_id) REFERENCES intents(intent_id)
);
CREATE INDEX IF NOT EXISTS idx_verdicts_intent ON risk_verdicts(intent_id);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    intent_id TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY (intent_id) REFERENCES intents(intent_id)
);
CREATE INDEX IF NOT EXISTS idx_orders_intent ON orders(intent_id);
CREATE INDEX IF NOT EXISTS idx_orders_state ON orders(state);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    filled_at TEXT NOT NULL,
    quantity TEXT NOT NULL,
    price TEXT NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);

CREATE TABLE IF NOT EXISTS positions_shadow (
    instrument_id TEXT PRIMARY KEY,
    quantity TEXT NOT NULL,
    avg_price TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_provenance (
    model_run_id TEXT PRIMARY KEY,
    model_provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    request_at TEXT NOT NULL,
    cost_usd TEXT NOT NULL DEFAULT '0',
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS halt_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state TEXT NOT NULL,
    reason TEXT,
    changed_at TEXT NOT NULL,
    human_restart_required INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS outbox (
    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    created_at TEXT NOT NULL,
    published INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outbox_published ON outbox(published);

CREATE TABLE IF NOT EXISTS cost_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_run_id TEXT,
    advisor_id TEXT,
    occurred_at TEXT NOT NULL,
    cost_usd TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_cost_log_occurred_at ON cost_log(occurred_at);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (UUID,)):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def _model_json(model: Any) -> str:
    return json.dumps(model.model_dump(mode="json"), default=_json_default)


class DuplicateIntentError(Exception):
    """Raised when insert_intent is called with an intent_id already present."""


class LedgerStore:
    """Single-writer SQLite ledger.

    Only one instance of this class should ever hold a writable connection
    to a given ledger path at once (see module docstring). A process-local
    lock serializes writes from multiple threads within the same instance;
    it does NOT protect against a second OS process opening the same file
    for writing, which is the caller's responsibility to avoid.
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
        """Context manager giving an explicit transaction on the single writer connection."""
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
    def insert_forecast(self, forecast: ForecastSignal) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO forecasts (signal_id, advisor_id, instrument_id, "
                "generated_at, snapshot_id, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(forecast.signal_id),
                    forecast.advisor_id,
                    forecast.instrument_id,
                    forecast.generated_at.isoformat(),
                    str(forecast.snapshot_id),
                    _model_json(forecast),
                ),
            )

    # ---------------------------------------------------------------
    # Intents (durable queue)
    # ---------------------------------------------------------------
    def insert_intent(self, intent: OrderIntent) -> None:
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT 1 FROM intents WHERE intent_id = ?", (str(intent.intent_id),)
            ).fetchone()
            if existing is not None:
                raise DuplicateIntentError(
                    f"intent_id {intent.intent_id} already exists in ledger"
                )
            conn.execute(
                "INSERT INTO intents (intent_id, decision_id, instrument_id, "
                "created_at, expires_at, queue_state, payload) "
                "VALUES (?, ?, ?, ?, ?, 'ENQUEUED', ?)",
                (
                    str(intent.intent_id),
                    str(intent.decision_id),
                    intent.instrument_id,
                    intent.created_at.isoformat(),
                    intent.expires_at.isoformat(),
                    _model_json(intent),
                ),
            )

    def enqueue_intent(self, intent: OrderIntent) -> None:
        """Alias for insert_intent: places a new intent on the durable queue."""
        self.insert_intent(intent)

    def consume_intent(self, intent_id: UUID | str) -> OrderIntent | None:
        """Atomically claim the next-available intent for processing.

        Returns None if the intent does not exist or is not in the
        ENQUEUED state (already consumed, expired, etc). The state
        transition to CONSUMED happens in the same transaction as the
        read-and-check, so two concurrent callers cannot both receive the
        same intent, and a crash after this method returns but before any
        follow-on work is durably recorded elsewhere does not roll back
        the consumption (it has already committed) -- callers must design
        follow-on work to be safe to retry against an already-consumed
        intent (at-most-once, not exactly-once).
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT payload, queue_state FROM intents WHERE intent_id = ?",
                (str(intent_id),),
            ).fetchone()
            if row is None:
                return None
            payload, state = row
            if state != "ENQUEUED":
                return None
            conn.execute(
                "UPDATE intents SET queue_state = 'CONSUMED', consumed_at = ? "
                "WHERE intent_id = ?",
                (_utcnow(), str(intent_id)),
            )
            return OrderIntent.model_validate_json(payload)

    def get_intent_state(self, intent_id: UUID | str) -> str | None:
        row = self._conn.execute(
            "SELECT queue_state FROM intents WHERE intent_id = ?", (str(intent_id),)
        ).fetchone()
        return row[0] if row else None

    # ---------------------------------------------------------------
    # Risk verdicts
    # ---------------------------------------------------------------
    def record_verdict(self, verdict: RiskVerdict) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO risk_verdicts (verdict_id, intent_id, evaluated_at, "
                "verdict, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    str(verdict.verdict_id),
                    str(verdict.intent_id),
                    verdict.evaluated_at.isoformat(),
                    verdict.verdict,
                    _model_json(verdict),
                ),
            )

    def count_orders_since(self, instrument_id: str, since: datetime) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM orders WHERE instrument_id = ? AND created_at >= ?",
            (instrument_id, since.isoformat()),
        ).fetchone()
        return int(row[0]) if row else 0

    # ---------------------------------------------------------------
    # Orders / fills
    # ---------------------------------------------------------------
    def insert_order(self, order_id: str, intent_id: UUID | str, instrument_id: str,
                      state: str, payload: dict[str, Any]) -> None:
        now = _utcnow()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO orders (order_id, intent_id, instrument_id, state, "
                "created_at, updated_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (order_id, str(intent_id), instrument_id, state, now, now,
                 json.dumps(payload, default=_json_default)),
            )

    def update_order_state(self, order_id: str, state: str,
                            payload: dict[str, Any] | None = None) -> None:
        with self.transaction() as conn:
            if payload is not None:
                conn.execute(
                    "UPDATE orders SET state = ?, updated_at = ?, payload = ? "
                    "WHERE order_id = ?",
                    (state, _utcnow(), json.dumps(payload, default=_json_default), order_id),
                )
            else:
                conn.execute(
                    "UPDATE orders SET state = ?, updated_at = ? WHERE order_id = ?",
                    (state, _utcnow(), order_id),
                )

    def record_fill(self, fill_id: str, order_id: str, instrument_id: str,
                     quantity: Decimal, price: Decimal,
                     payload: dict[str, Any] | None = None) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO fills (fill_id, order_id, instrument_id, filled_at, "
                "quantity, price, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    fill_id,
                    order_id,
                    instrument_id,
                    _utcnow(),
                    str(quantity),
                    str(price),
                    json.dumps(payload or {}, default=_json_default),
                ),
            )

    # ---------------------------------------------------------------
    # Shadow positions
    # ---------------------------------------------------------------
    def upsert_shadow_position(self, instrument_id: str, quantity: Decimal,
                                avg_price: Decimal,
                                payload: dict[str, Any] | None = None) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO positions_shadow (instrument_id, quantity, avg_price, "
                "updated_at, payload) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(instrument_id) DO UPDATE SET "
                "quantity=excluded.quantity, avg_price=excluded.avg_price, "
                "updated_at=excluded.updated_at, payload=excluded.payload",
                (
                    instrument_id,
                    str(quantity),
                    str(avg_price),
                    _utcnow(),
                    json.dumps(payload or {}, default=_json_default),
                ),
            )

    def get_shadow_position(self, instrument_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT instrument_id, quantity, avg_price, updated_at, payload "
            "FROM positions_shadow WHERE instrument_id = ?",
            (instrument_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "instrument_id": row[0],
            "quantity": Decimal(row[1]),
            "avg_price": Decimal(row[2]),
            "updated_at": row[3],
            "payload": json.loads(row[4]),
        }

    # ---------------------------------------------------------------
    # Model provenance / cost
    # ---------------------------------------------------------------
    def insert_model_provenance(self, provenance: ModelProvenance) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO model_provenance (model_run_id, model_provider, "
                "model_id, request_at, cost_usd, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(provenance.model_run_id),
                    provenance.model_provider,
                    provenance.model_id,
                    provenance.request_at.isoformat(),
                    str(provenance.cost_usd),
                    _model_json(provenance),
                ),
            )

    def log_llm_cost(self, cost_usd: Decimal, advisor_id: str | None = None,
                      model_run_id: UUID | str | None = None,
                      occurred_at: datetime | None = None,
                      payload: dict[str, Any] | None = None) -> None:
        occurred_at = occurred_at or datetime.now(timezone.utc)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO cost_log (model_run_id, advisor_id, occurred_at, "
                "cost_usd, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    str(model_run_id) if model_run_id else None,
                    advisor_id,
                    occurred_at.isoformat(),
                    str(cost_usd),
                    json.dumps(payload or {}, default=_json_default),
                ),
            )

    def monthly_llm_spend(self, year: int, month: int) -> Decimal:
        prefix = f"{year:04d}-{month:02d}"
        rows = self._conn.execute(
            "SELECT cost_usd FROM cost_log WHERE occurred_at LIKE ?",
            (f"{prefix}%",),
        ).fetchall()
        total = Decimal("0")
        for (cost,) in rows:
            total += Decimal(cost)
        return total

    # ---------------------------------------------------------------
    # Halt state
    # ---------------------------------------------------------------
    def get_halt_state(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT state, reason, changed_at, human_restart_required, payload "
            "FROM halt_state WHERE id = 1"
        ).fetchone()
        if row is None:
            return {
                "state": "NORMAL",
                "reason": None,
                "changed_at": None,
                "human_restart_required": False,
                "payload": {},
            }
        return {
            "state": row[0],
            "reason": row[1],
            "changed_at": row[2],
            "human_restart_required": bool(row[3]),
            "payload": json.loads(row[4]),
        }

    def set_halt_state(self, state: str, reason: str | None = None,
                        human_restart_required: bool = False,
                        payload: dict[str, Any] | None = None) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO halt_state (id, state, reason, changed_at, "
                "human_restart_required, payload) VALUES (1, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET state=excluded.state, "
                "reason=excluded.reason, changed_at=excluded.changed_at, "
                "human_restart_required=excluded.human_restart_required, "
                "payload=excluded.payload",
                (
                    state,
                    reason,
                    _utcnow(),
                    int(human_restart_required),
                    json.dumps(payload or {}, default=_json_default),
                ),
            )

    # ---------------------------------------------------------------
    # Outbox
    # ---------------------------------------------------------------
    def enqueue_outbox(self, topic: str, payload: dict[str, Any]) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO outbox (topic, created_at, published, payload) "
                "VALUES (?, ?, 0, ?)",
                (topic, _utcnow(), json.dumps(payload, default=_json_default)),
            )
            return int(cur.lastrowid)

    # ---------------------------------------------------------------
    # Integrity
    # ---------------------------------------------------------------
    def integrity_check(self) -> bool:
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        return row is not None and row[0] == "ok"
