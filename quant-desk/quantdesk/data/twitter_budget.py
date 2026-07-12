"""Reservation-based budget accounting for the Twitter/X source (plan §1).

This module is deliberately backed by its own small SQLite database — it is
NOT the desk's main ledger (which stays single-writer, per
``config/sentiment.yaml``'s ``storage.tweet_store_path`` note). All budget
state (daily request counts, per-cycle tweet volume, monthly USD spend) lives
here so that concurrent workers/retries reserve against a single source of
truth with SQLite's own locking (``BEGIN IMMEDIATE``) providing atomicity.

Flow: ``reserve()`` atomically checks the remaining caps and inserts a
reservation row *before* any network call is made; the caller performs the
call; ``reconcile()`` then replaces the estimate with reported actuals and
releases any unused portion of the reservation. Exhaustion never fails
silently — callers that receive :class:`BudgetExhausted` must mark the
affected query window incomplete with ``truncation_reason=budget_cap``
(enforced in ``quantdesk/data/twitter.py``, not here).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class BudgetExhausted(Exception):
    """Raised when a reservation would exceed a configured cap.

    ``cap_name`` identifies which cap was hit: one of
    ``max_requests_per_day``, ``max_tweets_per_cycle``, ``max_monthly_usd``.
    """

    def __init__(self, cap_name: str, limit: float, requested: float, current: float):
        self.cap_name = cap_name
        self.limit = limit
        self.requested = requested
        self.current = current
        super().__init__(
            f"budget cap exceeded: {cap_name} limit={limit} current={current} "
            f"requested={requested}"
        )


@dataclass(frozen=True)
class Reservation:
    reservation_id: int
    endpoint: str
    est_credits: float
    est_usd: float
    day: str
    reserved_at: datetime


@dataclass(frozen=True)
class EndpointCounters:
    endpoint: str
    request_count: int
    records_returned: int
    credits_estimated: float
    credits_reported: float
    cost_estimated_usd: float
    cost_reported_usd: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS reservations (
    reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT NOT NULL,
    day TEXT NOT NULL,
    month TEXT NOT NULL,
    est_credits REAL NOT NULL,
    est_usd REAL NOT NULL,
    reported_credits REAL,
    reported_usd REAL,
    records_returned INTEGER,
    reconciled INTEGER NOT NULL DEFAULT 0,
    reserved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cycle_tweets (
    day TEXT NOT NULL,
    cycle TEXT NOT NULL,
    tweets INTEGER NOT NULL,
    PRIMARY KEY (day, cycle)
);
"""


class TwitterBudget:
    """SQLite-backed reservation budget for the twitterapi_io source."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_requests_per_day: int,
        max_tweets_per_cycle: int,
        max_monthly_usd: float,
        timeout: float = 30.0,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_requests_per_day = max_requests_per_day
        self.max_tweets_per_cycle = max_tweets_per_cycle
        self.max_monthly_usd = max_monthly_usd
        self._timeout = timeout
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=self._timeout, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def reserve(
        self,
        endpoint: str,
        est_credits: float,
        est_usd: float,
        *,
        cycle_id: str | None = None,
        est_tweets: int = 0,
        now: datetime | None = None,
    ) -> Reservation:
        """Atomically reserve budget for one call, or raise BudgetExhausted.

        Checks (in order): daily request count, per-cycle tweet volume,
        monthly USD spend. The whole check-then-insert happens inside a
        single ``BEGIN IMMEDIATE`` transaction so two concurrent callers
        cannot both succeed against a nearly-exhausted cap.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware UTC")
        day = now.strftime("%Y-%m-%d")
        month = now.strftime("%Y-%m")
        cycle = cycle_id or day

        with self._immediate() as conn:
            req_count = conn.execute(
                "SELECT COUNT(*) FROM reservations WHERE day = ?", (day,)
            ).fetchone()[0]
            if req_count + 1 > self.max_requests_per_day:
                raise BudgetExhausted(
                    "max_requests_per_day", self.max_requests_per_day, req_count + 1, req_count
                )

            cycle_row = conn.execute(
                "SELECT tweets FROM cycle_tweets WHERE day = ? AND cycle = ?", (day, cycle)
            ).fetchone()
            cycle_tweets = cycle_row[0] if cycle_row else 0
            if cycle_tweets + est_tweets > self.max_tweets_per_cycle:
                raise BudgetExhausted(
                    "max_tweets_per_cycle",
                    self.max_tweets_per_cycle,
                    cycle_tweets + est_tweets,
                    cycle_tweets,
                )

            month_prefix = f"{month}-%"
            spent = conn.execute(
                "SELECT COALESCE(SUM(COALESCE(reported_usd, est_usd)), 0) "
                "FROM reservations WHERE day LIKE ?",
                (month_prefix,),
            ).fetchone()[0]
            if spent + est_usd > self.max_monthly_usd:
                raise BudgetExhausted(
                    "max_monthly_usd", self.max_monthly_usd, spent + est_usd, spent
                )

            cur = conn.execute(
                "INSERT INTO reservations "
                "(endpoint, day, month, est_credits, est_usd, reserved_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (endpoint, day, month, est_credits, est_usd, now.isoformat()),
            )
            reservation_id = cur.lastrowid

            if est_tweets:
                if cycle_row:
                    conn.execute(
                        "UPDATE cycle_tweets SET tweets = tweets + ? "
                        "WHERE day = ? AND cycle = ?",
                        (est_tweets, day, cycle),
                    )
                else:
                    conn.execute(
                        "INSERT INTO cycle_tweets (day, cycle, tweets) VALUES (?, ?, ?)",
                        (day, cycle, est_tweets),
                    )

        return Reservation(
            reservation_id=reservation_id,
            endpoint=endpoint,
            est_credits=est_credits,
            est_usd=est_usd,
            day=day,
            reserved_at=now,
        )

    def reconcile(
        self,
        reservation_id: int,
        reported_credits: float,
        reported_usd: float,
        records_returned: int,
    ) -> None:
        """Replace an estimate with actuals and release any unused portion.

        Idempotent: reconciling an already-reconciled reservation is a no-op
        overwrite of the reported figures (never double counts spend, since
        totals are computed from ``COALESCE(reported_usd, est_usd)`` — one
        row contributes once regardless of how many times it is reconciled).
        """
        with self._immediate() as conn:
            conn.execute(
                "UPDATE reservations SET reported_credits = ?, reported_usd = ?, "
                "records_returned = ?, reconciled = 1 WHERE reservation_id = ?",
                (reported_credits, reported_usd, records_returned, reservation_id),
            )

    def endpoint_counters(self, endpoint: str) -> EndpointCounters:
        """Aggregate counters for one endpoint across all recorded reservations."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(records_returned), 0), "
                "COALESCE(SUM(est_credits), 0), COALESCE(SUM(reported_credits), 0), "
                "COALESCE(SUM(est_usd), 0), COALESCE(SUM(reported_usd), 0) "
                "FROM reservations WHERE endpoint = ?",
                (endpoint,),
            ).fetchone()
        return EndpointCounters(
            endpoint=endpoint,
            request_count=row[0],
            records_returned=row[1],
            credits_estimated=row[2],
            credits_reported=row[3],
            cost_estimated_usd=row[4],
            cost_reported_usd=row[5],
        )

    def remaining_requests_today(self, now: datetime | None = None) -> int:
        if now is None:
            now = datetime.now(timezone.utc)
        day = now.strftime("%Y-%m-%d")
        with self._connect() as conn:
            used = conn.execute(
                "SELECT COUNT(*) FROM reservations WHERE day = ?", (day,)
            ).fetchone()[0]
        return max(0, self.max_requests_per_day - used)

    def remaining_monthly_usd(self, now: datetime | None = None) -> float:
        if now is None:
            now = datetime.now(timezone.utc)
        month_prefix = f"{now.strftime('%Y-%m')}-%"
        with self._connect() as conn:
            spent = conn.execute(
                "SELECT COALESCE(SUM(COALESCE(reported_usd, est_usd)), 0) "
                "FROM reservations WHERE day LIKE ?",
                (month_prefix,),
            ).fetchone()[0]
        return max(0.0, self.max_monthly_usd - spent)
