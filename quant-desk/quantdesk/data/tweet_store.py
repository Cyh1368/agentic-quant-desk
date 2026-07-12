"""SQLite-backed tweet store (plan §4-5).

Owns its own database file, separate from the deterministic ledger, so the
research-only sentiment subsystem can never contend with (or corrupt) the
single-writer trading ledger. WAL mode is enabled; this store is designed
for a single writer process at a time (the sentiment ingest cycle), with
readers (feature builders) safe to run concurrently thanks to WAL.

Tables
------
``tweets``
    Append-only content revisions. Primary key is ``(tweet_id,
    content_revision)`` — an edit never overwrites a prior revision, it
    inserts a new row with ``content_revision + 1``. Historical
    point-in-time queries can therefore always recover exactly what a
    feature snapshot saw at the time.

``engagement_observations``
    Append-only engagement snapshots keyed by observation time. Never
    updated, never joined onto historical tweet snapshots implicitly —
    callers must explicitly pick an observation lag.

``query_windows``
    Completeness records for each fetch window, so feature code can detect
    truncated/incomplete coverage.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("datetimes stored in TweetStore must be timezone-aware UTC")
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).astimezone(timezone.utc)


@dataclass(frozen=True)
class QueryWindow:
    asset: str
    start_time: datetime
    end_time: datetime
    inserted_at: datetime
    complete: bool
    tweet_count: int
    meta: dict


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tweets (
    tweet_id TEXT NOT NULL,
    content_revision INTEGER NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    deleted_observed_at TEXT,

    event_time TEXT NOT NULL,
    ingested_time TEXT NOT NULL,
    available_to_strategy_time TEXT NOT NULL,
    source_id TEXT NOT NULL,
    raw_payload_hash TEXT NOT NULL,
    normalizer_version TEXT NOT NULL,
    quality_flags TEXT NOT NULL DEFAULT '[]',
    source_class TEXT,

    raw_text_restricted TEXT NOT NULL,
    normalized_text_for_dedup TEXT NOT NULL,
    model_input_text TEXT NOT NULL,
    display_text_for_llm TEXT NOT NULL,

    author_id TEXT,
    author_handle TEXT,
    author_followers INTEGER,
    author_created_at TEXT,
    author_age_days REAL,

    asset_mentions TEXT NOT NULL DEFAULT '[]',
    target_ambiguity TEXT NOT NULL DEFAULT 'none',

    is_retweet INTEGER NOT NULL DEFAULT 0,
    is_reply INTEGER NOT NULL DEFAULT 0,
    lang TEXT,

    retweet_count INTEGER,
    reply_count INTEGER,
    like_count INTEGER,
    view_count INTEGER,

    p_negative REAL,
    p_neutral REAL,
    p_positive REAL,
    sentiment REAL,
    scorer_confidence REAL,
    scorer_version TEXT,
    preprocessor_version TEXT,

    canonical_text_hash TEXT,
    canonical_tweet_id TEXT,
    duplicate_count INTEGER NOT NULL DEFAULT 1,
    duplicate_unique_authors INTEGER NOT NULL DEFAULT 1,
    duplicate_first_seen_at TEXT,
    duplicate_last_seen_at TEXT,

    PRIMARY KEY (tweet_id, content_revision)
);

CREATE INDEX IF NOT EXISTS idx_tweets_dedup
    ON tweets (normalized_text_for_dedup);
CREATE INDEX IF NOT EXISTS idx_tweets_author_window
    ON tweets (author_id, event_time);
CREATE INDEX IF NOT EXISTS idx_tweets_avail
    ON tweets (available_to_strategy_time);

CREATE TABLE IF NOT EXISTS engagement_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id TEXT NOT NULL,
    observation_time TEXT NOT NULL,
    tweet_age_minutes REAL NOT NULL,
    likes INTEGER,
    reposts INTEGER,
    replies INTEGER,
    views INTEGER
);
CREATE INDEX IF NOT EXISTS idx_engagement_tweet
    ON engagement_observations (tweet_id, observation_time);

CREATE TABLE IF NOT EXISTS query_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    complete INTEGER NOT NULL,
    tweet_count INTEGER NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_query_windows_asset
    ON query_windows (asset, start_time, end_time);
"""


class TweetStore:
    """Single-writer SQLite store for normalized tweets and observations.

    Only one process should call the mutating methods (``upsert_tweet``,
    ``mark_deleted``, ``add_engagement_observation``, ``assign_canonical``,
    ``insert_query_window``) at a time; concurrent readers are fine under
    WAL. This mirrors the ledger's single-writer discipline but the
    sentiment store is intentionally a *separate* database file so it can
    never contend with (or corrupt) the trading ledger.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TweetStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    def upsert_tweet(self, normalized: dict, *, now: datetime | None = None) -> int:
        """Insert or update a tweet record from a normalized dict.

        If the ``tweet_id`` has never been seen, inserts content_revision 0.
        If it has been seen and ``model_input_text`` is unchanged from the
        latest revision, only ``last_seen_at`` is bumped (no new row). If
        ``model_input_text`` differs, a NEW content_revision row is
        inserted; the prior revision is never overwritten.

        Returns the content_revision that is now current.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        now_iso = _to_utc_iso(now)
        tweet_id = normalized["tweet_id"]

        with self._conn:
            row = self._conn.execute(
                "SELECT content_revision, model_input_text FROM tweets "
                "WHERE tweet_id = ? ORDER BY content_revision DESC LIMIT 1",
                (tweet_id,),
            ).fetchone()

            if row is None:
                self._insert_revision(tweet_id, 0, normalized, now_iso, now_iso)
                return 0

            latest_revision = row["content_revision"]
            if row["model_input_text"] == normalized["model_input_text"]:
                self._conn.execute(
                    "UPDATE tweets SET last_seen_at = ? "
                    "WHERE tweet_id = ? AND content_revision = ?",
                    (now_iso, tweet_id, latest_revision),
                )
                return latest_revision

            new_revision = latest_revision + 1
            self._insert_revision(tweet_id, new_revision, normalized, now_iso, now_iso)
            return new_revision

    def _insert_revision(
        self,
        tweet_id: str,
        content_revision: int,
        normalized: dict,
        first_seen_at: str,
        last_seen_at: str,
    ) -> None:
        cols = dict(normalized)
        cols.pop("tweet_id", None)
        cols.pop("source_id", None)

        record: dict[str, Any] = {
            "tweet_id": tweet_id,
            "content_revision": content_revision,
            "first_seen_at": first_seen_at,
            "last_seen_at": last_seen_at,
            "is_deleted": 0,
            "deleted_observed_at": None,
            "event_time": _to_utc_iso(normalized["event_time"]),
            "ingested_time": _to_utc_iso(normalized["ingested_time"]),
            "available_to_strategy_time": _to_utc_iso(normalized["available_to_strategy_time"]),
            "source_id": normalized["source_id"],
            "raw_payload_hash": normalized["raw_payload_hash"],
            "normalizer_version": normalized["normalizer_version"],
            "quality_flags": json.dumps(normalized.get("quality_flags", [])),
            "source_class": normalized.get("source_class"),
            "raw_text_restricted": normalized["raw_text_restricted"],
            "normalized_text_for_dedup": normalized["normalized_text_for_dedup"],
            "model_input_text": normalized["model_input_text"],
            "display_text_for_llm": normalized["display_text_for_llm"],
            "author_id": normalized.get("author_id"),
            "author_handle": normalized.get("author_handle"),
            "author_followers": normalized.get("author_followers"),
            "author_created_at": (
                _to_utc_iso(normalized["author_created_at"])
                if normalized.get("author_created_at")
                else None
            ),
            "author_age_days": normalized.get("author_age_days"),
            "asset_mentions": json.dumps(normalized.get("asset_mentions", [])),
            "target_ambiguity": normalized.get("target_ambiguity", "none"),
            "is_retweet": int(bool(normalized.get("is_retweet"))),
            "is_reply": int(bool(normalized.get("is_reply"))),
            "lang": normalized.get("lang"),
            "retweet_count": normalized.get("retweet_count"),
            "reply_count": normalized.get("reply_count"),
            "like_count": normalized.get("like_count"),
            "view_count": normalized.get("view_count"),
            "p_negative": normalized.get("p_negative"),
            "p_neutral": normalized.get("p_neutral"),
            "p_positive": normalized.get("p_positive"),
            "sentiment": normalized.get("sentiment"),
            "scorer_confidence": normalized.get("scorer_confidence"),
            "scorer_version": normalized.get("scorer_version"),
            "preprocessor_version": normalized.get("preprocessor_version"),
            "canonical_text_hash": normalized.get("canonical_text_hash"),
            "canonical_tweet_id": normalized.get("canonical_tweet_id"),
            "duplicate_count": normalized.get("duplicate_count", 1),
            "duplicate_unique_authors": normalized.get("duplicate_unique_authors", 1),
            "duplicate_first_seen_at": (
                _to_utc_iso(normalized["duplicate_first_seen_at"])
                if normalized.get("duplicate_first_seen_at")
                else None
            ),
            "duplicate_last_seen_at": (
                _to_utc_iso(normalized["duplicate_last_seen_at"])
                if normalized.get("duplicate_last_seen_at")
                else None
            ),
        }
        columns = ", ".join(record.keys())
        placeholders = ", ".join("?" for _ in record)
        self._conn.execute(
            f"INSERT INTO tweets ({columns}) VALUES ({placeholders})",
            tuple(record.values()),
        )

    def mark_deleted(self, tweet_id: str, deleted_observed_at: datetime) -> None:
        """Flag the latest revision of ``tweet_id`` as deleted. Never deletes rows."""
        with self._conn:
            row = self._conn.execute(
                "SELECT content_revision FROM tweets WHERE tweet_id = ? "
                "ORDER BY content_revision DESC LIMIT 1",
                (tweet_id,),
            ).fetchone()
            if row is None:
                return
            self._conn.execute(
                "UPDATE tweets SET is_deleted = 1, deleted_observed_at = ? "
                "WHERE tweet_id = ? AND content_revision = ?",
                (_to_utc_iso(deleted_observed_at), tweet_id, row["content_revision"]),
            )

    def update_scores(
        self,
        tweet_id: str,
        content_revision: int,
        *,
        p_negative: float,
        p_neutral: float,
        p_positive: float,
        scorer_confidence: float,
        scorer_version: str,
        preprocessor_version: str,
    ) -> None:
        """Fill sentiment score columns for a specific content revision.

        Called by the (out-of-scope) scorer; never mutates text fields.
        """
        sentiment = p_positive - p_negative
        with self._conn:
            self._conn.execute(
                """
                UPDATE tweets
                SET p_negative = ?, p_neutral = ?, p_positive = ?, sentiment = ?,
                    scorer_confidence = ?, scorer_version = ?, preprocessor_version = ?
                WHERE tweet_id = ? AND content_revision = ?
                """,
                (
                    p_negative,
                    p_neutral,
                    p_positive,
                    sentiment,
                    scorer_confidence,
                    scorer_version,
                    preprocessor_version,
                    tweet_id,
                    content_revision,
                ),
            )

    def add_engagement_observation(
        self,
        tweet_id: str,
        observation_time: datetime,
        tweet_age_minutes: float,
        *,
        likes: int | None = None,
        reposts: int | None = None,
        replies: int | None = None,
        views: int | None = None,
    ) -> int:
        """Append one engagement snapshot. Never updates existing rows."""
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO engagement_observations
                    (tweet_id, observation_time, tweet_age_minutes, likes, reposts, replies, views)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tweet_id,
                    _to_utc_iso(observation_time),
                    tweet_age_minutes,
                    likes,
                    reposts,
                    replies,
                    views,
                ),
            )
            return int(cur.lastrowid)

    def insert_query_window(
        self,
        asset: str,
        start_time: datetime,
        end_time: datetime,
        *,
        complete: bool,
        tweet_count: int,
        meta: dict | None = None,
        inserted_at: datetime | None = None,
    ) -> int:
        if inserted_at is None:
            inserted_at = datetime.now(timezone.utc)
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO query_windows
                    (asset, start_time, end_time, inserted_at, complete, tweet_count, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset,
                    _to_utc_iso(start_time),
                    _to_utc_iso(end_time),
                    _to_utc_iso(inserted_at),
                    int(complete),
                    tweet_count,
                    json.dumps(meta or {}),
                ),
            )
            return int(cur.lastrowid)

    def get_windows(self, asset: str, start_time: datetime, end_time: datetime) -> list[QueryWindow]:
        rows = self._conn.execute(
            """
            SELECT * FROM query_windows
            WHERE asset = ? AND start_time < ? AND end_time > ?
            ORDER BY start_time ASC
            """,
            (asset, _to_utc_iso(end_time), _to_utc_iso(start_time)),
        ).fetchall()
        return [
            QueryWindow(
                asset=r["asset"],
                start_time=_from_iso(r["start_time"]),
                end_time=_from_iso(r["end_time"]),
                inserted_at=_from_iso(r["inserted_at"]),
                complete=bool(r["complete"]),
                tweet_count=r["tweet_count"],
                meta=json.loads(r["meta"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # dedup
    # ------------------------------------------------------------------

    def assign_canonical(self, lookback_start: datetime, lookback_end: datetime) -> int:
        """Group tweets by ``normalized_text_for_dedup`` within the lookback window.

        Groups only the *latest, non-deleted* content revision per tweet_id
        whose ``event_time`` falls in ``[lookback_start, lookback_end)``. The
        canonical tweet within each group is the one with the earliest
        ``event_time`` (ties broken by tweet_id). Every member of the group
        (including the canonical one) is stamped with the group's
        ``canonical_tweet_id``, a stable hash of the dedup text, and the
        group's duplicate propagation counts.

        Returns the number of groups processed.
        """
        rows = self._conn.execute(
            """
            SELECT t.tweet_id, t.content_revision, t.normalized_text_for_dedup,
                   t.author_id, t.event_time, t.first_seen_at, t.last_seen_at
            FROM tweets t
            INNER JOIN (
                SELECT tweet_id, MAX(content_revision) AS max_rev
                FROM tweets
                GROUP BY tweet_id
            ) latest
              ON t.tweet_id = latest.tweet_id AND t.content_revision = latest.max_rev
            WHERE t.is_deleted = 0
              AND t.event_time >= ? AND t.event_time < ?
            """,
            (_to_utc_iso(lookback_start), _to_utc_iso(lookback_end)),
        ).fetchall()

        groups: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            groups.setdefault(r["normalized_text_for_dedup"], []).append(r)

        with self._conn:
            for dedup_text, members in groups.items():
                if not dedup_text:
                    continue
                members_sorted = sorted(members, key=lambda r: (r["event_time"], r["tweet_id"]))
                canonical = members_sorted[0]
                # sha256, not builtin hash(): this value is persisted, so it
                # must be stable across process restarts (PYTHONHASHSEED).
                canonical_hash = "dedup:" + hashlib.sha256(dedup_text.encode()).hexdigest()[:16]
                unique_authors = len({m["author_id"] for m in members if m["author_id"]})
                first_seen = min(m["first_seen_at"] for m in members)
                last_seen = max(m["last_seen_at"] for m in members)
                for m in members_sorted:
                    self._conn.execute(
                        """
                        UPDATE tweets
                        SET canonical_text_hash = ?, canonical_tweet_id = ?,
                            duplicate_count = ?, duplicate_unique_authors = ?,
                            duplicate_first_seen_at = ?, duplicate_last_seen_at = ?
                        WHERE tweet_id = ? AND content_revision = ?
                        """,
                        (
                            canonical_hash,
                            canonical["tweet_id"],
                            len(members),
                            unique_authors,
                            first_seen,
                            last_seen,
                            m["tweet_id"],
                            m["content_revision"],
                        ),
                    )
        return len(groups)

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def tweets_for_feature_window(
        self, asset: str, start: datetime, end: datetime, as_of: datetime
    ) -> list[dict]:
        """Point-in-time-correct rows for feature computation.

        Only rows with ``available_to_strategy_time <= as_of`` are eligible,
        and for each tweet_id only the content revision that was current
        *as of* ``as_of`` is returned (the latest revision whose
        ``first_seen_at <= as_of``) — never a revision created after
        ``as_of``, even if it is the current revision today. Restricted to
        tweets whose ``event_time`` falls in ``[start, end)`` and which
        mention ``asset``.
        """
        as_of_iso = _to_utc_iso(as_of)
        rows = self._conn.execute(
            """
            SELECT t.* FROM tweets t
            INNER JOIN (
                SELECT tweet_id, MAX(content_revision) AS rev
                FROM tweets
                WHERE first_seen_at <= ? AND available_to_strategy_time <= ?
                GROUP BY tweet_id
            ) pit
              ON t.tweet_id = pit.tweet_id AND t.content_revision = pit.rev
            WHERE t.event_time >= ? AND t.event_time < ?
            """,
            (as_of_iso, as_of_iso, _to_utc_iso(start), _to_utc_iso(end)),
        ).fetchall()

        out = []
        for r in rows:
            d = dict(r)
            mentions = json.loads(d["asset_mentions"] or "[]")
            if asset not in mentions:
                continue
            d["asset_mentions"] = mentions
            d["quality_flags"] = json.loads(d["quality_flags"] or "[]")
            out.append(d)
        return out

    def get_latest_revision(self, tweet_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM tweets WHERE tweet_id = ? ORDER BY content_revision DESC LIMIT 1",
            (tweet_id,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["asset_mentions"] = json.loads(d["asset_mentions"] or "[]")
        d["quality_flags"] = json.loads(d["quality_flags"] or "[]")
        return d

    def get_all_revisions(self, tweet_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM tweets WHERE tweet_id = ? ORDER BY content_revision ASC",
            (tweet_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["asset_mentions"] = json.loads(d["asset_mentions"] or "[]")
            d["quality_flags"] = json.loads(d["quality_flags"] or "[]")
            out.append(d)
        return out

    def get_engagement_observations(self, tweet_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM engagement_observations WHERE tweet_id = ? ORDER BY observation_time ASC",
            (tweet_id,),
        ).fetchall()
        return [dict(r) for r in rows]
