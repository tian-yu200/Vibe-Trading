"""SQLite persistence for watchlist scans, picks and manual items.

Follows ``src/portfolio/store.py``: injectable path (default
``get_runtime_root()/watchlist/watchlist.sqlite3``), one connection per
operation (thread-safe), schema versioned with ``CREATE TABLE IF NOT EXISTS``
and a ``PRAGMA user_version`` marker (v2 adds ``scan_runs.seq`` for strict
save-order without mutating timestamps).

All JSON payloads are written with ``ensure_ascii=False``; text opens use
``encoding='utf-8'`` (Windows default here is GBK).
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.config.paths import get_runtime_root

SCHEMA_VERSION = 3


def _dump(payload: Any) -> str:
    """JSON-serialise with CJK kept readable."""
    return json.dumps(payload, ensure_ascii=False)


def _load(text: str | None) -> Any:
    return json.loads(text) if text else None


class WatchlistStore:
    """SQLite-backed store for scan runs, picks, scores and manual items."""

    def __init__(self, path: Path | None = None) -> None:
        """Open (and create when missing) the watchlist database.

        Args:
            path: Database file path. Defaults to
                ``watchlist/watchlist.sqlite3`` under the runtime root.
        """
        self.path = path or (get_runtime_root() / "watchlist" / "watchlist.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # -- connections --------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """One connection per operation — safe across threads."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _transaction(self):
        """One explicit transaction over one connection (BEGIN IMMEDIATE).

        Every write path (save_run / update_run_review) must land its parent
        row AND its picked/dropped child rows atomically: a run whose picks
        were persisted but whose dropped rows were lost left the DROPPED
        section silently empty (observed live: run d4ce9714 saved 20 picks
        and 0 dropped rows). ``BEGIN IMMEDIATE`` takes the write lock up
        front, so the seq computation and every child insert share one
        all-or-nothing unit; any failure rolls back the whole run.
        """
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.isolation_level = None  # commit/rollback are explicit
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._connect() as db:
            current = db.execute("PRAGMA user_version").fetchone()[0]
            if current < SCHEMA_VERSION and current > 0:
                if current < 2:
                    # v1 -> v2: scan_runs.seq (INTEGER, strict save order).
                    # Existing rows get seq in started_at order; started_at is
                    # restored to its clean form (a v1 store appended
                    # '.000'-style suffixes).
                    db.execute("ALTER TABLE scan_runs ADD COLUMN seq INTEGER")
                    rows = db.execute(
                        "SELECT run_id, started_at FROM scan_runs ORDER BY started_at"
                    ).fetchall()
                    for order, row in enumerate(rows):
                        clean = re.sub(r"\.\d{3}$", "", row["started_at"])
                        db.execute(
                            "UPDATE scan_runs SET seq = ?, started_at = ? WHERE run_id = ?",
                            (order + 1, clean, row["run_id"]),
                        )
                if current < 3:
                    # v2 -> v3: scan_runs.failure_reason (coverage-guard /
                    # maintenance invalidation evidence on failed runs).
                    db.execute(
                        "ALTER TABLE scan_runs ADD COLUMN failure_reason TEXT"
                    )
            db.executescript("""
                CREATE TABLE IF NOT EXISTS scan_runs (
                    run_id TEXT PRIMARY KEY,
                    trade_date TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    seq INTEGER,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    duration_s REAL,
                    intraday INTEGER,
                    ai_status TEXT,
                    ai_note TEXT,
                    failure_reason TEXT,
                    config TEXT,
                    funnel TEXT,
                    sources TEXT,
                    ai_demoted TEXT,
                    requalified TEXT,
                    pinned TEXT,
                    payload TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_scan_runs_seq
                    ON scan_runs(seq DESC);
                CREATE INDEX IF NOT EXISTS idx_scan_runs_trade_date
                    ON scan_runs(trade_date DESC);
                CREATE TABLE IF NOT EXISTS scan_picks (
                    run_id TEXT NOT NULL,
                    code TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    score REAL,
                    change TEXT NOT NULL,
                    pick TEXT NOT NULL,
                    PRIMARY KEY (run_id, code),
                    FOREIGN KEY (run_id) REFERENCES scan_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS scan_dropped (
                    run_id TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT,
                    prev_rank INTEGER,
                    reason TEXT,
                    PRIMARY KEY (run_id, code)
                );
                CREATE TABLE IF NOT EXISTS scan_scores (
                    run_id TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT,
                    board TEXT,
                    industry TEXT,
                    score REAL,
                    fundamental REAL,
                    trend REAL,
                    flow REAL,
                    exclusion_reason TEXT,
                    PRIMARY KEY (run_id, code)
                );
                CREATE TABLE IF NOT EXISTS watchlist_items (
                    code TEXT PRIMARY KEY,
                    name TEXT,
                    origin TEXT NOT NULL,
                    state TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    removed_at TEXT,
                    note TEXT
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # -- scan runs -----------------------------------------------------------
    def save_run(
        self, run: dict[str, Any], *, changes: dict[str, list] | None = None
    ) -> None:
        """Persist one ScanRun envelope plus its picks, drops and scores.

        Args:
            run: The ScanRun dict (CONTRACT section 7).
            changes: ``{"new": [...], "kept": [...], "dropped": [...]}`` where
                new/kept carry ``(code, rank, score, pick)`` and dropped
                carries ``(code, name, prev_rank, reason)``.
        """
        changes = changes or {}
        # Strict save order comes from the seq column, not from mutating
        # started_at (FIX A: timestamps stay clean).
        # One transaction for the whole run: the seq allocation, the parent
        # row, and every picks/dropped/scores child row commit together or
        # not at all. Previously the seq was read on a *separate* connection
        # and the children were only implicitly transactional, so a failure
        # in the middle could leave a run with picks but no dropped rows
        # (the DROPPED section silently empty — live run d4ce9714).
        dropped_rows = [
            (run["run_id"], code, name, prev_rank, reason)
            for code, name, prev_rank, reason in changes.get("dropped", [])
        ]
        with self._transaction() as db:
            row = db.execute(
                "SELECT COALESCE(MAX(seq), 0) AS m FROM scan_runs"
            ).fetchone()
            seq = (row["m"] or 0) + 1
            db.execute(
                """
                INSERT OR REPLACE INTO scan_runs
                    (run_id, trade_date, started_at, seq, finished_at, status, trigger,
                     duration_s, intraday, ai_status, ai_note, failure_reason, config,
                     funnel, sources, ai_demoted, requalified, pinned, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run["run_id"],
                    run.get("trade_date"),
                    run.get("started_at"),
                    seq,
                    run.get("finished_at"),
                    run.get("status"),
                    run.get("trigger"),
                    run.get("duration_s"),
                    1 if run.get("intraday") else 0,
                    run.get("ai_status"),
                    run.get("ai_note"),
                    run.get("failure_reason"),
                    _dump(run.get("config")),
                    _dump(run.get("funnel")),
                    _dump(run.get("sources")),
                    _dump(run.get("ai_demoted")),
                    _dump(run.get("requalified")),
                    _dump(run.get("pinned")),
                    _dump(run),
                ),
            )
            # A re-save of the same run_id replaces its children wholesale so
            # stale rows cannot survive under a fresh diff (INSERT OR
            # REPLACE alone left orphans by (run_id, code) key).
            for table in ("scan_picks", "scan_dropped", "scan_scores"):
                db.execute(
                    f"DELETE FROM {table} WHERE run_id = ?", (run["run_id"],)
                )
            for code, rank, score, pick in changes.get("new", []):
                db.execute(
                    "INSERT OR REPLACE INTO scan_picks VALUES (?, ?, ?, ?, ?, ?)",
                    (run["run_id"], code, rank, score, "new", _dump(pick)),
                )
            for code, rank, score, pick in changes.get("kept", []):
                db.execute(
                    "INSERT OR REPLACE INTO scan_picks VALUES (?, ?, ?, ?, ?, ?)",
                    (run["run_id"], code, rank, score, "kept", _dump(pick)),
                )
            for row in dropped_rows:
                db.execute(
                    "INSERT OR REPLACE INTO scan_dropped VALUES (?, ?, ?, ?, ?)",
                    row,
                )
            for score_row in run.get("scores", []):
                db.execute(
                    "INSERT OR REPLACE INTO scan_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run["run_id"],
                        score_row.get("code"),
                        score_row.get("name"),
                        score_row.get("board"),
                        score_row.get("industry"),
                        score_row.get("score"),
                        score_row.get("fundamental"),
                        score_row.get("trend"),
                        score_row.get("flow"),
                        score_row.get("exclusion_reason"),
                    ),
                )

    def update_run_review(
        self, run: dict[str, Any], *, changes: dict[str, list] | None = None
    ) -> None:
        """Persist an AI re-review in place (no seq change, no reorder).

        Updates the AI verdict columns (ai_status/ai_note), the derived
        columns (config/funnel/sources/ai_demoted/requalified/pinned), the
        full JSON payload, and rewrites picks + dropped rows from
        ``changes``. ``started_at``/``finished_at``/``duration_s``/``seq``
        stay untouched so history order and the original scan duration
        remain intact.
        """
        changes = changes or {}
        # Same one-transaction guarantee as save_run: the review update, the
        # Delete of the old picks/dropped rows and their re-insert land as a
        # single unit — a crash mid-way cannot leave a reviewed run with its
        # DROPPED section wiped and the new rows lost.
        with self._transaction() as db:
            db.execute(
                """
                UPDATE scan_runs SET
                    ai_status = ?, ai_note = ?, config = ?, funnel = ?,
                    sources = ?, ai_demoted = ?, requalified = ?, pinned = ?,
                    payload = ?
                WHERE run_id = ?
                """,
                (
                    run.get("ai_status"),
                    run.get("ai_note"),
                    _dump(run.get("config")),
                    _dump(run.get("funnel")),
                    _dump(run.get("sources")),
                    _dump(run.get("ai_demoted")),
                    _dump(run.get("requalified")),
                    _dump(run.get("pinned")),
                    _dump(run),
                    run["run_id"],
                ),
            )
            db.execute(
                "DELETE FROM scan_picks WHERE run_id = ?", (run["run_id"],)
            )
            db.execute(
                "DELETE FROM scan_dropped WHERE run_id = ?", (run["run_id"],)
            )
            for code, rank, score, pick in changes.get("new", []):
                db.execute(
                    "INSERT OR REPLACE INTO scan_picks VALUES (?, ?, ?, ?, ?, ?)",
                    (run["run_id"], code, rank, score, "new", _dump(pick)),
                )
            for code, rank, score, pick in changes.get("kept", []):
                db.execute(
                    "INSERT OR REPLACE INTO scan_picks VALUES (?, ?, ?, ?, ?, ?)",
                    (run["run_id"], code, rank, score, "kept", _dump(pick)),
                )
            for code, name, prev_rank, reason in changes.get("dropped", []):
                db.execute(
                    "INSERT OR REPLACE INTO scan_dropped VALUES (?, ?, ?, ?, ?)",
                    (run["run_id"], code, name, prev_rank, reason),
                )

    def set_run_failed(
        self, run_id: str, *, failure_reason: str, payload: dict | None = None
    ) -> None:
        """Maintenance invalidation: mark a stored run failed in place.

        Keeps picks/scores/dropped rows and seq (history order) untouched;
        updates the status column, the failure_reason column and — when
        given — the full JSON payload.
        """
        with self._connect() as db:
            if payload is not None:
                db.execute(
                    "UPDATE scan_runs SET status = 'failed', failure_reason = ?, "
                    "payload = ? WHERE run_id = ?",
                    (failure_reason, _dump(payload), run_id),
                )
            else:
                db.execute(
                    "UPDATE scan_runs SET status = 'failed', failure_reason = ? "
                    "WHERE run_id = ?",
                    (failure_reason, run_id),
                )

    def _run_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        """Run row -> summary dict (JSON payload columns decoded)."""
        return {
            "run_id": row["run_id"],
            "trade_date": row["trade_date"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "status": row["status"],
            "trigger": row["trigger"],
            "duration_s": row["duration_s"],
            "intraday": bool(row["intraday"]),
            "ai_status": row["ai_status"],
            "ai_note": row["ai_note"],
            "failure_reason": row["failure_reason"],
            "config": _load(row["config"]),
            "funnel": _load(row["funnel"]),
            "sources": _load(row["sources"]),
            "ai_demoted": _load(row["ai_demoted"]),
            "requalified": _load(row["requalified"]),
            "pinned": _load(row["pinned"]),
        }

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """One run's full envelope (summary columns + payload), or None.

        Summary columns are authoritative for scalar fields; the payload
        supplies envelope-only keys (picks, scores, coverage, timings).
        """
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM scan_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        summary = self._run_summary(row)
        payload = _load(row["payload"])
        if isinstance(payload, dict):
            return {**payload, **summary}
        return summary

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent run summaries."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM scan_runs ORDER BY COALESCE(seq, 0) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._run_summary(r) for r in rows]

    def last_successful_run(self) -> dict[str, Any] | None:
        """The most recent successful run (status 'ok' or 'partial'), or None."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM scan_runs WHERE status IN ('ok', 'partial') "
                "ORDER BY COALESCE(seq, 0) DESC LIMIT 1"
            ).fetchone()
        return self._run_summary(row) if row else None

    def last_successful_run_before(self, run_id: str) -> dict[str, Any] | None:
        """The most recent ok|partial run started before ``run_id``, or None.

        Used by an AI re-review to diff picks against the run that was
        latest *before* the reviewed run (``last_successful_run()`` would
        return the reviewed run itself).
        """
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM scan_runs WHERE status IN ('ok', 'partial') AND COALESCE(seq, 0) < "
                "(SELECT COALESCE(seq, 0) FROM scan_runs WHERE run_id = ?) "
                "ORDER BY COALESCE(seq, 0) DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return self._run_summary(row) if row else None

    def last_scan_for_trade_date(self, trade_date: str) -> dict[str, Any] | None:
        """The best run (ok|partial first) for a trade date, or None."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM scan_runs WHERE trade_date = ? AND status IN ('ok', 'partial') "
                "ORDER BY COALESCE(seq, 0) DESC LIMIT 1",
                (trade_date,),
            ).fetchone()
            if row is None:
                row = db.execute(
                    "SELECT * FROM scan_runs WHERE trade_date = ? "
                    "ORDER BY COALESCE(seq, 0) DESC LIMIT 1",
                    (trade_date,),
                ).fetchone()
        return self._run_summary(row) if row else None

    def get_picks(self, run_id: str) -> list[dict[str, Any]]:
        """Picks of one run with their change labels, rank order."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT code, rank, score, change, pick FROM scan_picks "
                "WHERE run_id = ? ORDER BY rank",
                (run_id,),
            ).fetchall()
        return [
            {
                "code": r["code"],
                "rank": r["rank"],
                "score": r["score"],
                "change": r["change"],
                "pick": _load(r["pick"]),
            }
            for r in rows
        ]

    def get_scores(self, run_id: str) -> dict[str, dict[str, Any]]:
        """Scores of one run keyed by code."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM scan_scores WHERE run_id = ?", (run_id,)
            ).fetchall()
        return {
            r["code"]: {
                "code": r["code"],
                "name": r["name"],
                "board": r["board"],
                "industry": r["industry"],
                "score": r["score"],
                "fundamental": r["fundamental"],
                "trend": r["trend"],
                "flow": r["flow"],
                "exclusion_reason": r["exclusion_reason"],
            }
            for r in rows
        }

    def get_dropped(self, run_id: str) -> list[dict[str, Any]]:
        """Dropped rows of one run."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT code, name, prev_rank, reason FROM scan_dropped WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return [
            {
                "code": r["code"],
                "name": r["name"],
                "prev_rank": r["prev_rank"],
                "reason": r["reason"],
            }
            for r in rows
        ]

    def latest_run_id_before(self, run_id: str) -> str | None:
        """The run started immediately before ``run_id`` (any status), or None."""
        with self._connect() as db:
            row = db.execute(
                "SELECT run_id FROM scan_runs WHERE COALESCE(seq, 0) < "
                "(SELECT COALESCE(seq, 0) FROM scan_runs WHERE run_id = ?) "
                "ORDER BY COALESCE(seq, 0) DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return row["run_id"] if row else None

    # -- watchlist items ------------------------------------------------------
    def upsert_item(
        self,
        code: str,
        name: str | None,
        origin: str,
        state: str,
        now_iso: str,
        note: str | None = None,
        promote_origin: bool = False,
    ) -> None:
        """Insert or update one watchlist item row.

        Args:
            promote_origin: When True (explicit user action such as
                ``add_manual``), ``origin`` is overwritten to the given value;
                otherwise the existing origin is preserved (automatic scans
                never demote a manual item back to auto).
        """
        with self._connect() as db:
            origin_assignment = (
                "excluded.origin" if promote_origin else "watchlist_items.origin"
            )
            db.execute(
                f"""
                INSERT INTO watchlist_items (code, name, origin, state, added_at, note)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                    name = COALESCE(excluded.name, watchlist_items.name),
                    origin = {origin_assignment},
                    state = excluded.state,
                    note = COALESCE(excluded.note, watchlist_items.note),
                    removed_at = CASE WHEN excluded.state = 'removed'
                        THEN ? ELSE NULL END
                """,
                (code, name, origin, state, now_iso, note, now_iso),
            )

    def get_item(self, code: str) -> dict[str, Any] | None:
        """One item row or None."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM watchlist_items WHERE code = ?", (code,)
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_items(
        self, state: str | None = None, origin: str | None = None
    ) -> list[dict[str, Any]]:
        """Item rows, optionally filtered."""
        query = "SELECT * FROM watchlist_items"
        conditions, params = [], []
        if state:
            conditions.append("state = ?")
            params.append(state)
        if origin:
            conditions.append("origin = ?")
            params.append(origin)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY added_at"
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def remove_item_row(self, code: str) -> None:
        """Delete one watchlist_items row (auto item dropped by a scan)."""
        with self._connect() as db:
            db.execute("DELETE FROM watchlist_items WHERE code = ?", (code,))

    def mark_removed(self, code: str, now_iso: str) -> None:
        """Set state='removed' (suppression) keeping history."""
        with self._connect() as db:
            db.execute(
                "UPDATE watchlist_items SET state = 'removed', removed_at = ? WHERE code = ?",
                (now_iso, code),
            )

    def mark_active(self, code: str, now_iso: str) -> None:
        """Clear the suppression (restore)."""
        with self._connect() as db:
            db.execute(
                "UPDATE watchlist_items SET state = 'active', removed_at = NULL WHERE code = ?",
                (code,),
            )

    def active_codes(self, origin: str | None = None) -> list[str]:
        """Active item codes, optionally by origin."""
        if origin:
            with self._connect() as db:
                rows = db.execute(
                    "SELECT code FROM watchlist_items WHERE state = 'active' AND origin = ?",
                    (origin,),
                ).fetchall()
        else:
            with self._connect() as db:
                rows = db.execute(
                    "SELECT code FROM watchlist_items WHERE state = 'active'"
                ).fetchall()
        return [r["code"] for r in rows]

    def suppressed_codes(self) -> list[str]:
        """Codes with state='removed' (never auto-re-added)."""
        return [r["code"] for r in self.list_items(state="removed")]

    def update_item_note(self, code: str, note: str) -> None:
        """Set or clear an item note."""
        with self._connect() as db:
            db.execute(
                "UPDATE watchlist_items SET note = ? WHERE code = ?", (note, code)
            )

    # -- settings -------------------------------------------------------------
    def get_setting(self, key: str) -> Any:
        """One setting value (JSON-decoded), or None."""
        with self._connect() as db:
            row = db.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return _load(row["value"]) if row else None

    def set_setting(self, key: str, value: Any) -> None:
        """Upsert one setting."""
        with self._connect() as db:
            db.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, _dump(value)),
            )
