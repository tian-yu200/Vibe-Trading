"""RELFIX Bug C evidence hardening: ``WatchlistStore`` must not rewrite the DB
it only wanted to read (boss, 2026-10-24).

``_init_schema`` unconditionally ran the ``CREATE ... IF NOT EXISTS`` script and
``PRAGMA user_version = <n>`` on EVERY construction, i.e. on every API request —
proven by the live DB's mtime moving on 10-20 and 10-21 with no scan running.
Besides the wear, that write is what a one-off 500 was suspected to come from,
so the open path is now read-only when the DB is already current.

These tests use tmp DBs only. The live DB and the real home are never opened.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.watchlist.store import SCHEMA_VERSION, WatchlistStore  # noqa: E402

TABLES = {
    "scan_runs",
    "scan_picks",
    "scan_dropped",
    "scan_scores",
    "watchlist_items",
    "settings",
}

_WRITE = re.compile(
    r"\b(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|REPLACE)\b", re.IGNORECASE
)


def _statements_at_open(path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every SQL statement the next ``WatchlistStore(path)`` issues on open."""
    seen: list[str] = []
    original = WatchlistStore._connect

    def traced_connect(self):
        connection = original(self)
        connection.set_trace_callback(seen.append)
        return connection

    monkeypatch.setattr(WatchlistStore, "_connect", traced_connect)
    WatchlistStore(path=path)
    return seen


def _version(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _table_names(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()


def _columns(path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    finally:
        conn.close()


def _db_stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


# --------------------------------------------------------------------------- #
# 1. An up-to-date DB is opened without writing it
# --------------------------------------------------------------------------- #
def test_reopen_at_current_version_writes_no_statements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "watchlist.sqlite3"
    store = WatchlistStore(path=path)
    store_path = store.path
    before = _db_stamp(store_path)

    seen = _statements_at_open(store_path, monkeypatch)

    writes = [sql for sql in seen if _WRITE.search(sql)]
    assert writes == [], f"reading the DB must not write, got {writes}"
    assert not any(
        "user_version =" in sql or "user_version=" in sql for sql in seen
    ), f"the version marker must not be rewritten: {seen}"
    assert _db_stamp(store_path) == before, "the live DB file must be untouched"


def test_reopen_at_current_version_still_reads_the_data(tmp_path: Path) -> None:
    """Skipping the script must not cost us a working store."""
    path = tmp_path / "watchlist.sqlite3"
    store = WatchlistStore(path=path)
    store.save_run(
        {
            "run_id": "r1",
            "trade_date": "2026-10-23",
            "started_at": "2026-10-23T18:00:00",
            "finished_at": "2026-10-23T18:04:50",
            "status": "ok",
            "trigger": "api",
            "duration_s": 1.0,
            "intraday": False,
            "ai_status": "reviewed",
            "ai_note": None,
            "config": {},
            "funnel": [],
            "sources": [],
            "picks": [],
            "ai_demoted": [],
            "requalified": [],
            "pinned": [],
            "scores": [],
        },
        changes={},
    )
    reopened = WatchlistStore(path=path)
    run = reopened.get_run("r1")
    assert run is not None and run["run_id"] == "r1"
    assert _version(path) == SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# 2. A brand-new DB still gets the full schema
# --------------------------------------------------------------------------- #
def test_fresh_db_builds_the_whole_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "new" / "watchlist.sqlite3"
    seen = _statements_at_open(path, monkeypatch)

    assert _version(path) == SCHEMA_VERSION
    assert TABLES <= _table_names(path)
    assert any(sql.upper().lstrip().startswith("CREATE TABLE") for sql in seen)
    assert any(f"PRAGMA user_version = {SCHEMA_VERSION}" in sql for sql in seen)


def test_fresh_db_is_writable_after_the_shortcut(tmp_path: Path) -> None:
    store = WatchlistStore(path=tmp_path / "watchlist.sqlite3")
    store.set_setting("scan", {"top_n": 5})
    assert store.get_setting("scan") == {"top_n": 5}


# --------------------------------------------------------------------------- #
# 3. Migration paths unchanged
# --------------------------------------------------------------------------- #
def test_v2_db_still_migrates_to_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v2 DB (no failure_reason) must still be altered and re-marked."""
    path = tmp_path / "v2.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE scan_runs (
            run_id TEXT PRIMARY KEY, trade_date TEXT NOT NULL,
            started_at TEXT NOT NULL, seq INTEGER, finished_at TEXT,
            status TEXT NOT NULL, trigger TEXT NOT NULL, duration_s REAL,
            intraday INTEGER, ai_status TEXT, ai_note TEXT, config TEXT,
            funnel TEXT, sources TEXT, ai_demoted TEXT, requalified TEXT,
            pinned TEXT, payload TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO scan_runs VALUES ('r1','2026-10-21','2026-10-21T18:00:00',"
        "1,'2026-10-21T18:03:00','ok','cli',2.0,0,'reviewed',NULL,'{}','[]','[]',"
        "'[]','[]','[]','{}')"
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()

    seen = _statements_at_open(path, monkeypatch)

    assert _version(path) == SCHEMA_VERSION
    assert "failure_reason" in _columns(path, "scan_runs")
    assert any("ALTER TABLE scan_runs" in sql for sql in seen)
    assert any(f"PRAGMA user_version = {SCHEMA_VERSION}" in sql for sql in seen)
    assert TABLES <= _table_names(path)  # the missing tables are still created
    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT trade_date FROM scan_runs WHERE run_id='r1'"
        ).fetchone()[0] == "2026-10-21"
    finally:
        conn.close()


def test_v1_db_still_migrates_to_current(tmp_path: Path) -> None:
    """v1 (no seq): seq added, rows ordered by started_at, timestamps cleaned."""
    path = tmp_path / "v1.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE scan_runs (
            run_id TEXT PRIMARY KEY, trade_date TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
            trigger TEXT NOT NULL, duration_s REAL, intraday INTEGER,
            ai_status TEXT, ai_note TEXT, config TEXT, funnel TEXT,
            sources TEXT, ai_demoted TEXT, requalified TEXT, pinned TEXT,
            payload TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO scan_runs VALUES ('r1','2026-10-12','2026-10-12T18:00:00.000',"
        "'2026-10-12T18:05:00','ok','cli',1.0,0,'not_reviewed',NULL,'{}','[]','[]',"
        "'[]','[]','[]','{}')"
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    store = WatchlistStore(path=path)

    assert _version(path) == SCHEMA_VERSION
    assert {"seq", "failure_reason"} <= _columns(path, "scan_runs")
    run = store.last_successful_run()
    assert run is not None and "." not in run["started_at"]


def test_db_above_the_current_version_is_still_remarked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged (pre-existing) path: only an EXACT match takes the shortcut."""
    path = tmp_path / "future.sqlite3"
    WatchlistStore(path=path)
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    seen = _statements_at_open(path, monkeypatch)

    assert any(f"PRAGMA user_version = {SCHEMA_VERSION}" in sql for sql in seen)
    assert _version(path) == SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# 4. _connect() carries the same busy timeout as _transaction()
# --------------------------------------------------------------------------- #
def test_connect_uses_the_transaction_timeout(tmp_path: Path, monkeypatch) -> None:
    from src.watchlist import store as store_mod

    path = tmp_path / "watchlist.sqlite3"
    real_connect = sqlite3.connect
    kwargs: dict[str, dict] = {}

    def spy(db_path, **kw):
        if str(db_path) == str(path):
            kwargs.setdefault("connect", kw)
        return real_connect(db_path, **kw)

    monkeypatch.setattr(store_mod.sqlite3, "connect", spy)
    store = WatchlistStore(path=path)
    store.get_setting("scan")  # a read that used to grab a default-timeout connection

    assert kwargs["connect"].get("timeout") == 30.0
