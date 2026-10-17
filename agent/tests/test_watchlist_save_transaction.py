"""TODO_R3 run d4ce9714 forensics — save_path transaction + dropped-row guarantees.

Covers the boss follow-up (2026-10-13 night):
  * ``save_run`` must persist the run row, its picks AND its dropped rows in
    ONE transaction — a failure half-way cannot leave a run with picks but an
    empty DROPPED section (that is exactly the d4ce9714 shape).
  * A re-save of the same run_id must replace its child rows wholesale so
    stale dropped rows cannot survive under a fresh diff.
  * The post-save self-heal (``_verify_dropped_saved``) stays the safety net.

Read-only fixture probes for this analysis live in the task's scratch notes;
these tests only touch tmp_path stores.

Run: pytest tests/test_watchlist_save_transaction.py -q
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path

import pytest

from src.watchlist.store import WatchlistStore

TRADE = "2026-10-13"


def _pick(code: str, rank: int, score: float) -> dict:
    return {
        "code": code,
        "name": f"股票{code}",
        "board": "main_sh",
        "industry": "测试",
        "rank": rank,
        "score": score,
        "sub_scores": {"fundamental": 80, "trend": 70, "flow": 60},
        "reasons": [],
        "flags": [],
        "ai": {"status": "not_reviewed"},
    }


def _run(run_id: str, pick_codes: list[str], *, status: str = "ok") -> dict:
    return {
        "run_id": run_id,
        "trade_date": TRADE,
        "started_at": "2026-10-13T08:42:47",
        "finished_at": "2026-10-13T08:46:15",
        "status": status,
        "trigger": "cli",
        "duration_s": 1.0,
        "intraday": False,
        "ai_status": "not_reviewed",
        "config": {"top_n": 20},
        "funnel": [],
        "sources": [],
        "picks": [_pick(c, i + 1, 100.0 - i) for i, c in enumerate(pick_codes)],
        "ai_demoted": [],
        "requalified": [],
        "pinned": [],
        "scores": [
            {
                "code": c,
                "name": f"股票{c}",
                "board": "main_sh",
                "industry": "测试",
                "score": 100.0 - i,
                "fundamental": 80,
                "trend": 70,
                "flow": 60,
                "exclusion_reason": None,
            }
            for i, c in enumerate(pick_codes)
        ],
    }


def _changes(run_id: str, pick_codes: list[str], dropped: list) -> dict:
    """new/kept/dropped exactly as _compute_changes builds them.

    ``dropped`` carries ``(code, name, prev_rank, reason)`` — the run_id is
    supplied by the store, not by the caller.
    """
    return {
        "new": [(c, i + 1, 100.0 - i, _pick(c, i + 1, 100.0 - i))
                for i, c in enumerate(pick_codes)],
        "kept": [],
        "dropped": [(c, f"股票{c}", 1, "rank fell to 40 (score 70.0)")
                    for c in dropped],
    }


@pytest.fixture()
def store(tmp_path: Path) -> WatchlistStore:
    return WatchlistStore(path=tmp_path / "watchlist.sqlite3")


# --------------------------------------------------------------------------- #
# one-transaction guarantee
# --------------------------------------------------------------------------- #
def test_save_run_persists_picks_and_dropped_together(
    store: WatchlistStore, tmp_path: Path
) -> None:
    run = _run("aaa", ["600519", "600520", "600522"])
    store.save_run(run, changes=_changes("aaa", ["600519", "600520", "600522"],
                                         ["600520"]))
    picks = {r["code"] for r in store.get_picks("aaa")}
    dropped = {r["code"] for r in store.get_dropped("aaa")}
    assert picks == {"600519", "600520", "600522"}
    # All-or-nothing: the drop survives with the picks in the same unit.
    assert dropped == {"600520"}


def test_save_run_rolls_back_when_a_child_row_fails(
    store: WatchlistStore, tmp_path: Path
) -> None:
    """A poisoned payload (unserialisable value in a score row) aborts the
    WHOLE save: no parent run, no picks, no dropped rows — never the d4ce9714
    half-state (picks persisted, dropped lost)."""
    run = _run("aaa", ["600519", "600520", "600522"])
    changes = _changes("aaa", ["600519", "600520", "600522"], ["600520"])
    # Inject a value that makes the child insert blow up mid-transaction.
    run["scores"][1]["score"] = {"not": "a number"}
    with pytest.raises(Exception):
        store.save_run(run, changes=changes)
    # Nothing from this run may survive.
    assert store.get_picks("aaa") == []
    assert store.get_dropped("aaa") == []
    assert store.get_run("aaa") is None
    assert store.list_runs() == []


def test_save_run_is_atomic_under_concurrent_second_store(
    store: WatchlistStore, tmp_path: Path
) -> None:
    """Two writers serialise on the write lock: both runs land complete."""
    run_a = _run("aaa", ["600519"])
    run_b = _run("bbb", ["600520"])
    store.save_run(run_a, changes=_changes("aaa", ["600519"], []))
    other = WatchlistStore(path=tmp_path / "watchlist.sqlite3")
    other.save_run(run_b, changes=_changes("bbb", ["600520"], ["600519"]))
    assert {r["code"] for r in other.get_picks("bbb")} == {"600520"}
    assert {r["code"] for r in other.get_dropped("bbb")} == {"600519"}
    # Seq is allocated inside the write transaction, so it stays strictly
    # increasing across writers.
    db = other._connect()
    try:
        rows = db.execute(
            "SELECT run_id, seq FROM scan_runs ORDER BY seq"
        ).fetchall()
    finally:
        db.close()
    by_id = {r["run_id"]: r["seq"] for r in rows}
    assert by_id["aaa"] == 1
    assert by_id["bbb"] == 2


# --------------------------------------------------------------------------- #
# re-save replaces children wholesale
# --------------------------------------------------------------------------- #
def test_resave_of_same_run_replaces_dropped_rows(store: WatchlistStore) -> None:
    run = _run("aaa", ["600519", "600522"])
    store.save_run(run, changes=_changes("aaa", ["600519", "600522"], ["600520"]))
    assert {r["code"] for r in store.get_dropped("aaa")} == {"600520"}
    # Same run saved again with a smaller diff (the bug shape: an empty or
    # partial changes dict silently keeping prior dropped rows, or losing the
    # new one) — the store must mirror the new diff exactly.
    store.save_run(run, changes=_changes("aaa", ["600519", "600522"], ["600523"]))
    assert {r["code"] for r in store.get_dropped("aaa")} == {"600523"}
    store.save_run(run, changes=_changes("aaa", ["600519", "600522"], []))
    assert store.get_dropped("aaa") == []


def test_resave_of_same_run_replaces_pick_rows(store: WatchlistStore) -> None:
    run = _run("aaa", ["600519", "600520"])
    store.save_run(run, changes=_changes("aaa", ["600519", "600520"], []))
    store.save_run(run, changes=_changes("aaa", ["600519"], ["600520"]))
    assert {r["code"] for r in store.get_picks("aaa")} == {"600519"}
    assert {r["code"] for r in store.get_dropped("aaa")} == {"600520"}


# --------------------------------------------------------------------------- #
# raw-schema invariant: picks and dropped are the same transaction's children
# --------------------------------------------------------------------------- #
def test_dropped_rows_join_the_same_run_as_its_picks(
    store: WatchlistStore, tmp_path: Path
) -> None:
    """Read the tables directly (no ORM): any dropped row's run must exist and
    carry picks — an orphan drop means a half-committed save."""
    store.save_run(
        _run("aaa", ["600519", "600520"]),
        changes=_changes("aaa", ["600519", "600520"], ["600573"]),
    )
    store.save_run(
        _run("bbb", ["600519"]),
        changes=_changes("bbb", ["600519"], ["600520"]),
    )
    db = sqlite3.connect(tmp_path / "watchlist.sqlite3")
    try:
        rows = db.execute(
            """
            SELECT d.run_id, COUNT(*) AS drops,
                   (SELECT COUNT(*) FROM scan_picks p WHERE p.run_id = d.run_id)
                       AS picks
            FROM scan_dropped d GROUP BY d.run_id
            """
        ).fetchall()
    finally:
        db.close()
    assert rows, "expected dropped rows for both runs"
    for run_id, drops, picks in rows:
        assert picks > 0, f"{run_id} has drops but no picks (half-committed save)"
        assert drops >= 1


# --------------------------------------------------------------------------- #
# d4ce9714 shape: the self-heal still repairs an emptied diff
# --------------------------------------------------------------------------- #
def test_service_self_heal_repairs_an_emptied_diff(
    tmp_path: Path,
) -> None:
    """The live bug shape end-to-end: save() drops the diff, the service's
    post-save repair puts the dropped rows back."""
    import datetime as dt
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, ".")
    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from test_watchlist_taskb_regressions import _make_service, _scan_run

    fake_now = dt.datetime(2026, 10, 13, 18, 0, 0)
    runs = [
        _scan_run("r1", ["600519", "600520", "600521"]),
        _scan_run("r2", ["600519", "600522", "600523"]),
    ]
    calls = {"n": 0}

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        r["trigger"] = "cli"
        calls["n"] += 1
        return r

    svc = _make_service(tmp_path / "watchlist.sqlite3", scan_fn, fake_now)
    svc.run_scan(trigger="cli")

    real_save = svc.store.save_run
    state = {"n": 0}

    def sabotaging_save(run, *, changes=None):
        state["n"] += 1
        if state["n"] == 2:
            real_save(run, changes={})  # the d4ce9714 bug: drops vanish
            return
        real_save(run, changes=changes)

    svc.store.save_run = sabotaging_save  # type: ignore[method-assign]
    svc.run_scan(trigger="cli")
    svc.store.save_run = real_save  # type: ignore[method-assign]

    assert {r["code"] for r in svc.store.get_dropped("r2")} == {
        "600520", "600521"
    }
    assert {r["code"] for r in svc.store.get_picks("r2")} == {
        "600519", "600522", "600523"
    }
