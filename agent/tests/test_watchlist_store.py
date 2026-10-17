"""Offline tests for WatchlistStore (tmp_path DB, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.watchlist.store import WatchlistStore


@pytest.fixture()
def store(tmp_path: Path) -> WatchlistStore:
    return WatchlistStore(path=tmp_path / "watchlist.sqlite3")


def _run(
    run_id: str,
    *,
    status: str = "ok",
    picks: list[str] | None = None,
    scores_extra: list[dict] | None = None,
    trade_date: str = "2026-10-12",
) -> dict:
    pick_codes = picks if picks is not None else ["600519", "000001"]
    picks_list = [
        {
            "code": c,
            "name": f"股票{c}",
            "rank": i + 1,
            "score": 90 - i,
            "board": "main_sh",
            "industry": "测试",
            "sub_scores": {"fundamental": 80, "trend": 70, "flow": 60},
            "reasons": ["r1"],
            "flags": [],
            "ai": None,
        }
        for i, c in enumerate(pick_codes)
    ]
    scores = [
        {
            "code": c,
            "name": f"股票{c}",
            "board": "main_sh",
            "industry": "测试",
            "score": 90 - i,
            "fundamental": 80,
            "trend": 70,
            "flow": 60,
            "exclusion_reason": None,
        }
        for i, c in enumerate(pick_codes)
    ] + (scores_extra or [])
    return {
        "run_id": run_id,
        "trade_date": trade_date,
        "started_at": "2026-10-12T18:00:00",
        "finished_at": "2026-10-12T18:05:00",
        "status": status,
        "trigger": "cli",
        "duration_s": 300.0,
        "intraday": False,
        "ai_status": "not_reviewed",
        "ai_note": None,
        "config": {"top_n": 20},
        "funnel": [{"stage": "universe", "count": 5572, "removed": 0, "note": None}],
        "sources": [{"name": "universe", "status": "ok", "rows": 5572}],
        "picks": picks_list,
        "ai_demoted": [],
        "requalified": [],
        "pinned": [],
        "scores": scores,
    }


def test_schema_created_and_versioned(store: WatchlistStore) -> None:
    import sqlite3

    conn = sqlite3.connect(store.path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert version >= 1
    assert {
        "scan_runs",
        "scan_picks",
        "scan_dropped",
        "scan_scores",
        "watchlist_items",
        "settings",
    } <= tables


def test_save_and_get_run_round_trip(store: WatchlistStore) -> None:
    run = _run("r1")
    store.save_run(
        run,
        changes={"new": [(p["code"], p["rank"], p["score"], p) for p in run["picks"]]},
    )
    loaded = store.get_run("r1")
    assert loaded["run_id"] == "r1"
    assert loaded["status"] == "ok"
    assert loaded["config"] == {"top_n": 20}
    assert loaded["funnel"][0]["stage"] == "universe"
    picks = store.get_picks("r1")
    assert [p["code"] for p in picks] == ["600519", "000001"]
    assert picks[0]["change"] == "new"
    assert picks[0]["pick"]["reasons"] == ["r1"]
    scores = store.get_scores("r1")
    assert scores["600519"]["fundamental"] == 80
    # CJK stored unescaped
    raw = store.path.read_bytes()
    assert "股票".encode("utf-8") in raw


def test_last_successful_run_counts_partial(store: WatchlistStore) -> None:
    """Success = ok OR partial (pipeline completed); only failed is skipped."""
    store.save_run(_run("r_bad", status="failed"))
    assert store.last_successful_run() is None
    store.save_run(_run("r_ok"))
    assert store.last_successful_run()["run_id"] == "r_ok"
    assert store.last_successful_run_before("r_ok") is None
    # a partial run counts as successful and becomes the most recent one
    store.save_run(_run("r_partial", status="partial"))
    assert store.last_successful_run()["run_id"] == "r_partial"
    # ...and _before resolves it as the predecessor for re-reviews/diffs
    assert store.last_successful_run_before("r_partial")["run_id"] == "r_ok"


def test_items_lifecycle(store: WatchlistStore) -> None:
    store.upsert_item(
        "600519", "贵州茅台", "manual", "active", "2026-10-12T10:00:00", note="白酒"
    )
    item = store.get_item("600519")
    assert item["origin"] == "manual"
    assert item["note"] == "白酒"
    assert store.active_codes(origin="manual") == ["600519"]

    store.mark_removed("600519", "2026-10-12T11:00:00")
    assert store.get_item("600519")["state"] == "removed"
    assert store.suppressed_codes() == ["600519"]
    assert store.active_codes() == []

    store.mark_active("600519", "2026-10-12T12:00:00")
    item = store.get_item("600519")
    assert item["state"] == "active"
    assert item["removed_at"] is None


def test_settings_round_trip(store: WatchlistStore) -> None:
    assert store.get_setting("scan") is None
    store.set_setting("scan", {"top_n": 10})
    assert store.get_setting("scan") == {"top_n": 10}
    store.set_setting("schedule", {"enabled": True})
    assert store.get_setting("schedule") == {"enabled": True}


def test_list_runs_order(store: WatchlistStore) -> None:
    store.save_run(_run("r1", trade_date="2026-10-10"))
    store.save_run(_run("r2", trade_date="2026-10-12"))
    runs = store.list_runs(limit=10)
    assert {r["run_id"] for r in runs} == {"r1", "r2"}
