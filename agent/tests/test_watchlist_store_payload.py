"""Regression: get_run returns the full ScanRun envelope from payload (item 4).

list_runs stays summary-only. Offline, tmp_path DB, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.watchlist.store import WatchlistStore


@pytest.fixture()
def store(tmp_path: Path) -> WatchlistStore:
    return WatchlistStore(path=tmp_path / "watchlist.sqlite3")


def _realistic_run(run_id: str = "r-full") -> dict:
    picks = [
        {
            "code": "600519",
            "name": "贵州茅台",
            "board": "main_sh",
            "industry": "白酒",
            "rank": 1,
            "score": 91.5,
            "sub_scores": {"fundamental": 90, "trend": 88, "flow": 96.5},
            "reasons": ["高ROE"],
            "flags": [],
            "ai": {"status": "not_reviewed"},
        },
        {
            "code": "000001",
            "name": "平安银行",
            "board": "main_sz",
            "industry": "银行",
            "rank": 2,
            "score": 84.0,
            "sub_scores": {"fundamental": 82, "trend": 80, "flow": 90},
            "reasons": ["低估值"],
            "flags": [],
            "ai": {"status": "not_reviewed"},
        },
        {
            "code": "300750",
            "name": "宁德时代",
            "board": "chinext",
            "industry": "电池",
            "rank": 3,
            "score": 80.2,
            "sub_scores": {"fundamental": 78, "trend": 85, "flow": 77.6},
            "reasons": ["动量强"],
            "flags": [],
            "ai": {"status": "not_reviewed"},
        },
    ]
    scores = [
        {
            "code": p["code"],
            "name": p["name"],
            "board": p["board"],
            "industry": p["industry"],
            "score": p["score"],
            "fundamental": p["sub_scores"]["fundamental"],
            "trend": p["sub_scores"]["trend"],
            "flow": p["sub_scores"]["flow"],
            "exclusion_reason": None,
        }
        for p in picks
    ]
    return {
        "run_id": run_id,
        "trade_date": "2026-10-12",
        "started_at": "2026-10-12T18:00:00",
        "finished_at": "2026-10-12T18:05:00",
        "status": "ok",
        "trigger": "cli",
        "duration_s": 300.0,
        "intraday": False,
        "ai_status": "not_reviewed",
        "ai_note": None,
        "config": {"top_n": 20, "include_star": True},
        "funnel": [
            {"stage": "universe", "count": 5572, "removed": 0, "note": None},
            {"stage": "scored", "count": 3, "removed": 5569, "note": None},
        ],
        "sources": [{"name": "universe", "status": "ok", "rows": 5572}],
        "picks": picks,
        "ai_demoted": [],
        "requalified": [],
        "pinned": [],
        "scores": scores,
        "coverage": {
            "passed_exclusions": 3,
            "insufficient": 0,
            "scored": 3,
            "trend_scored": 3,
            "flow_real": 3,
            "flow_neutral": 0,
        },
        "coverage_note": None,
        "timings": {"total": 300.0, "klines": 120.0, "scoring": 40.0},
    }


def test_get_run_returns_full_envelope(store: WatchlistStore) -> None:
    run = _realistic_run()
    store.save_run(
        run,
        changes={
            "new": [(p["code"], p["rank"], p["score"], p) for p in run["picks"]],
        },
    )
    loaded = store.get_run("r-full")
    assert loaded is not None
    # Summary columns
    assert loaded["run_id"] == "r-full"
    assert loaded["status"] == "ok"
    assert loaded["trade_date"] == "2026-10-12"
    # Envelope keys that were previously dropped
    assert len(loaded["picks"]) == 3
    assert [p["code"] for p in loaded["picks"]] == ["600519", "000001", "300750"]
    assert loaded["picks"][0]["name"] == "贵州茅台"
    assert loaded["coverage"]["scored"] == 3
    assert loaded["timings"]["total"] == 300.0
    assert loaded["scores"][0]["fundamental"] == 90
    assert loaded["config"]["top_n"] == 20
    assert loaded["funnel"][0]["stage"] == "universe"
    assert loaded["sources"][0]["name"] == "universe"
    assert loaded["ai_status"] == "not_reviewed"
    assert loaded["ai_note"] is None
    assert loaded["ai_demoted"] == []
    assert loaded["requalified"] == []
    assert loaded["pinned"] == []


def test_list_runs_stays_summary_only(store: WatchlistStore) -> None:
    run = _realistic_run()
    store.save_run(
        run,
        changes={
            "new": [(p["code"], p["rank"], p["score"], p) for p in run["picks"]],
        },
    )
    rows = store.list_runs(limit=5)
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "r-full"
    # Summary columns present
    assert row["status"] == "ok"
    assert row["funnel"] is not None
    # Envelope-only keys absent from list rows
    assert "picks" not in row
    assert "scores" not in row
    assert "coverage" not in row
    assert "timings" not in row


def test_get_run_unknown_id_returns_none(store: WatchlistStore) -> None:
    assert store.get_run("nope") is None


def test_get_run_without_payload_falls_back_to_summary(store: WatchlistStore) -> None:
    # Simulate a legacy row: summary columns only, payload NULL.
    run = _realistic_run("r-legacy")
    store.save_run(run)
    import sqlite3

    conn = sqlite3.connect(store.path)
    conn.execute("UPDATE scan_runs SET payload = NULL WHERE run_id = ?", ("r-legacy",))
    conn.commit()
    conn.close()
    loaded = store.get_run("r-legacy")
    assert loaded is not None
    assert loaded["run_id"] == "r-legacy"
    assert loaded["status"] == "ok"
    assert "picks" not in loaded
