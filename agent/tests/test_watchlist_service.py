"""Offline tests for WatchlistService (fake scan_fn, tmp_path DB, fake calendar)."""

from __future__ import annotations

import datetime as _dt
import threading
from pathlib import Path

import pytest

from src.watchlist.service import ScanAlreadyRunning, WatchlistService


class FakeCalendar:
    """Fixed latest trading day; injectable."""

    def __init__(self, day: _dt.date) -> None:
        self.day = day

    def is_trading_day(self, d: _dt.date) -> bool:
        return d.weekday() < 5

    def previous_trading_day(self, d: _dt.date) -> _dt.date:
        d = d - _dt.timedelta(days=1)
        while d.weekday() >= 5:
            d -= _dt.timedelta(days=1)
        return d

    def latest_completed_trading_day(self, now: _dt.datetime) -> _dt.date:
        if now.date().weekday() < 5 and now.time() >= _dt.time(15, 0):
            return now.date()
        return self.previous_trading_day(now.date())


def _pick(code: str, rank: int, name: str = None) -> dict:
    return {
        "code": code,
        "name": name or f"股票{code}",
        "board": "main_sh",
        "industry": "测试",
        "rank": rank,
        "score": 100 - rank,
        "sub_scores": {"fundamental": 80, "trend": 70, "flow": 60},
        "reasons": ["r1"],
        "flags": [],
        "ai": {"status": "not_reviewed"},
    }


def _scan_run(
    run_id: str,
    picks: list[str],
    *,
    status: str = "ok",
    trade_date: str = "2026-10-12",
    scores_extra: list[dict] | None = None,
) -> dict:
    picks_list = [_pick(c, i + 1) for i, c in enumerate(picks)]
    scores = [
        {
            "code": c,
            "name": f"股票{c}",
            "board": "main_sh",
            "industry": "测试",
            "score": 100 - i,
            "fundamental": 80,
            "trend": 70,
            "flow": 60,
            "exclusion_reason": None,
        }
        for i, c in enumerate(picks)
    ] + (scores_extra or [])
    return {
        "run_id": run_id,
        "trade_date": trade_date,
        "started_at": "2026-10-12T18:00:00",
        "finished_at": "2026-10-12T18:05:00",
        "status": status,
        "trigger": "cli",
        "duration_s": 1.0,
        "intraday": False,
        "ai_status": "not_reviewed",
        "ai_note": None,
        "config": {"top_n": 20},
        "funnel": [
            {"stage": "scored", "count": len(scores), "removed": 0, "note": None}
        ],
        "sources": [{"name": "universe", "status": "ok", "rows": 10}],
        "picks": picks_list,
        "ai_demoted": [],
        "requalified": [],
        "pinned": [],
        "scores": scores,
    }


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "watchlist.sqlite3"


@pytest.fixture()
def fake_now() -> _dt.datetime:
    # Monday 2026-10-12 at 18:00 (after close) — completed trading day is today.
    return _dt.datetime(2026, 10, 12, 18, 0, 0)


def _make_service(
    store_path: Path, scan_fn, fake_now: _dt.datetime
) -> WatchlistService:
    from src.watchlist.store import WatchlistStore

    return WatchlistService(
        store=WatchlistStore(path=store_path),
        scan_fn=scan_fn,
        calendar_loader=lambda: FakeCalendar(_dt.date(2026, 10, 12)),
        now_fn=lambda: fake_now,
    )


def _service(store_path: Path, scan_fn, fake_now: _dt.datetime) -> WatchlistService:
    return _make_service(store_path, scan_fn, fake_now)


def test_run_scan_persists_and_counts(store_path: Path, fake_now: _dt.datetime) -> None:
    def scan_fn(config, **kw):
        return _scan_run("r1", ["600519", "000001"])

    svc = _make_service(store_path, scan_fn, fake_now)
    summary = svc.run_scan(trigger="cli")
    assert summary["status"] == "ok"
    assert summary["counts"]["new"] == 2
    assert summary["counts"]["kept"] == 0
    view = svc.get_watchlist()
    assert [i["code"] for i in view["items"]] == ["600519", "000001"]
    assert all(i["change"] == "new" for i in view["items"])


def test_new_kept_dropped_across_runs(store_path: Path, fake_now: _dt.datetime) -> None:
    sequence = [
        ["600519", "000001"],  # r1
        ["600519", "300750", "000001"],  # r2: 300750 new
        ["300750", "688981"],  # r3: 600519+000001 dropped
    ]
    calls = {"n": 0}

    def scan_fn(config, **kw):
        rid = f"r{calls['n'] + 1}"
        picks = sequence[calls["n"]]
        calls["n"] += 1
        return _scan_run(rid, picks)

    svc = _make_service(store_path, scan_fn, fake_now)
    s1 = svc.run_scan(trigger="cli")
    assert s1["counts"]["new"] == 2
    s2 = svc.run_scan(trigger="cli")
    assert s2["counts"]["kept"] == 2
    assert s2["counts"]["new"] == 1
    s3 = svc.run_scan(trigger="cli")
    assert s3["counts"]["dropped"] == 2
    view = svc.get_watchlist()
    dropped = {d["code"]: d for d in view["dropped"]}
    assert set(dropped) == {"600519", "000001"}
    # FIX B: 600519/000001 are absent from r3's scores entirely -> 'no data'
    assert dropped["600519"]["reason"] == "no data in this run"


def test_view_diff_and_items_come_from_latest_successful_run(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """Bug after M1 invalidation: dropped/requalified/ai_demoted must come
    from the latest SUCCESSFUL run only (never a failed run's stored rows)."""
    runs = [
        _scan_run("r1", ["600519", "000001", "300750"]),  # baseline
        _scan_run("r2", ["600519", "000001"]),  # ok: drops 300750
        _scan_run("r3", ["600519"], status="failed"),  # drops 000001
    ]
    runs[2]["ai_demoted"] = [{"code": "999999", "name": "失败泄露"}]
    runs[2]["requalified"] = [{"code": "000651", "name": "失败泄露"}]
    runs[2]["started_at"] = "2026-10-12T18:10:00"
    runs[2]["finished_at"] = "2026-10-12T18:15:00"
    calls = {"n": 0}

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        r["trigger"] = "cli"
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")  # r1
    svc.run_scan(trigger="cli")  # r2
    svc.add_manual("000651")
    svc.remove("000651")  # suppressed -> would pass the rq filter if leaked
    svc.run_scan(trigger="cli")  # r3 failed
    view = svc.get_watchlist()
    # dropped comes from r2 (latest successful), never r3's failed-run rows
    assert [d["code"] for d in view["dropped"]] == ["300750"]
    assert view["requalified"] == []  # r2 had none; r3's suppressed leak blocked
    assert view["ai_demoted"] == []  # r2 had none; r3's must not leak
    # items are the latest successful run's picks
    assert [i["code"] for i in view["items"]] == ["600519", "000001"]


def test_view_items_sorted_by_rank_with_manual_last(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """Items display rank-ascending; manual items (rank None) last,
    regardless of added_at order."""
    sequence = [
        ["600519", "000001"],  # r1: 600519 rank1, 000001 rank2
        ["000001", "600519"],  # r2: ranks swapped
    ]
    calls = {"n": 0}

    def scan_fn(config, **kw):
        picks = sequence[calls["n"]]
        calls["n"] += 1
        return _scan_run(f"r{calls['n']}", picks)

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.add_manual("000858")
    svc.run_scan(trigger="cli")
    view = svc.get_watchlist()
    assert [i["code"] for i in view["items"]] == ["000001", "600519", "000858"]
    assert [i["rank"] for i in view["items"]] == [1, 2, None]


def test_failed_run_does_not_wipe_picks(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    runs = [
        _scan_run("r1", ["600519", "000001"]),
        _scan_run("r2", [], status="failed"),
    ]
    runs[1]["started_at"] = "2026-10-12T18:10:00"
    runs[1]["finished_at"] = "2026-10-12T18:15:00"
    calls = {"n": 0}

    def scan_fn(config, **kw):
        runs[calls["n"]]["trigger"] = "cli"
        r = runs[calls["n"]]
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.run_scan(trigger="cli")
    view = svc.get_watchlist()
    # items still from r1 (last successful)
    assert {i["code"] for i in view["items"]} == {"600519", "000001"}
    # latest_run is the current EFFECTIVE run (last successful); the failed
    # run surfaces separately with its failure_reason.
    assert view["latest_run"]["run_id"] == "r1"
    assert view["latest_run"]["status"] == "ok"
    assert view["last_failed_run"]["run_id"] == "r2"
    assert view["last_failed_run"]["status"] == "failed"


def test_coverage_failed_run_not_baseline_and_stays_stale(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """A coverage-failed run: kept for history, not baseline, not clearing
    staleness; CLI/API surfaces show the failure_reason."""
    runs = [
        _scan_run("r1", ["600519", "000001"], trade_date="2026-10-09"),
        _scan_run("r2", ["300750"], status="failed"),
    ]
    runs[1]["failure_reason"] = "insufficient coverage (trend 12.0% < 90%)"
    runs[1]["started_at"] = "2026-10-12T18:10:00"
    runs[1]["finished_at"] = "2026-10-12T18:15:00"
    calls = {"n": 0}

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    summary2 = svc.run_scan(trigger="cli")
    assert summary2["status"] == "failed"
    assert summary2["failure_reason"] == "insufficient coverage (trend 12.0% < 90%)"

    view = svc.get_watchlist()
    # items untouched by the failed run; latest_run stays r1
    assert {i["code"] for i in view["items"]} == {"600519", "000001"}
    assert view["latest_run"]["run_id"] == "r1"
    assert view["last_failed_run"]["failure_reason"] == (
        "insufficient coverage (trend 12.0% < 90%)"
    )
    # failed run never clears staleness: today has a run but it failed
    assert view["stale"]["is_stale"] is True
    # history keeps the failure reason
    history = svc.list_runs(limit=5)
    assert history[0]["run_id"] == "r2"
    assert history[0]["failure_reason"] == (
        "insufficient coverage (trend 12.0% < 90%)"
    )

    # baseline skips the failed run: r3 diffs against r1 (kept 600519,
    # dropped 000001, new nothing) — if r2 were the baseline, 600519
    # would come back as "new".
    runs.append(_scan_run("r3", ["600519"]))
    runs[2]["started_at"] = "2026-10-13T18:00:00"
    runs[2]["finished_at"] = "2026-10-13T18:05:00"
    runs[2]["trade_date"] = "2026-10-13"
    summary3 = svc.run_scan(trigger="cli")
    assert summary3["counts"] == {
        "new": 0,
        "kept": 1,
        "dropped": 1,
        "requalified": 0,
        "ai_demoted": 0,
    }


def test_latest_run_exposes_coverage(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """latest_run must carry the run's coverage dict (payload-only key)."""
    run1 = _scan_run("r1", ["600519"])
    run1["coverage"] = {"universe": 5000, "after_filter": 120, "scored": 20}
    run2 = _scan_run("r2", ["600519"])
    run2["started_at"] = "2026-10-12T19:00:00"
    run2["finished_at"] = "2026-10-12T19:05:00"
    runs = [run1, run2]
    calls = {"n": 0}

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    view = svc.get_watchlist()
    assert view["latest_run"]["run_id"] == "r1"
    assert view["latest_run"]["coverage"] == {
        "universe": 5000,
        "after_filter": 120,
        "scored": 20,
    }

    # r2 carries no coverage -> key still present, value None (no KeyError)
    svc.run_scan(trigger="cli")
    view = svc.get_watchlist()
    assert view["latest_run"]["run_id"] == "r2"
    assert view["latest_run"]["coverage"] is None


def test_manual_pinned_survives_scans(store_path: Path, fake_now: _dt.datetime) -> None:
    scan_picks = {"n": 0}

    def scan_fn(config, **kw):
        scan_picks["n"] += 1
        return _scan_run(f"r{scan_picks['n']}", ["600519"])

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")  # seed scores so name lookup works
    item = svc.add_manual("300750", note="自己加的")
    assert item["origin"] == "manual"

    def scan_fn2(config, **kw):
        return _scan_run("rX", ["600519"])

    svc._scan_fn = scan_fn2
    svc.run_scan(trigger="cli")
    view = svc.get_watchlist()
    codes = {i["code"]: i for i in view["items"]}
    assert "300750" in codes
    assert codes["300750"]["origin"] == "manual"
    assert codes["300750"]["change"] == "manual"


def test_removed_never_re_added_and_requalified(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    state = {"n": 0}

    def scan_fn(config, suppressed=None, **kw):
        state["n"] += 1
        run = _scan_run(f"r{state['n']}", ["600519", "000001"])
        # pipeline puts suppressed codes that qualify into requalified
        if suppressed and "000001" in suppressed:
            run["requalified"] = [
                {
                    "code": "000001",
                    "name": "股票000001",
                    "rank": 2,
                    "score": 99,
                    "sub_scores": {"fundamental": 80, "trend": 70, "flow": 60},
                    "reasons": ["r1"],
                }
            ]
        return run

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.remove("000001")
    summary = svc.run_scan(trigger="cli")
    assert summary["counts"]["requalified"] == 1
    view = svc.get_watchlist()
    codes = {i["code"] for i in view["items"]}
    assert "000001" not in codes  # never re-added
    rq = {r["code"]: r for r in view["requalified"]}
    assert rq["000001"]["label"] == "qualifies again - you removed it"


def test_restore_clears_suppression(store_path: Path, fake_now: _dt.datetime) -> None:
    def scan_fn(config, **kw):
        return _scan_run("r1", ["600519"])

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.add_manual("300750")
    svc.remove("300750")
    assert svc.store.suppressed_codes() == ["300750"]
    item = svc.restore("300750")
    assert item["state"] == "active"
    assert svc.store.suppressed_codes() == []
    view = svc.get_watchlist()
    mine = [i for i in view["items"] if i["code"] == "300750"]
    assert mine and mine[0]["change"] == "manual"


def test_add_manual_validation(store_path: Path, fake_now: _dt.datetime) -> None:
    def scan_fn(config, **kw):
        return _scan_run("r1", [])

    svc = _make_service(store_path, scan_fn, fake_now)
    with pytest.raises(ValueError):
        svc.add_manual("60051")  # too short
    with pytest.raises(ValueError):
        svc.add_manual("6005a1")  # not digits
    with pytest.raises(ValueError):
        svc.add_manual("920002")  # BSE rejected
    with pytest.raises(ValueError):
        svc.add_manual("430047")  # BSE rejected
    item = svc.add_manual("600519")
    assert item["code"] == "600519"


def test_manual_scores_or_unscored_reason(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    def scan_fn(config, **kw):
        run = _scan_run(
            "r1",
            ["600519"],
            scores_extra=[
                {
                    "code": "000001",
                    "name": "平安银行",
                    "board": "main_sz",
                    "industry": "银行",
                    "score": 55,
                    "fundamental": 60,
                    "trend": 50,
                    "flow": 55,
                    "exclusion_reason": "excluded:st",
                },
                {
                    "code": "300750",
                    "name": "宁德时代",
                    "board": "chinext",
                    "industry": "电池",
                    "score": None,
                    "fundamental": None,
                    "trend": None,
                    "flow": None,
                    "exclusion_reason": "excluded:illiquid",
                },
            ],
        )
        return run

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.add_manual("600519")
    svc.add_manual("000001")  # in scores with exclusion
    svc.add_manual("300750")  # in scores, excluded
    view = svc.get_watchlist()
    by_code = {i["code"]: i for i in view["items"]}
    assert by_code["600519"]["scores_available"] is True
    assert by_code["600519"]["score"] is not None
    assert by_code["000001"]["scores_available"] is False
    assert "excluded:st" in (by_code["000001"]["unscored_reason"] or "")
    assert by_code["300750"]["unscored_reason"] == "excluded:illiquid"


def test_staleness_with_fake_calendar(store_path: Path, fake_now: _dt.datetime) -> None:
    def scan_fn(config, **kw):
        return _scan_run("r1", ["600519"], trade_date="2026-10-12")

    svc = _make_service(store_path, scan_fn, fake_now)
    st = svc.staleness()
    assert st["latest_trading_day"] == "2026-10-12"
    assert st["is_stale"] is True  # no scan yet
    svc.run_scan(trigger="cli")
    st = svc.staleness()
    assert st["is_stale"] is False
    assert st["last_scan_trade_date"] == "2026-10-12"

    # now move the clock to the NEXT trading day without a scan -> stale
    svc2 = _make_service(store_path, scan_fn, _dt.datetime(2026, 10, 13, 18, 0))
    assert svc2.staleness()["is_stale"] is True


def test_single_flight_raises(store_path: Path, fake_now: _dt.datetime) -> None:
    release = threading.Event()
    started = threading.Event()

    def slow_scan(config, **kw):
        started.set()
        release.wait(timeout=5)
        return _scan_run("r1", ["600519"])

    import threading as _th

    svc = _make_service(store_path, slow_scan, fake_now)
    t = _th.Thread(target=lambda: svc.run_scan(trigger="cli"), daemon=True)
    t.start()
    started.wait(timeout=5)
    with pytest.raises(ScanAlreadyRunning):
        svc.run_scan(trigger="cli")
    assert svc.is_scanning() is True
    release.set()
    t.join(timeout=5)
    assert svc.is_scanning() is False


def test_settings_defaults_and_env_override(
    store_path: Path,
    fake_now: _dt.datetime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def scan_fn(config, **kw):
        return _scan_run("r1", ["600519"])

    monkeypatch.delenv("VIBE_TRADING_WATCHLIST_SCHEDULE", raising=False)
    svc = _make_service(store_path, scan_fn, fake_now)
    settings = svc.get_settings()
    assert settings["scan"]["top_n"] == 20  # ScanConfig defaults
    assert settings["scan"]["min_total_mv"] == 3e9
    assert settings["schedule"] == {
        "enabled": False,
        "time": "18:00",
        "timezone": "Asia/Shanghai",
        "catch_up": True,
        "warmup_wait_min": 60,  # phase (b): the warm-up wait became a setting
    }

    # update_settings persists
    svc.update_settings({"scan": {"top_n": 5}, "schedule": {"enabled": True}})
    assert svc.get_settings()["scan"]["top_n"] == 5
    assert svc.get_settings()["schedule"]["enabled"] is True

    # env override wins for enabled
    monkeypatch.setenv("VIBE_TRADING_WATCHLIST_SCHEDULE", "0")
    assert svc.get_settings()["schedule"]["enabled"] is False
    monkeypatch.setenv("VIBE_TRADING_WATCHLIST_SCHEDULE", "1")
    assert svc.get_settings()["schedule"]["enabled"] is True

    # overrides reach the scan_fn config
    captured = {}

    def scan_fn2(config, **kw):
        captured["top_n"] = config.top_n
        captured["suppressed"] = kw.get("suppressed")
        captured["pinned"] = kw.get("pinned")
        return _scan_run("r2", ["600519"])

    svc._scan_fn = scan_fn2
    svc.run_scan(trigger="cli", config_overrides={"top_n": 7})
    assert captured["top_n"] == 7


def test_partial_scan_settings_patch_preserves_existing_values(
    store_path: Path,
    fake_now: _dt.datetime,
) -> None:
    svc = _make_service(
        store_path,
        lambda config, **kw: _scan_run("r1", ["600519"]),
        fake_now,
    )
    svc.update_settings(
        {"scan": {"ai_time_budget_s": 1500, "ai_call_timeout_s": 180, "ai_enabled": True}}
    )

    updated = svc.update_settings({"scan": {"ai_call_timeout_s": 240}})

    assert updated["scan"]["ai_call_timeout_s"] == 240
    assert updated["scan"]["ai_time_budget_s"] == 1500
    assert updated["scan"]["ai_enabled"] is True


def test_suppressed_and_pinned_passed_to_scan(
    store_path: Path,
    fake_now: _dt.datetime,
) -> None:
    captured = {}

    def scan_fn(config, **kw):
        captured["suppressed"] = set(kw.get("suppressed") or set())
        captured["pinned"] = set(kw.get("pinned") or set())
        return _scan_run("r1", ["600519"])

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")  # seed
    svc.remove("000001")  # suppressed (stub item)
    svc.add_manual("600519")  # pinned manual
    svc.run_scan(trigger="cli")
    assert "000001" in captured["suppressed"]
    assert "600519" in captured["pinned"]


# --------------------------------------------------------------------------- #
# FIX A: clean started_at + seq ordering
# --------------------------------------------------------------------------- #
def test_fix_a_started_at_clean_and_seq_orders(
    store_path: Path,
    fake_now: _dt.datetime,
) -> None:
    """Two runs saved in the same second: started_at stays clean, seq orders."""

    def scan_fn(config, **kw):
        return _scan_run("r1", ["600519"])

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    # r2 with an identical started_at but a different timestamp set
    svc._scan_fn = lambda config, **kw: _scan_run("r2", ["000001"])
    svc.run_scan(trigger="cli")
    runs = svc.store.list_runs(limit=10)
    assert len(runs) == 2
    for r in runs:
        assert "." not in r["started_at"]  # no .000-style suffixes
    # last successful is the later-saved run
    assert svc.store.last_successful_run()["run_id"] == "r2"


def test_fix_a_migration_from_v1(tmp_path: Path) -> None:
    """A v1 store (no seq column, dotted started_at) migrates cleanly."""
    import sqlite3

    from src.watchlist.store import WatchlistStore

    db_path = tmp_path / "v1.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE scan_runs (
            run_id TEXT PRIMARY KEY, trade_date TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
            trigger TEXT NOT NULL, duration_s REAL, intraday INTEGER,
            ai_status TEXT, ai_note TEXT, config TEXT, funnel TEXT,
            sources TEXT, ai_demoted TEXT, requalified TEXT, pinned TEXT,
            payload TEXT
        );
    """)
    conn.execute(
        "INSERT INTO scan_runs VALUES ('r1','2026-10-12','2026-10-12T18:00:00.000',"
        "'2026-10-12T18:05:00','ok','cli',1.0,0,'not_reviewed',NULL,'{}','[]','[]',"
        "'[]','[]','[]','{}')"
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    store = WatchlistStore(path=db_path)
    run = store.last_successful_run()
    assert run is not None
    assert run["run_id"] == "r1"
    assert "." not in run["started_at"]  # clean after migration
    assert store.get_setting("scan") is None  # other tables untouched


# --------------------------------------------------------------------------- #
# FIX B: dropped reason with new rank
# --------------------------------------------------------------------------- #
def test_fix_b_dropped_reason_new_rank(
    store_path: Path,
    fake_now: _dt.datetime,
) -> None:
    """A dropped stock still scored in the new run reports its new rank."""
    # r1 picks 600519 (rank 1); r2 scores 600519 at rank 4 by score (not picked)
    calls = {"n": 0}

    def scan_fn(config, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _scan_run("r1", ["600519", "000001"])
        run = _scan_run("r2", ["000001"])
        # 600519 still scored (no exclusion_reason) with score 61.2 -> rank 3
        run["scores"].append(
            {
                "code": "600519",
                "name": "股票600519",
                "board": "main_sh",
                "industry": "测试",
                "score": 61.2,
                "fundamental": 50,
                "trend": 55,
                "flow": 60,
                "exclusion_reason": None,
            }
        )
        # two better-scored stocks not in picks
        for i, c in enumerate(["688981", "300750"]):
            run["scores"].append(
                {
                    "code": c,
                    "name": f"股票{c}",
                    "board": "main_sh",
                    "industry": "测试",
                    "score": 90 - i,
                    "fundamental": 80,
                    "trend": 80,
                    "flow": 80,
                    "exclusion_reason": None,
                }
            )
        return run

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.run_scan(trigger="cli")
    view = svc.get_watchlist()
    dropped = {d["code"]: d for d in view["dropped"]}
    assert dropped["600519"]["reason"] == "rank fell to 4 (score 61.2)"


def test_fix_b_dropped_reason_exclusion(
    store_path: Path,
    fake_now: _dt.datetime,
) -> None:
    """A dropped stock now excluded reports the exclusion reason instead."""
    calls = {"n": 0}

    def scan_fn(config, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _scan_run("r1", ["600519"])
        run = _scan_run("r2", ["000001"])
        run["scores"].append(
            {
                "code": "600519",
                "name": "股票600519",
                "board": "main_sh",
                "industry": "测试",
                "score": None,
                "fundamental": None,
                "trend": None,
                "flow": None,
                "exclusion_reason": "excluded:st",
            }
        )
        return run

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.run_scan(trigger="cli")
    view = svc.get_watchlist()
    dropped = {d["code"]: d for d in view["dropped"]}
    assert dropped["600519"]["reason"] == "excluded:st"


# --------------------------------------------------------------------------- #
# Partial counts as success: pipeline completed => watchlist updated (CONTRACT)
# ---------------------------------------------------------------------------
def test_partial_run_updates_watchlist_and_is_diff_baseline(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """A partial run moves the auto set AND is the baseline for the next diff."""
    calls = {"n": 0}

    def scan_fn(config, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _scan_run("r1", ["600519", "000001"])
        if calls["n"] == 2:
            r = _scan_run("r2", ["000001", "300750"], status="partial")
            r["started_at"] = "2026-10-12T18:10:00"
            r["finished_at"] = "2026-10-12T18:15:00"
            return r
        r = _scan_run("r3", ["300750"])
        r["started_at"] = "2026-10-12T19:00:00"
        r["finished_at"] = "2026-10-12T19:05:00"
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    summary = svc.run_scan(trigger="cli")
    assert summary["status"] == "partial"
    # the partial run updated the watchlist (applied new/kept/dropped)
    view = svc.get_watchlist()
    assert {i["code"] for i in view["items"]} == {"000001", "300750"}
    assert {d["code"] for d in view["dropped"]} == {"600519"}
    # ...and is the diff baseline: r3 kept 300750 (from partial r2, not r1)
    s3 = svc.run_scan(trigger="cli")
    assert s3["counts"]["kept"] == 1
    assert s3["counts"]["new"] == 0
    assert s3["counts"]["dropped"] == 1  # 000001
    assert svc.store.last_successful_run()["run_id"] == "r3"


def test_failed_run_does_not_update_watchlist_or_baseline(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """A failed run never touches the watchlist and never becomes a baseline."""
    calls = {"n": 0}

    def scan_fn(config, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _scan_run("r1", ["600519"])
        if calls["n"] == 2:
            r = _scan_run("r2", ["300750"], status="failed")
            r["started_at"] = "2026-10-12T18:10:00"
            r["finished_at"] = "2026-10-12T18:15:00"
            return r
        r = _scan_run("r3", ["300750"])
        r["started_at"] = "2026-10-12T19:00:00"
        r["finished_at"] = "2026-10-12T19:05:00"
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.run_scan(trigger="cli")  # failed — items unchanged
    view = svc.get_watchlist()
    assert {i["code"] for i in view["items"]} == {"600519"}
    # r3 diffs against r1 (the failed r2 is not a baseline): 300750 is new
    s3 = svc.run_scan(trigger="cli")
    assert s3["counts"]["new"] == 1
    assert s3["counts"]["kept"] == 0
    assert s3["counts"]["dropped"] == 1  # 600519


def test_staleness_satisfied_by_partial_run(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """A partial scan of the latest completed trading day clears staleness."""

    def scan_fn(config, **kw):
        return _scan_run("r1", ["600519"], status="partial", trade_date="2026-10-12")

    svc = _make_service(store_path, scan_fn, fake_now)
    assert svc.staleness()["is_stale"] is True  # no scan yet
    svc.run_scan(trigger="cli")
    st = svc.staleness()
    assert st["is_stale"] is False
    assert st["last_scan_trade_date"] == "2026-10-12"


# --------------------------------------------------------------------------- #
# invalidate_run (maintenance invalidation)
# --------------------------------------------------------------------------- #
def test_invalidate_run_restores_previous_auto_set(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """Invalidating the latest successful run restores the previous auto
    set (manual pins untouched) and surfaces the standard coverage reason."""
    runs = [
        _scan_run("r1", ["600519", "000001"]),
        _scan_run("r2", ["300750"]),
    ]
    runs[1]["started_at"] = "2026-10-12T18:10:00"
    runs[1]["finished_at"] = "2026-10-12T18:15:00"
    # r2 carries a below-floor coverage block (M1 shape: trend 501/4166 = 12%)
    runs[1]["coverage"] = {
        "passed_exclusions": 4166,
        "trend_scored": 501,
        "insufficient_by_key": {},
    }
    calls = {"n": 0}

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")  # r1 ok -> auto items
    svc.run_scan(trigger="cli")  # r2 ok -> replaces the auto set
    svc.add_manual("000858")
    view = svc.get_watchlist()
    # each successful scan replaces the auto set: only r2's pick + the manual
    assert {i["code"] for i in view["items"]} == {"300750", "000858"}

    result = svc.invalidate_run("r2")
    assert result["status"] == "failed"
    assert result["failure_reason"] == "insufficient coverage (trend 12.0% < 90%)"
    assert result["restored"] is True
    assert result["restored_from"] == "r1"

    view = svc.get_watchlist()
    codes = {i["code"]: i for i in view["items"]}
    # auto set back to r1's picks; 300750 (r2-only) gone; manual untouched
    assert set(codes) == {"600519", "000001", "000858"}
    assert codes["000858"]["origin"] == "manual"
    assert view["latest_run"]["run_id"] == "r1"
    assert view["last_failed_run"]["run_id"] == "r2"
    assert view["last_failed_run"]["failure_reason"].startswith("insufficient coverage")
    history = svc.list_runs(limit=5)
    assert [h["run_id"] for h in history] == ["r2", "r1"]
    assert history[0]["status"] == "failed"
    assert history[1]["status"] == "ok"


def test_invalidate_run_keeps_suppressed_and_manual(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """No successful predecessor: auto set clears; suppressed rows stay."""
    calls = {"n": 0}

    def scan_fn(config, **kw):
        calls["n"] += 1
        return _scan_run("r1", ["600519", "000001"])

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.add_manual("000858")
    svc.remove("000001")  # suppression

    result = svc.invalidate_run("r1", reason="manual maintenance")
    assert result["failure_reason"] == "manual maintenance"
    assert result["restored"] is True
    assert result["restored_from"] is None
    view = svc.get_watchlist()
    codes = {i["code"] for i in view["items"]}
    assert codes == {"000858"}  # auto cleared; manual kept
    assert svc.store.get_item("000001")["state"] == "removed"  # still suppressed


def test_invalidate_run_not_latest_successful_skips_restore(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """Invalidating an older successful run must not touch the current set."""
    runs = [
        _scan_run("r1", ["600519"]),
        _scan_run("r2", ["300750"]),
        _scan_run("r3", ["688981"]),
    ]
    for i, r in enumerate(runs):
        if i:
            r["started_at"] = f"2026-10-12T18:{10 + i * 5:02d}:00"
            r["finished_at"] = f"2026-10-12T18:{15 + i * 5:02d}:00"
    calls = {"n": 0}

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    for _ in range(3):
        svc.run_scan(trigger="cli")
    result = svc.invalidate_run("r2")
    assert result["status"] == "failed"
    assert result["restored"] is False
    view = svc.get_watchlist()
    assert {i["code"] for i in view["items"]} == {"688981"}  # r3's set intact
    assert view["latest_run"]["run_id"] == "r3"


def test_invalidate_run_unknown_id(store_path: Path, fake_now: _dt.datetime) -> None:
    def scan_fn(config, **kw):
        return _scan_run("r1", ["600519"])

    svc = _make_service(store_path, scan_fn, fake_now)
    with pytest.raises(ValueError, match="run not found"):
        svc.invalidate_run("nope")
