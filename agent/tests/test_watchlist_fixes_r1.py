"""Regression tests for cross-review findings #2,#3,#5,#6,#7,#8,#9,#11.

Scheduler items live in test_watchlist_scheduler.py (see the api-server-wiring
test there). This file covers the service + API items with real WatchlistStore
fixtures mirroring test_watchlist_service.py.
"""

from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.watchlist.service import (  # noqa: E402
    CrossProcessScanLock,
    WatchlistService,
)
import datetime as _dt


class FakeCalendar:
    def __init__(self, day):
        self.day = day

    def is_trading_day(self, d):
        return d.weekday() < 5

    def previous_trading_day(self, d):
        d = d - _dt.timedelta(days=1)
        while d.weekday() >= 5:
            d -= _dt.timedelta(days=1)
        return d

    def latest_completed_trading_day(self, now):
        if self.is_trading_day(now.date()) and now.time() >= _dt.time(15, 0):
            return now.date()
        return self.previous_trading_day(now.date())


def _pick(code: str, rank: int, name: str | None = None) -> dict:
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
        "status": status,
        "started_at": "2026-10-12T18:00:00",
        "finished_at": "2026-10-12T18:01:00",
        "trigger": "cli",
        "duration_s": 1.0,
        "ai_status": "not_reviewed",
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


def _make(store_path: Path, runs: list[dict]) -> WatchlistService:
    """Service whose scan_fn replays the given run summaries in order."""
    seq = {"n": 0}

    def scan_fn(config, **kw):
        run = runs[min(seq["n"], len(runs) - 1)]
        seq["n"] += 1
        return run

    return WatchlistService(
        store=_store(store_path),
        scan_fn=scan_fn,
        calendar_loader=lambda: FakeCalendar(datetime(2026, 10, 12).date()),
        now_fn=lambda: datetime(2026, 10, 12, 18, 0),
    )


def _store(path: Path):
    from src.watchlist.store import WatchlistStore

    return WatchlistStore(path=path)


def test_5_dropped_auto_item_leaves_items(tmp_path: Path) -> None:
    """#5: each scan replaces auto picks — an auto item not picked in the
    latest successful run leaves 'items' (it lives in 'dropped'), and it is
    never marked 'restored'. Manual items and suppressed codes stay."""
    r1 = _scan_run("r1", ["600519", "000001"])
    r2 = _scan_run("r2", ["000001"])  # 600519 dropped
    svc = _make(tmp_path / "w.sqlite3", [r1, r2])
    svc.run_scan(trigger="cli")  # both picked -> auto items
    svc.add_manual("600036")  # manual item survives
    svc.remove("300750")  # suppressed code stays suppressed
    view1 = svc.get_watchlist()
    codes1 = {i["code"] for i in view1["items"]}
    assert "600519" in codes1 and "600036" in codes1

    svc.run_scan(trigger="cli")  # 600519 dropped
    view2 = svc.get_watchlist()
    codes2 = {i["code"] for i in view2["items"]}
    assert "600519" not in codes2  # FIX #5: leaves items
    assert "600036" in codes2  # manual keeps its rule
    assert all(i["change"] != "restored" for i in view2["items"])
    assert {d["code"] for d in view2["dropped"]} >= {"600519"}
    item = svc.store.get_item("600519")
    assert item is None or item["state"] != "active"
    assert "300750" in svc.store.suppressed_codes()


def test_dropped_name_falls_back_to_previous_pick(tmp_path: Path) -> None:
    r1 = _scan_run("r1", ["600519", "000001"])
    # r2 drops 600519 entirely (no score row for it -> name would be None)
    r2 = _scan_run(
        "r2",
        ["000001"],
        scores_extra=[
            {
                "code": "600519",
                "name": None,
                "board": "main_sh",
                "industry": "t",
                "score": None,
                "fundamental": None,
                "trend": None,
                "flow": None,
                "exclusion_reason": "insufficient_data:fundamentals",
            }
        ],
    )
    svc = _make(tmp_path / "w.sqlite3", [r1, r2])
    svc.run_scan(trigger="cli")  # r1
    svc.run_scan(trigger="cli")  # r2
    view = svc.get_watchlist()
    dropped = {d["code"]: d for d in view["dropped"]}
    assert dropped["600519"]["name"] == "股票600519"  # from r1's pick, not None


# --- #6 lock release only when still owner -------------------------------
def test_release_ignores_reacquired_lock(tmp_path: Path) -> None:
    lock1 = CrossProcessScanLock(path=tmp_path / "s.lock")
    assert lock1.acquire() is True
    token = lock1._token
    # Simulate: lock went stale, another process broke + re-acquired it.
    lock1.release()  # close fd but keep token for the simulation
    lock2 = CrossProcessScanLock(path=tmp_path / "s.lock")
    assert lock2.acquire() is True
    # Force a distinguishable token (same-second writes would collide).
    lock2._token = {"pid": lock2._token["pid"], "started_at": "2999-01-01T00:00:00"}
    (tmp_path / "s.lock").write_text(__import__("json").dumps(lock2._token))
    # lock1 "finishes" and tries to release again with its old token
    lock1._fd = None
    lock1._token = token
    lock1.release()
    # lock2's lock must still be present (not deleted by the stale owner)
    info = (tmp_path / "s.lock").exists()
    assert info is True
    lock2.release()
    assert not (tmp_path / "s.lock").exists()


# --- #6 lock ownership ---
def test_release_does_not_steal_reacquired_lock(tmp_path):
    """A release after the lock was broken and re-acquired by another
    owner leaves the new owner's lock file intact."""
    lock_path = tmp_path / "scan.lock"
    lock1 = CrossProcessScanLock(lock_path)
    assert lock1.acquire() is True
    old_token = dict(lock1._token)
    old_token["started_at"] = "2000-01-01T00:00:00"  # distinct from lock2's
    # Simulate: lock1 went stale, another process broke it and re-acquired.
    lock_path.unlink()
    lock2 = CrossProcessScanLock(lock_path)
    assert lock2.acquire() is True
    # The stale owner releases — must NOT remove lock2's file.
    lock1._token = old_token
    lock1.release()
    assert lock_path.exists()
    info = __import__("json").loads(lock_path.read_text())
    assert info["pid"] == lock2._token["pid"]
    lock2.release()
    assert not lock_path.exists()
