"""Cross-process single-flight lock tests (CONTRACT section 10).

Two service instances share one tmp runtime root; a stale lock (dead pid /
old mtime) is broken automatically. No network.
"""

from __future__ import annotations
import json
import os
import time
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.watchlist import service as svc_mod
from src.watchlist.service import (
    CrossProcessScanLock,
    ScanAlreadyRunning,
    WatchlistService,
)


class FakeCalendar:
    def is_trading_day(self, d):
        return True

    def latest_completed_trading_day(self, now):
        return now.date()


def _scan_result(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "trade_date": "2026-10-12",
        "status": "completed",
        "trigger": "cli",
        "started_at": "2026-10-12T18:00:00",
        "ai_status": "not_reviewed",
        "duration_s": 0.1,
        "funnel": [],
        "sources": [],
        "counts": {
            "new": 0,
            "kept": 0,
            "dropped": 0,
            "requalified": 0,
            "ai_demoted": 0,
        },
    }


@pytest.fixture()
def tmp_runtime_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point get_runtime_root at a tmp dir for the lock file."""
    root = tmp_path / "runtime"
    monkeypatch.setenv("VIBE_TRADING_HOME", str(root))
    return root


def _make_service(store_path: Path, scan_result: dict) -> WatchlistService:
    from src.watchlist.store import WatchlistStore

    def scan_fn(config, **kw):
        return scan_result

    return WatchlistService(
        store=WatchlistStore(path=store_path),
        scan_fn=scan_fn,
        calendar_loader=FakeCalendar(),
        now_fn=lambda: datetime(2026, 10, 12, 18, 0),
    )


def test_lock_holds_pid_and_started_at(tmp_runtime_root: Path) -> None:
    lock = CrossProcessScanLock()
    assert lock.acquire() is True
    info = json.loads((tmp_runtime_root / "watchlist" / "scan.lock").read_text())
    assert info["pid"] == os.getpid()
    assert "started_at" in info
    lock.release()
    assert not (tmp_runtime_root / "watchlist" / "scan.lock").exists()


def test_two_services_same_root_single_flight(
    tmp_path: Path, tmp_runtime_root: Path
) -> None:
    """Second service on the same runtime root hits ScanAlreadyRunning."""
    svc_b = _make_service(tmp_path / "b.sqlite3", _scan_result("rB"))

    lock = CrossProcessScanLock()
    assert lock.acquire() is True  # simulate another process holding the lock
    try:
        with pytest.raises(ScanAlreadyRunning):
            svc_b.run_scan(trigger="cli")
        assert svc_b.is_scanning() is True  # reflects the other process
    finally:
        lock.release()
    # After release B can proceed.
    summary = svc_b.run_scan(trigger="cli")
    assert summary["run_id"] == "rB"
    assert svc_b.is_scanning() is False


def test_same_process_lock_reentrant_error(
    tmp_path: Path, tmp_runtime_root: Path
) -> None:
    """Within one process the thread lock still fires first."""
    svc = _make_service(tmp_path / "s.sqlite3", _scan_result("r1"))
    svc._scan_lock.acquire()
    try:
        with pytest.raises(ScanAlreadyRunning):
            svc.run_scan(trigger="cli")
    finally:
        svc._scan_lock.release()


def test_stale_lock_broken_by_dead_pid(tmp_path: Path, tmp_runtime_root: Path) -> None:
    lock_path = tmp_runtime_root / "watchlist" / "scan.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": 999999999, "started_at": "2026-10-12T18:00:00"})
    )
    svc = _make_service(tmp_path / "s.sqlite3", _scan_result("r1"))
    if svc_mod._pid_alive(999999999):
        pytest.skip("pid 999999999 exists on this platform")
    assert svc.is_scanning() is False  # dead pid -> not considered live
    summary = svc.run_scan(trigger="cli")  # stale lock broken, scan proceeds
    assert summary["run_id"] == "r1"
    assert not lock_path.exists()


def test_stale_lock_broken_by_age(
    tmp_path: Path, tmp_runtime_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_runtime_root / "watchlist" / "scan.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "started_at": "2026-01-01T00:00:00"})
    )
    old = time.time() - (svc_mod._LOCK_STALE_S + 60)
    os.utime(lock_path, (old, old))  # older than 2 h
    svc = _make_service(tmp_path / "s.sqlite3", _scan_result("r1"))
    summary = svc.run_scan(trigger="cli")
    assert summary["run_id"] == "r1"


def test_live_pid_lock_is_respected(tmp_path: Path, tmp_runtime_root: Path) -> None:
    lock_path = tmp_runtime_root / "watchlist" / "scan.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "started_at": "2026-10-12T18:00:00"})
    )
    svc = _make_service(tmp_path / "s.sqlite3", _scan_result("r1"))
    assert svc.is_scanning() is True  # our own live pid holds it
    with pytest.raises(ScanAlreadyRunning):
        svc.run_scan(trigger="cli")
    # cleanup so other tests are unaffected
    lock_path.unlink()


def test_pid_alive_dead_pid(monkeypatch):
    """Dead pid -> _pid_alive False -> stale lock broken (Windows-safe path)."""
    import os as _os
    from src.watchlist.service import _pid_alive

    if _os.name == "nt":
        # an almost-certainly-free pid
        assert _pid_alive(4_000_000) is False
    else:
        assert _pid_alive(4_000_000) is False


def test_pid_alive_own_pid(monkeypatch):
    import os as _os
    from src.watchlist.service import _pid_alive

    assert _pid_alive(_os.getpid()) is True


def test_pid_alive_never_sends_signal_windows(monkeypatch):
    """On the Windows path os.kill must never be called."""
    import os as _os
    from src.watchlist import service

    def _boom(pid, sig):
        raise AssertionError(f"os.kill called with ({pid}, {sig}) — forbidden on Windows")

    if _os.name == "nt":
        monkeypatch.setattr(service.os, "kill", _boom)
        service._pid_alive(4_000_000)  # dead pid: ctypes path, os.kill untouched
        service._pid_alive(_os.getpid())  # alive pid: ctypes path


def test_stale_lock_broken_dead_pid(tmp_path, monkeypatch):
    """A lock file holding a dead pid is stale and gets broken."""
    import json
    import time as _time
    from src.watchlist.service import _break_stale_lock

    lock = tmp_path / "scan.lock"
    lock.write_text(json.dumps({"pid": 4_000_000, "started_at": "2026-10-13T07:30:13"}),
                    encoding="utf-8")
    monkeypatch.setattr(service_time := __import__("src.watchlist.service", fromlist=["x"]),
                        "_LOCK_STALE_S", 10_000_000)  # age check off; pid decides
    assert _break_stale_lock(lock) is True
    assert not lock.exists()
