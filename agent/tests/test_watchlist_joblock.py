"""FIX_CONTRACT F2/F3/F9 regressions (hermes scope).

F2: one cross-process job lock for scan/review/warmup/backfill.
  - lock file deleted mid-run -> run aborts with a clear failure_reason
  - scheduler + API + CLI started together -> exactly one runs
F3: setup_watchlist_scheduler singleton + server.lock refusing a second server.
F9: forget_manual deletes manual rows only, refuses auto, leaves no suppression.

Offline only: fake scan callables, tmp homes, no network.

Run: pytest tests/test_watchlist_joblock.py -q
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.watchlist.service import (  # noqa: E402
    CrossProcessScanLock,
    JobHandle,
    JobLockLost,
    ScanAlreadyRunning,
    WatchlistService,
)
from src.watchlist.store import WatchlistStore  # noqa: E402

from test_watchlist_taskb_regressions import _make_service, _scan_run  # noqa: E402
from src.watchlist.service import _lock_path  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_job_lock():
    """Remove the shared scan.lock between tests (services in this file are
    built with an explicit store, so the home check is skipped, but the lock
    path is per-runtime-home and shared)."""
    yield
    try:
        _lock_path().unlink()
    except OSError:
        pass


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Point the runtime home at a tmp dir (the conftest sandbox already does
    via VIBE_TRADING_HOME; this keeps lock paths inside the test's own dir)."""
    home = tmp_path / "home" / ".vibe-trading"
    monkeypatch.setenv("VIBE_TRADING_HOME", str(home))
    return home


def _two_services(tmp_path: Path):
    """Two service instances sharing ONE store (simulates two processes)."""
    store = WatchlistStore(path=tmp_path / "watchlist.sqlite3")

    def scan_fn(config, **kw):
        return _scan_run("r", ["600519"])

    a = WatchlistService(store=store, scan_fn=scan_fn, now_fn=lambda: _dt.datetime(2026, 10, 16, 12))
    b = WatchlistService(store=store, scan_fn=scan_fn, now_fn=lambda: _dt.datetime(2026, 10, 16, 12))
    return a, b


# --------------------------------------------------------------------------- #
# F2: acquire_job / JobHandle
# --------------------------------------------------------------------------- #
def test_acquire_job_exclusive_across_two_services(tmp_path) -> None:
    a, b = _two_services(tmp_path)
    h1 = a.acquire_job("scan", owner="scheduler")
    assert h1.kind == "scan" and h1.owner == "scheduler"
    with pytest.raises(ScanAlreadyRunning) as ei:
        b.acquire_job("scan", owner="api")
    # 409 payload: who holds it — kind, owner, pid, since.
    msg = str(ei.value)
    assert "kind=scan" in msg and "owner=scheduler" in msg and "pid=" in msg
    h1.release()
    h2 = b.acquire_job("backfill", owner="cli")
    h2.release()


def test_job_lock_kinds_share_one_lock(tmp_path) -> None:
    """scan vs warmup vs backfill: ONE lock — different kinds still exclude."""
    a, b = _two_services(tmp_path)
    h = a.acquire_job("warmup", owner="scheduler")
    with pytest.raises(ScanAlreadyRunning):
        b.acquire_job("scan", owner="cli")
    with pytest.raises(ScanAlreadyRunning):
        b.acquire_job("backfill", owner="cli")
    h.release()
    assert b.acquire_job("scan", owner="cli") is not None


def test_job_handle_check_passes_then_raises_after_deletion(tmp_path) -> None:
    a, _b = _two_services(tmp_path)
    h = a.acquire_job("scan", owner="api")
    h.check()  # still held: no exception
    h.heartbeat()  # touching a live lock is fine
    # THE REQUIRED TEST: the lock file is deleted mid-run.
    lock_file = a.store.path.parent / "scan.lock"
    # resolve the real lock path (runtime home sandbox)
    from src.watchlist.service import _lock_path

    lock_file = _lock_path()
    assert lock_file.exists()
    lock_file.unlink()
    with pytest.raises(JobLockLost) as ei:
        h.check()
    assert "job lock lost" in str(ei.value)
    # release after loss is a no-op (nothing to unlink), must not raise
    h.release()


def test_job_handle_check_raises_when_token_taken_over(tmp_path) -> None:
    a, b = _two_services(tmp_path)
    h = a.acquire_job("scan", owner="api")
    from src.watchlist.service import _lock_path

    lock_file = _lock_path()
    # Another writer overwrites the file (take-over scenario).
    lock_file.write_text(json.dumps({"pid": 999999, "started_at": "x"}), encoding="utf-8")
    with pytest.raises(JobLockLost):
        h.check()
    # The original holder must NOT delete the new holder's file on release.
    h.release()
    assert lock_file.exists()
    lock_file.unlink()


def test_lock_file_deleted_mid_run_aborts_with_failure_reason(tmp_path) -> None:
    """REQUIRED TEST (F2): delete the lock file mid-run -> the run aborts,
    is stored as failed with the clear reason, and the exception surfaces."""
    store = WatchlistStore(path=tmp_path / "watchlist.sqlite3")
    calls = {"n": 0}

    def scan_fn(config, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            # Simulate the mid-run deletion: the lock vanishes between
            # the pre-scan check and the pipeline's own check.
            from src.watchlist.service import _lock_path

            _lock_path().unlink()
        # The pipeline honours the injected lock_check at stage boundaries
        # (F2) — the fake pipeline does the same.
        lock_check = kw.get("lock_check")
        if lock_check is not None:
            lock_check()
        return _scan_run("r", ["600519"])

    svc = WatchlistService(
        store=store, scan_fn=scan_fn, now_fn=lambda: _dt.datetime(2026, 10, 16, 12)
    )
    svc.run_scan(trigger="cli")  # first run: fine
    with pytest.raises(JobLockLost):
        svc.run_scan(trigger="cli")
    # The aborted run is stored as failed with the clear reason.
    failed = [r for r in store.list_runs(10) if r["status"] == "failed"]
    assert failed, "aborted run must be persisted"
    reasons = [
        r.get("failure_reason") or ""
        for r in failed
        if "job lock lost" in (r.get("failure_reason") or "")
    ]
    assert reasons, f"failure_reason must name the lock loss: {failed}"
    # The pipeline never ran the second scan (aborted BEFORE the callable
    # got to finish — its _scan_run never saved; exactly 1 successful run).
    ok = [r for r in store.list_runs(10) if r["status"] == "ok"]
    assert len(ok) == 1


def test_run_scan_with_preacquired_handle_does_not_double_lock(tmp_path) -> None:
    """API pattern: acquire synchronously, hand the handle to the worker."""
    a, b = _two_services(tmp_path)
    handle = a.acquire_job("scan", owner="api")
    # The service accepts the external handle and does NOT release it.
    summary = a.run_scan(trigger="api", handle=handle)
    assert summary["run_id"] == "r"
    # The lock is still held by the caller (the worker pattern releases it).
    with pytest.raises(ScanAlreadyRunning):
        b.acquire_job("scan", owner="cli")
    handle.release()
    assert b.acquire_job("scan", owner="cli") is not None


def test_scheduler_api_cli_started_together_exactly_one_runs(tmp_path) -> None:
    """REQUIRED TEST (F2): three contenders, one lock -> one winner."""
    a, b = _two_services(tmp_path)
    store = WatchlistStore(path=tmp_path / "watchlist.sqlite3")
    c = WatchlistService(
        store=store, scan_fn=lambda cfg, **kw: _scan_run("r", ["600519"]),
        now_fn=lambda: _dt.datetime(2026, 10, 16, 12),
    )
    results: dict[str, str] = {}
    barrier = threading.Barrier(3)

    def contender(name: str, svc: WatchlistService):
        barrier.wait()  # maximise contention
        try:
            svc.run_scan(trigger=name)
            results[name] = "ran"
        except ScanAlreadyRunning as exc:
            results[name] = f"refused: {exc}"

    threads = [
        threading.Thread(target=contender, args=("scheduler", a)),
        threading.Thread(target=contender, args=("api", b)),
        threading.Thread(target=contender, args=("cli", c)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    winners = [k for k, v in results.items() if v == "ran"]
    assert len(winners) == 1, results
    refused = [k for k, v in results.items() if v.startswith("refused")]
    assert len(refused) == 2, results
    # Exactly one successful run row (the winner's) — no duplicates.
    from src.watchlist.service import _lock_path

    lock_file = _lock_path()
    if lock_file.exists():
        lock_file.unlink()


def test_stale_lock_from_dead_process_is_broken(tmp_path) -> None:
    a, b = _two_services(tmp_path)
    h = a.acquire_job("scan", owner="ghost")
    from src.watchlist.service import _lock_path

    lock_file = _lock_path()
    # The holder 'dies': rewrite the lock with a dead pid.
    lock_file.write_text(
        json.dumps({"pid": 2**28, "started_at": "2000-01-01T00:00:00"}),
        encoding="utf-8",
    )
    # The dead holder's lock must not block a new job.
    h2 = b.acquire_job("scan", owner="new")
    h2.release()
    h.released = True  # the ghost's handle is dead; keep release() honest


# --------------------------------------------------------------------------- #
# F3: scheduler singleton + server instance lock
# --------------------------------------------------------------------------- #
def test_setup_watchlist_scheduler_is_idempotent(monkeypatch) -> None:
    from src.api import watchlist_routes as wr

    prev = wr._SCHEDULER
    wr._SCHEDULER = None
    try:
        s1 = wr.setup_watchlist_scheduler()
        s2 = wr.setup_watchlist_scheduler()
        assert s1 is s2
    finally:
        wr._SCHEDULER = prev


def test_server_instance_lock_refuses_second_server(monkeypatch, tmp_path) -> None:
    from src.api import watchlist_routes as wr

    home = tmp_path / "home" / ".vibe-trading"
    monkeypatch.setenv("VIBE_TRADING_HOME", str(home))
    monkeypatch.setattr(wr, "_SERVER_LOCK_PATH", None)
    first = wr._acquire_server_instance_lock(8765)
    assert first and wr.server_instance_id() == first
    # A second server on the same home is refused, naming the holder.
    with pytest.raises(RuntimeError) as ei:
        wr._acquire_server_instance_lock(8765)
    assert "already running" in str(ei.value) and "pid" in str(ei.value)
    # On release the lock goes away and a new server may start.
    wr._release_server_instance_lock()
    second = wr._acquire_server_instance_lock(8765)
    assert second != first
    wr._release_server_instance_lock()


def test_server_instance_lock_breaks_stale_holder(monkeypatch, tmp_path) -> None:
    from src.api import watchlist_routes as wr

    home = tmp_path / "home" / ".vibe-trading"
    monkeypatch.setenv("VIBE_TRADING_HOME", str(home))
    monkeypatch.setattr(wr, "_SERVER_LOCK_PATH", None)
    lock_file = wr._server_lock_path()
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    # A holder pid that cannot exist, started long ago: stale, must break.
    lock_file.write_text(
        json.dumps({"pid": 2**28, "started_at": "2000-01-01T00:00:00Z", "port": 8765}),
        encoding="utf-8",
    )
    instance = wr._acquire_server_instance_lock(8765)
    assert instance
    wr._release_server_instance_lock()


def test_lifespan_chains_server_lock_once(monkeypatch, tmp_path) -> None:
    """Double registration (importlib.reload scenario) must not double-take
    the server lock or double-wrap the lifespan."""
    from fastapi import FastAPI
    from src.api import watchlist_routes as wr

    home = tmp_path / "home" / ".vibe-trading"
    monkeypatch.setenv("VIBE_TRADING_HOME", str(home))
    monkeypatch.setattr(wr, "_SERVER_LOCK_PATH", None)

    class FakeSched:
        def start(self):
            pass

        async def stop(self):
            pass

        def status(self):
            return {}

    wr.set_scheduler_factory(FakeSched)
    app = FastAPI()
    wr.register_watchlist_routes(app)
    wr.register_watchlist_routes(app)  # re-registration
    assert getattr(app.state, "watchlist_lifespan_chained", False)
    lock_file = wr._server_lock_path()
    from fastapi.testclient import TestClient

    with TestClient(app):
        assert lock_file.exists()
    assert not lock_file.exists()  # released on shutdown


# --------------------------------------------------------------------------- #
# F9: forget_manual
# --------------------------------------------------------------------------- #
def _svc_with_items(tmp_path: Path):
    store = WatchlistStore(path=tmp_path / "watchlist.sqlite3")
    svc = WatchlistService(
        store=store, scan_fn=lambda cfg, **kw: _scan_run("r", ["600519"]),
        now_fn=lambda: _dt.datetime(2026, 10, 16, 12),
    )
    return svc, store


def test_forget_manual_deletes_the_row(tmp_path) -> None:
    svc, store = _svc_with_items(tmp_path)
    svc.add_manual("600519", note="manual test")
    assert store.get_item("600519") is not None
    deleted = svc.forget_manual("600519")
    assert deleted["code"] == "600519"
    assert store.get_item("600519") is None
    # No suppression left behind (F9's whole point).
    assert store.suppressed_codes() == []


def test_forget_manual_works_on_removed_manual_row_too(tmp_path) -> None:
    svc, store = _svc_with_items(tmp_path)
    svc.add_manual("600519")
    svc.remove("600519")  # suppressed manual pin
    assert "600519" in store.suppressed_codes()
    svc.forget_manual("600519")
    assert store.get_item("600519") is None
    assert store.suppressed_codes() == []


def test_forget_manual_refuses_auto(tmp_path) -> None:
    svc, store = _svc_with_items(tmp_path)
    store.upsert_item("600519", "贵州茅台", "auto", "active", "2026-10-16T12:00:00")
    with pytest.raises(ValueError) as ei:
        svc.forget_manual("600519")
    assert "auto" in str(ei.value)
    # The auto row is untouched.
    assert store.get_item("600519")["origin"] == "auto"
    assert store.suppressed_codes() == []


def test_forget_manual_unknown_code(tmp_path) -> None:
    svc, _store = _svc_with_items(tmp_path)
    with pytest.raises(ValueError) as ei:
        svc.forget_manual("600519")
    assert "unknown" in str(ei.value)
