"""RELFIX Bug A: the API job lock must not outlive the job (boss, 2026-10-24).

Live evidence: ``POST /api/watchlist/runs/<run>/review`` on 10-23 finished in
4 s, yet ``H\\watchlist\\scan.lock`` still named the live server pid 21 h later.
Because the route acquires the handle before answering 202 and the service only
releases handles it acquired itself (``own_handle``), nobody released it:
``is_scanning()`` stayed True, every later web scan/review got 409, and the
restart script refused.

These tests drive the REAL routes with the REAL ``CrossProcessScanLock`` /
``JobHandle`` under a tmp home, so ``scan.lock`` is a file we can watch; only the
job body is stubbed (no ``WatchlistService`` internals, no scheduler, no LLM).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import watchlist_routes  # noqa: E402
from src.config.paths import get_runtime_root  # noqa: E402
from src.watchlist.service import (  # noqa: E402
    CrossProcessScanLock,
    JobHandle,
    ScanAlreadyRunning,
    _pid_alive,
)
from src.watchlist import service as service_mod  # noqa: E402

JOB_TIMEOUT = 10.0


def _lock_file() -> Path:
    return get_runtime_root() / "watchlist" / "scan.lock"


class JobBodyStub:
    """The routes' service, with the job lock kept REAL.

    ``acquire_job``/``is_scanning`` use the production lock classes so the
    leak (or its absence) is a fact about the file system, not about a mock.
    ``run_scan``/``rerun_ai_review`` mirror the real service in the one respect
    that matters here: a handle handed IN is not released by them
    (``own_handle`` is False), so the caller must release it.
    """

    def __init__(self) -> None:
        self.handles: list[JobHandle] = []
        self.job_gate: threading.Event | None = None
        self.scan_error: str | None = None
        self.review_error: Exception | None = None
        self.review_calls: list[str | None] = []

    def acquire_job(self, kind: str, *, owner: str = "api") -> JobHandle:
        cross = CrossProcessScanLock()
        if not cross.acquire(kind=kind, owner=owner):
            raise ScanAlreadyRunning(f"a job is already running (kind={kind})")
        handle = JobHandle(cross, kind=kind, owner=owner)
        self.handles.append(handle)
        return handle

    def is_scanning(self) -> bool:
        return CrossProcessScanLock().is_locked_by_live_process()

    def _blocked(self) -> None:
        if self.job_gate is not None:
            assert self.job_gate.wait(JOB_TIMEOUT)

    def run_scan(self, *, trigger, config_overrides=None, progress=None, llm=None, handle=None):
        self._blocked()
        if progress:
            progress("ai", 0.9)
        if self.scan_error:
            raise RuntimeError(self.scan_error)
        return {"run_id": "r2", "status": "ok"}

    def rerun_ai_review(self, run_id=None, *, trigger, progress=None, llm=None, handle=None):
        self.review_calls.append(run_id)
        self._blocked()
        if progress:
            progress("ai_review", 0.6)
        if self.review_error:
            raise self.review_error
        return {"run_id": run_id or "r1", "status": "ok", "ai_status": "reviewed"}

    def get_run(self, run_id):
        return {"run_id": run_id, "status": "ok", "ai_status": "reviewed"}

    def get_settings(self):
        return {"scan": {}, "schedule": {}}

    def list_runs(self, limit=20):
        return []


def _traced(worker, done: threading.Event):
    """Wrap a worker so ``done`` is set only AFTER its ``finally`` has run."""

    def run(*args, **kwargs):
        try:
            worker(*args, **kwargs)
        finally:
            done.set()

    return run


@pytest.fixture()
def web(tmp_path, monkeypatch):
    """Real routes + real lock file, parked under a tmp home."""
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "runtime"))
    svc = JobBodyStub()
    monkeypatch.setattr(watchlist_routes, "_service", lambda: svc)
    # No scheduler: a test process must not start the 20 s polling thread.
    monkeypatch.setattr(watchlist_routes, "_SCHEDULER_FACTORY", lambda: None)
    monkeypatch.setattr(watchlist_routes, "_SCHEDULER", None)
    done = threading.Event()
    for name in ("_scan_worker", "_review_worker"):
        monkeypatch.setattr(
            watchlist_routes, name, _traced(getattr(watchlist_routes, name), done)
        )
    watchlist_routes._reset_scan_state()
    app = FastAPI()
    watchlist_routes.register_watchlist_routes(app)
    with TestClient(app) as client:
        yield client, svc, done
    watchlist_routes._reset_scan_state()


def _post(client, path, done) -> int:
    done.clear()
    return client.post(path, json={}).status_code


# --------------------------------------------------------------------------- #
# 1. A finished web job releases the lock
# --------------------------------------------------------------------------- #
def test_web_scan_releases_the_job_lock(web) -> None:
    client, svc, done = web
    svc.job_gate = threading.Event()
    assert _post(client, "/api/watchlist/scan", done) == 202
    # The 202 means "this job holds the lock" (F2), so it must be on disk now.
    assert _lock_file().exists()
    assert svc.is_scanning() is True
    svc.job_gate.set()
    assert done.wait(JOB_TIMEOUT)
    assert not _lock_file().exists()
    assert svc.handles[0].released is True
    assert svc.is_scanning() is False
    status = client.get("/api/watchlist/scan/status").json()
    assert status["running"] is False and status["error"] is None


def test_web_review_releases_the_job_lock(web) -> None:
    client, svc, done = web
    svc.job_gate = threading.Event()
    assert _post(client, "/api/watchlist/review", done) == 202
    assert _lock_file().exists()
    svc.job_gate.set()
    assert done.wait(JOB_TIMEOUT)
    assert not _lock_file().exists()
    assert svc.handles[0].released is True
    assert svc.is_scanning() is False


def test_web_review_of_a_named_run_releases_the_job_lock(web) -> None:
    client, svc, done = web
    svc.job_gate = threading.Event()
    done.clear()
    assert client.post("/api/watchlist/runs/r1/review").status_code == 202
    assert svc.review_calls == ["r1"]
    svc.job_gate.set()
    assert done.wait(JOB_TIMEOUT)
    assert not _lock_file().exists()
    assert svc.is_scanning() is False


# --------------------------------------------------------------------------- #
# 2. The release also happens when the service raises
# --------------------------------------------------------------------------- #
def test_scan_releases_the_lock_when_the_service_raises(web) -> None:
    client, svc, done = web
    svc.scan_error = "pipeline blew up"
    assert _post(client, "/api/watchlist/scan", done) == 202
    assert done.wait(JOB_TIMEOUT)
    assert not _lock_file().exists(), "an exception path must still release"
    assert svc.handles[0].released is True
    assert svc.is_scanning() is False
    status = client.get("/api/watchlist/scan/status").json()
    assert status["running"] is False
    assert "pipeline blew up" in status["error"]


def test_review_releases_the_lock_when_the_service_raises(web) -> None:
    client, svc, done = web
    svc.review_error = ScanAlreadyRunning("the handle is contested underneath us")
    assert _post(client, "/api/watchlist/review", done) == 202
    assert done.wait(JOB_TIMEOUT)
    assert not _lock_file().exists()
    assert svc.is_scanning() is False


# --------------------------------------------------------------------------- #
# 3. Back-to-back jobs: the second one is accepted, not 409
# --------------------------------------------------------------------------- #
def test_second_review_after_a_finished_review_is_202(web) -> None:
    client, svc, done = web
    assert _post(client, "/api/watchlist/review", done) == 202
    assert done.wait(JOB_TIMEOUT)
    # This is the live failure: before the fix the leftover lock made the route
    # answer 409 forever after the first web review.
    assert _post(client, "/api/watchlist/review", done) == 202
    assert done.wait(JOB_TIMEOUT)
    assert len(svc.handles) == 2
    assert all(h.released for h in svc.handles)
    assert not _lock_file().exists()


def test_scan_after_a_finished_review_is_202(web) -> None:
    client, svc, done = web
    assert _post(client, "/api/watchlist/review", done) == 202
    assert done.wait(JOB_TIMEOUT)
    assert _post(client, "/api/watchlist/scan", done) == 202
    assert done.wait(JOB_TIMEOUT)
    assert not _lock_file().exists()


def test_a_live_unreleased_handle_still_blocks_a_new_job(web) -> None:
    """The 202s above are not vacuous: a held lock really does answer 409."""
    client, svc, done = web
    handle = svc.acquire_job("review", owner="cli")
    try:
        assert _lock_file().exists()
        done.clear()
        assert client.post("/api/watchlist/review", json={}).status_code == 409
        assert client.post("/api/watchlist/scan", json={}).status_code == 409
    finally:
        handle.release()
    assert _post(client, "/api/watchlist/review", done) == 202
    assert done.wait(JOB_TIMEOUT)


# --------------------------------------------------------------------------- #
# 4. is_locked_by_live_process: age decides, like _break_stale_lock
# --------------------------------------------------------------------------- #
@pytest.fixture()
def lock_path(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "runtime"))
    path = _lock_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_lock(path: Path, pid: int) -> None:
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "started_at": "2026-10-23T21:04:50",
                "kind": "review",
                "owner": "api",
            }
        ),
        encoding="utf-8",
    )


def _age_lock(path: Path, extra_s: float = 60.0) -> None:
    old = time.time() - (service_mod._LOCK_STALE_S + extra_s)
    os.utime(path, (old, old))


def test_missing_lock_is_not_held(lock_path) -> None:
    assert CrossProcessScanLock().is_locked_by_live_process() is False


def test_young_lock_with_live_pid_is_held(lock_path) -> None:
    _write_lock(lock_path, os.getpid())
    assert _pid_alive(os.getpid()) is True
    assert CrossProcessScanLock().is_locked_by_live_process() is True


def test_aged_lock_with_live_pid_is_not_held(lock_path) -> None:
    """The live case: our own pid holds it, but it is past the stale limit."""
    _write_lock(lock_path, os.getpid())
    _age_lock(lock_path)
    assert _pid_alive(os.getpid()) is True
    assert CrossProcessScanLock().is_locked_by_live_process() is False


def test_young_lock_just_inside_the_limit_is_still_held(lock_path) -> None:
    _write_lock(lock_path, os.getpid())
    old = time.time() - (service_mod._LOCK_STALE_S - 120.0)
    os.utime(lock_path, (old, old))
    assert CrossProcessScanLock().is_locked_by_live_process() is True


def test_lock_with_dead_pid_is_not_held(lock_path) -> None:
    dead = 4_000_000
    if _pid_alive(dead):
        pytest.skip(f"pid {dead} exists on this machine")
    _write_lock(lock_path, dead)
    assert CrossProcessScanLock().is_locked_by_live_process() is False


def test_young_unparsable_lock_is_held(lock_path) -> None:
    """Kept rule: garbage we cannot judge is treated as held while it is young."""
    lock_path.write_text("this is not json", encoding="utf-8")
    assert CrossProcessScanLock().is_locked_by_live_process() is True


def test_aged_unparsable_lock_is_not_held(lock_path) -> None:
    """Age outranks the unparsable rule — otherwise the leak would survive it."""
    lock_path.write_text("this is not json", encoding="utf-8")
    _age_lock(lock_path)
    assert CrossProcessScanLock().is_locked_by_live_process() is False


def test_age_judgement_matches_the_acquire_path(lock_path) -> None:
    """is_locked_by_live_process() and acquire() must agree on an aged lock."""
    _write_lock(lock_path, os.getpid())
    _age_lock(lock_path)
    assert CrossProcessScanLock().is_locked_by_live_process() is False
    # acquire() breaks it and takes the job — the very thing the pre-check used
    # to refuse before acquire could run.
    cross = CrossProcessScanLock()
    assert cross.acquire(kind="scan", owner="api") is True
    assert _lock_file().exists()
    cross.release()


def test_young_live_lock_blocks_the_acquire_path(lock_path) -> None:
    _write_lock(lock_path, os.getpid())
    assert CrossProcessScanLock().is_locked_by_live_process() is True
    assert CrossProcessScanLock().acquire(kind="scan", owner="api") is False


def test_release_is_idempotent(lock_path) -> None:
    cross = CrossProcessScanLock()
    assert cross.acquire(kind="scan", owner="api") is True
    handle = JobHandle(cross, kind="scan", owner="api")
    handle.release()
    handle.release()
    assert not _lock_file().exists()


def test_lock_file_is_gone_after_a_released_handle_web_job(lock_path) -> None:
    """A handle released by its owner leaves nothing behind for is_scanning()."""
    handle = CrossProcessScanLock()
    assert handle.acquire(kind="review", owner="api") is True
    JobHandle(handle, kind="review", owner="api").release()
    assert CrossProcessScanLock().is_locked_by_live_process() is False
