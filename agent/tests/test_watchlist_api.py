"""API tests for the watchlist routes (CONTRACT section 11).

Minimal FastAPI app + ``register_watchlist_routes`` with a fake service (the
real WatchlistService is hermes' file and must not be exercised here).
Loopback-free: no auth dependency is enforced on the bare test app.
"""

from __future__ import annotations
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import watchlist_routes  # noqa: E402


class ScanAlreadyRunning(RuntimeError):
    pass


class FakeHandle:
    """Stand-in for ``JobHandle``.

    The route acquires the job lock before answering 202 and the background
    worker releases it, so the fake has to carry a ``release()``; a bare dict
    (what this used to return) described the pre-fix contract where nobody
    released an API-acquired handle.
    """

    def __init__(self, kind: str, owner: str) -> None:
        self.kind = kind
        self.owner = owner
        self.released = False

    def check(self) -> None:
        pass

    def release(self) -> None:
        self.released = True


class FakeService:
    def __init__(self):
        self.items = []
        self.runs = [
            {
                "run_id": "r1",
                "trade_date": "2026-10-12",
                "status": "completed",
                "counts": {"new": 3},
            }
        ]
        self.settings = {
            "scan": {"top_n": 10},
            "schedule": {
                "enabled": True,
                "time": "18:00",
                "timezone": "Asia/Shanghai",
                "catch_up": True,
            },
        }
        self.scan_error = None
        self.scan_delay = 0.0
        self.scan_gate = None
        self.handles: list[FakeHandle] = []
        self.watchlist = {"picks": [], "dropped": [], "pinned": []}
        self.stale = {"is_stale": False}

    # -- scan ------------------------------------------------------------
    def acquire_job(self, kind, *, owner="api"):
        """F2: fake job lock — never contested in the API tests."""
        handle = FakeHandle(kind, owner)
        self.handles.append(handle)
        return handle

    def run_scan(self, *, trigger, config_overrides=None, progress=None, llm=None, handle=None):
        if self.scan_gate is not None:
            self.scan_gate.wait()
        if self.scan_delay:
            time.sleep(self.scan_delay)
        if progress:
            progress("sources", 0.5)
            progress("ai", 0.9)
        if self.scan_error:
            raise RuntimeError(self.scan_error)
        return {"run_id": "r2", "status": "completed"}

    def is_scanning(self):
        return False

    def get_watchlist(self):
        return self.watchlist

    def list_runs(self, limit=20):
        return self.runs[:limit]

    def get_settings(self):
        return self.settings

    def update_settings(self, patch):
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(self.settings.get(k), dict):
                self.settings[k].update(v)
            else:
                self.settings[k] = v
        return self.settings

    def staleness(self):
        return self.stale

    def add_manual(self, code, note=None):
        if len(code) != 6 or not code.isdigit() or code.startswith(("4", "8", "9")):
            raise ValueError(f"unsupported code {code}")
        item = {"code": code, "state": "pinned", "note": note}
        self.items.append(item)
        return item

    def remove(self, code):
        item = {"code": code, "state": "removed"}
        self.items.append(item)
        return item

    def restore(self, code):
        item = {"code": code, "state": "watch"}
        self.items.append(item)
        return item

    def get_run(self, run_id):
        for r in self.runs:
            if r["run_id"] == run_id:
                return r
        return None


@pytest.fixture()
def client(monkeypatch):
    svc = FakeService()
    app = FastAPI()
    # Patch _service to return our fake. Restored per test: a bare assignment
    # used to leak the fake into every later module's view of the routes.
    monkeypatch.setattr(watchlist_routes, "_service", lambda: svc)
    watchlist_routes.register_watchlist_routes(app)
    # reset shared scan state between tests
    watchlist_routes._reset_scan_state()
    with TestClient(app) as c:
        yield c, svc
    watchlist_routes._reset_scan_state()


def test_get_watchlist(client):
    c, svc = client
    svc.watchlist = {"picks": [{"code": "600519"}], "dropped": [], "pinned": []}
    r = c.get("/api/watchlist")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["watchlist"]["picks"][0]["code"] == "600519"


def test_add_remove_restore(client):
    c, svc = client
    r = c.post("/api/watchlist/items", json={"code": "600519", "note": "manual"})
    assert r.status_code == 200 and r.json()["item"]["state"] == "pinned"
    r = c.post("/api/watchlist/items", json={"code": "920001"})
    assert r.status_code == 400  # BSE rejected
    r = c.delete("/api/watchlist/items/600519")
    assert r.status_code == 200 and r.json()["item"]["state"] == "removed"
    r = c.post("/api/watchlist/items/600519/restore")
    assert r.status_code == 200 and r.json()["item"]["state"] == "watch"


def test_scan_202_then_409_and_progress(client):
    c, svc = client
    svc.scan_delay = 0.4
    r = c.post("/api/watchlist/scan", json={"overrides": {"top_n": 5}})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    # while running -> 409
    r2 = c.post("/api/watchlist/scan")
    assert r2.status_code == 409
    # poll status: running or finished with progress + result
    deadline = time.time() + 5
    st = None
    while time.time() < deadline:
        st = c.get("/api/watchlist/scan/status").json()
        if not st["running"]:
            break
        time.sleep(0.05)
    assert st["job_id"] == job_id
    assert st["result"]["run_id"] == "r2"
    assert st["error"] is None
    # now a fresh scan is allowed (202 again)
    assert c.post("/api/watchlist/scan").status_code == 202
    for _ in range(100):
        st = c.get("/api/watchlist/scan/status").json()
        if not st["running"]:
            break
        time.sleep(0.05)
    assert st["running"] is False


def test_scan_2024_flow(client):
    c, svc = client
    svc.scan_delay = 0.3
    r = c.post("/api/watchlist/scan", json={"overrides": {"top_n": 5}})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    # immediately: running
    st = c.get("/api/watchlist/scan/status").json()
    assert st["running"] is True and st["job_id"] == job_id
    # second start while running -> 409
    r2 = c.post("/api/watchlist/scan", json={})
    assert r2.status_code == 409
    # wait for completion
    for _ in range(100):
        st = c.get("/api/watchlist/scan/status").json()
        if not st["running"]:
            break
        time.sleep(0.05)
    assert st["running"] is False
    assert st["result"]["run_id"] == "r2"
    assert st["progress"] == 0.9 or st["finished_at"]


def test_scan_flow_202_409_and_status_progress(client, monkeypatch):
    c, svc = client
    svc.scan_gate = threading.Event()
    worker_finished = threading.Event()
    original_worker = watchlist_routes._scan_worker

    def wrapped_worker(*args, **kwargs):
        try:
            original_worker(*args, **kwargs)
        finally:
            worker_finished.set()

    monkeypatch.setattr(watchlist_routes, "_scan_worker", wrapped_worker)

    r = c.post("/api/watchlist/scan", json={"overrides": {"top_n": 5}})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert job_id
    # immediately: 409 while running
    r2 = c.post("/api/watchlist/scan", json={})
    assert r2.status_code == 409

    svc.scan_gate.set()
    assert worker_finished.wait()
    st = c.get("/api/watchlist/scan/status").json()
    assert st["running"] is False
    assert st["job_id"] == job_id
    assert st["result"]["run_id"] == "r2"
    assert st["error"] is None

    worker_finished.clear()
    assert c.post("/api/watchlist/scan").status_code == 202
    assert worker_finished.wait()
    assert c.get("/api/watchlist/scan/status").json()["running"] is False


def test_scan_failure_reported_in_status(client):
    c, svc = client
    svc.scan_error = "boom"
    r = c.post("/api/watchlist/scan", json={})
    assert r.status_code == 202
    for _ in range(50):
        st = c.get("/api/watchlist/scan/status").json()
        if not st["running"]:
            break
        time.sleep(0.05)
    assert st["error"] and "RuntimeError" in st["error"]
    assert st["result"] is None


def test_runs_and_run_detail(client):
    c, svc = client
    r = c.get("/api/watchlist/runs")
    assert r.status_code == 200 and r.json()["runs"][0]["run_id"] == "r1"
    r = c.get("/api/watchlist/runs/r1")
    assert r.status_code == 200 and r.json()["run"]["run_id"] == "r1"
    r = c.get("/api/watchlist/runs/nope")
    assert r.status_code == 404


def test_settings_round_trip(client):
    c, svc = client
    r = c.get("/api/watchlist/settings")
    assert r.status_code == 200
    assert r.json()["settings"]["schedule"]["time"] == "18:00"
    r = c.put(
        "/api/watchlist/settings", json={"patch": {"schedule": {"time": "19:30"}}}
    )
    assert r.status_code == 200
    assert svc.settings["schedule"]["time"] == "19:30"


def test_schedule_route_merges_status_and_staleness(client):
    c, svc = client
    svc.stale = {
        "is_stale": True,
        "latest_trading_day": "2026-10-12",
        "last_scan_trade_date": "2026-10-11",
        "last_scan_at": "x",
    }

    class FakeSched:
        def status(self):
            return {
                "enabled": True,
                "time": "18:00",
                "timezone": "Asia/Shanghai",
                "next_run_at": None,
                "last_fired_at": None,
                "last_result": None,
                "running": False,
            }

    watchlist_routes.set_scheduler_factory(FakeSched)
    r = c.get("/api/watchlist/schedule")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["stale"]["is_stale"] is True


def test_api_server_registers_watchlist_routes():
    """api_server must expose the /api/watchlist routes (wiring check)."""
    import api_server

    paths = {getattr(r, "path", None) for r in api_server.app.routes}
    for p in (
        "/api/watchlist",
        "/api/watchlist/items",
        "/api/watchlist/scan",
        "/api/watchlist/scan/status",
        "/api/watchlist/schedule",
    ):
        assert p in paths, f"{p} not registered"


def test_schedule_route_fresh_store_shows_default_time(client):
    """Regression: fresh store (no settings) -> /schedule shows time 18:00,
    never null (CONTRACT section 10 defaults)."""
    c, svc = client
    # no scheduler factory set -> route falls back to service settings
    watchlist_routes.set_scheduler_factory(None)
    svc.settings = {"scan": {}, "schedule": {}}  # nothing stored
    r = c.get("/api/watchlist/schedule")
    assert r.status_code == 200
    body = r.json()
    assert body["time"] == "18:00"
    assert body["timezone"] == "Asia/Shanghai"
    assert body["enabled"] is False


@pytest.fixture()
def lifespan_spy():
    """Install a spy scheduler + preflight hook on api_server; restore both.

    Saves and restores the previous global state instead of resetting it to
    a default: earlier tests in a full-suite run may have left a scheduler
    factory behind, and forcing it to None would leak a different breakage
    into the next file. ``importlib.reload(api_server)`` (test_principal_wiring)
    rebuilds ``app`` mid-suite, so the spy is installed on whatever
    ``api_server`` currently is, not on a captured reference.
    """
    import api_server

    events: list[str] = []

    class SpySched:
        def start(self):
            events.append("scheduler-started")

        async def stop(self):
            events.append("scheduler-stopped")

        def status(self):
            return {
                "enabled": True,
                "time": "18:00",
                "timezone": "Asia/Shanghai",
                "next_run_at": None,
                "last_fired_at": None,
                "last_result": None,
                "running": False,
            }

    prev_preflight = api_server._run_startup_preflight
    prev_factory = watchlist_routes._SCHEDULER_FACTORY
    prev_scheduler = watchlist_routes._SCHEDULER

    async def spy_preflight():
        events.append("original-lifespan-entered")
        await prev_preflight()

    api_server._run_startup_preflight = spy_preflight
    watchlist_routes.set_scheduler_factory(SpySched)
    try:
        yield events
    finally:
        api_server._run_startup_preflight = prev_preflight
        watchlist_routes._SCHEDULER_FACTORY = prev_factory
        watchlist_routes._SCHEDULER = prev_scheduler


def test_api_server_lifespan_starts_and_stops_watchlist_scheduler(lifespan_spy):
    """TestClient(api_server.app) as a context manager must run the original
    lifespan AND start the watchlist scheduler on enter / stop it on exit."""
    import api_server

    events = lifespan_spy
    with TestClient(api_server.app) as c:
        # original lifespan ran first, then the scheduler started
        assert events[:2] == ["original-lifespan-entered", "scheduler-started"], events
        r = c.get("/api/watchlist")
        assert r.status_code == 200, r.text
        r = c.get("/api/watchlist/schedule")
        assert r.status_code == 200, r.text
    # shutdown: scheduler stopped before the context exits
    assert events[-1] == "scheduler-stopped", events
