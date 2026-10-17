"""API tests for the watchlist AI re-review endpoints (CONTRACT section 11).

``POST /api/watchlist/runs/{run_id}/review`` and ``POST /api/watchlist/review``
share the scan job machinery: 202 + job_id, 409 while a scan or review runs,
status via ``GET /api/watchlist/scan/status`` with ``kind == "review"``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import watchlist_routes  # noqa: E402


class ScanAlreadyRunning(RuntimeError):
    pass


class FakeService:
    def __init__(self):
        self.runs = [
            {
                "run_id": "r1",
                "trade_date": "2026-10-12",
                "status": "ok",
                "ai_status": "partial",
            }
        ]
        self.review_delay = 0.0
        self.review_error: Exception | None = None
        self.review_calls: list[str | None] = []

    # -- scan ------------------------------------------------------------
    def run_scan(self, *, trigger, config_overrides=None, progress=None, llm=None, handle=None):
        if progress:
            progress("sources", 0.5)
        return {"run_id": "r2", "status": "ok"}

    def is_scanning(self):
        return False

    def acquire_job(self, kind, *, owner="api"):
        """F2: fake job lock — never contested in the API tests."""
        return {"kind": kind, "owner": owner}

    # -- review ----------------------------------------------------------
    def rerun_ai_review(self, run_id=None, *, trigger, progress=None, llm=None, handle=None):
        self.review_calls.append(run_id)
        if self.review_delay:
            time.sleep(self.review_delay)
        if progress:
            progress("news", 0.3)
            progress("ai_review", 0.6)
        if self.review_error:
            raise self.review_error
        return {
            "run_id": run_id or "r1",
            "status": "ok",
            "ai_status": "reviewed",
            "ai_note": None,
            "duration_s": 1.2,
            "review_revision": 1,
            "reviewed_at": "2026-10-12T19:00:00",
        }

    def get_run(self, run_id):
        for r in self.runs:
            if r["run_id"] == run_id:
                return r
        return None

    # -- misc ------------------------------------------------------------
    def get_watchlist(self):
        return {"items": [], "dropped": [], "pinned": []}

    def list_runs(self, limit=20):
        return self.runs[:limit]

    def get_settings(self):
        return {"scan": {}, "schedule": {}}

    def update_settings(self, patch):
        return self.get_settings()

    def staleness(self):
        return {"is_stale": False}


@pytest.fixture()
def client():
    svc = FakeService()
    app = FastAPI()
    watchlist_routes._service = lambda: svc  # type: ignore[assignment]
    watchlist_routes.register_watchlist_routes(app)
    watchlist_routes._reset_scan_state()
    with TestClient(app) as c:
        yield c, svc
    watchlist_routes._reset_scan_state()


def _wait_done(c: TestClient, deadline_s: float = 5.0) -> dict:
    deadline = time.time() + deadline_s
    st = None
    while time.time() < deadline:
        st = c.get("/api/watchlist/scan/status").json()
        if not st["running"]:
            return st
        time.sleep(0.05)
    return st


def test_review_latest_202_progress_and_result(client):
    c, svc = client
    svc.review_delay = 0.3
    r = c.post("/api/watchlist/review")
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert job_id

    # while running: shared status shows kind=review + 409 for a new scan
    st = c.get("/api/watchlist/scan/status").json()
    assert st["running"] is True and st["kind"] == "review"
    assert c.post("/api/watchlist/scan").status_code == 409
    assert c.post("/api/watchlist/review").status_code == 409
    assert c.post("/api/watchlist/runs/r1/review").status_code == 409

    st = _wait_done(c)
    assert st["job_id"] == job_id
    assert st["kind"] == "review"
    assert st["error"] is None
    assert st["result"]["ai_status"] == "reviewed"
    assert st["result"]["review_revision"] == 1
    assert svc.review_calls == [None]


def test_review_by_run_id_202(client):
    c, svc = client
    r = c.post("/api/watchlist/runs/r1/review")
    assert r.status_code == 202
    st = _wait_done(c)
    assert st["error"] is None
    assert st["result"]["run_id"] == "r1"
    assert svc.review_calls == ["r1"]


def test_review_unknown_run_404(client):
    c, svc = client
    r = c.post("/api/watchlist/runs/nope/review")
    assert r.status_code == 404
    assert svc.review_calls == []  # never started


def test_review_error_surfaces_in_status(client):
    c, svc = client
    svc.review_error = ValueError("run r1 is not the latest successful run")
    r = c.post("/api/watchlist/review")
    assert r.status_code == 202
    st = _wait_done(c)
    assert st["error"] and "ValueError" in st["error"]
    assert st["result"] is None


def test_scan_202_while_idle_after_review(client):
    c, svc = client
    assert c.post("/api/watchlist/review").status_code == 202
    _wait_done(c)
    # a fresh scan is allowed again (kind switches back to scan)
    r = c.post("/api/watchlist/scan")
    assert r.status_code == 202
    st = _wait_done(c)
    assert st["kind"] == "scan"
    assert st["result"]["run_id"] == "r2"


def test_api_server_registers_review_routes():
    """api_server must expose the review endpoints (wiring check)."""
    import api_server

    paths = {getattr(r, "path", None) for r in api_server.app.routes}
    assert "/api/watchlist/review" in paths
    assert "/api/watchlist/runs/{run_id}/review" in paths
