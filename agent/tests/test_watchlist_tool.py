"""Offline tests for the watchlist agent tool (fake service)."""

from __future__ import annotations

import json
import time

import pytest

from src.tools import watchlist_tool as wt


class FakeService:
    def __init__(self) -> None:
        self.view = {
            "disclaimer": "d",
            "stale": {"is_stale": False},
            "latest_run": None,
            "items": [],
            "dropped": [],
            "requalified": [],
            "ai_demoted": [],
        }
        self.runs = [{"run_id": "r1", "status": "ok", "trade_date": "2026-10-12"}]
        self.added: list[str] = []
        self.removed: list[str] = []
        self.restored: list[str] = []
        self.scan_calls = 0

    def get_watchlist(self):
        return self.view

    def list_runs(self, limit=20):
        return self.runs[:limit]

    def get_run(self, run_id):
        return self.runs[0] if run_id == "r1" else None

    def add_manual(self, code, note=None):
        if code.startswith("92") or code.startswith("43"):
            raise ValueError(f"BSE code {code} is unsupported")
        if len(code) != 6 or not code.isdigit():
            raise ValueError(f"invalid A-share code: {code!r}")
        self.added.append(code)
        return {"code": code, "origin": "manual", "state": "active"}

    def remove(self, code):
        self.removed.append(code)
        return {"code": code, "state": "removed"}

    def restore(self, code):
        self.restored.append(code)
        return {"code": code, "state": "active"}

    def run_scan(self, *, trigger, config_overrides=None, progress=None, llm=None):
        self.scan_calls += 1
        time.sleep(0.05)
        return {"run_id": "rX", "status": "ok", "counts": {}}


@pytest.fixture()
def tool(monkeypatch: pytest.MonkeyPatch) -> object:
    svc = FakeService()
    tool = wt.WatchlistTool()
    monkeypatch.setattr(tool, "_service", lambda: svc)
    # reset the shared registry between tests
    wt._REGISTRY._running = False
    wt._REGISTRY._job = {}
    tool._svc = svc
    return tool


def test_list_action(tool) -> None:
    out = tool.execute(action="list")
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["data"]["disclaimer"] == "d"


def test_history_and_show_run(tool) -> None:
    payload = json.loads(tool.execute(action="history", limit=5))
    assert payload["ok"] is True and payload["data"][0]["run_id"] == "r1"
    payload = json.loads(tool.execute(action="show_run", run_id="r1"))
    assert payload["ok"] is True and payload["data"]["run_id"] == "r1"
    payload = json.loads(tool.execute(action="show_run", run_id="missing"))
    assert payload["ok"] is False and "not found" in payload["error"]


def test_add_remove_restore(tool) -> None:
    payload = json.loads(tool.execute(action="add", code="600519", note="x"))
    assert payload["ok"] is True and payload["data"]["origin"] == "manual"
    payload = json.loads(tool.execute(action="add", code="920002"))
    assert payload["ok"] is False and "BSE" in payload["error"]
    payload = json.loads(tool.execute(action="add", code="abc"))
    assert payload["ok"] is False and "invalid" in payload["error"]
    payload = json.loads(tool.execute(action="remove", code="600519"))
    assert payload["ok"] is True and tool._svc.removed == ["600519"]
    payload = json.loads(tool.execute(action="restore", code="600519"))
    assert payload["ok"] is True and tool._svc.restored == ["600519"]


def test_scan_starts_background_and_status(tool) -> None:
    payload = json.loads(tool.execute(action="scan"))
    assert payload["ok"] is True and payload["data"]["started"] is True
    # wait for the thread to finish
    deadline = time.time() + 5
    while wt._REGISTRY.running and time.time() < deadline:
        time.sleep(0.02)
    status = json.loads(tool.execute(action="scan_status"))
    assert status["ok"] is True
    assert status["data"]["running"] is False
    assert status["data"]["result"]["run_id"] == "rX"
    assert tool._svc.scan_calls == 1


def test_scan_conflict_returns_409_like(tool, monkeypatch: pytest.MonkeyPatch) -> None:
    # simulate a running scan
    wt._REGISTRY.start({"job_id": "j1", "started_at": "now"})
    payload = json.loads(tool.execute(action="scan"))
    assert payload["ok"] is False
    assert "already running" in payload["error"]
    assert payload["job"]["job_id"] == "j1"
    wt._REGISTRY.finish(result={})


def test_unknown_action(tool) -> None:
    payload = json.loads(tool.execute(action="nope"))
    assert payload["ok"] is False and "unknown action" in payload["error"]


def test_tool_registration_fields() -> None:
    assert wt.WatchlistTool.name == "watchlist"
    assert "list" in wt.WatchlistTool.parameters["properties"]["action"]["enum"]
    assert wt.WatchlistTool.is_readonly is False
