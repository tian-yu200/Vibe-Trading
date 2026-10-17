"""``watchlist`` agent tool: list / history / show_run / add / remove / restore /
scan / scan_status. JSON envelope ``{"ok": ...}`` like every other tool; no
order placement anywhere (read-only market data + user list management).
"""

from __future__ import annotations

import json
import threading
from typing import Any

from src.agent.tools import BaseTool


def _error(message: str) -> str:
    """Error envelope; the caller-facing contract for every failure."""
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


class _ScanRegistry:
    """Process-wide registry of the running background scan (per tool)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._job: dict[str, Any] = {}

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self, job: dict[str, Any]) -> bool:
        """Claim the slot; False when a scan is already running."""
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._job = job
            return True

    def finish(
        self, result: dict[str, Any] | None = None, error: str | None = None
    ) -> None:
        with self._lock:
            self._running = False
            self._job["finished_at"] = _now_iso()
            if error:
                self._job["error"] = error
            else:
                self._job["result"] = result


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now().isoformat(timespec="seconds")


_REGISTRY = _ScanRegistry()


class WatchlistTool(BaseTool):
    """Read and manage the A-share watchlist; run scans in the background."""

    name = "watchlist"
    description = (
        "View and manage the A-share watchlist produced by the market scanner: "
        "list current items with scores, browse run history, show one run's "
        "detail, manually add/remove/restore codes, or start a full scan "
        "(background; check scan_status). Actions: list, history, show_run, "
        "add, remove, restore, scan, scan_status. "
        'Example: {"action": "list"}.'
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list",
                    "history",
                    "show_run",
                    "add",
                    "remove",
                    "restore",
                    "scan",
                    "scan_status",
                ],
                "description": "Which watchlist operation to perform.",
            },
            "code": {
                "type": "string",
                "description": "6-digit A-share code (add/remove/restore).",
            },
            "note": {
                "type": "string",
                "description": "Optional note for add.",
            },
            "run_id": {
                "type": "string",
                "description": "Run identifier (show_run).",
            },
            "limit": {
                "type": "integer",
                "description": "History limit (default 20).",
                "default": 20,
            },
            "no_ai": {
                "type": "boolean",
                "description": "Skip the AI review in the scan (default false).",
                "default": False,
            },
            "top_n": {
                "type": "integer",
                "description": "Override the scan top_n.",
            },
        },
        "required": ["action"],
    }
    repeatable = True
    is_readonly = False  # add/remove/restore/scan mutate the watchlist

    def execute(self, **kwargs: Any) -> str:
        """Dispatch one action; returns a JSON envelope string."""
        action = kwargs.get("action")
        try:
            return self._dispatch(action, kwargs)
        except Exception as exc:  # noqa: BLE001 — envelope contract
            return _error(f"{type(exc).__name__}: {exc}")

    def _service(self):
        from src.watchlist.service import WatchlistService

        return WatchlistService()

    def _dispatch(self, action: Any, kwargs: dict[str, Any]) -> str:
        if not isinstance(action, str) or action not in {
            "list",
            "history",
            "show_run",
            "add",
            "remove",
            "restore",
            "scan",
            "scan_status",
        }:
            return _error(f"unknown action: {action!r}")
        svc = self._service()

        if action == "list":
            return json.dumps(
                {"ok": True, "data": svc.get_watchlist()},
                ensure_ascii=False,
                default=str,
            )

        if action == "history":
            limit = kwargs.get("limit") or 20
            return json.dumps(
                {"ok": True, "data": svc.list_runs(limit=int(limit))},
                ensure_ascii=False,
                default=str,
            )

        if action == "show_run":
            run_id = kwargs.get("run_id")
            if not run_id:
                return _error("run_id is required for show_run")
            run = svc.get_run(str(run_id))
            if run is None:
                return _error(f"run not found: {run_id}")
            return json.dumps(
                {"ok": True, "data": run}, ensure_ascii=False, default=str
            )

        if action == "add":
            return self._add(svc, kwargs)

        if action == "remove":
            code = kwargs.get("code")
            if not code:
                return _error("code is required for remove")
            item = svc.remove(str(code))
            return json.dumps({"ok": True, "data": item}, ensure_ascii=False)

        if action == "restore":
            code = kwargs.get("code")
            if not code:
                return _error("code is required for restore")
            item = svc.restore(str(code))
            return json.dumps({"ok": True, "data": item}, ensure_ascii=False)

        if action == "scan":
            return self._start_scan(svc, kwargs)

        # scan_status
        with _REGISTRY._lock:
            job = dict(_REGISTRY._job)
            running = _REGISTRY._running  # private read: `running` re-acquires the lock
        return json.dumps(
            {"ok": True, "data": {"running": running, **job}},
            ensure_ascii=False,
            default=str,
        )

    def _add(self, svc, kwargs: dict[str, Any]) -> str:
        code = kwargs.get("code")
        if not code:
            return _error("code is required for add")
        try:
            item = svc.add_manual(str(code), note=kwargs.get("note"))
        except ValueError as exc:
            return _error(str(exc))
        return json.dumps({"ok": True, "data": item}, ensure_ascii=False)

    def _start_scan(self, svc, kwargs: dict[str, Any]) -> str:
        if _REGISTRY.running:
            return json.dumps(
                {
                    "ok": False,
                    "error": "a scan is already running",
                    "job": dict(_REGISTRY._job),
                },
                ensure_ascii=False,
                default=str,
            )
        overrides: dict[str, Any] = {}
        if kwargs.get("top_n"):
            overrides["top_n"] = int(kwargs["top_n"])

        def job() -> None:
            try:
                result = svc.run_scan(
                    trigger="agent",
                    config_overrides=overrides or None,
                    llm=None,
                )
                _REGISTRY.finish(result=result)
            except Exception as exc:  # noqa: BLE001 — surfaced via scan_status
                _REGISTRY.finish(error=f"{type(exc).__name__}: {exc}"[:300])

        job_id = f"scan-{_now_iso().replace(':', '').replace('-', '')}"
        claimed = _REGISTRY.start({"job_id": job_id, "started_at": _now_iso()})
        if not claimed:
            return json.dumps(
                {
                    "ok": False,
                    "error": "a scan is already running",
                    "job": dict(_REGISTRY._job),
                },
                ensure_ascii=False,
                default=str,
            )
        thread = threading.Thread(target=job, daemon=True, name=f"watchlist-{job_id}")
        thread.start()
        return json.dumps(
            {
                "ok": True,
                "data": {
                    "job_id": job_id,
                    "started": True,
                    "status_url": "watchlist scan_status",
                },
            },
            ensure_ascii=False,
        )


TOOL = WatchlistTool
