"""Authenticated Web API for the watchlist (CONTRACT section 11).

Routes are mounted by ``api_server.py`` via ``register_watchlist_routes(app)``,
following the ``portfolio_routes.py`` pattern: ``Depends(require_auth)``,
service built lazily, background scan thread with in-memory job state.

The scheduler is shared with the lifespan: ``get_scheduler()`` creates a
process-wide ``WatchlistScheduler`` bound to the same service factory so
``GET /schedule`` merges live scheduler status with service staleness.
"""

from __future__ import annotations
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.security import require_auth

logger = logging.getLogger(__name__)

_SCAN_STATE: dict[str, Any] = {
    "running": False,
    "job_id": None,
    "kind": None,  # "scan" | "review" — which job the shared state tracks
    "stage": None,
    "progress": 0.0,
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
}
_SCAN_LOCK = threading.Lock()


class WatchlistItemRequest(BaseModel):
    code: str = Field(min_length=6, max_length=6)
    note: str | None = None


class WatchlistScanRequest(BaseModel):
    overrides: dict[str, Any] | None = None


class WatchlistSettingsPatch(BaseModel):
    patch: dict[str, Any]


def _service():
    """Lazy service import — hermes owns service.py; never at module import.

    A HomeMismatch (F1) raised while constructing the service becomes an
    HTTP 503 with the pin's message verbatim, so EVERY route reports the
    misconfigured home identically instead of a bare 500.
    """
    from src.watchlist.service import WatchlistService

    try:
        return WatchlistService()
    except Exception as exc:  # noqa: BLE001
        raise _map_service_errors(exc) from exc


def _home_mismatch_detail(exc: BaseException) -> str:
    return str(exc) or "runtime home mismatch"


def _map_service_errors(exc: BaseException) -> HTTPException:
    """Map service-layer errors to HTTP statuses (F1/F2)."""
    if isinstance(exc, HTTPException):
        return exc  # already mapped (e.g. by _service)
    try:
        from src.watchlist.home import HomeMismatch
    except ImportError:
        HomeMismatch = ()  # pi's module not present yet: no special mapping
    if HomeMismatch and isinstance(exc, HomeMismatch):
        return HTTPException(status_code=503, detail=_home_mismatch_detail(exc))
    return HTTPException(status_code=500, detail=str(exc)[:300])


def _reset_scan_state(job_id: str | None = None) -> None:
    _SCAN_STATE.update(
        running=False,
        job_id=job_id,
        kind=None,
        stage=None,
        progress=0.0,
        started_at=None,
        finished_at=None,
        result=None,
        error=None,
    )


def _scan_worker(
    service: Any, job_id: str, overrides: dict | None, handle: Any
) -> None:
    """Background thread body: run the scan, feeding progress into state."""

    def progress(stage: str, fraction: float) -> None:
        _SCAN_STATE["stage"] = stage
        _SCAN_STATE["progress"] = max(0.0, min(1.0, float(fraction)))

    try:
        result = service.run_scan(
            trigger="api",
            config_overrides=overrides,
            progress=progress,
            handle=handle,
        )
        _SCAN_STATE["result"] = result
        _SCAN_STATE["error"] = None
    except Exception as exc:  # noqa: BLE001 — surfaced in job state
        logger.error("watchlist scan job %s failed", job_id, exc_info=True)
        _SCAN_STATE["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        _SCAN_STATE["running"] = False
        _SCAN_STATE["finished_at"] = datetime.now(timezone.utc).isoformat()


def _review_worker(
    service: Any, job_id: str, run_id: str | None, handle: Any
) -> None:
    """Background thread body: re-run the AI review (no market rescan)."""

    def progress(stage: str, fraction: float) -> None:
        _SCAN_STATE["stage"] = stage
        _SCAN_STATE["progress"] = max(0.0, min(1.0, float(fraction)))

    try:
        result = service.rerun_ai_review(
            run_id, trigger="api", progress=progress, handle=handle
        )
        _SCAN_STATE["result"] = result
        _SCAN_STATE["error"] = None
    except Exception as exc:  # noqa: BLE001 — surfaced in job state
        logger.error("watchlist review job %s failed", job_id, exc_info=True)
        _SCAN_STATE["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        _SCAN_STATE["running"] = False
        _SCAN_STATE["finished_at"] = datetime.now(timezone.utc).isoformat()


_SCHEDULER: Any = None
_SCHEDULER_FACTORY: Callable[[], Any] | None = None
_SERVER_LOCK_PATH: Any = None  # set once; the server-instance lock file
_INSTANCE_ID: str | None = None


def server_instance_id() -> str | None:
    """F3: this process's scheduler instance id (``None`` before setup)."""
    return _INSTANCE_ID


def _server_lock_path() -> Any:
    """`H/watchlist/server.lock` (F3) — refuses a second server per home."""
    global _SERVER_LOCK_PATH
    if _SERVER_LOCK_PATH is None:
        from src.config.paths import get_runtime_root

        _SERVER_LOCK_PATH = get_runtime_root() / "watchlist" / "server.lock"
    return _SERVER_LOCK_PATH


def _acquire_server_instance_lock(port: int | None = None) -> str:
    """Create-or-refuse the server instance lock; returns the instance id.

    A second server on the SAME home must not start (F3): two lifespans
    double-register the scheduler and two scans race (observed 15 Oct 20:24:
    scheduler_started logged twice, catch-up fired twice). Raises
    ``RuntimeError`` naming the current holder.
    """
    global _INSTANCE_ID
    path = _server_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "port": port,
        }
    )
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
    except FileExistsError:
        holder: dict[str, Any] = {}
        try:
            holder = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        # A dead holder's lock is stale: break it and retry once.
        pid = int(holder.get("pid") or 0)
        alive = False
        if pid > 0:
            try:
                from src.watchlist.service import _pid_alive

                alive = _pid_alive(pid)
            except Exception:  # noqa: BLE001
                alive = False
        if alive:
            raise RuntimeError(
                "another Vibe-Trading server is already running on this home "
                f"(pid {pid}, started {holder.get('started_at')}, "
                f"port {holder.get('port')}); refusing to start a second one"
            )
        try:
            path.unlink()
        except OSError:
            pass
        return _acquire_server_instance_lock(port)
    _INSTANCE_ID = uuid.uuid4().hex[:12]
    logger.info(
        "watchlist server instance lock acquired: instance=%s pid=%s port=%s",
        _INSTANCE_ID,
        os.getpid(),
        port,
    )
    return _INSTANCE_ID


def _release_server_instance_lock() -> None:
    path = _server_lock_path()
    try:
        holder = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if holder.get("pid") == os.getpid():
        try:
            path.unlink()
        except OSError:
            pass


def setup_watchlist_scheduler() -> Any:
    """Create the process-wide scheduler bound to a fresh shared service.

    F3 idempotence: a module-level singleton — repeated calls (double
    lifespan entry, ``importlib.reload`` of api_server) return the SAME
    scheduler instead of building a second one.
    """

    global _SCHEDULER
    if _SCHEDULER is not None:
        return _SCHEDULER

    def _factory() -> Any:
        from src.watchlist.service import WatchlistService

        return WatchlistService()

    try:
        scheduler = _watchlist_scheduler_module().WatchlistScheduler(
            _factory(),
            poll_interval_s=20.0,
        )
    except Exception as exc:  # noqa: BLE001
        # F1: a home mismatch (or any construction failure) must NOT kill the
        # server — start without a scheduler and let the routes answer 503
        # with the diagnosable message instead of a bare startup crash.
        logger.warning("watchlist scheduler not started: %s: %s", type(exc).__name__, exc)
        return None
    set_scheduler_factory(lambda: scheduler)
    _SCHEDULER = scheduler
    return scheduler


async def _start_watchlist_scheduler() -> None:
    """Lifespan helper: start the process-wide scheduler (never raises)."""
    sched = _scheduler()
    if sched is not None:
        try:
            sched.start()
        except Exception:  # noqa: BLE001 — startup must never fail the server
            logger.warning("watchlist scheduler start failed", exc_info=True)


async def _stop_watchlist_scheduler() -> None:
    """Lifespan helper: stop the scheduler (never raises)."""
    sched = _scheduler()
    if sched is not None:
        try:
            await sched.stop()
        except Exception:  # noqa: BLE001 — shutdown must never raise
            logger.warning("watchlist scheduler stop failed", exc_info=True)


def _watchlist_scheduler_module() -> Any:
    from src.watchlist import scheduler as module

    return module


def set_scheduler_factory(factory: Callable[[], Any]) -> None:
    """api_server lifespan wires this so routes share the live scheduler."""
    global _SCHEDULER_FACTORY
    _SCHEDULER_FACTORY = factory
    global _SCHEDULER
    _SCHEDULER = None


def _scheduler():
    global _SCHEDULER
    if _SCHEDULER is None and _SCHEDULER_FACTORY is not None:
        _SCHEDULER = _SCHEDULER_FACTORY()
    return _SCHEDULER


def register_watchlist_routes(app: FastAPI) -> None:
    """Mount the watchlist endpoints (CONTRACT section 11)."""

    if _SCHEDULER_FACTORY is None and _SCHEDULER is None:
        setup_watchlist_scheduler()

    # Chain onto the app's existing lifespan so api_server stays a thin
    # assembler (its <400-line meta-test): enter the original lifespan,
    # start the scheduler, and stop it again before the original exits.
    # Idempotent per app — a re-registration (e.g. importlib.reload of
    # api_server elsewhere in the suite) must not wrap the lifespan twice
    # and start/stop the scheduler on every nesting level.
    if not getattr(app.state, "watchlist_lifespan_chained", False):
        _original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def _watchlist_lifespan(scope):
            # F3: refuse a second server on the same home BEFORE anything
            # else starts (a second lifespan would double-fire the
            # scheduler). The lock is released on exit; a dead holder's
            # stale lock is broken by the next start.
            _acquire_server_instance_lock()
            try:
                async with _original_lifespan(scope) as state:
                    await _start_watchlist_scheduler()
                    try:
                        yield state
                    finally:
                        await _stop_watchlist_scheduler()
            finally:
                _release_server_instance_lock()

        app.router.lifespan_context = _watchlist_lifespan
        app.state.watchlist_lifespan_chained = True

    @app.get("/api/watchlist", dependencies=[Depends(require_auth)])
    def get_watchlist():
        try:
            return {"status": "ok", "watchlist": _service().get_watchlist()}
        except Exception as exc:  # noqa: BLE001
            raise _map_service_errors(exc) from exc

    @app.post("/api/watchlist/items", dependencies=[Depends(require_auth)])
    def add_item(body: WatchlistItemRequest):
        try:
            item = _service().add_manual(body.code, body.note)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "ok", "item": item}

    @app.delete("/api/watchlist/items/{code}", dependencies=[Depends(require_auth)])
    def remove_item(code: str):
        item = _service().remove(code)
        if item is None:
            raise HTTPException(status_code=404, detail=f"unknown code {code}")
        return {"status": "ok", "item": item}

    @app.post(
        "/api/watchlist/items/{code}/restore", dependencies=[Depends(require_auth)]
    )
    def restore_item(code: str):
        item = _service().restore(code)
        if item is None:
            raise HTTPException(status_code=404, detail=f"unknown code {code}")
        return {"status": "ok", "item": item}

    @app.delete(
        "/api/watchlist/items/{code}/forget", dependencies=[Depends(require_auth)]
    )
    def forget_item(code: str):
        """F9: delete a manual item row entirely (no suppression left)."""
        try:
            item = _service().forget_manual(code)
        except ValueError as exc:
            status = 404 if "unknown watchlist code" in str(exc) else 409
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise _map_service_errors(exc) from exc
        return {"status": "ok", "item": item}

    @app.post(
        "/api/watchlist/scan", status_code=202, dependencies=[Depends(require_auth)]
    )
    def start_scan(body: WatchlistScanRequest | None = None):
        service = _service()
        job_id = uuid.uuid4().hex
        overrides = body.overrides if body else None
        # FIX #7: the is_scanning check + job start are atomic under the lock
        # so concurrent POSTs get exactly one 202 and the rest 409.
        with _SCAN_LOCK:
            if service.is_scanning() or _SCAN_STATE["running"]:
                raise HTTPException(
                    status_code=409, detail="watchlist scan already running"
                )
            # F2: acquire the cross-process job lock SYNCHRONOUSLY — a 202
            # must mean "this job holds the lock", not "it will try later".
            # Held -> 409 with the holder info (kind/owner/pid/since).
            try:
                handle = service.acquire_job("scan", owner="api")
            except Exception as exc:  # noqa: BLE001
                raise _map_service_errors(exc) from exc
            _reset_scan_state(job_id)
            _SCAN_STATE["running"] = True
            _SCAN_STATE["kind"] = "scan"
            _SCAN_STATE["started_at"] = datetime.now(timezone.utc).isoformat()
        thread = threading.Thread(
            target=_scan_worker,
            args=(service, job_id, overrides, handle),
            name=f"watchlist-scan-{job_id}",
            daemon=True,
        )
        thread.start()
        return {"status": "accepted", "job_id": job_id}

    def _start_review_job(run_id: str | None):
        """Shared 202/409 machinery for the AI re-review endpoints."""
        service = _service()
        if run_id is not None:
            try:
                known = service.get_run(run_id) is not None
            except Exception as exc:  # noqa: BLE001
                raise _map_service_errors(exc) from exc
            if not known:
                raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        job_id = uuid.uuid4().hex
        with _SCAN_LOCK:
            if service.is_scanning() or _SCAN_STATE["running"]:
                raise HTTPException(
                    status_code=409,
                    detail="watchlist scan or review already running",
                )
            # F2: same synchronous lock acquisition as /scan.
            try:
                handle = service.acquire_job("review", owner="api")
            except Exception as exc:  # noqa: BLE001
                raise _map_service_errors(exc) from exc
            _reset_scan_state(job_id)
            _SCAN_STATE["running"] = True
            _SCAN_STATE["kind"] = "review"
            _SCAN_STATE["started_at"] = datetime.now(timezone.utc).isoformat()
        thread = threading.Thread(
            target=_review_worker,
            args=(service, job_id, run_id, handle),
            name=f"watchlist-review-{job_id}",
            daemon=True,
        )
        thread.start()
        return {"status": "accepted", "job_id": job_id}

    @app.post(
        "/api/watchlist/runs/{run_id}/review",
        status_code=202,
        dependencies=[Depends(require_auth)],
    )
    def start_review(run_id: str):
        return _start_review_job(run_id)

    @app.post(
        "/api/watchlist/review", status_code=202, dependencies=[Depends(require_auth)]
    )
    def start_review_latest():
        return _start_review_job(None)

    @app.get("/api/watchlist/scan/status", dependencies=[Depends(require_auth)])
    def scan_status():
        with _SCAN_LOCK:
            return {
                "running": bool(_SCAN_STATE["running"]),
                "job_id": _SCAN_STATE["job_id"],
                "kind": _SCAN_STATE["kind"],
                "stage": _SCAN_STATE["stage"],
                "progress": _SCAN_STATE["progress"],
                "started_at": _SCAN_STATE["started_at"],
                "finished_at": _SCAN_STATE["finished_at"],
                "result": _SCAN_STATE["result"],
                "error": _SCAN_STATE["error"],
            }

    @app.get("/api/watchlist/runs", dependencies=[Depends(require_auth)])
    def list_runs(limit: int = Query(default=20, ge=1, le=200)):
        return {"status": "ok", "runs": _service().list_runs(limit)}

    @app.get("/api/watchlist/runs/{run_id}", dependencies=[Depends(require_auth)])
    def get_run(run_id: str):
        run = _service().get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        return {"status": "ok", "run": run}

    @app.get("/api/watchlist/settings", dependencies=[Depends(require_auth)])
    def get_settings():
        return {"status": "ok", "settings": _service().get_settings()}

    @app.put("/api/watchlist/settings", dependencies=[Depends(require_auth)])
    def put_settings(body: WatchlistSettingsPatch):
        try:
            return {"status": "ok", "settings": _service().update_settings(body.patch)}
        except ValueError as exc:  # FIX #8: bad keys/types -> 400, not 500
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/watchlist/schedule", dependencies=[Depends(require_auth)])
    def get_schedule():
        sch = _scheduler()
        status = sch.status() if sch is not None else None
        if status is None:
            # No live scheduler (e.g. server started without the lifespan):
            # fall back to the service's stored settings + contract defaults
            # so `time` is never null.
            settings = _service().get_settings().get("schedule") or {}
            status = {
                "enabled": bool(settings.get("enabled", False)),
                "time": str(settings.get("time") or "18:00"),
                "timezone": str(settings.get("timezone") or "Asia/Shanghai"),
                "next_run_at": None,
                "last_fired_at": None,
                "last_result": None,
                "running": False,
            }
        try:
            stale = _service().staleness()
        except Exception:  # noqa: BLE001 — staleness must not break the route
            stale = {}
        from src.watchlist.scheduler import scheduler_log_path

        return {
            "status": "ok",
            **status,
            "stale": stale,
            "log_path": str(scheduler_log_path()),
        }
