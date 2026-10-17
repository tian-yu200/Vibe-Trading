"""Watchlist scanning + persistence service (CONTRACT section 10)."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
import time
from pathlib import Path
import uuid

from typing import Any, Callable

from src.market_scanner.config import ScanConfig
from src.watchlist.store import WatchlistStore

logger = logging.getLogger(__name__)

_DISCLAIMER = "Research only, not investment advice."
_SCHEDULE_DEFAULTS = {
    "enabled": False,
    "time": "18:00",
    "timezone": "Asia/Shanghai",
    "catch_up": True,
}
_SCHEDULE_ENV = "VIBE_TRADING_WATCHLIST_SCHEDULE"


class ScanAlreadyRunning(RuntimeError):
    """Raised by :meth:`WatchlistService.run_scan` while a scan is in flight."""


class JobLockLost(RuntimeError):
    """The cross-process job lock vanished or was taken over mid-run.

    F2: the run must abort with this reason stored as ``failure_reason`` —
    a deleted/overwritten lock file means another writer may believe it owns
    the job, so continuing would duplicate work (observed 15 Oct: two scans
    ran concurrently and both hammered Tencent).
    """


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _coverage_failure_reason(coverage: dict, min_coverage: float) -> str | None:
    """Mirror of the pipeline's coverage guard (kept dependency-free).

    trend = trend_scored / passed_exclusions; fundamentals =
    (passed_exclusions - insufficient_data:fundamentals) / passed_exclusions.
    Returns the standard failure reason when either is below the floor.
    """
    try:
        passed = int(coverage.get("passed_exclusions") or 0)
        if passed <= 0:
            return None
        trend_cov = float(coverage.get("trend_scored") or 0) / passed
        insuff_fund = int(
            (coverage.get("insufficient_by_key") or {}).get(
                "insufficient_data:fundamentals", 0
            )
        )
        fund_cov = (passed - insuff_fund) / passed
    except (TypeError, ValueError):
        return None
    if trend_cov < min_coverage:
        return (
            f"insufficient coverage (trend {trend_cov * 100:.1f}% "
            f"< {min_coverage * 100:g}%)"
        )
    if fund_cov < min_coverage:
        return (
            f"insufficient coverage (fundamentals {fund_cov * 100:.1f}% "
            f"< {min_coverage * 100:g}%)"
        )
    return None


# Cross-process single-flight (CONTRACT section 10): an O_CREAT|O_EXCL lock
# file under get_runtime_root()/watchlist/ holding pid + started_at. Stale
# when the owning pid is gone or the file is older than 2 h.
_LOCK_STALE_S = 2 * 60 * 60


def _lock_path() -> Path:
    from src.config.paths import get_runtime_root

    return get_runtime_root() / "watchlist" / "scan.lock"


def _pid_alive(pid: int) -> bool:
    """Windows-safe liveness check — NEVER sends any signal.

    os.kill(pid, 0) is POSIX-only semantics: on Windows signal 0 maps to
    CTRL_C_EVENT (sends Ctrl+C to a process group!), so it must not be used
    there. Windows: ctypes OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) +
    GetExitCodeProcess == STILL_ACTIVE(259). POSIX keeps os.kill(pid, 0).
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid)
        )
        if not handle:
            return False  # no such process (or no access -> treat as dead)
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by another user
        return True


def _read_lock_file(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _break_stale_lock(path: Path) -> bool:
    """Remove the lock file when its owner is gone or it is too old."""
    info = _read_lock_file(path)
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    stale = age > _LOCK_STALE_S or (
        info is not None and not _pid_alive(int(info.get("pid", 0)))
    )
    if stale:
        try:
            path.unlink()
        except OSError:
            pass
        return True
    return False


class CrossProcessScanLock:
    """O_CREAT|O_EXCL lock file; ScanAlreadyRunning semantics across processes."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _lock_path()
        self._fd: int | None = None
        self._token: dict | None = None

    def acquire(self, *, kind: str = "scan", owner: str = "") -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._token = {
            "pid": os.getpid(),
            "started_at": _now_iso(),
            "kind": kind,
            "owner": owner,
        }
        payload = json.dumps(self._token).encode("utf-8")
        while True:
            try:
                fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, payload)
                finally:
                    os.close(
                        fd
                    )  # closed immediately: unlink-while-open fails on Windows
                return True
            except FileExistsError:
                if _break_stale_lock(self._path):
                    continue  # retry immediately after breaking a stale lock
                return False
            except OSError:
                return False

    def release(self) -> None:
        self._fd = (
            None  # fd is closed immediately after acquire (Windows unlink semantics)
        )
        # FIX #6: release only a lock we still own (pid+started_at token).
        # If the lock went stale mid-scan and another process broke it and
        # re-acquired, the file now names a different owner — leave it alone.
        info = _read_lock_file(self._path)
        if (
            self._token is not None
            and info is not None
            and (
                info.get("pid") == self._token["pid"]
                and info.get("started_at") == self._token["started_at"]
            )
        ):
            try:
                self._path.unlink()
            except OSError:
                pass

    def is_locked_by_live_process(self) -> bool:
        if not self._path.exists():
            return False
        info = _read_lock_file(self._path)
        if info is None:
            return True  # unparsable but present: treat as held
        return _pid_alive(int(info.get("pid", 0)))


class JobHandle:
    """One acquired cross-process job lock (F2).

    Wraps a :class:`CrossProcessScanLock` plus the job metadata so callers
    (API background threads, the scheduler, warm-up, backfill) can:

    * ``heartbeat()`` — refresh the lock file's mtime (stale-breaker input);
    * ``check()`` — raise :class:`JobLockLost` when the file is gone or the
      token was taken over (the pipeline calls this at stage boundaries and
      every N kline fetches via an injected callable);
    * ``release()`` — idempotent; only removes a file still naming this token.

    The token includes ``kind`` and ``owner`` so a 409 can tell the caller
    exactly who holds the job.
    """

    def __init__(self, lock: CrossProcessScanLock, *, kind: str, owner: str) -> None:
        self._lock = lock
        self.kind = kind
        self.owner = owner
        self.released = False

    @property
    def info(self) -> dict:
        """Holder info for 409 payloads / logs."""
        raw = _read_lock_file(self._lock._path) or {}
        return {
            "kind": raw.get("kind") or self.kind,
            "owner": raw.get("owner") or self.owner,
            "pid": raw.get("pid") or os.getpid(),
            "since": raw.get("started_at"),
        }

    def heartbeat(self) -> None:
        """Touch the lock file so `_break_stale_lock` does not age it out."""
        if self.released:
            return
        try:
            now = time.time()
            os.utime(self._lock._path, (now, now))
        except OSError:
            pass  # a vanished file surfaces on the next check()

    def check(self) -> None:
        """Raise JobLockLost when the lock is gone or taken over."""
        if self.released:
            return
        info = _read_lock_file(self._lock._path)
        token = self._lock._token or {}
        if info is None:
            raise JobLockLost(
                "job lock lost (deleted or taken over) - aborted"
            )
        if (
            info.get("pid") != token.get("pid")
            or info.get("started_at") != token.get("started_at")
        ):
            raise JobLockLost(
                "job lock lost (deleted or taken over) - aborted"
            )

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self._lock.release()


class WatchlistService:
    """Single entry point for watchlist scans, items and the stored view.

    CLI, API, tool and scheduler call only this class.
    """

    def __init__(
        self,
        store: WatchlistStore | None = None,
        *,
        scan_fn: Callable | None = None,
        rerun_fn: Callable | None = None,
        calendar_loader: Callable | None = None,
        now_fn: Callable[[], _dt.datetime] | None = None,
    ) -> None:
        """Create the service.

        Args:
            store: Injectable store (tests pass a tmp_path DB).
            scan_fn: Injectable scan callable; default
                ``src.market_scanner.pipeline.run_scan`` (imported lazily so
                offline tests never touch network modules).
            rerun_fn: Injectable AI re-review callable; default
                ``src.market_scanner.pipeline.rerun_ai_review``.
            calendar_loader: Injectable zero-arg callable returning a trading
                calendar (default ``sources.calendar.load_calendar``).
            now_fn: Injectable clock for timestamps and staleness.
        """
        # F1: verify the home BEFORE building the store — WatchlistStore's
        # __init__ mkdirs its parent, so checking afterwards would already
        # have created directories in a mis-resolved home (observed: a
        # negative test left an empty `wronghome/watchlist/` behind).
        self.home = self._check_home(explicit_store=store is not None)
        self.store = store or WatchlistStore()
        self._scan_fn = scan_fn
        self._rerun_fn = rerun_fn
        self._calendar_loader = calendar_loader
        self._now_fn = now_fn or _dt.datetime.now
        self._scan_lock = threading.Lock()

    @staticmethod
    def _check_home(*, explicit_store: bool) -> dict[str, Any] | None:
        """F1: verify the resolved runtime home against the pin (pi's
        ``watchlist.home``). Lazy import: the module may not exist yet (pi
        owns it) — until then the check is a no-op. Skipped when an explicit
        store path is given (tests, tools with their own DB).

        Returns ``home_fields()`` for logs/run rows, or ``None`` when the
        check is unavailable/skipped.

        Raises:
            HomeMismatch: pin vs resolved home disagreement (the API maps
                this to HTTP 503 with the message verbatim).
        """
        if explicit_store:
            return None
        try:
            from src.watchlist.home import home_fields, require_home
        except ImportError:
            return None
        home = require_home("watchlist service")
        try:
            return home_fields()
        except Exception:  # noqa: BLE001 — fields are cosmetic
            return {"home": str(home), "pid": os.getpid()}

    # -- helpers --------------------------------------------------------------
    def _now(self) -> _dt.datetime:
        return self._now_fn()

    def _scan_callable(self) -> Callable:
        if self._scan_fn is not None:
            return self._scan_fn
        from src.market_scanner.pipeline import run_scan

        return run_scan

    def _rerun_callable(self) -> Callable:
        if self._rerun_fn is not None:
            return self._rerun_fn
        from src.market_scanner.pipeline import rerun_ai_review

        return rerun_ai_review

    def _calendar(self):
        if self._calendar_loader is not None:
            return self._calendar_loader()
        from src.market_scanner.sources.calendar import load_calendar

        calendar, _status = load_calendar()
        return calendar

    # -- settings -------------------------------------------------------------
    def get_settings(self) -> dict[str, Any]:
        """``{"scan": ScanConfig dict, "schedule": {...}}`` with env override."""
        stored = self.store.get_setting("scan")
        scan = ScanConfig.from_dict(stored).to_dict()
        schedule = {**_SCHEDULE_DEFAULTS, **(self.store.get_setting("schedule") or {})}
        from src.config.accessor import get_env_config, reset_env_config

        reset_env_config()  # fresh read: env may change in-process (tests)
        override = get_env_config().agent_tuning.vibe_trading_watchlist_schedule
        if override is not None:
            schedule["enabled"] = bool(override)
        return {"scan": scan, "schedule": schedule}

    def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial settings patch and return the new settings."""
        scan_patch = patch.get("scan")
        if scan_patch:
            current = ScanConfig.from_dict(self.store.get_setting("scan"))
            merged = {**current.to_dict(), **scan_patch}
            self.store.set_setting("scan", ScanConfig.from_dict(merged).to_dict())
        schedule_patch = patch.get("schedule")
        if schedule_patch:
            merged = {
                **_SCHEDULE_DEFAULTS,
                **(self.store.get_setting("schedule") or {}),
            }
            merged.update(
                {k: v for k, v in schedule_patch.items() if k in _SCHEDULE_DEFAULTS}
            )
            self.store.set_setting("schedule", merged)
        return self.get_settings()

    def _build_config(self, overrides: dict[str, Any] | None) -> ScanConfig:
        stored = self.get_settings()["scan"]
        merged = {**stored, **(overrides or {})}
        return ScanConfig.from_dict(merged)

    # -- scanning -------------------------------------------------------------
    def acquire_job(self, kind: str, *, owner: str) -> JobHandle:
        """Acquire the single cross-process job lock (F2).

        One lock guards scan, review, warm-up and backfill: whatever the
        kind, two jobs must never run at once (observed 15 Oct: scheduled
        catch-up + manual API scan ran concurrently and both hammered
        Tencent until the breaker tripped).

        Args:
            kind: ``scan`` / ``review`` / ``warmup`` / ``backfill``.
            owner: Human-readable holder, e.g. ``api`` / ``scheduler`` /
                ``cli`` — stored in the lock file and surfaced in the 409.

        Raises:
            ScanAlreadyRunning: when any live job holds the lock; the message
                names the holder (kind, owner, pid, since) for the API's 409.
        """
        cross = CrossProcessScanLock()
        if not cross.acquire(kind=kind, owner=owner):
            holder = _read_lock_file(cross._path) or {}
            raise ScanAlreadyRunning(
                f"a job is already running (kind={holder.get('kind') or 'unknown'}"
                f", owner={holder.get('owner') or 'unknown'}"
                f", pid={holder.get('pid')}"
                f", since={holder.get('started_at')})"
            )
        return JobHandle(cross, kind=kind, owner=owner)

    def _holder_info(self) -> dict:
        """Current lock holder (for logs / 409s), or {} when free."""
        return _read_lock_file(_lock_path()) or {}

    def run_scan(
        self,
        *,
        trigger: str,
        config_overrides: dict[str, Any] | None = None,
        progress: Callable[[str, float], None] | None = None,
        llm: Any = None,
        handle: JobHandle | None = None,
    ) -> dict[str, Any]:
        """Run one scan end-to-end and persist it (CONTRACT section 10).

        Args:
            trigger: ``cli`` / ``api`` / ``schedule`` / ``agent``.
            config_overrides: Partial ScanConfig fields over stored settings.
            progress: Optional ``(stage, fraction)`` callback forwarded to the
                scanner.
            llm: Injected completion callable for the AI review.
            handle: An already-acquired :class:`JobHandle` (F2: the API
                acquires the lock synchronously before answering 202 and
                hands the handle to the background thread). When omitted the
                scan acquires its own and releases it when done.

        Returns:
            Summary dict with ``run_id, trade_date, status, ai_status,
            duration_s, funnel, sources, counts``.

        Raises:
            ScanAlreadyRunning: while another job is in flight in this
                process or across processes.
            JobLockLost: when the cross-process lock vanished mid-run; the
                run is stored as failed with that failure_reason.
        """
        own_handle = handle is None
        if own_handle:
            if not self._scan_lock.acquire(blocking=False):
                raise ScanAlreadyRunning(
                    "a scan is already running in this process"
                )
            try:
                handle = self.acquire_job("scan", owner=trigger)
            except ScanAlreadyRunning:
                self._scan_lock.release()
                raise
        try:
            try:
                return self._run_scan_inner(
                    trigger=trigger,
                    config_overrides=config_overrides,
                    progress=progress,
                    llm=llm,
                    handle=handle,
                )
            except JobLockLost as exc:
                # F2: the lock vanished mid-run — another writer believes it
                # owns the job. Persist a failed run so history shows WHY
                # (the pipeline's partial output, if any, is discarded).
                reason = str(exc) or "job lock lost (deleted or taken over) - aborted"
                logger.warning(
                    "watchlist scan aborted: %s (trigger=%s)", reason, trigger
                )
                self.store.save_run(
                    {
                        "run_id": uuid.uuid4().hex,
                        "trade_date": _dt.date.today().isoformat(),
                        "started_at": _now_iso(),
                        "finished_at": _now_iso(),
                        "status": "failed",
                        "trigger": trigger,
                        "ai_status": "not_reviewed",
                        "failure_reason": reason,
                        "config": {},
                        "funnel": [],
                        "sources": [],
                        "picks": [],
                        "scores": [],
                    },
                    changes={},
                )
                raise
        finally:
            if own_handle:
                handle.release()
                self._scan_lock.release()

    def is_scanning(self) -> bool:
        """Whether a scan or AI re-review is currently in flight.

        Scan and review share the same single-flight lock (in-process and
        cross-process), so this covers both.
        """
        return (
            self._scan_lock.locked()
            or CrossProcessScanLock().is_locked_by_live_process()
        )

    def rerun_ai_review(
        self,
        run_id: str | None = None,
        *,
        trigger: str = "cli",
        progress: Callable[[str, float], None] | None = None,
        llm: Any = None,
        sources: Any = None,
        handle: JobHandle | None = None,
    ) -> dict[str, Any]:
        """Re-run only the AI review for a stored scan, in place (CONTRACT §10).

        The latest successful run (status ok or partial — success = ok|partial)
        is re-reviewed by default; any other run (unknown, failed, or not the
        latest reviewable one) is refused with ``ValueError``. Shares the
        single-flight lock with :meth:`run_scan` — scan and review exclude
        each other (``ScanAlreadyRunning``).

        The run keeps its ``run_id``, ``seq`` and scan timestamps; picks,
        ai fields and ``review_revision`` are recomputed by
        ``src.market_scanner.pipeline.rerun_ai_review`` and the watchlist
        view is recomputed honouring manual pins and suppressed codes
        exactly like a scan.

        Args:
            run_id: Run to re-review; ``None`` = the latest successful run.
            trigger: ``cli`` / ``api`` / ``schedule`` / ``agent``.
            progress: Optional ``(stage, fraction)`` callback.
            llm: Injected completion callable for the AI review.
            sources: Injectable source bundle (news/ann/red-flag fetches).

        Returns:
            Summary dict with ``run_id, trade_date, status, ai_status,
            ai_note, duration_s`` (the re-review's own duration),
            ``review_revision, reviewed_at, funnel, sources, counts``.

        Raises:
            ValueError: when the run is unknown, not successful (ok|partial),
                or not the latest reviewable run.
            ScanAlreadyRunning: while a scan or review is in flight.
        """
        # Resolve the target run first; always load the FULL envelope (the
        # summary alone has no picks/scores/ai_pool for the re-review).
        if run_id is None:
            latest = self.store.last_successful_run()
            if latest is None:
                raise ValueError("no successful run to re-review")
            run_id = latest["run_id"]
        run = self.store.get_run(run_id)
        if run is None:
            raise ValueError(f"unknown watchlist run: {run_id}")
        latest = self.store.last_successful_run()
        if run.get("status") not in ("ok", "partial"):
            raise ValueError(
                f"run {run['run_id']} is not a successful scan "
                f"(status={run.get('status')!r})"
            )
        if latest is None or latest["run_id"] != run["run_id"]:
            raise ValueError(
                f"run {run['run_id']} is not the latest successful run "
                f"(latest: {latest['run_id']}); only the latest run can be re-reviewed"
            )
        own_handle = handle is None
        if own_handle:
            if not self._scan_lock.acquire(blocking=False):
                raise ScanAlreadyRunning(
                    "a scan or review is already running in this process"
                )
            try:
                handle = self.acquire_job("review", owner=trigger)
            except ScanAlreadyRunning:
                self._scan_lock.release()
                raise
        try:
            return self._rerun_review_inner(
                run,
                trigger=trigger,
                progress=progress,
                llm=llm,
                sources=sources,
                handle=handle,
            )
        finally:
            if own_handle:
                handle.release()
                self._scan_lock.release()

    def _run_scan_inner(
        self,
        *,
        trigger: str,
        config_overrides: dict[str, Any] | None,
        progress: Callable[[str, float], None] | None,
        llm: Any,
        handle: JobHandle | None = None,
    ) -> dict[str, Any]:
        config = self._build_config(config_overrides)
        suppressed = set(self.store.suppressed_codes())
        # Manual pins beat suppression (user pinned after removing).
        pinned = set(self.store.active_codes(origin="manual"))
        suppressed_effective = suppressed - pinned

        t0 = time.monotonic()
        # F2: check the lock right before the expensive scan; the pipeline
        # receives `lock_check` so it can also check at stage boundaries and
        # every N kline fetches (opencode inserts the call sites).
        if handle is not None:
            handle.check()
        run = self._scan_callable()(
            config,
            suppressed=suppressed_effective,
            pinned=pinned,
            trigger=trigger,
            progress=progress,
            llm=llm,
            lock_check=(handle.check if handle is not None else None),
        )
        duration = time.monotonic() - t0
        run.setdefault("duration_s", round(duration, 2))
        run.setdefault("run_id", uuid.uuid4().hex)
        run.setdefault("status", "failed")
        run.setdefault("ai_status", "not_reviewed")

        changes = self._compute_changes(run)
        self.store.save_run(run, changes=changes)
        # Self-heal the diff: a run whose picks are in the store but whose
        # dropped rows went missing (observed live: runs d4ce9714 /
        # c9916e36 kept their 20 picks and 0 dropped rows even though codes
        # had left the previous run's top-20) must be re-saved with the full
        # diff. Recompute from the run's own stored neighbours, and only
        # rewrite when the recomputed set is strictly larger.
        changes = self._verify_dropped_saved(run, changes)

        # Watchlist item transitions from the diff (only successful runs —
        # ok or partial, i.e. pipeline completed — move the auto set; failed
        # runs must not wipe current picks).
        if run.get("status") in ("ok", "partial"):
            self._apply_item_changes(run, changes)

        counts = {
            "new": len(changes.get("new", [])),
            "kept": len(changes.get("kept", [])),
            "dropped": len(changes.get("dropped", [])),
            "requalified": len(run.get("requalified") or []),
            "ai_demoted": len(run.get("ai_demoted") or []),
        }
        # The full diff travels with the summary so the CLI can render the
        # correct new/kept labels (the picks payload carries no change field).
        changes_view = {
            "new": [
                {"code": c, "rank": r, "score": s} for c, r, s, _p in changes.get("new", [])
            ],
            "kept": [
                {"code": c, "rank": r, "score": s} for c, r, s, _p in changes.get("kept", [])
            ],
            "dropped": [
                {"code": c, "name": n, "prev_rank": pr, "reason": rs}
                for c, n, pr, rs in changes.get("dropped", [])
            ],
        }
        return {
            "run_id": run["run_id"],
            "trade_date": run.get("trade_date"),
            "status": run.get("status"),
            "failure_reason": run.get("failure_reason"),
            "ai_status": run.get("ai_status"),
            "duration_s": run.get("duration_s"),
            "funnel": run.get("funnel"),
            "sources": run.get("sources"),
            "counts": counts,
            "changes": changes_view,
        }

    def _compute_changes(
        self, run: dict[str, Any], *, prev: dict[str, Any] | None = None
    ) -> dict[str, list]:
        """Diff a run's picks against the previous SUCCESSFUL run.

        Args:
            run: The run whose picks are diffed.
            prev: Explicit predecessor run (summary). ``None`` = the latest
                successful run in the store (a fresh scan's default). An AI
                re-review passes the latest successful run *before* the
                re-reviewed run, because ``last_successful_run()`` would
                return the re-reviewed run itself.
        """
        if prev is None:
            prev = self.store.last_successful_run()
        prev_picks: dict[str, int] = {}
        prev_names: dict[str, str] = {}
        if prev and prev["run_id"] != run.get("run_id"):
            prev_pick_rows = self.store.get_picks(prev["run_id"])
            prev_picks = {p["code"]: p["rank"] for p in prev_pick_rows}
            prev_names = {
                p["code"]: (p.get("pick") or {}).get("name") for p in prev_pick_rows
            }

        new: list = []
        kept: list = []
        dropped: list = []
        scores = {s["code"]: s for s in run.get("scores", [])}
        seen: set[str] = set()
        for pick in run.get("picks", []):
            code = pick["code"]
            seen.add(code)
            rank = pick.get("rank")
            score = pick.get("score")
            if code in prev_picks:
                kept.append((code, rank, score, pick))
            else:
                new.append((code, rank, score, pick))
        # FIX B: dropped reason gives the NEW rank (by score desc among scored
        # rows) when the stock is still scored, else its exclusion reason.
        scored = [
            s
            for s in run.get("scores", [])
            if s.get("exclusion_reason") is None and s.get("score") is not None
        ]
        scored.sort(key=lambda s: s.get("score") or 0.0, reverse=True)
        new_rank = {s["code"]: i + 1 for i, s in enumerate(scored)}

        for code, prev_rank in prev_picks.items():
            if code in seen:
                continue
            row = scores.get(code, {})
            if row.get("exclusion_reason"):
                reason = row["exclusion_reason"]
            elif code in new_rank:
                score_now = row.get("score")
                score_txt = f" (score {score_now:.1f})" if score_now is not None else ""
                reason = f"rank fell to {new_rank[code]}{score_txt}"
            else:
                reason = "no data in this run"
            # FIX #11: name falls back to the previous run's pick name.
            dropped.append(
                (code, row.get("name") or prev_names.get(code), prev_rank, reason)
            )
        return {"new": new, "kept": kept, "dropped": dropped}

    def _verify_dropped_saved(
        self, run: dict[str, Any], changes: dict[str, list]
    ) -> dict[str, list]:
        """Re-persist the drop rows when the saved diff came up short.

        The DROPPED section is what tells the boss *why* a stock left the
        top-20, so a silent loss is a correctness bug, not cosmetics. Live
        evidence (run d4ce9714): picks saved 20/20 while the previous run had
        20 different picks — the diff should have recorded two drops, the
        store recorded none, and no code path logged it.

        This recomputes the diff against the run actually stored as the
        predecessor and re-saves the run (same run_id, same payload) when the
        neighbour's picks imply drops the store is missing. Idempotent: a
        correct save recomputes the identical set and is left alone.
        """
        try:
            run_id = run.get("run_id")
            if not run_id:
                return changes
            stored = self.store.get_dropped(run_id)
            stored_codes = {row.get("code") for row in stored}
            prev = self.store.last_successful_run_before(run_id)
            if prev is None:
                return changes
            # Neighbour must have picks of its own to imply drops.
            prev_codes = {
                row.get("code") for row in self.store.get_picks(prev["run_id"])
            }
            if not prev_codes:
                return changes
            current_picks = {
                (p or {}).get("code") for p in (run.get("picks") or [])
            }
            missing = sorted(prev_codes - current_picks - stored_codes)
            if not missing:
                return changes
            recomputed = self._compute_changes(
                self.store.get_run(run_id) or run, prev=prev
            )
            have = {c for c, _n, _pr, _r in recomputed.get("dropped", [])}
            if not (set(missing) & have):
                return changes
            logger.warning(
                "watchlist: run %s stored %d dropped row(s) but the previous "
                "run's picks imply %d more (%s); re-saving the diff",
                run_id[:8],
                len(stored_codes),
                len(missing),
                ", ".join(missing),
            )
            self.store.save_run(run, changes=recomputed)
            return recomputed
        except Exception as exc:  # noqa: BLE001 — never fail a scan over this
            logger.warning(
                "watchlist: dropped-row self-heal failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return changes

    def _apply_item_changes(
        self, run: dict[str, Any], changes: dict[str, list]
    ) -> None:
        """Update watchlist_items from a successful run's diff."""
        now = _now_iso()
        for code, _rank, _score, pick in changes.get("new", []) + changes.get(
            "kept", []
        ):
            existing = self.store.get_item(code)
            if existing and existing["state"] == "removed":
                continue  # never re-add suppressed codes
            if existing and existing["origin"] == "manual":
                continue  # manual items are managed explicitly
            self.store.upsert_item(
                code,
                pick.get("name"),
                "auto",
                "active",
                now,
                note=existing.get("note") if existing else None,
            )
        # FIX #5 (section 9: each scan replaces the auto picks): an auto item
        # that is not picked in the latest successful run leaves 'items' (it
        # shows in this run's 'dropped'). Never marked 'restored'. Manual
        # items and suppressed codes keep their rules.
        for code, _name, _prev_rank, _reason in changes.get("dropped", []):
            existing = self.store.get_item(code)
            if existing is None or existing["state"] == "removed":
                continue  # nothing to remove / suppression stays
            if existing["origin"] == "manual":
                continue  # manual items are managed explicitly
            self.store.remove_item_row(code)

    def _rerun_review_inner(
        self,
        run: dict[str, Any],
        *,
        trigger: str,
        progress: Callable[[str, float], None] | None,
        llm: Any,
        sources: Any,
        handle: JobHandle | None = None,
    ) -> dict[str, Any]:
        """Re-review one stored run under the single-flight lock."""
        logger.info(
            "watchlist: re-reviewing run %s (trigger=%s)", run.get("run_id"), trigger
        )
        config = ScanConfig.from_dict(run.get("config") or {})
        # Same pin/suppression rules as run_scan (CONTRACT section 9: a
        # manual add beats suppression; pins are never AI-demoted out).
        suppressed = set(self.store.suppressed_codes())
        pinned = set(self.store.active_codes(origin="manual"))
        suppressed_effective = suppressed - pinned

        t0 = time.monotonic()
        # F2: abort when the job lock vanished mid-review (a 20-40 min
        # re-review must not continue unowned).
        if handle is not None:
            handle.check()
        updated = self._rerun_callable()(
            run,
            config,
            llm=llm,
            sources=sources,
            progress=progress,
            now=self._now(),
            pinned=pinned,
            suppressed=suppressed_effective,
        )
        duration = time.monotonic() - t0
        # Diff against the run BEFORE this one: the re-reviewed run itself
        # would label every pick "kept". ok|partial predecessors count (a
        # partial previous session is the right baseline, not None).
        prev = self.store.last_successful_run_before(run["run_id"])
        changes = self._compute_changes(updated, prev=prev)
        self.store.update_run_review(updated, changes=changes)
        # Only successful runs move the auto set — same guard as run_scan.
        if updated.get("status") in ("ok", "partial"):
            self._apply_rerun_item_changes(run, updated, changes)

        counts = {
            "new": len(changes.get("new", [])),
            "kept": len(changes.get("kept", [])),
            "dropped": len(changes.get("dropped", [])),
            "requalified": len(updated.get("requalified") or []),
            "ai_demoted": len(updated.get("ai_demoted") or []),
        }
        changes_view = {
            "new": [
                {"code": c, "rank": r, "score": s} for c, r, s, _p in changes.get("new", [])
            ],
            "kept": [
                {"code": c, "rank": r, "score": s} for c, r, s, _p in changes.get("kept", [])
            ],
            "dropped": [
                {"code": c, "name": n, "prev_rank": pr, "reason": rs}
                for c, n, pr, rs in changes.get("dropped", [])
            ],
        }
        return {
            "run_id": updated["run_id"],
            "trade_date": updated.get("trade_date"),
            "status": updated.get("status"),
            "ai_status": updated.get("ai_status"),
            "ai_note": updated.get("ai_note"),
            # The re-review's own duration; the stored run keeps the original
            # scan duration_s untouched.
            "duration_s": round(duration, 2),
            "review_revision": updated.get("review_revision"),
            "reviewed_at": updated.get("reviewed_at"),
            "funnel": updated.get("funnel"),
            "sources": updated.get("sources"),
            "counts": counts,
            "changes": changes_view,
        }

    def _apply_rerun_item_changes(
        self,
        original: dict[str, Any],
        updated: dict[str, Any],
        changes: dict[str, list],
    ) -> None:
        """Watchlist item transitions for an in-place AI re-review.

        Reuses :meth:`_apply_item_changes` for new/kept/dropped, then also
        removes auto items whose code was a pick in the ORIGINAL scan but
        left picks in the re-review (backfill/demotion churn the original
        diff could not see).
        """
        self._apply_item_changes(updated, changes)
        original_picks = {p.get("code") for p in original.get("picks") or []}
        updated_picks = {p.get("code") for p in updated.get("picks") or []}
        for code in original_picks - updated_picks:
            if not code:
                continue
            existing = self.store.get_item(code)
            if existing is None or existing["state"] == "removed":
                continue  # nothing to remove / suppression stays
            if existing["origin"] == "manual":
                continue  # manual items are managed explicitly
            self.store.remove_item_row(code)

    # -- maintenance -----------------------------------------------------------
    def invalidate_run(
        self, run_id: str, *, reason: str | None = None
    ) -> dict[str, Any]:
        """Mark a stored run failed and restore the auto set (maintenance).

        Re-evaluates the stored run's coverage ratios against the config
        ``min_coverage`` floor: a run below the floor gets the standard
        ``insufficient coverage (...)`` failure_reason, otherwise ``reason``
        (or a default). The run stays in history as failed. When it had
        been the LATEST successful run, auto watchlist items are restored
        to the previous successful run's picks (manual pins and
        suppressions untouched). A failed run is never a diff baseline and
        never clears staleness, so those follow from the status change.
        """
        if self.is_scanning():
            raise ValueError("a scan or review is running; retry after it finishes")
        run = self.store.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        was_successful = run.get("status") in ("ok", "partial")
        latest_ok = self.store.last_successful_run()
        is_latest_ok = bool(
            was_successful and latest_ok and latest_ok["run_id"] == run_id
        )
        min_coverage = float((run.get("config") or {}).get("min_coverage") or 0.90)
        failure_reason = _coverage_failure_reason(
            run.get("coverage") or {}, min_coverage
        ) or (reason or "invalidated manually")
        run["status"] = "failed"
        run["failure_reason"] = failure_reason
        self.store.set_run_failed(run_id, failure_reason=failure_reason, payload=run)
        restored_from = None
        counts = {"added": 0, "removed": 0}
        if is_latest_ok:
            prev = self.store.last_successful_run_before(run_id)
            restored_from = prev["run_id"] if prev else None
            counts = self._restore_items_to(prev)
        return {
            "run_id": run_id,
            "status": "failed",
            "failure_reason": failure_reason,
            "restored": is_latest_ok,
            "restored_from": restored_from,
            "restored_counts": counts,
        }

    def _restore_items_to(self, prev: dict[str, Any] | None) -> dict[str, int]:
        """Reset the auto watchlist set to the post-``prev`` picks.

        Auto items become exactly ``prev``'s picks; manual pins and
        suppressed (state='removed') rows are never touched. ``prev=None``
        clears every active auto item (no successful predecessor).
        """
        now = _now_iso()
        prev_codes: set[str] = set()
        added = 0
        removed = 0
        if prev is not None:
            for row in self.store.get_picks(prev["run_id"]):
                code = row["code"]
                prev_codes.add(code)
                if self.store.get_item(code) is not None:
                    continue  # kept / manual / suppressed rows stay as they are
                pick = row.get("pick") or {}
                self.store.upsert_item(code, pick.get("name"), "auto", "active", now)
                added += 1
        for item in self.store.list_items(state="active", origin="auto"):
            if item["code"] not in prev_codes:
                self.store.remove_item_row(item["code"])
                removed += 1
        return {"added": added, "removed": removed}

    # -- items ----------------------------------------------------------------
    def add_manual(self, code: str, note: str | None = None) -> dict[str, Any]:
        """Pin a manual item; validates the code, rejects BSE.

        Args:
            code: 6-digit A-share code.
            note: Optional user note.

        Returns:
            The stored item dict.

        Raises:
            ValueError: on a malformed code or a BSE (unsupported) code.
        """
        code = str(code).strip()
        if not (len(code) == 6 and code.isdigit()):
            raise ValueError(f"invalid A-share code: {code!r} (expected 6 digits)")
        if code[0] in {"4", "8"} or code.startswith("92"):
            raise ValueError(
                f"BSE code {code} is unsupported (BSE excluded from scanning)"
            )
        name = None
        last_ok = self.store.last_successful_run()
        if last_ok:
            scores = self.store.get_scores(last_ok["run_id"])
            if code in scores:
                name = scores[code].get("name")
        now = _now_iso()
        self.store.upsert_item(
            code, name, "manual", "active", now, note=note, promote_origin=True
        )
        return self.store.get_item(code)

    def _attach_ai_context(
        self, run_id: str, dropped: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Attach the AI verdict/reason/cites to each dropped row.

        The ``scan_dropped`` table stores only ``code/name/prev_rank/reason``;
        the AI verdict for a code that left the top-20 lives in the run's
        envelope: ``ai_demoted`` (an ``avoid`` verdict demoted it out of
        picks) or the picks/ai_pool ``ai`` block (reviewed as ``watch`` but
        its rank fell below top_n). A drop purely from rank or exclusion
        data has no AI verdict and stays with ``ai_verdict=None``.

        Returns the same rows plus ``ai_verdict`` / ``ai_reason`` /
        ``ai_cites`` so the CLI and the UI can render "why did it leave" —
        the boss needs the AI's cited reason, not just the rank fall.
        """
        if not dropped:
            return dropped
        run = self.store.get_run(run_id) or {}
        ai_by_code: dict[str, dict] = {}
        for entry in run.get("ai_demoted") or []:
            code = entry.get("code")
            if code:
                ai_by_code[code] = entry.get("ai") or {}
        if not ai_by_code:
            for entry in (run.get("picks") or []) + (run.get("ai_pool") or []):
                code = entry.get("code")
                ai = entry.get("ai")
                if code and isinstance(ai, dict) and ai.get("status") in (
                    "reviewed",
                    "failed",
                ):
                    ai_by_code[code] = ai
        if not ai_by_code:
            ai_by_code = {}
        enriched: list[dict[str, Any]] = []
        for row in dropped:
            ai = ai_by_code.get(row.get("code"))
            if not ai:
                # Rank/exclusion drop: no AI verdict for this code.
                row["ai_verdict"] = None
                row["ai_reason"] = None
                row["ai_cites"] = []
                enriched.append(row)
                continue
            # First cited reason: a red flag is the negative driver, else the
            # first review reason; cites join so the CLI shows [A1, M:pe, …].
            reason_text = None
            cites: list[str] = []
            red_flags = ai.get("red_flags") or []
            if red_flags:
                first = red_flags[0] or {}
                reason_text = first.get("text")
                cites = list(first.get("cites") or [])
            if not reason_text:
                reasons = ai.get("reasons") or []
                if reasons:
                    first = reasons[0] or {}
                    reason_text = first.get("text")
                    cites = list(first.get("cites") or [])
            if not reason_text:
                reason_text = ai.get("summary")
            enriched.append(
                {
                    **row,
                    "ai_verdict": ai.get("verdict"),
                    "ai_status": ai.get("status"),
                    "ai_reason": reason_text,
                    "ai_cites": cites,
                }
            )
        return enriched

    def remove(self, code: str) -> dict[str, Any]:
        """Remove an item (auto or manual) — suppressed, never auto-re-added."""
        item = self.store.get_item(code)
        if item is None:
            # Remember the removal even for codes never seen: create a stub.
            self.store.upsert_item(code, None, "manual", "removed", _now_iso())
        else:
            self.store.mark_removed(code, _now_iso())
        return self.store.get_item(code)

    def restore(self, code: str) -> dict[str, Any]:
        """Clear the suppression; the item shows as change ``restored``."""
        item = self.store.get_item(code)
        if item is None:
            raise ValueError(f"unknown watchlist code: {code}")
        self.store.mark_active(code, _now_iso())
        return self.store.get_item(code)

    def forget_manual(self, code: str) -> dict[str, Any]:
        """Delete a manual item row entirely — F9 clean undo of add_manual.

        Unlike :meth:`remove` (a suppression) this leaves NO trace: the row
        is deleted whatever its state, so a forgotten test pin cannot linger
        in the UI or re-appear in the view (the 600519 wrap-up leftover).

        Args:
            code: 6-digit A-share code.

        Returns:
            The deleted row (for the API/CLI response).

        Raises:
            ValueError: unknown code, or the row is an auto item (scans own
                those — removing an auto row is ``remove``/suppression).
        """
        item = self.store.get_item(code)
        if item is None:
            raise ValueError(f"unknown watchlist code: {code}")
        if item.get("origin") != "manual":
            raise ValueError(
                f"code {code} is an auto item (origin={item.get('origin')!r}); "
                "only manual items can be forgotten"
            )
        self.store.remove_item_row(code)
        return item

    # -- views ----------------------------------------------------------------
    def get_watchlist(self) -> dict[str, Any]:
        """The full watchlist view (CONTRACT section 10).

        ``latest_run`` is the current effective run — the latest successful
        one (items/scores come from it). When the most recent run overall
        failed, ``last_failed_run`` surfaces it (with ``failure_reason``)
        instead of pretending it is current.
        """
        latest_ok = self.store.last_successful_run()
        latest_any = self.store.list_runs(limit=1)
        latest_run = latest_ok
        newest = latest_any[0] if latest_any else None
        last_failed = (
            newest
            if newest is not None
            and newest.get("status") not in ("ok", "partial")
            else None
        )

        picks_by_code: dict[str, dict] = {}
        scores: dict[str, dict] = {}
        dropped: list[dict] = []
        requalified: list[dict] = []
        ai_demoted: list = []
        if latest_ok:
            run_id = latest_ok["run_id"]
            picks_by_code = {p["code"]: p for p in self.store.get_picks(run_id)}
            scores = self.store.get_scores(run_id)
            # Dropped / requalified / ai_demoted all come from the latest
            # SUCCESSFUL run only. A failed run (e.g. invalidated for
            # insufficient coverage) keeps its rows in the store for history
            # but must never leak its diff into the current view.
            dropped = self.store.get_dropped(run_id)
            dropped = self._attach_ai_context(run_id, dropped)
            requalified = latest_ok.get("requalified") or []
            ai_demoted = latest_ok.get("ai_demoted") or []

        suppressed_set = set(self.store.suppressed_codes())
        items: list[dict[str, Any]] = []
        for item in self.store.list_items():
            if item["state"] != "active":
                continue
            code = item["code"]
            score_row = scores.get(code)
            pick = picks_by_code.get(code)
            if item["origin"] == "manual":
                change = "manual"
            elif pick and pick.get("change"):
                change = pick["change"]
            else:
                change = "restored"
            if score_row and score_row.get("exclusion_reason") is None:
                sub = {
                    "fundamental": score_row.get("fundamental"),
                    "trend": score_row.get("trend"),
                    "flow": score_row.get("flow"),
                }
                scores_available = True
                unscored_reason = None
            elif score_row and score_row.get("exclusion_reason"):
                # Excluded stock: show why it has no score.
                scores_available = False
                unscored_reason = score_row["exclusion_reason"]
                sub = None
            elif score_row is None and code in scores:
                scores_available = False
                unscored_reason = scores[code].get("exclusion_reason") or "excluded"
                sub = None
            else:
                scores_available = False
                unscored_reason = "no data in the latest scan"
                sub = None
            items.append(
                {
                    "code": code,
                    "name": (score_row or {}).get("name") or item.get("name"),
                    "board": (score_row or {}).get("board"),
                    "industry": (score_row or {}).get("industry"),
                    "origin": item["origin"],
                    "change": change,
                    "rank": (pick or {}).get("rank"),
                    "score": (
                        (pick or {}).get("score")
                        if pick
                        else (score_row or {}).get("score")
                    ),
                    "sub_scores": sub,
                    "reasons": (
                        (pick or {}).get("pick", {}).get("reasons") if pick else None
                    ),
                    "flags": (
                        (pick or {}).get("pick", {}).get("flags") if pick else None
                    ),
                    "ai": (pick or {}).get("pick", {}).get("ai") if pick else None,
                    "added_at": item["added_at"],
                    "note": item.get("note"),
                    "scores_available": scores_available,
                    "unscored_reason": unscored_reason,
                }
            )
        # Display order: rank ascending; unscored/manual items (no rank) last.
        items.sort(key=lambda it: (it.get("rank") is None, it.get("rank") or 0))

        requalified_view = []
        for rq in requalified:
            code = rq.get("code")
            item = self.store.get_item(code) if code else None
            requalified_view.append(
                {
                    "code": code,
                    "name": rq.get("name"),
                    "rank": rq.get("rank"),
                    "score": rq.get("score"),
                    "sub_scores": rq.get("sub_scores"),
                    "reasons": rq.get("reasons"),
                    "removed_at": (item or {}).get("removed_at"),
                    "label": "qualifies again - you removed it",
                }
            )

        return {
            "disclaimer": _DISCLAIMER,
            "stale": self.staleness(),
            "latest_run": (
                {
                    "run_id": latest_run["run_id"],
                    "trade_date": latest_run["trade_date"],
                    "intraday": latest_run.get("intraday"),
                    "finished_at": latest_run["finished_at"],
                    "status": latest_run["status"],
                    "ai_status": latest_run["ai_status"],
                    "ai_note": latest_run["ai_note"],
                    "duration_s": latest_run["duration_s"],
                    "trigger": latest_run["trigger"],
                    "funnel": latest_run["funnel"],
                    "sources": latest_run["sources"],
                    "coverage": (
                        self.store.get_run(latest_run["run_id"]) or {}
                    ).get("coverage"),
                }
                if latest_run
                else None
            ),
            "last_failed_run": (
                {
                    "run_id": last_failed["run_id"],
                    "trade_date": last_failed["trade_date"],
                    "finished_at": last_failed["finished_at"],
                    "status": last_failed["status"],
                    "ai_status": last_failed["ai_status"],
                    "trigger": last_failed["trigger"],
                    "failure_reason": last_failed.get("failure_reason"),
                }
                if last_failed
                else None
            ),
            "items": items,
            "dropped": dropped,
            "requalified": [
                rq for rq in requalified_view if rq["code"] in suppressed_set
            ],
            "ai_demoted": ai_demoted,
        }

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Run history (most recent first)."""
        return self.store.list_runs(limit)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """One run's summary or None."""
        return self.store.get_run(run_id)

    # -- staleness --------------------------------------------------------------
    def staleness(self) -> dict[str, Any]:
        """Stale when the latest completed trading day has no successful scan."""
        calendar = self._calendar()
        now = self._now()
        try:
            latest_day = calendar.latest_completed_trading_day(now)
        except Exception:  # noqa: BLE001 — degraded calendar: weekday semantics
            # FIX #9: honour the 15:00 Asia/Shanghai close cutoff so a weekday
            # morning resolves to the previous trading day, not today.
            sh = _dt.timezone(_dt.timedelta(hours=8))
            local = now if now.tzinfo else now.replace(tzinfo=sh)
            local = local.astimezone(sh)
            latest_day = local.date()
            if not (latest_day.weekday() < 5 and local.time() >= _dt.time(15, 0)):
                latest_day -= _dt.timedelta(days=1)
                while latest_day.weekday() >= 5:
                    latest_day -= _dt.timedelta(days=1)
        last_ok = self.store.last_successful_run()
        last_any_for_day = self.store.last_scan_for_trade_date(latest_day.isoformat())
        last_scan_trade_date = last_ok.get("trade_date") if last_ok else None
        return {
            "is_stale": last_any_for_day is None
            or last_any_for_day.get("status") not in ("ok", "partial"),
            "latest_trading_day": latest_day.isoformat(),
            "last_scan_trade_date": last_scan_trade_date,
            "last_scan_at": (last_ok or {}).get("finished_at"),
        }
