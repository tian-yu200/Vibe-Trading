"""Asyncio daily scheduler for the watchlist scan (CONTRACT section 11).

Fires ``service.run_scan(trigger="schedule")`` once per trading day at a
configured HH:MM in Asia/Shanghai. Started/stopped from the api_server
lifespan; never crashes the server. No OS-level scheduled tasks.

Daily-fire rule (TODO R2 item B):
- A SCHEDULED run (``trigger=="schedule"``) with status ok/partial for the
  trade date suppresses the day's fire; manual/CLI/API runs never do.
- AI-review failure alone (``ai_status`` not_reviewed/partial, quota) is
  still a pipeline success — recovery is re-run-AI-review, not a rescan.
- A failed run (status failed or exception) is retried once after 30 min,
  at most 2 attempts per trade date — never re-fired every tick.
- Catch-up on start uses the identical success rule and never runs during
  trading hours (09:15–5:00 on a trading day). Before 09:15 it may run
  for the previous session; after 15:00 only once today's fire time passed.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import threading
import time as _time
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _default_now_fn() -> _dt.datetime:
    """Wall-clock now in Asia/Shanghai (aware)."""
    return _dt.datetime.now(ZoneInfo(_DEFAULT_TZ))


def _default_calendar_loader():
    """Load the shared trading calendar (lazy: network/cache on first call)."""
    from src.market_scanner.sources.calendar import load_calendar

    cal, _st = load_calendar()
    return cal


_DEFAULT_TZ = "Asia/Shanghai"
_LOOP_SLEEP_S = 20.0  # idle poll; injectable sleep used in tests
_RETRY_DELAY_S = 30 * 60  # one retry per failed run, never sooner
_MAX_ATTEMPTS_PER_DAY = 2  # per trade date: first try + one retry
_TRADING_BLOCK_START = _dt.time(9, 15)  # no scans during trading hours
_TRADING_BLOCK_END = _dt.time(15, 0)
_SUCCESS_STATUSES = ("ok", "partial")
_MAX_LOG_BYTES = 1_000_000  # rotate: keep roughly the last 1 MB of events
_TRIM_BYTES = _MAX_LOG_BYTES // 2

#: Serialises scheduler.log writes: the main loop and a background warm-up
#: thread log to the same file, and concurrent ``open(..., "a")`` appends
#: can interleave mid-line on Windows. One lock per process (two servers on
#: one home are refused by the F3 server lock, so cross-process races are
#: out of scope).
_LOG_LOCK = threading.Lock()

#: Warm-up policy (F5): background cache fill at <= 1 req/s.
_WARMUP_RATE_PER_SEC = 1.0
_WARMUP_CHECK_S = 3600.0  # reassess at most hourly on trading days
_WARMUP_PROGRESS_EVERY = 10  # log warmup_progress every Nth progress tick
_FIRE_INFLIGHT_FILENAME = "fire_inflight.json"


# F2/F3: single job-lock failure type, owned by service.py (hermes). Imported
# here (not defined) so the scheduler, the legacy handle and hermes's service
# raise and catch the identical class. ``service`` never imports this module,
# so there is no cycle.
from src.watchlist.service import JobLockLost  # noqa: E402  (re-exported)


def _default_instance_id_fn() -> str | None:
    """F3: this process's server instance id, or None before it exists."""
    try:
        from src.api.watchlist_routes import server_instance_id
    except Exception:  # noqa: BLE001 — api package unavailable in tests
        return None
    try:
        return server_instance_id()
    except Exception:  # noqa: BLE001
        return None


class _LegacyWarmupHandle:
    """Job-handle shim over the legacy file lock (until hermes F2 lands)."""

    def __init__(self, file_lock: Any) -> None:
        self._file_lock = file_lock

    def heartbeat(self) -> None:
        """Advisory only on the legacy lock (the file mtime is the beat)."""

    def check(self) -> None:
        """Raise :class:`JobLockLost` when the lock is gone or retokened."""
        from src.watchlist.service import _read_lock_file

        token = getattr(self._file_lock, "_token", None)
        try:
            info = _read_lock_file(self._file_lock._path)
        except Exception:  # noqa: BLE001
            info = None
        if token is None or info != token:
            raise JobLockLost("job lock lost (deleted or taken over)")

    def release(self) -> None:
        self._file_lock.release()


def scheduler_log_path() -> Path:
    """Absolute path of the JSON-lines scheduler event log."""
    from src.config.paths import get_runtime_root

    return get_runtime_root() / "watchlist" / "scheduler.log"


def fire_inflight_path() -> Path:
    """Absolute path of the F6 fire-in-progress marker file."""
    from src.config.paths import get_runtime_root

    return get_runtime_root() / "watchlist" / _FIRE_INFLIGHT_FILENAME


def _require_home(context: str) -> Any:
    """Home pin guard (F1); fallback until pi's home module lands."""
    try:
        from src.watchlist.home import require_home as _pi_require_home
    except ImportError:
        from src.config.paths import get_runtime_root

        return get_runtime_root()
    return _pi_require_home(context)


def _home_fields_safe() -> dict[str, Any]:
    """``{"home": str, "pid": int}`` for scheduler events; never raises."""
    try:
        from src.watchlist.home import home_fields as _pi_home_fields

        return dict(_pi_home_fields())
    except Exception:  # noqa: BLE001 — fallback until home.py lands
        pass
    try:
        from src.config.paths import get_runtime_root

        return {"home": str(get_runtime_root()), "pid": os.getpid()}
    except Exception:  # noqa: BLE001
        return {"home": "unknown", "pid": os.getpid()}


def _is_home_mismatch(exc: BaseException) -> bool:
    """True for pi's HomeMismatch (matched by name until home.py lands)."""
    if type(exc).__name__ == "HomeMismatch":
        return True
    text = str(exc).lower()
    return "pinned home" in text and "home" in text


def _rotate_log(path: Path) -> None:
    """Keep roughly the last half of an oversized log (best-effort)."""
    try:
        if path.stat().st_size <= _MAX_LOG_BYTES:
            return
        data = path.read_bytes()
        tail = data[-_TRIM_BYTES:]
        nl = tail.find(b"\n")
        if nl != -1:
            tail = tail[nl + 1:]
        path.write_bytes(tail)
    except OSError:  # pragma: no cover — best-effort
        pass


def log_scheduler_event(event: str, **fields: Any) -> None:
    """Append one JSON line to scheduler.log and the module logger (INFO).

    Best-effort: never raises; a broken log target must not affect scanning.
    """
    record = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    line = json.dumps(record, ensure_ascii=False)
    logger.info("watchlist scheduler %s", line)
    try:
        path = scheduler_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            _rotate_log(path)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except OSError:
        logger.debug("watchlist scheduler: event log write failed", exc_info=True)


class WatchlistScheduler:
    """Single asyncio task driving the scheduled watchlist scan.

    All clock access goes through ``now_fn`` (returns a naive or aware
    datetime interpreted in the schedule timezone) and waiting goes through
    ``sleep_fn`` — both injectable for deterministic tests.
    """

    def __init__(
        self,
        service: Any,
        *,
        calendar_loader: Callable | None = None,
        now_fn: Callable[[], _dt.datetime] | None = None,
        sleep_fn: Callable[[float], Any] | None = None,
        poll_interval_s: float = _LOOP_SLEEP_S,
        enable_warmup: bool | None = None,
        universe_fn: Callable[[], list[str]] | None = None,
        warmup_runner: Callable[..., Any] | None = None,
        shortfall_fn: Callable[..., Any] | None = None,
        warmup_check_s: float = _WARMUP_CHECK_S,
        warmup_wait_cap_s: float = 30 * 60,
        warmup_rate_per_sec: float = _WARMUP_RATE_PER_SEC,
        instance_id_fn: Callable[[], str | None] | None = None,
    ) -> None:
        self._service = service
        self._calendar_loader = calendar_loader or _default_calendar_loader
        self._now_fn = now_fn or _default_now_fn
        self._sleep_fn = sleep_fn
        self._poll_interval_s = poll_interval_s
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._last_fired_at: str | None = None
        self._last_result: str | None = None  # "ok" | "skipped" | error text
        self._running = False
        # F3/F6 identity: one id per scheduler instance, logged everywhere.
        # ``instance_id_fn`` prefers the server instance id (hermes F3);
        # without it each scheduler keeps its own fallback id.
        self._instance_id = uuid.uuid4().hex[:8]
        self._instance_id_fn = instance_id_fn or _default_instance_id_fn
        # F6 in-flight run (set at fire, cleared at fire_result).
        self._inflight: dict[str, Any] | None = None
        # F5 warm-up machinery. enable_warmup=None auto-enables when the
        # service offers hermes's acquire_job (F2); tests pass explicit
        # doubles or False. Everything else is injectable for offline tests.
        if enable_warmup is None:
            enable_warmup = hasattr(service, "acquire_job")
        self._enable_warmup = bool(enable_warmup)
        self._universe_fn = universe_fn
        self._warmup_runner = warmup_runner
        self._shortfall_fn = shortfall_fn
        self._warmup_check_s = warmup_check_s
        self._warmup_wait_cap_s = warmup_wait_cap_s
        self._warmup_rate_per_sec = warmup_rate_per_sec
        self._warmup_thread: threading.Thread | None = None
        self._warmup_guard = threading.Lock()
        self._warmup_stop = threading.Event()
        self._last_warmup_check: _dt.datetime | None = None
        # In-memory attempt bookkeeping per trade date (process restart
        # resets it; the max-2 rule still holds within a process lifetime).
        self._attempts: dict[str, int] = {}
        self._last_attempt_at: dict[str, _dt.datetime] = {}
        # Last logged decision key (event, trade_date, reason/attempt) —        # suppresses identical per-tick skip log lines (idle-loop spam guard).
        self._last_logged_decision: tuple | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            _require_home("watchlist scheduler start")
        except Exception as exc:  # noqa: BLE001
            if _is_home_mismatch(exc):
                logger.error("watchlist scheduler refused to start: %s", exc)
                log_scheduler_event(
                    "scheduler_start_refused", error=str(exc)[:300],
                    **_home_fields_safe(),
                    instance=self._scheduler_instance_id(),
                )
                return
            raise
        # F6: an in-flight marker left by a killed run becomes fire_aborted
        # BEFORE anything else (including scheduler_started).
        try:
            pending = self._read_inflight()
        except (OSError, ValueError):
            pending = None
        if pending is not None:
            log_scheduler_event(
                "fire_aborted", reason="server stopped during run",
                **pending,
            )
            try:
                fire_inflight_path().unlink(missing_ok=True)
            except OSError:
                logger.debug("watchlist scheduler: inflight unlink failed",
                             exc_info=True)
        self._stopping = False
        self._warmup_stop.clear()
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._run(), name="watchlist-scheduler")
        log_scheduler_event("scheduler_started", **self._started_fields())

    async def stop(self) -> None:
        fields: dict[str, Any] = dict(self._inflight or {})
        if fields:
            log_scheduler_event("scheduler_stopped", **fields)
        else:
            log_scheduler_event("scheduler_stopped")
        self._stopping = True
        self._warmup_stop.set()
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _sleep(self, seconds: float) -> None:
        if self._sleep_fn is not None:
            result = self._sleep_fn(seconds)
            if hasattr(result, "__await__"):
                await result
            return
        await asyncio.sleep(seconds)

    def running(self) -> bool:
        return self._running

    # -- scheduling logic ---------------------------------------------------
    def _settings(self) -> dict:
        try:
            return self._service.get_settings().get("schedule", {})
        except Exception as exc:  # noqa: BLE001 — never crash on settings
            logger.warning("watchlist scheduler: get_settings failed: %s", exc)
            return {}

    def _tz(self, name: str | None) -> _dt.tzinfo:
        try:
            return ZoneInfo(name or _DEFAULT_TZ)
        except Exception:  # noqa: BLE001 — bad tz name falls back
            return ZoneInfo(_DEFAULT_TZ)

    def _now(self, tz: _dt.tzinfo) -> _dt.datetime:
        now = self._now_fn()
        if now.tzinfo is None:
            return now.replace(tzinfo=tz)
        return now.astimezone(tz)

    def _calendar(self):
        try:
            result = self._calendar_loader()
            return result[0] if isinstance(result, tuple) else result
        except Exception as exc:  # noqa: BLE001
            logger.warning("watchlist scheduler: calendar load failed: %s", exc)
            return None

    def _is_trading_day(self, day: _dt.date, cal) -> bool:
        if cal is not None:
            return cal.is_trading_day(day)
        return day.weekday() < 5

    def _latest_completed_trading_day(self, now: _dt.datetime, cal) -> _dt.date:
        if cal is not None:
            return cal.latest_completed_trading_day(now)
        day = now.date()
        if day.weekday() < 5 and now.time() >= _dt.time(15, 0):
            return day
        day -= _dt.timedelta(days=1)
        while day.weekday() >= 5:
            day -= _dt.timedelta(days=1)
        return day

    def _parse_hhmm(self, hhmm: str) -> _dt.time | None:
        try:
            hh, mm = hhmm.split(":")
            return _dt.time(int(hh), int(mm))
        except (ValueError, AttributeError):
            return None

    def _fire_time_passed(self, now: _dt.datetime, hhmm: str) -> bool:
        fire = self._parse_hhmm(hhmm)
        if fire is None:
            return False
        return now.time() >= fire

    def _in_trading_block(self, now: _dt.datetime, cal) -> bool:
        """True during trading hours 09:15–5:00 on a trading day."""
        return (
            self._is_trading_day(now.date(), cal)
            and _TRADING_BLOCK_START <= now.time() < _TRADING_BLOCK_END
        )

    def _next_run_at(self, now: _dt.datetime, hhmm: str, cal) -> str | None:
        fire = self._parse_hhmm(hhmm)
        if fire is None:
            return None
        day = now.date()
        if not self._is_trading_day(day, cal) or now.time() >= fire:
            day += _dt.timedelta(days=1)
            while not self._is_trading_day(day, cal):
                day += _dt.timedelta(days=1)
        return _dt.datetime.combine(day, fire, tzinfo=now.tzinfo).isoformat()

    def _has_scheduled_success_for(self, trade_date: _dt.date) -> bool:
        """True when a SCHEDULED ok/partial run exists for trade_date.

        Manual/CLI/API runs never suppress the daily fire (TODO R2 item B).
        AI-review failure alone is still a pipeline success (status ok) —        recovery is re-run-AI-review, never a rescan.
        """
        try:
            for run in self._service.list_runs(limit=50):
                if (
                    str(run.get("trade_date")) == trade_date.isoformat()
                    and run.get("trigger") == "schedule"
                    and run.get("status") in _SUCCESS_STATUSES
                ):
                    return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("watchlist scheduler: list_runs failed: %s", exc)
        return False

    def _skip_reason(self, trade_date: _dt.date, now: _dt.datetime) -> str | None:
        """Why an attempt is skipped, or None when it may run."""
        key = trade_date.isoformat()
        attempts = self._attempts.get(key, 0)
        if attempts >= _MAX_ATTEMPTS_PER_DAY:
            return "max attempts reached"
        if attempts >= 1:
            last = self._last_attempt_at.get(key)
            if last is not None and (now - last).total_seconds() < _RETRY_DELAY_S:
                return "retry delay not elapsed"
        return None

    def _may_attempt(self, trade_date: _dt.date, now: _dt.datetime) -> bool:
        """One retry after 30 min, max 2 attempts per trade date."""
        return self._skip_reason(trade_date, now) is None

    def _record_attempt(self, trade_date: _dt.date, now: _dt.datetime) -> None:
        key = trade_date.isoformat()
        self._attempts[key] = self._attempts.get(key, 0) + 1
        self._last_attempt_at[key] = now

    def _due_trade_date(
        self, now: _dt.datetime, cal, hhmm: str
    ) -> _dt.date | None:
        """Trade date the scheduler should fire for right now, or None.

        Never during trading hours (09:15–5:00 on a trading day). Before
        09:15 the previous completed session may be due (catch-up). After
        15:00 on a trading day, today is due only once today's fire time
        passed; on non-trading days the latest completed session is due
        once the daily fire time passed.
        """
        if self._in_trading_block(now, cal):
            return None
        fire = self._parse_hhmm(hhmm)
        if fire is None:
            return None
        if now.time() < _TRADING_BLOCK_START:
            return self._latest_completed_trading_day(now, cal)
        if self._is_trading_day(now.date(), cal):
            return now.date() if now.time() >= fire else None
        if now.time() >= fire:
            return self._latest_completed_trading_day(now, cal)
        return None

    def _due_with_reason(
        self, now: _dt.datetime, cal, hhmm: str, schedule: dict
    ) -> tuple[_dt.date | None, str]:
        """Like ``_due_trade_date``, but also why it is not due. For logging."""
        if not schedule.get("enabled"):
            return None, "schedule disabled"
        if self._in_trading_block(now, cal):
            return None, "trading hours"
        if self._parse_hhmm(hhmm) is None:
            return None, f"invalid fire time {hhmm!r}"
        target = self._due_trade_date(now, cal, hhmm)
        if target is not None:
            return target, "due"
        return None, "before fire time"

    def _log_decision_once(self, key: tuple, event: str, **fields: Any) -> None:
        """Log a per-tick decision once per distinct key (anti-spam)."""
        if key == self._last_logged_decision:
            return
        self._last_logged_decision = key
        log_scheduler_event(event, **fields)

    def _scheduler_instance_id(self) -> str:
        """F3 instance id for log fields: the server id when present.

        ``src.api.watchlist_routes.server_instance_id()`` is set when the
        lifespan takes the server lock; before that (or outside a server)
        each scheduler keeps its own fallback id.
        """
        try:
            ident = self._instance_id_fn()
        except Exception:  # noqa: BLE001
            ident = None
        return ident or self._instance_id

    def _started_fields(self) -> dict:
        """Settings snapshot logged with scheduler_started (best-effort)."""
        schedule = self._settings()
        fields: dict[str, Any] = {
            "enabled": bool(schedule.get("enabled")),
            "time": str(schedule.get("time", "18:00")),
            "timezone": str(schedule.get("timezone", _DEFAULT_TZ)),
            "next_run_at": None,
            **_home_fields_safe(),
            "instance": self._scheduler_instance_id(),
        }
        try:
            if fields["enabled"]:
                tz = self._tz(fields["timezone"])
                now = self._now(tz)
                fields["next_run_at"] = self._next_run_at(
                    now, fields["time"], self._calendar()
                )
        except Exception:  # noqa: BLE001 — evidence must never block start
            fields["next_run_at"] = None
        return fields

    # -- F6 fire-in-progress marker --------------------------------------
    def _read_inflight(self) -> dict[str, Any] | None:
        """The persisted in-flight fire, or None. Raises OSError on IO."""
        raw = fire_inflight_path().read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None

    def _write_inflight(self, record: dict[str, Any]) -> None:
        """Persist the in-flight fire (best-effort; never raises)."""
        try:
            path = fire_inflight_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False),
                            encoding="utf-8")
        except OSError:
            logger.debug("watchlist scheduler: inflight write failed",
                         exc_info=True)

    def _clear_inflight(self) -> None:
        """Remove the in-flight marker (best-effort; never raises)."""
        try:
            fire_inflight_path().unlink(missing_ok=True)
        except OSError:
            logger.debug("watchlist scheduler: inflight clear failed",
                         exc_info=True)

    def _run_scan(self, trade_date: _dt.date | None, now: _dt.datetime,
                  attempt: int | None = None) -> None:
        """Run one scan via the service, recording outcome (never raises).

        ok/partial = pipeline success (even when the AI review alone
        failed); status failed or an exception = a failed attempt eligible
        for the single retry; ScanAlreadyRunning = skipped, not an attempt.
        """
        self._running = True
        failed_attempt = False
        trade_date_iso = trade_date.isoformat() if trade_date is not None else None
        t0 = _time.monotonic()
        base_fields: dict[str, Any] = {
            **_home_fields_safe(),
            "instance": self._scheduler_instance_id(),
        }
        if attempt is not None or trade_date_iso is not None:
            self._inflight = {
                "trade_date": trade_date_iso,
                "attempt": attempt,
                "pid": os.getpid(),
                "instance": self._scheduler_instance_id(),
                "started_at": _dt.datetime.now(
                    _dt.timezone.utc).isoformat(),
            }
            self._write_inflight(dict(self._inflight))
        try:
            result = self._service.run_scan(trigger="schedule")
            status = (result or {}).get("status")
            duration_s = (result or {}).get("duration_s")
            if duration_s is None:
                duration_s = round(_time.monotonic() - t0, 3)
            result_fields: dict[str, Any] = {
                "trade_date": trade_date_iso,
                "run_id": (result or {}).get("run_id"),
                "status": status,
                "ai_status": (result or {}).get("ai_status"),
                "duration_s": duration_s,
                **base_fields,
            }
            if (result or {}).get("failure_reason"):
                result_fields["failure_reason"] = result["failure_reason"]
            log_scheduler_event("fire_result", **result_fields)
            if status in _SUCCESS_STATUSES:
                self._last_result = "ok"
            else:
                failed_attempt = True
                self._last_result = f"failed: status={status or 'unknown'}"
        except Exception as exc:  # noqa: BLE001 — contract: never raise
            duration_s = round(_time.monotonic() - t0, 3)
            if type(exc).__name__ == "ScanAlreadyRunning":
                logger.info("watchlist scheduler: scan already running")
                self._last_result = "skipped: scan already running"
                self._log_decision_once(
                    ("skipped", trade_date_iso, "scan already running"),
                    "skipped",
                    trade_date=trade_date_iso,
                    reason="scan already running",
                    **base_fields,
                )
            elif type(exc) is JobLockLost or type(exc).__name__ == "JobLockLost":
                # F2: the job lock vanished mid-run. The service already
                # persisted the run as failed; the scheduler logs the same
                # outcome as fire_result so every fire has a result.
                reason = (str(exc)
                          or "job lock lost (deleted or taken over) - aborted")
                logger.warning("watchlist scheduler: %s", reason)
                self._last_result = f"failed: {reason}"[:300]
                failed_attempt = True
                log_scheduler_event(
                    "fire_result",
                    trade_date=trade_date_iso,
                    run_id=None,
                    status="failed",
                    ai_status=None,
                    duration_s=duration_s,
                    failure_reason=reason[:300],
                    error=f"{type(exc).__name__}: {exc}"[:300],
                    **base_fields,
                )
            else:
                logger.error("watchlist scheduler scan failed", exc_info=True)
                self._last_result = f"{type(exc).__name__}: {exc}"[:300]
                failed_attempt = True
                log_scheduler_event(
                    "fire_result",
                    trade_date=trade_date_iso,
                    run_id=None,
                    status=None,
                    ai_status=None,
                    duration_s=duration_s,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                    **base_fields,
                )
        finally:
            self._running = False
            self._last_fired_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
            self._inflight = None
            self._clear_inflight()
        if failed_attempt and trade_date is not None:
            self._record_attempt(trade_date, now)

    # -- main loop ----------------------------------------------------------
    async def _run(self) -> None:
        # Catch-up on start (once): same decision as a tick (TODO R2 item B).
        try:
            await self._catch_up_async()
        except Exception:  # noqa: BLE001 — never kill the loop
            logger.error("watchlist scheduler catch-up failed", exc_info=True)
        while not self._stopping:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — never kill the loop
                logger.error("watchlist scheduler tick failed", exc_info=True)
            if self._stopping:
                break
            await self._sleep(self._poll_interval_s)

    async def _catch_up_async(self) -> None:
        # Identical decision as a tick: same success rule, same trading-hours
        # guard, same retry bookkeeping (TODO R2 item B). The decision itself
        # is logged once as catch-up evidence for the day.
        try:
            schedule = self._settings()
            tz = self._tz(schedule.get("timezone"))
            now = self._now(tz)
            cal = self._calendar()
            hhmm = str(schedule.get("time", "18:00"))
            target, reason = self._due_with_reason(now, cal, hhmm, schedule)
        except Exception:  # noqa: BLE001
            logger.error("watchlist scheduler catch-up failed", exc_info=True)
            target, reason = None, "catch-up evaluation failed"
        if target is None:
            log_scheduler_event("catch_up_decision", reason=reason)
        else:
            log_scheduler_event(
                "catch_up_decision",
                trade_date=target.isoformat(),
                reason="due",
            )
            # F5: right after a (re)start with a due session, start warming
            # the cache when the shortfall exceeds one scan budget.
            try:
                if self._auto_warmup_enabled():
                    await asyncio.to_thread(
                        self._assess_and_maybe_warmup, target)
            except Exception:  # noqa: BLE001 — warmup never blocks catch-up
                logger.debug("watchlist scheduler: catch-up warmup failed",
                             exc_info=True)
        await self._tick()

    async def _tick(self) -> None:
        schedule = self._settings()
        if not schedule.get("enabled"):
            return  # disabled makes no decisions -> no log lines
        tz = self._tz(schedule.get("timezone"))
        now = self._now(tz)
        cal = self._calendar()
        hhmm = str(schedule.get("time", "18:00"))
        # F5: hourly shortfall reassessment on trading days (background).
        if self._auto_warmup_enabled():
            try:
                last = self._last_warmup_check
                if last is None or (now - last).total_seconds() >= self._warmup_check_s:
                    self._last_warmup_check = now
                    if self._is_trading_day(now.date(), cal):
                        latest = self._latest_completed_trading_day(now, cal)
                        await asyncio.to_thread(
                            self._assess_and_maybe_warmup, latest)
            except Exception:  # noqa: BLE001 — warmup never breaks a tick
                logger.debug("watchlist scheduler: periodic warmup failed",
                             exc_info=True)
        target = self._due_trade_date(now, cal, hhmm)
        if target is None:
            # Idle ticks are silent; only decisions at/after fire time log.
            if self._fire_time_passed(now, hhmm):
                _due, reason = self._due_with_reason(now, cal, hhmm, schedule)
                self._log_decision_once(
                    ("skipped", None, reason), "skipped", reason=reason
                )
            return
        if self._has_scheduled_success_for(target):
            self._log_decision_once(
                ("skipped", target.isoformat(), "already succeeded"),
                "skipped",
                trade_date=target.isoformat(),
                reason="already succeeded",
            )
            return
        skip = self._skip_reason(target, now)
        if skip is not None:
            self._log_decision_once(
                ("skipped", target.isoformat(), skip),
                "skipped",
                trade_date=target.isoformat(),
                reason=skip,
            )
            return
        attempt = self._attempts.get(target.isoformat(), 0) + 1
        self._last_logged_decision = ("fire", target.isoformat(), attempt)
        # F5: a running warm-up finishes first (fire + 60 min max), then scan.
        await self._wait_for_warmup(target, hhmm, tz, now)
        log_scheduler_event(
            "fire", trade_date=target.isoformat(), attempt=attempt,
            **_home_fields_safe(), instance=self._scheduler_instance_id(),
        )
        await asyncio.to_thread(self._run_scan, target, now, attempt)

    # -- F5 warm-up ------------------------------------------------------
    def _auto_warmup_enabled(self) -> bool:
        """Warm-up machinery on/off (auto-detects hermes's acquire_job)."""
        return bool(self._enable_warmup)

    def _warmup_fits_budget(self) -> float:
        """Scan kline budget (s): a bigger shortfall triggers warm-up."""
        try:
            settings = self._service.get_settings()
            scan = (settings or {}).get("scan") or {}
            return float(scan.get("kline_time_budget_s", 600.0))
        except Exception:  # noqa: BLE001
            return 600.0

    def _universe_symbols(self) -> list[str]:
        """Prefixed symbols to warm; empty on any failure (offline-safe)."""
        if self._universe_fn is not None:
            try:
                symbols = self._universe_fn()
                return [str(s) for s in (symbols or []) if s]
            except Exception:  # noqa: BLE001
                logger.debug("watchlist scheduler: universe_fn failed",
                             exc_info=True)
                return []
        try:
            from src.market_scanner.sources.universe import fetch_universe

            frame, _status = fetch_universe()
            if frame is None or getattr(frame, "empty", True):
                return []
            if "symbol" in getattr(frame, "columns", []):
                return [str(s) for s in frame["symbol"].tolist() if s]
            from src.market_scanner.models import symbol_from_code

            return [symbol_from_code(code) for code in frame.index]
        except Exception:  # noqa: BLE001 — warm-up is best-effort
            logger.debug("watchlist scheduler: universe fetch failed",
                         exc_info=True)
            return []

    def _default_shortfall(self, symbols: list[str],
                           trade_date: _dt.date) -> Any:
        """assessshortfall over the default home cache (no network)."""
        from src.market_scanner.sources.kline import assess_shortfall
        from src.market_scanner.sources.kline_cache import KlineCache

        return assess_shortfall(symbols, trade_date,
                                calendar=self._calendar(),
                                cache=KlineCache())

    def _assess_and_maybe_warmup(self, trade_date: _dt.date) -> bool:
        """Start a warm-up thread when the shortfall exceeds the budget.

        Returns True when a warm-up is running afterwards. Never raises:
        every dependency is guarded so a warm-up problem can never kill a
        tick or a catch-up.
        """
        if not self._auto_warmup_enabled():
            return False
        with self._warmup_guard:
            if (self._warmup_thread is not None
                    and self._warmup_thread.is_alive()):
                return True
        try:
            symbols = self._universe_symbols()
            if not symbols:
                return False
            shortfall_fn = self._shortfall_fn or self._default_shortfall
            short = shortfall_fn(symbols, trade_date)
            est_requests = int(getattr(short, "est_requests", 0) or 0)
            est_seconds = float(getattr(short, "est_seconds", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            logger.debug("watchlist scheduler: shortfall assess failed",
                         exc_info=True)
            return False
        if est_seconds <= self._warmup_fits_budget() or est_requests <= 0:
            return False
        with self._warmup_guard:
            if (self._warmup_thread is not None
                    and self._warmup_thread.is_alive()):
                return True
            self._warmup_stop.clear()
            thread = threading.Thread(
                target=self._warmup_body,
                args=(symbols, trade_date, est_requests, est_seconds),
                name="watchlist-warmup",
                daemon=True,
            )
            self._warmup_thread = thread
            thread.start()
            return True

    def _acquire_warmup_lock(self) -> Any:
        """Job lock kind="warmup": hermes's acquire_job, else a legacy file
        lock with the same JobLockLost-on-loss semantics."""
        acquire = getattr(self._service, "acquire_job", None)
        if callable(acquire):
            return acquire("warmup", owner=f"scheduler:{self._scheduler_instance_id()}")
        from src.watchlist.service import (
            CrossProcessScanLock,
            ScanAlreadyRunning,
            _read_lock_file,
        )

        file_lock = CrossProcessScanLock()
        if not file_lock.acquire():
            holder = _read_lock_file(file_lock._path) or {}
            raise ScanAlreadyRunning(
                "a warm-up is already running in another process"
                + (f" (pid {holder.get('pid')})" if holder.get("pid") else "")
            )
        return _LegacyWarmupHandle(file_lock)

    def _default_warmup_runner(self, symbols: list[str],
                               trade_date: _dt.date, *,
                               rate_per_sec: float,
                               progress: Callable[[int, int], None] | None,
                               lock_check: Callable[[], None] | None,
                               stop: Any) -> Any:
        """Production warm-up: sequential engine fill at <= 1 req/s."""
        from src.market_scanner.sources.kline import warm_cache

        return warm_cache(
            symbols, trade_date,
            rate_per_sec=min(float(rate_per_sec or 1.0), 1.0),
            progress=progress, lock_check=lock_check, stop=stop)

    def _warmup_body(self, symbols: list[str], trade_date: _dt.date,
                     est_requests: int, est_seconds: float) -> None:
        """Background warm-up: lock, fill, log, release (never raises)."""
        base = {**_home_fields_safe(), "instance": self._scheduler_instance_id()}
        try:
            _require_home("watchlist warmup")
        except Exception as exc:  # noqa: BLE001
            if _is_home_mismatch(exc):
                logger.error("watchlist warmup refused: %s", exc)
                log_scheduler_event(
                    "warmup_finished", trade_date=trade_date.isoformat(),
                    error=str(exc)[:300], **base)
                with self._warmup_guard:
                    if self._warmup_thread is threading.current_thread():
                        self._warmup_thread = None
                return
            logger.debug("watchlist warmup home check failed", exc_info=True)
        try:
            handle = self._acquire_warmup_lock()
        except Exception as exc:  # noqa: BLE001 — busy lock: skip quietly
            logger.info("watchlist warmup skipped (job lock busy): %s", exc)
            with self._warmup_guard:
                if self._warmup_thread is threading.current_thread():
                    self._warmup_thread = None
            return
        log_scheduler_event(
            "warmup_started", trade_date=trade_date.isoformat(),
            est_requests=est_requests, est_seconds=round(est_seconds, 1),
            symbols=len(symbols), **base)
        check_at = max(1, len(symbols) // _WARMUP_PROGRESS_EVERY or 1)

        def _progress(done: int, total: int) -> None:
            try:
                handle.heartbeat()
            except Exception:  # noqa: BLE001 — heartbeat is advisory
                pass
            if total and (done >= total or done % check_at == 0):
                log_scheduler_event(
                    "warmup_progress", trade_date=trade_date.isoformat(),
                    done=done, total=total, **base)

        try:
            runner = self._warmup_runner or self._default_warmup_runner
            finished = runner(
                symbols, trade_date,
                rate_per_sec=self._warmup_rate_per_sec,
                progress=_progress,
                lock_check=handle.check, stop=self._warmup_stop)
            detail = (finished.to_dict() if hasattr(finished, "to_dict")
                      else dict(finished or {}))
            log_scheduler_event(
                "warmup_finished", trade_date=trade_date.isoformat(), **detail,
                **base)
        except Exception as exc:  # noqa: BLE001 — never kill the scheduler
            logger.error("watchlist warmup failed", exc_info=True)
            log_scheduler_event(
                "warmup_finished", trade_date=trade_date.isoformat(),
                error=f"{type(exc).__name__}: {exc}"[:300], **base)
        finally:
            try:
                handle.release()
            except Exception:  # noqa: BLE001
                pass
            with self._warmup_guard:
                if self._warmup_thread is threading.current_thread():
                    self._warmup_thread = None

    async def _wait_for_warmup(self, trade_date: _dt.date, hhmm: str,
                               tz: _dt.tzinfo, now: _dt.datetime) -> None:
        """Before a fire, wait for a running warm-up (fire + 60 min max).

        Logs one ``warmup_wait`` event with the outcome. Never raises.
        """
        thread = self._warmup_thread
        if thread is None or not thread.is_alive():
            return
        try:
            if now.tzinfo is None:
                now = now.replace(tzinfo=tz)
            fire = self._parse_hhmm(hhmm)
            fire_dt = (_dt.datetime.combine(trade_date, fire, tzinfo=tz)
                       if fire is not None else now)
            deadline = fire_dt + _dt.timedelta(seconds=self._warmup_wait_cap_s)
            wait_s = max(0.0, (deadline - now).total_seconds())
            t0 = _time.monotonic()
            if wait_s > 0:
                await asyncio.to_thread(thread.join, wait_s)
            waited = round(_time.monotonic() - t0, 3)
            outcome = ("finished" if not thread.is_alive() else "timeout")
            log_scheduler_event(
                "warmup_wait", trade_date=trade_date.isoformat(),
                waited_s=waited, outcome=outcome,
                **_home_fields_safe(), instance=self._scheduler_instance_id())
        except Exception:  # noqa: BLE001 — waiting must never block a fire
            logger.debug("watchlist scheduler: warmup wait failed",
                         exc_info=True)

    # -- introspection -------------------------------------------------------
    def status(self) -> dict:
        """{enabled, time, timezone, next_run_at, attempts_today,
        last_fired_at, last_result, running}."""
        schedule = self._settings()
        enabled = bool(schedule.get("enabled"))
        hhmm = str(schedule.get("time", "18:00"))
        tzname = str(schedule.get("timezone", _DEFAULT_TZ))
        next_run_at = None
        attempts_today = 0
        if enabled:
            try:
                tz = self._tz(tzname)
                now = self._now(tz)
                cal = self._calendar()
                next_run_at = self._next_run_at(now, hhmm, cal)
                try:
                    target = self._due_trade_date(now, cal, hhmm)
                    if target is None:
                        target = self._latest_completed_trading_day(now, cal)
                    attempts_today = self._attempts.get(target.isoformat(), 0)
                except Exception:  # noqa: BLE001 — attempts are best-effort
                    attempts_today = 0
            except Exception:  # noqa: BLE001 — status must never raise
                next_run_at = None
        return {
            "enabled": enabled,
            "time": hhmm,
            "timezone": tzname,
            "next_run_at": next_run_at,
            "attempts_today": attempts_today,
            "last_fired_at": self._last_fired_at,
            "last_result": self._last_result,
            "running": self._running or self._service.is_scanning(),
        }
