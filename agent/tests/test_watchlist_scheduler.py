"""Offline unit tests for the watchlist scheduler (CONTRACT section 11,
TODO R2 item B daily-fire rule).

A fake service and a fake calendar drive a fake clock; sleep_fn records
instead of waiting. No network, no real asyncio sleeps.
"""

from __future__ import annotations
import asyncio
import datetime as _dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.watchlist.scheduler import WatchlistScheduler  # noqa: E402
from src.watchlist.service import ScanAlreadyRunning  # noqa: E402


def _asyncio_run(coro):
    return asyncio.run(coro)


def _run_tick(sched):
    """Run one scheduler tick to completion."""
    _asyncio_run(sched._tick())


class FakeClock:
    """Callable clock returning a fixed datetime (mutable between ticks)."""

    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class FakeService:
    def __init__(
        self,
        schedule=None,
        runs=None,
        fail=False,
        already_running=False,
        result_status="ok",
        trade_date="2026-10-12",
    ):
        self.schedule = {
            "enabled": False,
            "time": "18:00",
            "timezone": "Asia/Shanghai",
            "catch_up": True,
        }
        if schedule:
            self.schedule.update(schedule)
        self.runs = runs or []
        self.calls = []
        self.scans = []
        self.invocations = []  # every run_scan entry, even failing ones
        self.fail = fail
        self.already_running = already_running
        self.result_status = result_status
        self.trade_date = trade_date

    def get_settings(self):
        return {"scan": {}, "schedule": dict(self.schedule)}

    def run_scan(self, *, trigger, config_overrides=None, progress=None, llm=None):
        self.invocations.append(trigger)
        if self.already_running:
            raise ScanAlreadyRunning("already running")
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append(trigger)
        run = {
            "trigger": trigger,
            "run_id": f"r{len(self.scans) + 1}",
            "status": self.result_status,
            "trade_date": self.trade_date,
        }
        self.scans.append(run)
        self.runs.append(run)  # visible to list_runs immediately
        return run

    def is_scanning(self):
        return False

    def list_runs(self, limit=20):
        return self.runs


class FakeCalendar:
    def __init__(self, trading_days):
        self.trading_days = frozenset(trading_days)

    def is_trading_day(self, d):
        return d.isoformat() in self.trading_days

    def previous_trading_day(self, d):
        from datetime import timedelta

        d -= timedelta(days=1)
        while d.isoformat() not in self.trading_days:
            d -= timedelta(days=1)
        return d

    def latest_completed_trading_day(self, now):
        day = now.date()
        if self.is_trading_day(day) and now.time() >= _dt.time(15, 0):
            return day
        return self.previous_trading_day(day)


def _make_scheduler(svc, cal, now, sleeps=None):
    sched = WatchlistScheduler(
        svc,
        calendar_loader=lambda: cal,
        now_fn=now,
        sleep_fn=lambda s: sleeps.append(s) if sleeps is not None else None,
        poll_interval_s=0.01,
    )
    return sched


# ---------------------------------------------------------------------------
# Daily fire timing
# ---------------------------------------------------------------------------
def test_fires_at_or_after_time_on_trading_day_only() -> None:
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    cal = FakeCalendar({"2026-10-09", "2026-10-12", "2026-10-13"})
    # Saturday mid-day (non-trading, before fire time) -> no fire
    sched = _make_scheduler(
        svc, cal, FakeClock(_dt.datetime(2026, 10, 10, 10, 0)))
    _asyncio_run(sched._tick())
    assert svc.calls == []
    # Monday 17:00 -> trading day but before fire time -> no fire
    sched2 = _make_scheduler(
        svc, cal, FakeClock(_dt.datetime(2026, 10, 12, 17, 0)))
    _asyncio_run(sched2._tick())
    assert svc.calls == []
    # Monday 18:05 -> fires once
    sched3 = _make_scheduler(
        svc, cal, FakeClock(_dt.datetime(2026, 10, 12, 18, 5)))
    _asyncio_run(sched3._tick())
    assert len(svc.calls) == 1 and svc.calls[0] == "schedule"


def test_fires_on_trading_day_at_or_after_time():
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    cal = FakeCalendar({"2026-10-12", "2026-10-13"})  # Mon, Tue
    # Mon 17:59 -> not yet; Mon 18:00 -> fires
    sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 12, 17, 59))
    _run_tick(sched)
    assert svc.scans == []
    sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 12, 18, 0))
    _run_tick(sched)
    assert len(svc.scans) == 1 and svc.scans[0]["trigger"] == "schedule"


def test_skips_non_trading_days():
    """Non-trading day before the fire time never fires; weekend evenings
    only catch up a still-missed previous session (covered separately)."""
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    cal = FakeCalendar({"2026-10-12", "2026-10-13"})  # Sat 10-10/Sun 10-11 off
    sched = _make_scheduler(
        svc, cal, lambda: _dt.datetime(2026, 10, 10, 10, 0)
    )  # Saturday mid-day
    _run_tick(sched)
    assert svc.scans == []
    sched = _make_scheduler(
        svc, cal, lambda: _dt.datetime(2026, 10, 11, 10, 0)
    )  # Sunday mid-day
    _run_tick(sched)
    assert svc.scans == []


def test_weekend_catch_up_for_missed_friday_session():
    """A long-running scheduler catching up Friday's missed fire on Saturday
    evening (non-trading day, fire time passed, no scheduled success)."""
    svc = FakeService(
        schedule={"enabled": True, "time": "18:00"},
        trade_date="2026-10-09",
    )
    cal = FakeCalendar({"2026-10-09", "2026-10-12", "2026-10-13"})
    sched = _make_scheduler(
        svc, cal, lambda: _dt.datetime(2026, 10, 10, 18, 30)
    )  # Saturday evening
    _run_tick(sched)
    assert len(svc.scans) == 1 and svc.scans[0]["trigger"] == "schedule"


def test_disabled_never_fires():
    svc = FakeService(schedule={"enabled": False, "time": "00:00"})
    cal = FakeCalendar({"2026-10-12"})
    sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 12, 23, 0))
    _run_tick(sched)
    assert svc.scans == []
    assert sched.status()["enabled"] is False


# ---------------------------------------------------------------------------
# Success rule: scheduled ok|partial suppresses; manual never does
# ---------------------------------------------------------------------------
def test_no_refire_same_day_after_successful_scheduled_scan():
    svc = FakeService(
        schedule={"enabled": True, "time": "18:00"},
        runs=[{"trade_date": "2026-10-12", "status": "ok", "trigger": "schedule"}],
    )
    cal = FakeCalendar({"2026-10-12"})
    sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 12, 19, 0))
    _run_tick(sched)
    assert svc.scans == []  # already scanned for 2026-10-12


def test_scan_status_ok_partial_counts_as_success():
    """Pipeline persists 'ok'/'partial'; both stop the day's refire."""
    cal = FakeCalendar({"2026-10-12"})
    for status in ("ok", "partial"):
        svc = FakeService(
            schedule={"enabled": True, "time": "18:00"},
            runs=[{"trade_date": "2026-10-12", "status": status,
                   "trigger": "schedule"}],
        )
        sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 12, 19, 0))
        _run_tick(sched)
        assert svc.scans == [], f"refired after status={status}"


def test_manual_run_does_not_suppress_scheduled_fire():
    """Required (TODO R2 B): a manual scan at 15:05 never suppresses the
    18:00 scheduled fire — only trigger=='schedule' runs do."""
    svc = FakeService(
        schedule={"enabled": True, "time": "18:00"},
        runs=[{"trade_date": "2026-10-12", "status": "ok", "trigger": "manual"}],
    )
    cal = FakeCalendar({"2026-10-12"})
    sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 12, 18, 5))
    _run_tick(sched)
    assert len(svc.scans) == 1 and svc.scans[0]["trigger"] == "schedule"


def test_ai_only_failure_is_not_retried():
    """Required (TODO R2 B): run status ok with a failed AI review
    (ai_status not_reviewed/partial) is still success — never a rescan;
    recovery is re-run-AI-review."""
    cal = FakeCalendar({"2026-10-12"})
    for ai_status in ("not_reviewed", "partial"):
        svc = FakeService(
            schedule={"enabled": True, "time": "18:00"},
            runs=[{"trade_date": "2026-10-12", "status": "ok",
                   "trigger": "schedule", "ai_status": ai_status}],
        )
        sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 12, 19, 30))
        _run_tick(sched)
        assert svc.scans == [], f"rescanned after ai_status={ai_status}"


# ---------------------------------------------------------------------------
# Failure retry: once after 30 min, max 2 attempts per trade date
# ---------------------------------------------------------------------------
def test_failed_run_retried_once_after_30min_never_twice():
    """Required (TODO R2 B): exception -> retry once 30 min later, never a
    second retry, never re-fired every tick."""
    cal = FakeCalendar({"2026-10-12"})
    clock = FakeClock(_dt.datetime(2026, 10, 12, 18, 0))
    svc = FakeService(schedule={"enabled": True, "time": "18:00"}, fail=True)
    sched = _make_scheduler(svc, cal, clock)
    _run_tick(sched)  # attempt 1 fails
    assert svc.invocations == ["schedule"]
    assert sched._attempts.get("2026-10-12") == 1
    assert "RuntimeError" in (sched.status()["last_result"] or "")
    clock.value = _dt.datetime(2026, 10, 12, 18, 10)  # +10 min: too soon
    _run_tick(sched)
    assert svc.invocations == ["schedule"]  # no retry yet
    clock.value = _dt.datetime(2026, 10, 12, 18, 35)  # +35 min: retry once
    _run_tick(sched)
    assert svc.invocations == ["schedule", "schedule"]
    assert sched._attempts.get("2026-10-12") == 2
    clock.value = _dt.datetime(2026, 10, 12, 19, 30)  # later: maxed out
    _run_tick(sched)
    assert svc.invocations == ["schedule", "schedule"]  # never twice
    assert sched._attempts.get("2026-10-12") == 2


def test_failed_status_run_counts_as_attempt():
    """run_scan returning status 'failed' (no exception) is a failed
    attempt: retried once after 30 min, then maxed."""
    cal = FakeCalendar({"2026-10-12"})
    clock = FakeClock(_dt.datetime(2026, 10, 12, 18, 0))
    svc = FakeService(
        schedule={"enabled": True, "time": "18:00"}, result_status="failed")
    sched = _make_scheduler(svc, cal, clock)
    _run_tick(sched)
    assert sched._attempts.get("2026-10-12") == 1
    assert sched.status()["last_result"] == "failed: status=failed"
    clock.value = _dt.datetime(2026, 10, 12, 18, 40)
    svc.result_status = "ok"  # retry succeeds
    _run_tick(sched)
    assert len(svc.scans) == 2 and svc.scans[-1]["status"] == "ok"
    assert sched.status()["last_result"] == "ok"


# ---------------------------------------------------------------------------
# Catch-up on start: same rule, never during trading hours
# ---------------------------------------------------------------------------
def test_no_catch_up_at_10_on_trading_day():
    """Required (TODO R2 B): 10:00 on a trading day is inside 09:15–15:00 —
    the stale marker shows and the scheduler waits."""
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    cal = FakeCalendar({"2026-10-12", "2026-10-13"})
    sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 13, 10, 0))
    _asyncio_run(sched._catch_up_async())
    assert svc.scans == []


def test_catch_up_at_08_for_previous_session():
    """Required (TODO R2 B): before 09:15 the previous session may be
    caught up; a scheduled success for it suppresses the catch-up."""
    cal = FakeCalendar({"2026-10-12", "2026-10-13"})
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 13, 8, 0))
    _asyncio_run(sched._catch_up_async())
    assert len(svc.scans) == 1 and svc.scans[0]["trigger"] == "schedule"
    # second start: scheduled run recorded -> no second catch-up
    svc2 = FakeService(
        schedule={"enabled": True, "time": "18:00"},
        runs=[{"trade_date": "2026-10-12", "status": "ok", "trigger": "schedule"}],
    )
    sched2 = _make_scheduler(svc2, cal, lambda: _dt.datetime(2026, 10, 13, 8, 0))
    _asyncio_run(sched2._catch_up_async())
    assert svc2.scans == []


def test_catch_up_ignores_manual_run_for_previous_session():
    """Catch-up uses the same success rule: a manual run does not count."""
    svc = FakeService(
        schedule={"enabled": True, "time": "18:00"},
        runs=[{"trade_date": "2026-10-12", "status": "ok", "trigger": "manual"}],
    )
    cal = FakeCalendar({"2026-10-12", "2026-10-13"})
    sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 13, 8, 0))
    _asyncio_run(sched._catch_up_async())
    assert len(svc.scans) == 1


# ---------------------------------------------------------------------------
# Robustness + wiring
# ---------------------------------------------------------------------------
def test_already_scanning_is_recorded_not_raised():
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    svc.already_running = True
    cal = FakeCalendar({"2026-10-12"})
    sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 12, 18, 0))
    _run_tick(sched)  # must not raise
    assert "already running" in (sched.status()["last_result"] or "")
    assert sched._attempts.get("2026-10-12") is None  # skipped, not an attempt


def test_exception_does_not_kill_loop():
    svc = FakeService(schedule={"enabled": True, "time": "18:00"}, fail=True)
    cal = FakeCalendar({"2026-10-12"})
    sleeps = []
    sched = _make_scheduler(
        svc, cal, lambda: _dt.datetime(2026, 10, 12, 18, 0), sleeps=sleeps
    )

    async def _bounded_run():
        # mimic the loop but stop after 3 ticks (test never hangs)
        for _ in range(3):
            try:
                await sched._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # loop must keep going
            await sched._sleep(sched._poll_interval_s)

    _asyncio_run(_bounded_run())
    assert len(sleeps) >= 2
    assert "RuntimeError" in (sched.status()["last_result"] or "")


def test_status_shape():
    svc = FakeService(schedule={"enabled": True, "time": "18:30"})
    cal = FakeCalendar({"2026-10-12", "2026-10-13"})
    sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 12, 10, 0))
    st = sched.status()
    assert set(st) >= {
        "enabled",
        "time",
        "timezone",
        "next_run_at",
        "attempts_today",
        "last_fired_at",
        "last_result",
        "running",
    }
    assert st["enabled"] is True and st["time"] == "18:30"
    assert st["next_run_at"] and "2026-10-12" in st["next_run_at"]
    assert st["attempts_today"] == 0


def test_tick_runs_scan_in_thread():
    """_tick must not block the loop — it delegates to a thread."""
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    cal = FakeCalendar({"2026-10-12"})
    sched = _make_scheduler(svc, cal, lambda: _dt.datetime(2026, 10, 12, 18, 0))
    seen = {}

    real_to_thread = asyncio.to_thread

    async def spy(fn, *a, **kw):
        seen["threaded"] = True
        return await real_to_thread(fn, *a, **kw)

    orig = asyncio.to_thread
    asyncio.to_thread = spy
    try:
        _run_tick(sched)
    finally:
        asyncio.to_thread = orig
    assert seen.get("threaded") is True
    assert len(svc.scans) == 1


def test_scheduler_wired_like_api_server_one_scan_per_day():
    """api_server builds the scheduler with a real service instance; emulate
    that wiring exactly (same constructor path, real-ish service object with
    the pipeline's status vocabulary) and verify: tick 1 fires exactly once,
    tick 2 the same day fires not at all."""
    cal = FakeCalendar({"2026-10-12"})

    def now():
        return _dt.datetime(2026, 10, 12, 18, 5)

    class PipelineLikeService:
        """Mimics WatchlistService.get_settings/is_scanning/list_runs/run_scan
        with the store's real status vocabulary ('ok'/'partial')."""

        def __init__(self):
            self.settings = {
                "scan": {},
                "schedule": {
                    "enabled": True,
                    "time": "18:00",
                    "timezone": "Asia/Shanghai",
                    "catch_up": True,
                },
            }
            self.runs = []
            self.scan_count = 0

        def get_settings(self):
            return self.settings

        def is_scanning(self):
            return False

        def list_runs(self, limit=20):
            return self.runs

        def run_scan(self, *, trigger, config_overrides=None, progress=None, llm=None):
            self.scan_count += 1
            run = {
                "run_id": f"r{self.scan_count}",
                "trade_date": "2026-10-12",
                "status": "ok",
                "trigger": trigger,
            }
            self.runs.append(run)
            return run

    # EXACTLY like api_server.py: an instance (not a factory), poll 20s.
    svc = PipelineLikeService()
    sched = WatchlistScheduler(
        svc,
        calendar_loader=lambda: cal,
        now_fn=now,
        sleep_fn=lambda s: None,
        poll_interval_s=20.0,
    )
    _run_tick(sched)  # 18:05, after fire time, trading day, no scan yet
    assert svc.scan_count == 1
    _run_tick(sched)  # same day again -> no refire
    assert svc.scan_count == 1


# ---------------------------------------------------------------------------
# Event log (watchlist/scheduler.log)
# ---------------------------------------------------------------------------
def _patch_runtime_root(monkeypatch, tmp_path):
    """Point get_runtime_root() at a per-test tmp dir (scheduler.log lands there)."""
    from src.config import paths

    monkeypatch.setattr(paths, "get_runtime_root", lambda: tmp_path)


def _read_events(log_path: Path) -> list:
    import json

    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]


def test_start_logs_scheduler_started(tmp_path, monkeypatch) -> None:
    """start() writes scheduler_started with settings + next_run_at."""
    _patch_runtime_root(monkeypatch, tmp_path)
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    cal = FakeCalendar({"2026-10-09", "2026-10-12", "2026-10-13"})
    sched = _make_scheduler(
        svc, cal, FakeClock(_dt.datetime(2026, 10, 12, 16, 0))
    )

    async def _scenario():
        sched.start()
        # No await between start and stop: the task cannot tick before cancel.
        await sched.stop()

    _asyncio_run(_scenario())
    events = _read_events(tmp_path / "watchlist" / "scheduler.log")
    names = [e["event"] for e in events]
    assert names[0] == "scheduler_started"
    started = events[0]
    assert started["enabled"] is True
    assert started["time"] == "18:00"
    assert started["timezone"] == "Asia/Shanghai"
    assert started["next_run_at"] == "2026-10-12T18:00:00+08:00"
    assert "scheduler_stopped" in names


def test_fire_logs_fire_and_fire_result(tmp_path, monkeypatch) -> None:
    """A due fire writes fire (trade_date, attempt) then fire_result (run_id)."""
    _patch_runtime_root(monkeypatch, tmp_path)
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    cal = FakeCalendar({"2026-10-12"})
    sched = _make_scheduler(
        svc, cal, FakeClock(_dt.datetime(2026, 10, 12, 18, 0))
    )
    _run_tick(sched)
    events = _read_events(tmp_path / "watchlist" / "scheduler.log")
    assert [e["event"] for e in events] == ["fire", "fire_result"]
    assert events[0]["trade_date"] == "2026-10-12"
    assert events[0]["attempt"] == 1
    assert events[1]["trade_date"] == "2026-10-12"
    assert events[1]["run_id"] == "r1"
    assert events[1]["status"] == "ok"
    assert isinstance(events[1]["duration_s"], (int, float))


def test_idle_ticks_write_nothing(tmp_path, monkeypatch) -> None:
    """Before fire time / during trading hours / disabled: no log file at all."""
    _patch_runtime_root(monkeypatch, tmp_path)
    log = tmp_path / "watchlist" / "scheduler.log"
    cal = FakeCalendar({"2026-10-12", "2026-10-13"})

    # Trading day inside the trading block (10:00): silent.
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    sched = _make_scheduler(
        svc, cal, FakeClock(_dt.datetime(2026, 10, 12, 10, 0))
    )
    _run_tick(sched)
    # Trading day after 15:00 but before fire time (17:00): silent.
    sched2 = _make_scheduler(
        svc, cal, FakeClock(_dt.datetime(2026, 10, 12, 17, 0))
    )
    _run_tick(sched2)
    # Disabled schedule: silent even after fire time.
    svc_off = FakeService(schedule={"enabled": False, "time": "18:00"})
    sched3 = _make_scheduler(
        svc_off, cal, FakeClock(_dt.datetime(2026, 10, 12, 18, 0))
    )
    _run_tick(sched3)
    assert svc.calls == []
    assert not log.exists()
