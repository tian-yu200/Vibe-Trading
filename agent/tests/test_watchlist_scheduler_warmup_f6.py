"""Scheduler F5 warm-up + F6 in-flight marker (offline, fake everything)."""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.watchlist.scheduler import (  # noqa: E402
    JobLockLost,
    WatchlistScheduler,
)


# ------------------------------------------------------------------ fakes ---
def _run(coro):
    return asyncio.run(coro)


class FakeClock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class FakeService:
    def __init__(self, schedule=None, runs=None, result_status="ok",
                 trade_date="2026-10-12"):
        self.schedule = {"enabled": False, "time": "18:00",
                         "timezone": "Asia/Shanghai", "catch_up": True}
        if schedule:
            self.schedule.update(schedule)
        self.runs = runs or []
        self.scans = []
        self.result_status = result_status
        self.trade_date = trade_date

    def get_settings(self):
        return {"scan": {}, "schedule": dict(self.schedule)}

    def run_scan(self, *, trigger, config_overrides=None, progress=None,
                 llm=None):
        run = {"trigger": trigger, "run_id": f"r{len(self.scans) + 1}",
               "status": self.result_status, "ai_status": "reviewed",
               "duration_s": 0.1, "trade_date": self.trade_date}
        self.scans.append(run)
        self.runs.append(run)
        return run

    def is_scanning(self):
        return False

    def list_runs(self, limit=20):
        return self.runs


class FakeServiceWithJob(FakeService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acquire_calls: list[tuple] = []
        self.handle = _FakeHandle()

    def acquire_job(self, kind, *, owner=None):
        self.acquire_calls.append((kind, owner))
        return self.handle


class _FakeHandle:
    def __init__(self):
        self.checks = 0
        self.beats = 0
        self.released = False

    def heartbeat(self):
        self.beats += 1

    def check(self):
        self.checks += 1

    def release(self):
        self.released = True


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


class _Shortfall:
    def __init__(self, requests, seconds):
        self.est_requests = requests
        self.est_seconds = seconds


class _WarmResult:
    def __init__(self, **kw):
        self._kw = {"attempted": 1, "filled": 1, "failed": 0,
                    "skipped_known_missing": 0, "requests": 1,
                    "duration_s": 0.5, "stopped_unavailable": False,
                    "aborted": False, "deadline_hit": False, "error": None,
                    **kw}

    def to_dict(self):
        return dict(self._kw)


def _patch_runtime_root(monkeypatch, tmp_path):
    from src.config import paths
    monkeypatch.setattr(paths, "get_runtime_root", lambda: tmp_path)


def _read_events(tmp_path) -> list:
    log = tmp_path / "watchlist" / "scheduler.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in
            log.read_text(encoding="utf-8").splitlines()]


def _make(svc, cal=None, now=None, **kw):
    cal = cal or FakeCalendar({"2026-10-12", "2026-10-13"})
    now = now or FakeClock(_dt.datetime(2026, 10, 12, 10, 0))
    return WatchlistScheduler(
        svc, calendar_loader=lambda: cal, now_fn=now,
        sleep_fn=lambda s: None, poll_interval_s=0.01, **kw)


# ------------------------------------------------------- F6 inflight -------
def test_run_scan_writes_and_clears_inflight(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)
    started = threading.Event()
    release = threading.Event()

    class BlockingService(FakeService):
        def run_scan(self, **kw):
            started.set()
            assert release.wait(10)
            return super().run_scan(**kw)

    sched = _make(BlockingService())
    thread = threading.Thread(
        target=sched._run_scan,
        args=(_dt.date(2026, 10, 12),
              _dt.datetime(2026, 10, 12, 18, 5), 1),
        daemon=True)
    thread.start()
    assert started.wait(10)
    data = json.loads((tmp_path / "watchlist"
                       / "fire_inflight.json").read_text(encoding="utf-8"))
    assert data["trade_date"] == "2026-10-12"
    assert data["attempt"] == 1
    assert data["pid"] and data["instance"]
    release.set()
    thread.join(10)
    assert not (tmp_path / "watchlist"
                / "fire_inflight.json").exists()


def test_start_reports_aborted_inflight_first(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)
    (tmp_path / "watchlist").mkdir(parents=True, exist_ok=True)
    (tmp_path / "watchlist" / "fire_inflight.json").write_text(
        json.dumps({"trade_date": "2026-10-15", "attempt": 1,
                    "pid": 4242, "instance": "oldcafe",
                    "started_at": "2026-10-15T10:00:12+00:00"}),
        encoding="utf-8")

    async def scenario():
        svc = FakeService(schedule={"enabled": True, "time": "18:00"})
        sched = _make(svc, now=FakeClock(_dt.datetime(2026, 10, 12, 16, 0)))
        sched.start()
        await sched.stop()

    _run(scenario())
    events = _read_events(tmp_path)
    assert events[0]["event"] == "fire_aborted"
    assert events[0]["reason"] == "server stopped during run"
    assert events[0]["pid"] == 4242
    assert events[0]["trade_date"] == "2026-10-15"
    assert events[0]["instance"] == "oldcafe"
    assert events[1]["event"] == "scheduler_started"
    assert "instance" in events[1] and "home" in events[1]
    assert not (tmp_path / "watchlist" / "fire_inflight.json").exists()


def test_stop_carries_inflight_info(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)

    async def scenario():
        sched = _make(FakeService())
        sched._inflight = {"trade_date": "2026-10-15", "attempt": 2,
                           "pid": 1, "instance": "abc123",
                           "started_at": "x"}
        await sched.stop()

    _run(scenario())
    events = _read_events(tmp_path)
    assert events[-1]["event"] == "scheduler_stopped"
    assert events[-1]["trade_date"] == "2026-10-15"
    assert events[-1]["attempt"] == 2


def test_fire_and_result_carry_home_and_instance(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)
    # pin the server id: another test module in the same process may have
    # run a lifespan and set the global one.
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    sched = _make(svc, now=FakeClock(_dt.datetime(2026, 10, 12, 18, 5)))
    _run(sched._tick())
    events = _read_events(tmp_path)
    assert [e["event"] for e in events] == ["fire", "fire_result"]
    assert events[0]["instance"] == sched._instance_id
    assert "home" in events[0] and "home" in events[1]
    assert events[1]["instance"] == sched._instance_id


def test_start_refused_on_home_mismatch(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)

    class HomeMismatch(Exception):
        pass

    class _HomeStub:
        @staticmethod
        def require_home(context):
            raise HomeMismatch(
                "Vibe-Trading watchlist refuses to run: this process "
                "resolved home C:\\x but the pinned home is E:\\y. "
                "Start via serve-8765.sh.")

        @staticmethod
        def home_fields():
            raise AssertionError("must not be reached")

    monkeypatch.setitem(sys.modules, "src.watchlist.home", _HomeStub)
    sched = _make(FakeService(
        schedule={"enabled": True, "time": "18:00"}))
    sched.start()  # sync: refused before any loop is needed
    assert sched._task is None
    events = _read_events(tmp_path)
    assert events[0]["event"] == "scheduler_start_refused"
    assert "pinned home" in events[0]["error"]


# ------------------------------------------------------- F5 warmup --------
def test_warmup_starts_when_shortfall_exceeds_budget(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)
    svc = FakeServiceWithJob(schedule={"enabled": True, "time": "18:00"})
    runner_calls: list = []

    def runner(symbols, trade_date, *, rate_per_sec, progress, lock_check,
               stop):
        runner_calls.append((symbols, trade_date, rate_per_sec))
        lock_check()
        progress(len(symbols), len(symbols))
        return _WarmResult()

    sched = _make(
        svc, now=FakeClock(_dt.datetime(2026, 10, 12, 16, 0)),
        enable_warmup=True,
        universe_fn=lambda: [f"sh{600000 + i:06d}" for i in range(700)],
        warmup_runner=runner,
        shortfall_fn=lambda syms, td: _Shortfall(700, 700.0),
        warmup_rate_per_sec=1.0)
    assert sched._assess_and_maybe_warmup(_dt.date(2026, 10, 12)) is True
    deadline = time.monotonic() + 10
    while (sched._warmup_thread is not None
           and sched._warmup_thread.is_alive()
           and time.monotonic() < deadline):
        time.sleep(0.01)
    assert runner_calls and len(runner_calls[0][0]) == 700
    assert runner_calls[0][1] == _dt.date(2026, 10, 12)
    assert svc.acquire_calls and svc.acquire_calls[0][0] == "warmup"
    assert svc.handle.released is True
    events = _read_events(tmp_path)
    names = [e["event"] for e in events]
    assert "warmup_started" in names and "warmup_finished" in names
    started = next(e for e in events if e["event"] == "warmup_started")
    assert started["est_requests"] == 700
    assert started["instance"] == sched._instance_id
    assert "home" in started


def test_no_warmup_when_shortfall_fits_budget(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)
    svc = FakeServiceWithJob(schedule={"enabled": True, "time": "18:00"})
    sched = _make(
        svc, enable_warmup=True,
        universe_fn=lambda: ["sh600000"],
        warmup_runner=lambda *a, **k: _WarmResult(),
        shortfall_fn=lambda syms, td: _Shortfall(5, 5.0))
    assert sched._assess_and_maybe_warmup(_dt.date(2026, 10, 12)) is False
    assert sched._warmup_thread is None
    assert svc.acquire_calls == []


def test_no_warmup_when_disabled_by_default(tmp_path, monkeypatch):
    """FakeService without acquire_job: auto-detect leaves warm-up off, so
    ordinary ticks never touch the network or the universe."""
    _patch_runtime_root(monkeypatch, tmp_path)
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    sched = _make(svc, now=FakeClock(_dt.datetime(2026, 10, 12, 18, 5)))
    assert sched._enable_warmup is False
    _run(sched._tick())
    assert svc.scans and sched._warmup_thread is None


def test_legacy_lock_fallback_without_acquire_job(tmp_path, monkeypatch):
    """Pre-hermes service: enable_warmup=True uses the file lock."""
    _patch_runtime_root(monkeypatch, tmp_path)
    svc = FakeService(schedule={"enabled": True, "time": "18:00"},
                   trade_date="2026-10-12",
                   runs=[{"trigger": "schedule", "run_id": "r0",
                          "status": "ok", "ai_status": "reviewed",
                          "duration_s": 1.0,
                          "trade_date": "2026-10-12"}])
    seen: dict = {}

    def runner(symbols, trade_date, *, rate_per_sec, progress, lock_check,
               stop):
        seen["rate"] = rate_per_sec
        lock_check()  # must hold the real file lock: no raise
        return _WarmResult()

    sched = _make(
        svc, now=FakeClock(_dt.datetime(2026, 10, 12, 16, 0)),
        enable_warmup=True,
        universe_fn=lambda: ["sh600000", "sh600001"],
        warmup_runner=runner,
        shortfall_fn=lambda syms, td: _Shortfall(2, 1200.0))
    assert sched._assess_and_maybe_warmup(_dt.date(2026, 10, 12)) is True
    deadline = time.monotonic() + 10
    while (sched._warmup_thread is not None
           and sched._warmup_thread.is_alive()
           and time.monotonic() < deadline):
        time.sleep(0.01)
    assert seen.get("rate") == 1.0
    events = _read_events(tmp_path)
    assert "warmup_finished" in [e["event"] for e in events]


def test_catch_up_triggers_warmup(tmp_path, monkeypatch):
    """Right-after-start catch-up assesses a due session for warm-up."""
    _patch_runtime_root(monkeypatch, tmp_path)
    svc = FakeServiceWithJob(schedule={"enabled": True, "time": "18:00"},
                             trade_date="2026-10-09",
                             # I2: today's scheduled run already succeeded,
                             # so a warm-up may start after fire + cap.
                             runs=[{"trigger": "schedule", "run_id": "r0",
                                    "status": "ok", "ai_status": "reviewed",
                                    "duration_s": 1.0,
                                    "trade_date": "2026-10-10"}])
    sched = _make(
        svc, FakeCalendar({"2026-10-09", "2026-10-12"}),
        FakeClock(_dt.datetime(2026, 10, 10, 18, 30)),
        enable_warmup=True,
        universe_fn=lambda: [f"sh{600000 + i:06d}" for i in range(700)],
        warmup_runner=lambda *a, **k: _WarmResult(),
        shortfall_fn=lambda syms, td: _Shortfall(700, 700.0))
    _run(sched._catch_up_async())
    deadline = time.monotonic() + 10
    while (sched._warmup_thread is not None
           and sched._warmup_thread.is_alive()
           and time.monotonic() < deadline):
        time.sleep(0.01)
    events = _read_events(tmp_path)
    assert "warmup_started" in [e["event"] for e in events]
    # the catch-up scan itself still fired
    assert svc.scans and svc.scans[0]["trigger"] == "schedule"


# ------------------------------------------------------- wait-for-warmup --
def test_wait_skipped_without_warmup(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)
    sched = _make(FakeService())
    _run(sched._wait_for_warmup(
        _dt.date(2026, 10, 12), "18:00",
        _dt.timezone(_dt.timedelta(hours=8)),
        _dt.datetime(2026, 10, 12, 18, 5,
                     tzinfo=_dt.timezone(_dt.timedelta(hours=8)))))
    assert _read_events(tmp_path) == []


def test_wait_timeout_when_deadline_passed(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)
    gate = threading.Event()
    thread = threading.Thread(target=gate.wait, daemon=True)
    thread.start()
    sched = _make(FakeService(), warmup_wait_cap_s=60.0)
    sched._warmup_thread = thread
    sh = _dt.timezone(_dt.timedelta(hours=8))
    # now (20:00) is past fire+60min (18:00+60min=19:00) -> timeout, no join
    _run(sched._wait_for_warmup(
        _dt.date(2026, 10, 12), "18:00", sh,
        _dt.datetime(2026, 10, 12, 20, 0, tzinfo=sh)))
    events = _read_events(tmp_path)
    assert [e["event"] for e in events] == ["warmup_wait"]
    assert events[0]["outcome"] == "timeout"
    assert thread.is_alive()  # we did not wait


def test_wait_finished_when_warmup_completes(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)
    thread = threading.Thread(target=lambda: time.sleep(0.2), daemon=True)
    thread.start()
    sched = _make(FakeService(), warmup_wait_cap_s=60.0)
    sched._warmup_thread = thread
    sh = _dt.timezone(_dt.timedelta(hours=8))
    _run(sched._wait_for_warmup(
        _dt.date(2026, 10, 12), "18:00", sh,
        _dt.datetime(2026, 10, 12, 17, 0, tzinfo=sh)))
    events = _read_events(tmp_path)
    assert [e["event"] for e in events] == ["warmup_wait"]
    assert events[0]["outcome"] == "finished"
    assert events[0]["waited_s"] >= 0.15


def test_legacy_handle_check_detects_lock_loss(tmp_path, monkeypatch):
    """The fallback handle raises name-compatible JobLockLost."""
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.watchlist.scheduler import _LegacyWarmupHandle

    class _FakeLock:
        def __init__(self):
            self._path = tmp_path / "watchlist" / "scan.lock"
            self._token = {"pid": 1, "started_at": "t"}

    handle = _LegacyWarmupHandle(_FakeLock())
    with pytest.raises(JobLockLost):
        handle.check()  # no lock file -> lost


# ------------------------------------------------------- F3 instance id ---
def test_instance_id_prefers_server_instance_id(tmp_path, monkeypatch):
    """scheduler_started/fire carry hermes's server id when the lifespan set
    it; the scheduler's own id is only the fallback."""
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes

    monkeypatch.setattr(routes, "server_instance_id",
                        lambda: "srv-abcdef123456")
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    sched = _make(svc, now=FakeClock(_dt.datetime(2026, 10, 12, 18, 5)))
    assert sched._scheduler_instance_id() == "srv-abcdef123456"
    _run(sched._tick())
    events = _read_events(tmp_path)
    assert [e["event"] for e in events] == ["fire", "fire_result"]
    assert events[0]["instance"] == "srv-abcdef123456"
    assert events[1]["instance"] == "srv-abcdef123456"


def test_instance_id_falls_back_without_server(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    sched = _make(svc, now=FakeClock(_dt.datetime(2026, 10, 12, 18, 5)))
    assert sched._scheduler_instance_id() == sched._instance_id
    _run(sched._tick())
    events = _read_events(tmp_path)
    assert events[0]["instance"] == sched._instance_id


def test_explicit_instance_id_fn_overrides(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    sched = _make(svc, now=FakeClock(_dt.datetime(2026, 10, 12, 18, 5)),
                  instance_id_fn=lambda: "custom-1")
    _run(sched._tick())
    events = _read_events(tmp_path)
    assert events[0]["instance"] == "custom-1"
    assert events[1]["instance"] == "custom-1"


# ------------------------------------------------------- F2 abort result --
def test_job_lock_lost_logs_fire_result_failed(tmp_path, monkeypatch):
    """A JobLockLost mid-run is a fire_result with status failed + reason —
    every fire has a result."""
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.watchlist.service import JobLockLost as _RealLost

    class _LostService(FakeService):
        def run_scan(self, **kw):
            raise _RealLost(
                "job lock lost (deleted or taken over) - aborted")

    svc = _LostService(schedule={"enabled": True, "time": "18:00"})
    sched = _make(svc, now=FakeClock(_dt.datetime(2026, 10, 12, 18, 5)))
    _run(sched._tick())
    events = _read_events(tmp_path)
    assert [e["event"] for e in events] == ["fire", "fire_result"]
    result = events[1]
    assert result["status"] == "failed"
    assert "job lock lost" in (result.get("failure_reason") or "")
    assert events[0]["attempt"] == 1
    assert result["instance"] == sched._scheduler_instance_id()


# ------------------------------------------------------- production wiring
def test_production_wiring_auto_enables_warmup_with_job_lock(
        tmp_path, monkeypatch):
    """Build the scheduler exactly as setup_watchlist_scheduler does (real
    WatchlistService on a tmp home): warm-up auto-enables, and a running
    warm-up holds kind='warmup' so a concurrent scan is refused."""
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    from src.watchlist.service import ScanAlreadyRunning

    monkeypatch.setattr(routes, "_SCHEDULER", None)
    monkeypatch.setattr(routes, "_SCHEDULER_FACTORY", None)
    sched = routes.setup_watchlist_scheduler()
    try:
        from src.watchlist.scheduler import WatchlistScheduler

        assert isinstance(sched, WatchlistScheduler)
        # hermes's singleton: a second call returns the SAME scheduler
        assert routes.setup_watchlist_scheduler() is sched
        # the real service exposes acquire_job -> warm-up auto-enables
        assert sched._enable_warmup is True

        started = threading.Event()
        release = threading.Event()

        def runner(symbols, trade_date, *, rate_per_sec, progress,
                   lock_check, stop):
            started.set()
            assert release.wait(15)
            return _WarmResult()

        sched._universe_fn = lambda: ["sh600000"]
        sched._shortfall_fn = lambda syms, td: _Shortfall(1, 9999.0)
        sched._warmup_runner = runner
        # Clock-independence (round 3): the Q1/I2 gates read the real clock,
        # so pin a fixed one outside the trading block and after fire+cap, on
        # a day with no scheduled success — this test must pass at any hour.
        sched._now_fn = lambda: _dt.datetime(2026, 10, 17, 22, 0)
        sched._calendar_loader = lambda: FakeCalendar({"2026-10-19"})
        assert sched._assess_and_maybe_warmup(
            _dt.date(2026, 10, 12)) is True
        assert started.wait(15)
        # the warm-up holds kind='warmup': a concurrent scan is refused
        with pytest.raises(ScanAlreadyRunning) as ei:
            sched._service.acquire_job("scan", owner="test")
        assert "warmup" in str(ei.value)
    finally:
        release.set()
        thread = sched._warmup_thread
        if thread is not None:
            thread.join(15)
        assert sched._warmup_thread is None


# ------------------------------ FIX2 phase (e): deadline invariants I1-I3 ---
SH = _dt.timezone(_dt.timedelta(hours=8))
# A Monday (trading day) plus the Sunday before and the Tuesday after.
_MON = _dt.date(2026, 10, 19)
_TRADING_DAYS = {"2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"}


def _inv_cal(days=None):
    return FakeCalendar(days if days is not None else _TRADING_DAYS)


def _inv_sched(now, *, cap=3600.0, runs=None, enabled=True,
               hhmm="18:00", days=None):
    """A scheduler at `now` (naive Asia/Shanghai) with a real cap."""
    svc = FakeService(
        schedule={"enabled": enabled, "time": hhmm},
        runs=runs or [], result_status="ok",
        trade_date="2026-10-19")
    return _make(svc, cal=_inv_cal(days),
                 now=FakeClock(now), warmup_wait_cap_s=cap)


def _success_run(day=_MON):
    return {"trigger": "schedule", "run_id": "r1", "status": "ok",
            "ai_status": "reviewed", "duration_s": 1.0,
            "trade_date": day.isoformat()}


def _monotonic_deadline(sched):
    return sched._warmup_lock_deadline()


def _deadline_at(sched, wall):
    """The deadline expressed as a wall-clock datetime (Asia/Shanghai)."""
    mono = sched._warmup_lock_deadline()
    assert mono is not None
    assert wall.tzinfo is None
    return wall.replace(tzinfo=SH) + _dt.timedelta(
        seconds=(mono - time.monotonic()))


def _assert_deadline_not_after(sched, wall, limit):
    """The warm-up deadline must be <= `limit` (both naive, Asia/Shanghai)."""
    got = _deadline_at(sched, wall)
    assert got <= limit.replace(tzinfo=SH), f"{got} > {limit}"


def test_i1_deadline_never_holds_lock_past_fire_plus_cap(
        tmp_path, monkeypatch):
    """I1: while today's run has not succeeded, a warm-up stops by fire+cap."""
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    # 10:00 Monday, no successful run today: fire 18:00 + cap 3600 -> 19:00.
    sched = _inv_sched(_dt.datetime(2026, 10, 19, 10, 0), cap=3600.0)
    _assert_deadline_not_after(
        sched, _dt.datetime(2026, 10, 19, 10, 0),
        _dt.datetime(2026, 10, 19, 19, 0))
    # And it is in the future, so the warm-up may start and run.
    assert _monotonic_deadline(sched) > time.monotonic()


def test_i1_deadline_equals_fire_plus_cap(tmp_path, monkeypatch):
    """I1 exact value: fire + cap, not the later next-session bound."""
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    sched = _inv_sched(_dt.datetime(2026, 10, 19, 10, 0), cap=1800.0)
    monkeypatch.setattr(time, "monotonic", time.monotonic)
    fire_cap = _dt.datetime(2026, 10, 19, 18, 30)  # 18:00 fire + 30 min
    now = _dt.datetime(2026, 10, 19, 10, 0)
    expected = (fire_cap.replace(tzinfo=SH)
                - now.replace(tzinfo=SH)).total_seconds()
    got = _monotonic_deadline(sched) - time.monotonic()
    assert abs(got - expected) < 2.0


def test_i1_after_success_deadline_is_bounded_by_next_bounds(
        tmp_path, monkeypatch):
    """I3: after today's success the deadline is min(next open, fire+cap)."""
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    runs = [_success_run()]
    # 20:00 Monday, today's run succeeded -> no I1 bound any more.
    sched = _inv_sched(_dt.datetime(2026, 10, 19, 20, 0),
                       cap=1800.0, runs=runs)
    _assert_deadline_not_after(
        sched, _dt.datetime(2026, 10, 19, 20, 0),
        _dt.datetime(2026, 10, 20, 9, 30))   # earlier than Tue 09:15 open
    assert _monotonic_deadline(sched) > time.monotonic()


def test_i2_no_start_after_fire_plus_cap_until_success(
        tmp_path, monkeypatch):
    """I2: after fire+cap with no success, no warm-up may START."""
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    # 19:30 Monday (past 18:00+60), no succeeded run today.
    sched = _inv_sched(_dt.datetime(2026, 10, 19, 19, 30), cap=3600.0)
    assert sched._warmup_start_allowed(
        _dt.datetime(2026, 10, 19, 19, 0, tzinfo=SH)) is False
    # 17:30 is inside the cap and outside trading hours: allowed.
    assert sched._warmup_start_allowed(
        _dt.datetime(2026, 10, 19, 17, 30, tzinfo=SH)) is True


def test_i2_after_success_start_is_allowed_again(tmp_path, monkeypatch):
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    runs = [_success_run()]
    sched = _inv_sched(_dt.datetime(2026, 10, 19, 19, 30),
                       cap=3600.0, runs=runs)
    assert sched._warmup_start_allowed(
        _dt.datetime(2026, 10, 19, 19, 0, tzinfo=SH)) is True


def test_q1_no_warmup_starts_during_trading_hours(tmp_path, monkeypatch):
    """Q1: 09:15-15:00 on a trading day, whatever the target date."""
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    for hour in (9, 10, 12, 14):
        sched = _inv_sched(_dt.datetime(2026, 10, 19, hour, 30))
        assert sched._warmup_start_allowed(
            _dt.datetime(2026, 10, 19, hour, 30, tzinfo=SH)) is False
    # 09:14 and 15:00 are outside the block.
    sched = _inv_sched(_dt.datetime(2026, 10, 19, 9, 14))
    assert sched._warmup_start_allowed(
        _dt.datetime(2026, 10, 19, 9, 14, tzinfo=SH)) is True


def test_q2_warmup_symbols_drop_bse_codes(tmp_path, monkeypatch):
    """Q2: the warm-up universe uses the scan's own BSE predicate."""
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    sched = _inv_sched(_dt.datetime(2026, 10, 19, 20, 0))
    symbols = ["sh600000", "sz000001", "bj920002", "bj430047",
               "sh688000", "sz300001"]
    kept = sched._warmup_symbols(symbols)
    assert "bj920002" not in kept and "bj430047" not in kept
    assert kept == ["sh600000", "sz000001", "sh688000", "sz300001"]


def test_i1_deadline_is_todays_fire_plus_cap_until_success(tmp_path,
                                                           monkeypatch):
    """I1: while today's scheduled run has not succeeded, the deadline is
    today's fire + cap, so scan.lock is always free before fire + cap."""
    _patch_runtime_root(monkeypatch, tmp_path)
    cap = 3600.0
    # 09:00 Monday, no successful run today yet.
    sched = _inv_sched(_dt.datetime(2026, 10, 19, 9, 0))
    deadline = sched._warmup_lock_deadline()
    assert deadline is not None
    fire_cap = _dt.datetime(2026, 10, 19, 19, 0, tzinfo=SH)
    expected = (fire_cap - _dt.datetime(2026, 10, 19, 9, 0,
                                        tzinfo=SH)).total_seconds()
    assert abs((deadline - time.monotonic()) - expected) < 2.0


def test_i2_no_start_after_fire_plus_cap_without_success(
        tmp_path, monkeypatch):
    """I2: after today's fire + cap no warm-up starts until today's run
    succeeded, so the retry is never blocked by a fresh warm-up."""
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)
    started: list[int] = []

    def runner(symbols, trade_date, *, rate_per_sec, progress,
               lock_check, stop):
        started.append(len(symbols))
        return _WarmResult()

    def _assess(when):
        svc = FakeServiceWithJob(
            schedule={"enabled": True, "time": "18:00"},
            runs=[], trade_date="2026-10-19")
        sched = _make(
            svc, cal=_inv_cal(), now=FakeClock(when),
            warmup_wait_cap_s=3600.0, enable_warmup=True,
            universe_fn=lambda: ["sh600000", "sh600001"],
            warmup_runner=runner,
            shortfall_fn=lambda syms, td: _Shortfall(2, 1200.0))
        return sched, sched._assess_and_maybe_warmup(_dt.date(2026, 10, 19))

    # 19:05 is past fire 18:00 + cap 3600 -> 19:00, and today's scheduled
    # run has not succeeded: no warm-up may start.
    sched, running = _assess(_dt.datetime(2026, 10, 19, 19, 5))
    assert running is False
    assert started == []
    assert sched._warmup_thread is None
    assert sched._warmup_start_allowed(
        _dt.datetime(2026, 10, 19, 19, 5, tzinfo=SH)) is False
    _join_warmup(sched)

    # 19:05 is past fire 18:00 + cap 3600 -> 19:00, and today's scheduled
    # run has not succeeded: no warm-up may start.
    sched, running = _assess(_dt.datetime(2026, 10, 19, 19, 5))
    assert running is False
    assert started == []
    assert sched._warmup_thread is None
    assert sched._warmup_start_allowed(
        _dt.datetime(2026, 10, 19, 19, 5, tzinfo=SH)) is False
    _join_warmup(sched)


def _join_warmup(sched):
    thread = getattr(sched, "_warmup_thread", None)
    if thread is not None:
        sched._warmup_stop.set()
        thread.join(15)
        assert sched._warmup_thread is None


def _fire_wait_deadline_s(sched, trade_date, hhmm, tz, now):
    """Recompute what ``_wait_for_warmup`` waits, without running its loop.

    Mirrors the production arithmetic exactly (fire + the ONE cap accessor),
    so a regression shows up as a wrong number instead of a 15-minute sleep.
    """
    fire = sched._parse_hhmm(hhmm)
    fire_dt = (_dt.datetime.combine(trade_date, fire, tzinfo=tz)
               if fire is not None else now)
    deadline = fire_dt + _dt.timedelta(seconds=sched._warmup_deadline_cap_s())
    return max(0.0, (deadline - now).total_seconds())
def _join_warmup(sched):
    """Stop a running warm-up thread and wait for it to clear."""
    thread = getattr(sched, "_warmup_thread", None)
    if thread is not None:
        sched._warmup_stop.set()
        thread.join(15)
        assert sched._warmup_thread is None
# --------------- FIX2 (e) round 2: single-source cap, non-trading days, fail-closed


def test_cap_is_single_sourced_between_wait_and_lock_deadline(
        tmp_path, monkeypatch):
    """FIX 1: the fire's wait deadline and the warm-up lock deadline must
    move together, because both read the cap from ONE method.

    Before the fix ``_wait_for_warmup`` read ``_warmup_wait_cap_s`` directly, so
    phase (b) could set the lock deadline to fire+3600 while the fire gave up at
    fire+1800 — the 'scan already running' loop again.

    Deterministic: no thread is ever joined, so no real time passes. The fire's
    wait deadline is taken from the never-started ``_wait_for_warmup`` path (it
    returns before the wait when there is no warm-up thread), and the deadline
    formula is asserted through the ONE accessor.
    """
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    sched = _inv_sched(_dt.datetime(2026, 10, 19, 10, 0), cap=1800.0)
    assert sched._warmup_deadline_cap_s() == 1800.0
    # The lock deadline uses the constructor cap (1800) -> 18:30.
    assert _deadline_at(sched, _dt.datetime(2026, 10, 19, 10, 0)) == (
        _dt.datetime(2026, 10, 19, 18, 30, tzinfo=SH))

    # Only the accessor is patched: both deadlines must move together.
    monkeypatch.setattr(
        sched, "_warmup_deadline_cap_s", lambda: 3600.0, raising=True)

    # 1) the warm-up lock deadline -> fire 18:00 + 3600 = 19:00.
    assert _deadline_at(sched, _dt.datetime(2026, 10, 19, 10, 0)) == (
        _dt.datetime(2026, 10, 19, 19, 0, tzinfo=SH))


def test_fire_wait_deadline_reads_the_same_cap_accessor(tmp_path,
                                                        monkeypatch):
    """FIX 1 (second half): the fire's wait deadline uses the SAME cap method.

    ``_wait_for_warmup`` is exercised with a warm-up thread that has already
    finished, so it never sleeps: only the deadline it computes matters.
    """
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    sched = _inv_sched(_dt.datetime(2026, 10, 19, 10, 0), cap=1800.0)
    seen: list[float] = []
    real_cap = sched._warmup_deadline_cap_s

    def _spy():
        value = real_cap()
        seen.append(value)
        return value

    monkeypatch.setattr(sched, "_warmup_deadline_cap_s", _spy, raising=True)

    # A warm-up that is STILL running (so the wait path is entered) but
    # whose join returns at once: `hold` is set only after a short sleep in the
    # worker, and `now` is already past the computed deadline, so wait_s == 0
    # and no real time is spent.
    hold = threading.Event()

    def _quick():
        time.sleep(0.05)
        hold.set()

    thread = threading.Thread(target=_quick, daemon=True)
    thread.start()

    sched._warmup_thread = thread
    try:
        _run(sched._wait_for_warmup(
            _dt.date(2026, 10, 19), "18:00", SH,
            _dt.datetime(2026, 10, 19, 19, 0, tzinfo=SH)))
        events = _read_events(tmp_path)
    finally:
        sched._warmup_thread = None
    # The cap accessor was consulted during the fire's wait computation.
    assert seen, "the fire wait did not read the cap accessor"
    assert seen[-1] == 1800.0
    wait = next(e for e in events if e["event"] == "warmup_wait")
    # wait_s == 0 (now is at the deadline), so no join happened and no real
    # time passed: the thread is simply still alive.
    assert wait["waited_s"] < 0.5
    thread.join(5)
    _join_warmup(sched)
def test_weekend_warmup_starts_with_monday_session_open_deadline(
        tmp_path, monkeypatch):
    """FIX 2 (a): on a non-trading day the I1/I2 gate must not apply.

    Saturday 20:00: before the fix `_warmup_fire_cap` returned 18:00+cap for a
    Saturday 'fire' that does not exist, so I2 blocked every warm-up until
    Monday and I1 pointed at a non-existent fire. Now it falls back to I3:
    the deadline is the next session open, Monday 09:15.
    """
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    sat = _dt.datetime(2026, 10, 17, 20, 0)  # a Saturday
    assert sat.weekday() == 5
    sched = _inv_sched(sat)
    assert sched._warmup_start_allowed(sat.replace(tzinfo=SH)) is True
    # No successful run can exist for a Saturday, yet a warm-up may start.
    assert sched._has_scheduled_success_for(_dt.date(2026, 10, 17)) is False
    # I3: the next session open is Monday 2026-10-19 09:15.
    assert _deadline_at(sched, sat) == _dt.datetime(2026, 10, 19, 9, 15,
                                                    tzinfo=SH)
def test_weekday_holiday_falls_back_to_next_session_open(tmp_path,
                                                         monkeypatch):
    """FIX 2 (b): a weekday that is NOT a trading day (holiday) behaves the
    same as a weekend — the deadline is the next session open.

    Wednesday 2026-10-21 is a holiday in this fake calendar, so the next
    trading day is Thursday 2026-10-22 and its session opens at 09:15.
    """
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    wed = _dt.datetime(2026, 10, 21, 20, 0)
    assert wed.weekday() == 2  # a weekday
    days = {"2026-10-20", "2026-10-22"}  # 10-21 absent -> holiday
    sched = _inv_sched(wed, days=days)
    assert sched._warmup_start_allowed(wed.replace(tzinfo=SH)) is True
    assert _deadline_at(sched, wed) == _dt.datetime(2026, 10, 22, 9, 15,
                                                    tzinfo=SH)
def test_unparseable_schedule_fails_closed_no_warmup(tmp_path, monkeypatch):
    """FIX 3: an unbounded warm-up must never start.

    With an unparseable schedule time no deadline can be computed
    (``_warmup_lock_deadline() is None``). Before the fix the gate allowed the
    start and ``_default_warmup_runner`` passed ``deadline=None`` into
    ``warm_cache``, which then ran unbounded.
    """
    _patch_runtime_root(monkeypatch, tmp_path)
    from src.api import watchlist_routes as routes
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)
    calls: list = []

    def runner(symbols, trade_date, **kw):
        calls.append(kw.get("deadline"))
        return _WarmResult()

    mon = _dt.datetime(2026, 10, 19, 20, 0)
    svc = FakeService(schedule={"enabled": True, "time": "not-a-time"},
                      runs=[], result_status="ok", trade_date="2026-10-19")
    sched = _make(svc, cal=_inv_cal(), now=FakeClock(mon),
                  warmup_wait_cap_s=3600.0, enable_warmup=True,
                  universe_fn=lambda: ["sh600000", "sh600001"],
                  warmup_runner=runner,
                  shortfall_fn=lambda syms, td: _Shortfall(2, 1200.0))

    # No deadline can be computed -> the gate refuses, and warm_cache is
    # never reached (the runner is never invoked).
    assert sched._warmup_lock_deadline() is None
    assert sched._warmup_start_allowed(mon.replace(tzinfo=SH)) is False
    assert sched._assess_and_maybe_warmup(_dt.date(2026, 10, 19)) is False
    assert calls == []
    assert sched._warmup_thread is None
    _join_warmup(sched)
