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
    svc = FakeService(schedule={"enabled": True, "time": "18:00"})
    seen: dict = {}

    def runner(symbols, trade_date, *, rate_per_sec, progress, lock_check,
               stop):
        seen["rate"] = rate_per_sec
        lock_check()  # must hold the real file lock: no raise
        return _WarmResult()

    sched = _make(
        svc, enable_warmup=True,
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
                             trade_date="2026-10-09")
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
