"""Phase (b): the warm-up wait cap is the ``schedule.warmup_wait_min`` setting.

One ceiling, in minutes, covers both the 18:00 fire's wait for a warm-up and
the warm-up's own deadline. It used to be ``warmup_wait_cap_s = 30 * 60``
baked into the scheduler constructor; now it is stored per home, defaults to
60, is validated on write, and is read fresh on every deadline computation —
so a PUT takes effect without restarting the server. An explicit ``0`` means a
0 s cap (both deadlines land on the fire time), never "falsy, so use 60".

Offline only: tmp stores (``WatchlistStore(path=tmp_path/...)``), tmp runtime
roots, injected calendars, fixed clocks, fake clock functions. No network, no
server, no real home.

Run: pytest tests/test_watchlist_warmup_wait_setting.py -q
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import logging
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.watchlist.scheduler import WatchlistScheduler  # noqa: E402
from src.watchlist.service import WatchlistService  # noqa: E402
from src.watchlist.store import WatchlistStore  # noqa: E402

SH = _dt.timezone(_dt.timedelta(hours=8))
MON = _dt.date(2026, 10, 19)  # a Monday, and a trading day here
_AT = _dt.datetime(2026, 10, 19, 10, 0)  # fixed clock: before today's fire
_TRADING_DAYS = {"2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"}


class FakeCalendar:
    def __init__(self, days=None):
        self.trading_days = frozenset(days or _TRADING_DAYS)

    def is_trading_day(self, d):
        return d.isoformat() in self.trading_days

    def previous_trading_day(self, d):
        d -= _dt.timedelta(days=1)
        while d.isoformat() not in self.trading_days:
            d -= _dt.timedelta(days=1)
        return d

    def latest_completed_trading_day(self, now):
        day = now.date()
        if self.is_trading_day(day) and now.time() >= _dt.time(15, 0):
            return day
        return self.previous_trading_day(day)


class _AliveThread:
    """A warm-up that is 'still running' but never actually joined."""

    def is_alive(self):
        return True

    def join(self, timeout=None):
        pass


def _service(tmp_path) -> WatchlistService:
    """Real service on a tmp store; the calendar loader is never the network."""
    return WatchlistService(
        store=WatchlistStore(path=tmp_path / "watchlist.sqlite3"),
        calendar_loader=FakeCalendar,
    )


def _sched(svc, at=_AT) -> WatchlistScheduler:
    """Scheduler with NO explicit cap — the setting is the only cap source."""
    return WatchlistScheduler(
        svc,
        calendar_loader=FakeCalendar,
        now_fn=lambda: at,
        sleep_fn=lambda s: None,
        poll_interval_s=0.01,
    )


def _lock_deadline(sched, at=_AT) -> _dt.datetime:
    """Wall time a NEW warm-up would stop at (``_warmup_lock_deadline``)."""
    mono = sched._warmup_lock_deadline()
    assert mono is not None
    return at.replace(tzinfo=SH) + _dt.timedelta(seconds=mono - time.monotonic())


def _fire_wait_deadline(sched, at=_AT) -> _dt.datetime:
    """What the fire waits for — the arithmetic of ``_wait_for_warmup``
    (scheduler.py:1152): today's fire + the one cap accessor."""
    fire = sched._parse_hhmm("18:00")
    return _dt.datetime.combine(at.date(), fire, tzinfo=SH) + _dt.timedelta(
        seconds=sched._warmup_deadline_cap_s()
    )


def _seconds_off(got: _dt.datetime, want: _dt.datetime) -> float:
    return abs((got - want).total_seconds())


def _patch_runtime_root(monkeypatch, tmp_path):
    from src.config import paths

    monkeypatch.setattr(paths, "get_runtime_root", lambda: tmp_path)


# ------------------------------------------------------------- the setting ---
def test_the_key_is_a_schedule_setting_defaulting_to_60(tmp_path) -> None:
    svc = _service(tmp_path)
    assert "warmup_wait_min" in svc.get_settings()["schedule"]
    assert svc.get_settings()["schedule"]["warmup_wait_min"] == 60


def test_stored_settings_without_the_key_read_as_60(tmp_path) -> None:
    """A store written before phase (b) still reports 60, not None/0."""
    svc = _service(tmp_path)
    svc.store.set_setting(
        "schedule",
        {"enabled": False, "time": "18:00", "timezone": "Asia/Shanghai",
         "catch_up": True},
    )
    assert svc.get_settings()["schedule"]["warmup_wait_min"] == 60
    assert _sched(svc)._warmup_deadline_cap_s() == 3600.0


def test_stored_45_gives_a_2700_second_cap(tmp_path) -> None:
    svc = _service(tmp_path)
    svc.update_settings({"schedule": {"warmup_wait_min": 45}})
    sched = _sched(svc)
    assert svc.get_settings()["schedule"]["warmup_wait_min"] == 45
    assert sched._warmup_deadline_cap_s() == 2700.0
    assert sched.status()["warmup_wait_min"] == 45.0


def test_a_new_value_is_used_on_the_next_call_without_a_new_scheduler(
    tmp_path,
) -> None:
    """Fresh read per call: a PUT needs no restart to take effect."""
    sched = _sched(_service(tmp_path))
    sched._service.update_settings({"schedule": {"warmup_wait_min": 45}})
    assert sched._warmup_deadline_cap_s() == 2700.0
    sched._service.update_settings({"schedule": {"warmup_wait_min": 90}})
    assert sched._warmup_deadline_cap_s() == 5400.0
    assert sched.status()["warmup_wait_min"] == 90.0


def test_explicit_zero_is_zero_not_the_default(tmp_path) -> None:
    svc = _service(tmp_path)
    svc.update_settings({"schedule": {"warmup_wait_min": 0}})
    sched = _sched(svc)
    assert sched._warmup_deadline_cap_s() == 0.0
    assert sched.status()["warmup_wait_min"] == 0.0


def test_explicit_zero_lands_both_deadlines_on_the_fire_time(tmp_path) -> None:
    """Boss rule: cap 0 -> the fire waits until 18:00, not until 18:00+60min."""
    svc = _service(tmp_path)
    svc.update_settings({"schedule": {"warmup_wait_min": 0}})
    sched = _sched(svc)
    fire = _dt.datetime(2026, 10, 19, 18, 0, tzinfo=SH)
    assert _seconds_off(_fire_wait_deadline(sched), fire) < 2.0
    assert _seconds_off(_lock_deadline(sched), fire) < 2.0


def test_an_invalid_stored_value_falls_back_to_3600_and_never_raises(
    tmp_path, caplog
) -> None:
    """Bypassing the write-side validation (hand-edited store) is survivable."""
    svc = _service(tmp_path)
    sched = _sched(svc)
    for garbage in (999, -5, "45", None, 1.5, True):
        svc.store.set_setting("schedule", {"warmup_wait_min": garbage})
        with caplog.at_level(logging.WARNING, logger="src.watchlist.scheduler"):
            caplog.clear()
            assert sched._warmup_deadline_cap_s() == 3600.0
        assert "warmup_wait_min" in caplog.text
    assert sched.status()["warmup_wait_min"] == 60.0


def test_status_reports_whole_minutes_as_an_int(tmp_path) -> None:
    """The setting is an int, so status() (and the API JSON) shows 60, not 60.0.

    Only a constructor cap that is not whole minutes keeps a fraction.
    """
    import json

    svc = _service(tmp_path)
    sched = _sched(svc)
    value = sched.status()["warmup_wait_min"]
    assert value == 60 and type(value) is int
    assert '"warmup_wait_min": 60}' in json.dumps({"warmup_wait_min": value})
    svc.store.set_setting("schedule", {"warmup_wait_min": 0})
    assert type(sched.status()["warmup_wait_min"]) is int
    pinned = WatchlistScheduler(
        svc, calendar_loader=FakeCalendar, now_fn=lambda: _AT,
        sleep_fn=lambda s: None, poll_interval_s=0.01, warmup_wait_cap_s=30,
    )
    assert pinned.status()["warmup_wait_min"] == 0.5


# ------------------------------------------------------------- validation ---
@pytest.mark.parametrize("value", [0, 1, 45, 120])
def test_update_settings_accepts_the_whole_range(tmp_path, value) -> None:
    svc = _service(tmp_path)
    out = svc.update_settings({"schedule": {"warmup_wait_min": value}})
    assert out["schedule"]["warmup_wait_min"] == value
    assert svc.store.get_setting("schedule")["warmup_wait_min"] == value


@pytest.mark.parametrize("value", [-1, 121, True, False, 1.5, "60", None])
def test_update_settings_rejects_everything_else_and_stores_nothing(
    tmp_path, value
) -> None:
    svc = _service(tmp_path)
    svc.update_settings({"schedule": {"warmup_wait_min": 45}})
    with pytest.raises(ValueError, match="warmup_wait_min"):
        svc.update_settings({"schedule": {"warmup_wait_min": value}})
    # Nothing stored: neither the bad value nor a reset to the default.
    assert svc.store.get_setting("schedule")["warmup_wait_min"] == 45
    assert svc.get_settings()["schedule"]["warmup_wait_min"] == 45


def test_a_rejected_patch_stores_nothing_even_beside_a_valid_scan_patch(
    tmp_path,
) -> None:
    svc = _service(tmp_path)
    before_scan = svc.store.get_setting("scan")
    before_sched = svc.store.get_setting("schedule")
    with pytest.raises(ValueError, match="warmup_wait_min"):
        svc.update_settings(
            {"scan": {"top_n": 7}, "schedule": {"warmup_wait_min": 999}}
        )
    assert svc.store.get_setting("scan") == before_scan
    assert svc.store.get_setting("schedule") == before_sched


# ------------------------------------------------------------------ API ---
@pytest.fixture()
def api(tmp_path, monkeypatch):
    """Bare app + the real service on a tmp store; lifespan stays outside.

    ``_SCHEDULER_FACTORY`` is set to a no-op so that registering the routes
    does not auto-build a scheduler against the runtime home, and the tests
    choose the live/fallback branch by patching ``_scheduler``.
    """
    from src.api import watchlist_routes as routes

    svc = _service(tmp_path)
    monkeypatch.setattr(routes, "_service", lambda: svc)
    monkeypatch.setattr(routes, "_SCHEDULER", None)
    monkeypatch.setattr(routes, "_SCHEDULER_FACTORY", lambda: None)
    app = FastAPI()
    routes.register_watchlist_routes(app)
    return TestClient(app), svc, routes


def test_get_settings_shows_the_new_key(api) -> None:
    client, _svc, _routes = api
    body = client.get("/api/watchlist/settings").json()
    assert body["settings"]["schedule"]["warmup_wait_min"] == 60


@pytest.mark.parametrize("value", [-1, 121, True, 1.5, "60", None])
def test_put_rejects_bad_values_with_400_and_stores_nothing(api, value) -> None:
    client, svc, _routes = api
    svc.update_settings({"schedule": {"warmup_wait_min": 45}})
    r = client.put(
        "/api/watchlist/settings",
        json={"patch": {"schedule": {"warmup_wait_min": value}}},
    )
    assert r.status_code == 400, r.text
    assert "warmup_wait_min" in r.json()["detail"]
    assert svc.store.get_setting("schedule")["warmup_wait_min"] == 45
    assert client.get("/api/watchlist/settings").json()["settings"]["schedule"][
        "warmup_wait_min"
    ] == 45


@pytest.mark.parametrize("value", [0, 45, 120])
def test_put_accepts_valid_values_and_they_round_trip(api, value) -> None:
    client, svc, _routes = api
    r = client.put(
        "/api/watchlist/settings",
        json={"patch": {"schedule": {"warmup_wait_min": value}}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["settings"]["schedule"]["warmup_wait_min"] == value
    assert svc.store.get_setting("schedule")["warmup_wait_min"] == value


def test_get_schedule_with_a_live_scheduler_reports_the_effective_minutes(
    api, monkeypatch
) -> None:
    client, svc, routes = api
    svc.update_settings({"schedule": {"warmup_wait_min": 45}})
    monkeypatch.setattr(routes, "_scheduler", lambda: _sched(svc))
    body = client.get("/api/watchlist/schedule").json()
    assert body["warmup_wait_min"] == 45.0


def test_get_schedule_fallback_reports_the_stored_minutes(api) -> None:
    """No scheduler (server started without the lifespan) -> from settings."""
    client, svc, _routes = api
    svc.update_settings({"schedule": {"warmup_wait_min": 30}})
    assert client.get("/api/watchlist/schedule").json()["warmup_wait_min"] == 30


def test_get_schedule_fallback_defaults_to_60_when_nothing_is_stored(
    api, monkeypatch
) -> None:
    client, _svc, routes = api

    class LegacySchedule:
        """A pre-phase-(b) service: no warmup_wait_min anywhere."""

        def get_settings(self):
            return {"scan": {}, "schedule": {"time": "18:00"}}

        def staleness(self):
            return {"is_stale": False}

    monkeypatch.setattr(routes, "_service", lambda: LegacySchedule())
    body = client.get("/api/watchlist/schedule").json()
    assert body["warmup_wait_min"] == 60


# ----------------------------------------------------------------- CLI ---
def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    for key, value in kw.items():
        setattr(ns, key, value)
    return ns


def test_cli_schedule_status_prints_the_warmup_wait(tmp_path, capsys) -> None:
    from cli.commands import watchlist as wl_cli

    svc = _service(tmp_path)
    wl_cli._cmd_schedule(_args(schedule_cmd="status", time=None), svc)
    assert "warm-up wait: 60 min" in capsys.readouterr().out

    svc.update_settings({"schedule": {"warmup_wait_min": 45}})
    wl_cli._cmd_schedule(_args(schedule_cmd="status", time=None), svc)
    out = capsys.readouterr().out
    assert "warm-up wait: 45 min" in out
    # 0 is a real value, not "unset": it prints as 0 min.
    svc.update_settings({"schedule": {"warmup_wait_min": 0}})
    wl_cli._cmd_schedule(_args(schedule_cmd="status", time=None), svc)
    assert "warm-up wait: 0 min" in capsys.readouterr().out


# ------------------------------------------------- production wiring test ---
def test_setup_watchlist_scheduler_uses_the_setting_for_both_deadlines(
    tmp_path, monkeypatch
) -> None:
    """The scheduler the server actually builds reads the setting (no cap arg).

    Clock and calendar are pinned the way phase (e) does it (``_now_fn`` /
    ``_calendar_loader``), so nothing touches the wall clock or the network.
    """
    from src.api import watchlist_routes as routes

    _patch_runtime_root(monkeypatch, tmp_path)
    monkeypatch.setattr(routes, "_SCHEDULER", None)
    monkeypatch.setattr(routes, "_SCHEDULER_FACTORY", None)
    monkeypatch.setattr(routes, "server_instance_id", lambda: None)

    sched = routes.setup_watchlist_scheduler()
    assert isinstance(sched, WatchlistScheduler)
    sched._now_fn = lambda: _AT
    sched._calendar_loader = FakeCalendar
    sched._service.update_settings({"schedule": {"warmup_wait_min": 45}})

    assert sched._warmup_deadline_cap_s() == 2700.0
    fire_plus_45 = _dt.datetime(2026, 10, 19, 18, 45, tzinfo=SH)

    # 1) the warm-up's own deadline (I1: today's fire + cap)
    assert _seconds_off(_lock_deadline(sched), fire_plus_45) < 2.0

    # 2) the fire's wait: run the real wait path with a warm-up that is still
    #    "running"; now is past the deadline, so wait_s == 0 and no sleep.
    seen: list[float] = []
    real_cap = sched._warmup_deadline_cap_s

    def _spy() -> float:
        value = real_cap()
        seen.append(value)
        return value

    monkeypatch.setattr(sched, "_warmup_deadline_cap_s", _spy, raising=True)
    sched._warmup_thread = _AliveThread()
    try:
        asyncio.run(
            sched._wait_for_warmup(
                MON, "18:00", SH, _dt.datetime(2026, 10, 19, 19, 0, tzinfo=SH)
            )
        )
    finally:
        sched._warmup_thread = None
    assert seen, "the fire's wait never read the cap accessor"
    assert seen[-1] == 2700.0
    assert _seconds_off(_fire_wait_deadline(sched), fire_plus_45) < 2.0

    assert sched.status()["warmup_wait_min"] == 45.0
