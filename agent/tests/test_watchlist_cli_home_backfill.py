"""CLI home gate, F5c backfill, F9 forget, kline_reasons display (offline)."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from cli.commands import watchlist as wl_cli
from src.market_scanner.sources.kline_cache import KlineCache
from src.watchlist.service import ScanAlreadyRunning


# ------------------------------------------------------------------ fakes ---
def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class _FakeHandle:
    def __init__(self):
        self.checks = 0
        self.released = False

    def heartbeat(self):
        pass

    def check(self):
        self.checks += 1

    def release(self):
        self.released = True


class FakeSvc:
    """Minimal service double with job lock + forget (hermes API shape)."""

    def __init__(self, run=None):
        self.run = run or {}
        self.acquire_calls: list[tuple] = []
        self.handle = _FakeHandle()
        self.lock_busy = False
        self.forget_calls: list[str] = []
        self.forget_result: dict | Exception = {"code": "600519",
                                                "name": "n",
                                                "origin": "manual"}

    def acquire_job(self, kind, *, owner=None):
        self.acquire_calls.append((kind, owner))
        if self.lock_busy:
            raise ScanAlreadyRunning(
                "a job is already running (kind=scan, owner=other, "
                "pid=1, since=t)")
        return self.handle

    def forget_manual(self, code):
        self.forget_calls.append(code)
        if isinstance(self.forget_result, Exception):
            raise self.forget_result
        return dict(self.forget_result, code=code)

    def get_run(self, run_id):
        return self.run if run_id == self.run.get("run_id") else None

    def get_watchlist(self):
        return {
            "disclaimer": "Research only.",
            "stale": {"is_stale": False,
                      "latest_trading_day": "2026-10-16",
                      "last_scan_trade_date": "2026-10-16",
                      "last_scan_at": "t"},
            "latest_run": {"run_id": "r1", "trade_date": "2026-10-16",
                           "finished_at": "t", "status": "ok",
                           "ai_status": "reviewed", "trigger": "cli",
                           "funnel": [], "sources": []},
            "items": [{"code": "600000", "name": "n",
                       "board": "main_sh", "industry": "i", "origin": "auto",
                       "change": "new", "rank": 1, "score": 90.0,
                       "sub_scores": {}, "reasons": [], "flags": [],
                       "ai": None, "added_at": "t", "note": None,
                       "scores_available": True, "unscored_reason": None}],
            "dropped": [], "requalified": [], "ai_demoted": [],
        }

    def run_scan(self, **kw):
        return {"run_id": "r1", "counts": {}, "changes": {}}


def _universe_df(n=4):
    recs = []
    for i in range(n):
        code = f"{600000 + i:06d}"
        recs.append({"code": code, "symbol": f"sh{code}", "board": "main_sh",
                     "is_st": False, "is_delisting": False,
                     "is_suspended": False})
    return pd.DataFrame.from_records(recs)


class _UniStatus:
    rows = 4
    latency_s = 1.0


class _FakeCalendar:
    def latest_completed_trading_day(self, now):
        return _dt.date(2026, 10, 15)

    def previous_trading_day(self, d):
        return _dt.date(2026, 10, 14)

    def is_trading_day(self, d):
        return d.weekday() < 5


def _payload(symbol, dates=("2026-10-14", "2026-10-15"), close="10"):
    rows = [[d, close, close, close, close, "1000"] for d in dates]
    return json.dumps({"code": 0, "data": {symbol: {"qfqday": rows}}})


def _fetch(url, params):
    symbol = params["param"].split(",")[0]
    return _payload(symbol)


def _home_stub(monkeypatch, exc=None):
    class _Stub:
        @staticmethod
        def require_home(context):
            if exc is not None:
                raise exc
            from src.watchlist.home import require_home as _real
            return _real(context)

        @staticmethod
        def home_fields():
            from src.watchlist.home import home_fields as _real
            return _real()
    monkeypatch.setitem(sys.modules, "src.watchlist.home", _Stub)


# ------------------------------------------------------- home gate ---------
def test_scan_prints_home_header_and_kline_reasons(capsys):
    run = {"run_id": "r1", "trade_date": "2026-10-12", "status": "ok",
           "ai_status": "reviewed", "ai_note": None, "duration_s": 1.0,
           "trigger": "cli", "failure_reason": None, "funnel": [],
           "coverage": None, "sources": [], "picks": [],
           "kline_reasons": {
               "counts": {"cache_hit": 100, "transport_error": 5,
                          "gap_filled": 0, "appended": 0,
                          "refetch_no_cache": 0, "refetch_gap_mismatch": 0,
                          "refetch_prev_close_mismatch": 0,
                          "refetch_last_bar_mismatch": 0,
                          "known_missing": 0, "budget_not_fetched": 0,
                          "stopped_unavailable": 0},
               "examples": {"transport_error": ["600001", "600002"]}}}
    svc = FakeSvc(run=run)
    rc = wl_cli._cmd_scan(
        _args(top=None, exclude_star=False, exclude_chinext=False,
              no_ai=True, json=False), svc)
    out = capsys.readouterr().out
    assert rc == 0
    assert "home=" in out and "pid=" in out
    assert "KLINE-REASONS" in out
    assert "cache_hit" in out and "transport_error" in out
    assert "600001" in out


def test_home_mismatch_refuses_scan(capsys, monkeypatch):
    class HomeMismatch(Exception):
        pass

    class _Stub:
        @staticmethod
        def require_home(context):
            raise HomeMismatch(
                "Vibe-Trading watchlist refuses to run (watchlist scan): "
                "this process resolved home C:\\x but the pinned home is "
                "E:\\y. Start via serve-8765.sh.")

        @staticmethod
        def home_fields():
            raise AssertionError("unreachable")

    monkeypatch.setitem(sys.modules, "src.watchlist.home", _Stub)
    with pytest.raises(SystemExit) as ei:
        wl_cli._cmd_scan(
            _args(top=None, exclude_star=False, exclude_chinext=False,
                  no_ai=True, json=False), FakeSvc())
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "refuses to run" in err and "pinned home" in err


def test_home_mismatch_refuses_forget_and_backfill(capsys, monkeypatch):
    class HomeMismatch(Exception):
        pass

    class _Stub:
        @staticmethod
        def require_home(context):
            raise HomeMismatch("refuses to run: pinned home is E:\\y.")

        @staticmethod
        def home_fields():
            raise AssertionError("unreachable")

    monkeypatch.setitem(sys.modules, "src.watchlist.home", _Stub)
    with pytest.raises(SystemExit) as ei:
        wl_cli._cmd_forget(_args(code="600519"), FakeSvc())
    assert ei.value.code == 2
    with pytest.raises(SystemExit):
        wl_cli._cmd_backfill(
            _args(rate=1.0, max=None, dry_run=True), FakeSvc(),
            universe_fn=lambda: (_universe_df(), _UniStatus()))
    assert "pinned home" in capsys.readouterr().err


# ------------------------------------------------------- F9 forget ---------
def test_forget_success_and_refusals(capsys):
    svc = FakeSvc()
    rc = wl_cli._cmd_forget(_args(code="600519"), svc)
    out = capsys.readouterr().out
    assert rc == 0
    assert svc.forget_calls == ["600519"]
    assert "forgot 600519" in out and "no suppression" in out

    svc2 = FakeSvc()
    svc2.forget_result = ValueError(
        "code 000001 is an auto item (origin='auto'); "
        "only manual items can be forgotten")
    rc2 = wl_cli._cmd_forget(_args(code="000001"), svc2)
    assert rc2 == 2
    assert "auto item" in capsys.readouterr().out

    svc3 = FakeSvc()
    svc3.forget_result = ValueError("unknown watchlist code: 999999")
    assert wl_cli._cmd_forget(_args(code="999999"), svc3) == 2


def test_forget_subcommand_parses():
    import argparse as _ap
    parser = _ap.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    wl_cli.add_subparser(sub)
    args = parser.parse_args(["watchlist", "forget", "600519"])
    assert args.watchlist_cmd == "forget"
    assert args.code == "600519"
    args = parser.parse_args(["watchlist", "backfill", "--dry-run",
                              "--max", "10", "--rate", "0.5"])
    assert args.watchlist_cmd == "backfill"
    assert args.dry_run is True and args.max == 10 and args.rate == 0.5


# ------------------------------------------------------- F5c backfill ------
def _uni(n=4):
    return _universe_df(n), _UniStatus()


def test_list_prints_home_header(tmp_path, capsys):
    svc = FakeSvc()
    rc = wl_cli._cmd_list(_args(json=False), svc)
    out = capsys.readouterr().out
    assert rc == 0
    assert "home=" in out and "(watchlist list)" in out


def test_backfill_dry_run_prints_shortfall_only(tmp_path, capsys):
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    svc = FakeSvc()  # NOTE: no acquire_job use needed, but present anyway
    rc = wl_cli._cmd_backfill(
        _args(rate=1.0, max=None, dry_run=True), svc,
        universe_fn=lambda: _uni(4),
        calendar_fn=lambda: (_FakeCalendar(), None), cache=cache)
    out = capsys.readouterr().out
    assert rc == 0
    assert "shortfall for trade_date=2026-10-15" in out
    assert "missing=4" in out
    assert "dry run: no requests sent" in out
    assert svc.acquire_calls == []  # no lock in dry-run


def _warming_runner(cache):
    """warm_runner injecting a fake fetcher (offline end-to-end)."""
    def run(symbols, trade_date, *, rate_per_sec, progress, lock_check,
            stop, max_symbols=None):
        from src.market_scanner.sources.kline import warm_cache
        return warm_cache(
            symbols, trade_date, rate_per_sec=rate_per_sec,
            progress=progress, lock_check=lock_check, stop=stop,
            fetch=_fetch, calendar=_FakeCalendar(), cache=cache,
            max_symbols=max_symbols)
    return run


def test_backfill_fills_and_logs(tmp_path, capsys):
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    svc = FakeSvc()
    rc = wl_cli._cmd_backfill(
        _args(rate=1.0, max=None, dry_run=False), svc,
        universe_fn=lambda: _uni(3),
        calendar_fn=lambda: (_FakeCalendar(), None), cache=cache,
        warm_runner=_warming_runner(cache))
    out = capsys.readouterr().out
    assert rc == 0
    assert svc.acquire_calls == [("backfill", "cli")]
    assert svc.handle.released is True
    assert "backfill done: attempted=3 filled=3" in out
    assert "remaining after this run: 0" in out
    assert cache.get_frame("600000")["meta"]["last_date"] == "2026-10-15"
    log = wl_cli._backfill_log_path(wl_cli._cli_home("watchlist backfill"))
    text = Path(log).read_text(encoding="utf-8")
    assert "backfill started" in text and "backfill finished" in text
    assert "trade_date=2026-10-15" in text


def test_backfill_max_caps_attempts(tmp_path, capsys):
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    svc = FakeSvc()
    rc = wl_cli._cmd_backfill(
        _args(rate=1.0, max=2, dry_run=False), svc,
        universe_fn=lambda: _uni(5),
        calendar_fn=lambda: (_FakeCalendar(), None), cache=cache,
        warm_runner=_warming_runner(cache))
    out = capsys.readouterr().out
    assert rc == 0
    assert "capped to --max 2" in out
    assert "attempted=2 filled=2" in out


def test_backfill_lock_busy_returns_3(tmp_path, capsys):
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    svc = FakeSvc()
    svc.lock_busy = True
    rc = wl_cli._cmd_backfill(
        _args(rate=1.0, max=None, dry_run=False), svc,
        universe_fn=lambda: _uni(3),
        calendar_fn=lambda: (_FakeCalendar(), None), cache=cache,
        warm_runner=_warming_runner(cache))
    assert rc == 3
    assert "already running" in capsys.readouterr().out


def test_backfill_universe_failure_returns_1(tmp_path, capsys):
    cache = KlineCache(path=tmp_path / "k.sqlite3")

    class _Empty:
        empty = True

    def no_universe():
        return _Empty(), _UniStatus()

    rc = wl_cli._cmd_backfill(
        _args(rate=1.0, max=None, dry_run=False), FakeSvc(),
        universe_fn=no_universe,
        calendar_fn=lambda: (_FakeCalendar(), None), cache=cache,
        warm_runner=_warming_runner(cache))
    assert rc == 1
    assert "universe fetch failed" in capsys.readouterr().out
