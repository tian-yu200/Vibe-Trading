"""F5b: assess_shortfall + warm_cache (offline). Fake fetchers, tmp caches."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path

import pandas as pd
import pytest

from src.market_scanner.sources.kline import (
    KlineRateLimiter,
    Shortfall,
    WarmResult,
    assess_shortfall,
    resolve_kline_budget,
    warm_cache,
)
from src.market_scanner.sources.kline_cache import KlineCache


# ---------------------------------------------------------------- helpers ---
TRADE_DATE = dt.date(2026, 10, 15)
NOW = dt.datetime(2026, 10, 15, 20, 0)
#: The one clock test_assess_shortfall_counts injects as ``now_fn``. Every
#: timestamp that test compares against it must be derived from it, never
#: from the wall clock (a wall-clock-derived row silently ages past
#: MAX_AGE_DAYS and the test starts failing on a real-world date).
_FIXED_NOW = NOW.replace(hour=12)


class _FakeCalendar:
    DAYS = {"2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"}

    def previous_trading_day(self, d: dt.date) -> dt.date:
        if d.isoformat() == "2026-10-15":
            return dt.date(2026, 10, 14)
        raise AssertionError(d)

    def is_trading_day(self, d: dt.date) -> bool:
        return d.isoformat() in self.DAYS


def _bars(code: str, dates: list[str], close: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "code": code, "date": pd.Timestamp(d), "open": close,
        "high": close, "low": close, "close": close, "volume": 1000.0,
        "amount": close * 1000.0 * 100,
    } for d in dates])


def _payload(symbol: str, dates: list[str], close: float = 10.0) -> str:
    rows = [[d, str(close), str(close), str(close), str(close), "1000"]
            for d in dates]
    return json.dumps({"code": 0, "data": {symbol: {"qfqday": rows}}})


def _code(i: int) -> str:
    return f"{600000 + i:06d}"


def _sym(i: int) -> str:
    return f"sh{_code(i)}"


def test_assess_shortfall_counts(tmp_path):
    """missing / gap / stale / appendable / current / known-missing."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    # 0: missing entirely
    # 1: gap (ends 10-13)
    cache.put_frame(_code(1), _bars(_code(1), ["2026-10-12", "2026-10-13"]),
                    "tencent")
    # 2: stale (ends prev day but fetched 40 d ago)
    cache.put_frame(_code(2), _bars(_code(2), ["2026-10-13", "2026-10-14"]),
                    "tencent")
    # Derived from the SAME fixed clock the test passes as now_fn (below), never
    # from the wall clock: otherwise the row ages past MAX_AGE_DAYS on its own
    # and this test starts failing on a real-world date (the 2026-10-24 bomb).
    stale_at = (_FIXED_NOW - dt.timedelta(days=40)).isoformat(
        timespec="seconds")
    with sqlite3.connect(str(tmp_path / "k.sqlite3")) as db:
        db.execute("UPDATE kline_meta SET fetched_at = ? WHERE code = ?",
                   (stale_at, _code(2)))
    # 3: appendable (ends prev day, fresh)
    cache.put_frame(_code(3), _bars(_code(3), ["2026-10-13", "2026-10-14"]),
                    "tencent")
    # 4: current (ends trade date)
    cache.put_frame(_code(4), _bars(_code(4), ["2026-10-14", "2026-10-15"]),
                    "tencent")
    # 5: live known-missing (skipped, not shortfall)
    cache.mark_known_missing(_sym(5), retry_after_s=86400)

    short = assess_shortfall(
        [_sym(i) for i in range(6)], TRADE_DATE, calendar=_FakeCalendar(),
        cache=cache, rate_per_sec=2.0,
        now_fn=lambda: _FIXED_NOW)
    assert isinstance(short, Shortfall)
    assert short.missing == 1
    assert short.gap == 1
    assert short.stale == 1
    assert short.appendable == 1
    assert short.est_requests == 3
    assert short.est_seconds == pytest.approx(1.5)


def test_resolve_kline_budget():
    assert resolve_kline_budget(100.0, 600.0, 5400.0) == 600.0
    assert resolve_kline_budget(700.0, 600.0, 5400.0) == 700.0
    assert resolve_kline_budget(99999.0, 600.0, 5400.0) == 5400.0


def test_warm_cache_fills_and_resumes(tmp_path):
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    # STAR migration marker must survive the warm-up untouched.
    import sqlite3 as _sq
    with _sq.connect(str(tmp_path / "k.sqlite3")) as db:
        db.execute("INSERT INTO kline_meta VALUES ('_migration', 'k', 't', 'migration')")
    calls: list[str] = []

    def fetch(url, params):
        calls.append(params["param"])
        symbol = params["param"].split(",")[0]
        return _payload(symbol, ["2026-10-14", "2026-10-15"])

    seen: list[tuple[int, int]] = []
    res = warm_cache(
        [_sym(0), _sym(1)], TRADE_DATE, fetch=fetch,
        calendar=_FakeCalendar(), cache=cache,
        limiter=KlineRateLimiter(rate_per_sec=100.0),
        progress=lambda d, t: seen.append((d, t)),
    )
    assert isinstance(res, WarmResult)
    assert res.filled == 2 and res.failed == 0
    assert res.aborted is False and res.stopped_unavailable is False
    assert res.requests == 2
    assert seen[-1] == (2, 2)
    assert cache.get_frame(_code(0))["meta"]["last_date"] == "2026-10-15"
    with _sq.connect(str(tmp_path / "k.sqlite3")) as db:
        marker = db.execute(
            "SELECT * FROM kline_meta WHERE code = '_migration'").fetchall()
    assert len(marker) == 1
    # rerun: everything current or appendable -> nothing to do, no requests
    calls.clear()
    res2 = warm_cache(
        [_sym(0), _sym(1)], TRADE_DATE, fetch=fetch,
        calendar=_FakeCalendar(), cache=cache,
        limiter=KlineRateLimiter(rate_per_sec=100.0))
    assert res2.attempted == 0 and res2.filled == 0
    assert calls == []


def test_warm_cache_gap_tail_and_skips(tmp_path):
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    cache.put_frame(_code(1), _bars(_code(1), ["2026-10-12", "2026-10-13"]),
                    "tencent")
    cache.put_frame(_code(3), _bars(_code(3), ["2026-10-13", "2026-10-14"]),
                    "tencent")  # appendable -> skipped
    cache.mark_known_missing(_sym(5), retry_after_s=86400)  # skipped
    seen_bars: dict[str, str] = {}

    def fetch(url, params):
        seen_bars[params["param"].split(",")[0]] = params["param"].split(",")[4]
        symbol = params["param"].split(",")[0]
        dates = (["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]
                 if symbol == _sym(1) else ["2026-10-14", "2026-10-15"])
        return _payload(symbol, dates)

    res = warm_cache(
        [_sym(0), _sym(1), _sym(3), _sym(5)], TRADE_DATE, fetch=fetch,
        calendar=_FakeCalendar(), cache=cache,
        limiter=KlineRateLimiter(rate_per_sec=100.0))
    assert res.attempted == 2  # missing + gap only
    assert res.skipped_known_missing == 1
    assert res.filled == 2
    assert seen_bars[_sym(1)] == "4"  # gap(1) + overlap(3)
    assert seen_bars[_sym(0)] == "260"  # full refetch for missing
    assert cache.get_frame(_code(1))["meta"]["last_date"] == "2026-10-15"


def test_warm_cache_rate_capped_at_1rps(tmp_path):
    """rate_per_sec=10 still spaces at <= 1 req/s (sequential + limiter)."""
    import time as _time
    cache = KlineCache(path=tmp_path / "k.sqlite3")

    def fetch(url, params):
        symbol = params["param"].split(",")[0]
        return _payload(symbol, ["2026-10-15"])

    t0 = _time.monotonic()
    res = warm_cache([_sym(10), _sym(11), _sym(12)], TRADE_DATE,
                     fetch=fetch, calendar=_FakeCalendar(), cache=cache,
                     rate_per_sec=10.0)
    elapsed = _time.monotonic() - t0
    assert res.filled == 3
    assert elapsed >= 1.5  # 3 requests at capped 1 rps


def test_warm_cache_deadline_lock_and_stop(tmp_path):
    cache = KlineCache(path=tmp_path / "k.sqlite3")

    def fetch(url, params):
        symbol = params["param"].split(",")[0]
        return _payload(symbol, ["2026-10-15"])

    import time as _time
    res = warm_cache([_sym(0)], TRADE_DATE, fetch=fetch,
                     calendar=_FakeCalendar(), cache=cache,
                     limiter=KlineRateLimiter(rate_per_sec=100.0),
                     deadline=_time.monotonic() - 1.0)
    assert res.deadline_hit is True and res.filled == 0

    class JobLockLost(RuntimeError):
        pass

    def bad_check():
        raise JobLockLost("job lock lost (deleted or taken over)")

    res2 = warm_cache([_sym(1)], TRADE_DATE, fetch=fetch,
                      calendar=_FakeCalendar(), cache=cache,
                      limiter=KlineRateLimiter(rate_per_sec=100.0),
                      lock_check=bad_check)
    assert res2.aborted is True and "lock lost" in (res2.error or "")

    stop = threading.Event()
    stop.set()
    res3 = warm_cache([_sym(2)], TRADE_DATE, fetch=fetch,
                      calendar=_FakeCalendar(), cache=cache,
                      limiter=KlineRateLimiter(rate_per_sec=100.0), stop=stop)
    assert res3.aborted is True


def test_warm_cache_transport_not_marked_known_missing(tmp_path):
    """F5d: transport failures are retried next run (no 24 h block)."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")

    def fetch(url, params):
        raise TimeoutError("ReadTimeout: timed out")

    res = warm_cache(
        [_sym(7)], TRADE_DATE, fetch=fetch, calendar=_FakeCalendar(),
        cache=cache, limiter=KlineRateLimiter(rate_per_sec=100.0),
        stop_consecutive=1000, stop_min_elapsed_s=10 ** 9)
    assert res.failed == 1 and res.filled == 0
    assert cache.known_missing() == {}
    assert res.stopped_unavailable is False


# ------------------------------------------- FIX2 phase (e): intraday drop ---
# 2026-10-19 10:00 Asia/Shanghai (trading hours), target = the previous
# Friday. This is the exact shape of the live warm-up that failed 187/186.
_MONDAY_1000 = dt.datetime(2026, 10, 19, 10, 0)
_FRIDAY = dt.date(2026, 10, 16)
_DAY = dt.date(2026, 10, 19)  # the forming (partial) Monday bar


class _MondayCalendar:
    """Calendar for a Monday-morning warm-up targeting the previous Friday."""

    DAYS = {"2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15",
            "2026-10-16", "2026-10-19"}

    def previous_trading_day(self, d):
        order = sorted(self.DAYS)
        return dt.date.fromisoformat(order[order.index(d.isoformat()) - 1])

    def is_trading_day(self, d):
        return d.isoformat() in self.DAYS

    def latest_completed_trading_day(self, now):
        # 10:00 Monday: Friday's session is the newest completed one.
        return _FRIDAY


def test_warm_cache_keeps_completed_bars_on_intraday_fetch(tmp_path):
    """Regression (Q3): a fetch ending with today's partial bar still fills.

    The frame is NOT dropped wholesale: the completed bars are stored, the
    partial bar is not, ``filled`` counts and nothing is a failure.
    """
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    dates = ["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19"]

    def fetch(url, params):
        return _payload(params["param"].split(",")[0], dates)

    res = warm_cache(
        [_sym(0)], _FRIDAY, fetch=fetch, calendar=_MondayCalendar(),
        cache=cache, now_fn=lambda: _MONDAY_1000,
        limiter=KlineRateLimiter(rate_per_sec=100.0))

    assert res.filled == 1 and res.failed == 0
    assert res.intraday_dropped == 1           # today's bar was dropped
    assert res.intraday_no_store == 0
    assert res.stopped_unavailable is False
    stored = cache.get_frame(_code(0))["bars"]
    assert stored["date"].dt.date.max().isoformat() == "2026-10-16"
    assert len(stored) == 3


def test_warm_cache_intraday_drop_keeps_completed_bars(tmp_path):
    """Q3 / scenario A: the completed bars survive, today's bar is dropped.

    A fetch returning Friday's bars plus Monday's partial bar must store the
    completed rows and count as filled, not failed.
    """
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    completed = ["2026-10-14", "2026-10-15", "2026-10-16"]

    def fetch(url, params):
        symbol = params["param"].split(",")[0]
        return _payload(symbol, completed + ["2026-10-19"])

    res = warm_cache(
        [_sym(0)], _FRIDAY, fetch=fetch, calendar=_MondayCalendar(),
        cache=cache, limiter=KlineRateLimiter(rate_per_sec=100.0))
    assert res.filled == 1 and res.failed == 0
    assert res.intraday_dropped == 1
    assert res.intraday_no_store == 0
    assert res.stopped_unavailable is False and res.error is None
    frame = cache.get_frame(_code(0))["bars"]
    stored = {d.date().isoformat() for d in frame["date"]}
    assert stored == set(completed)
    assert cache.get_frame(_code(0))["meta"]["last_date"] == "2026-10-16"


def test_intraday_only_frame_is_not_an_error_and_notes_ok(tmp_path):
    """Boss part (2): a pure intraday drop is tallied apart, never an error.

    Before the fix this exact shape (187 attempted / 187 failed / 186
    intraday) fed the limiter a pseudo error per symbol and tripped the F4
    outage stop ("52% errors over the last 60 requests").
    """
    cache = KlineCache(path=tmp_path / "k2.sqlite3")

    def fetch(url, params):
        return _payload(params["param"].split(",")[0], ["2026-10-19"])

    res = warm_cache(
        [_sym(0)], _FRIDAY, fetch=fetch, calendar=_MondayCalendar(),
        cache=cache, now_fn=lambda: _MONDAY_1000,
        limiter=KlineRateLimiter(rate_per_sec=100.0),
        stop_consecutive=1000, stop_min_elapsed_s=10 ** 9,
    )
    assert res.attempted == 1
    assert res.filled == 0
    assert res.failed == 0
    assert res.intraday_no_store == 1
    assert res.intraday_dropped == 1
    assert res.stopped_unavailable is False
    assert res.error is None
    # Nothing is stored, and the symbol is not marked known-missing.
    assert cache.get_frame(_code(0)) is None
    assert cache.known_missing() == {}


class _RecordingLimiter(KlineRateLimiter):
    """Limiter that records every verdict it was fed (spy)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.verdicts: list[bool] = []
        self.unavailable_marks: list[str] = []

    def note_result(self, ok, **kw):
        self.verdicts.append(bool(ok))
        return super().note_result(ok, **kw)

    def _mark_unavailable(self, reason):
        self.unavailable_marks.append(reason)
        super()._mark_unavailable(reason)


def test_live_fingerprint_replay_has_zero_limiter_errors(tmp_path):
    """2026-10-19 09:58 replay: 187 intraday-only symbols, 0 limiter errors.

    The live run logged attempted 187 / failed 187 / intraday_dropped 186 /
    "52% errors over the last 60 requests". With the fix the same traffic
    must produce 0 errors seen by the limiter and no outage stop, so the
    warm-up keeps working instead of aborting after 187 symbols.
    """
    cache = KlineCache(path=tmp_path / "k3.sqlite3")
    codes = [_code(i) for i in range(187)]
    limiter = _RecordingLimiter(rate_per_sec=1000.0)

    def fetch(url, params):
        symbol = params["param"].split(",")[0]
        return _payload(symbol, [_DAY.isoformat()])

    res = warm_cache([f"sh{c}" for c in codes], _FRIDAY,
                     fetch=fetch, calendar=_MondayCalendar(), cache=cache,
                     limiter=limiter, now_fn=lambda: _MONDAY_1000)
    # Live fingerprint: every symbol attempted, none stored.
    assert res.attempted == 187 and res.filled == 0
    assert res.failed == 0 and res.requests == 187
    assert res.intraday_dropped == 187 and res.intraday_no_store == 187
    # The fix: the limiter never saw a single error.
    assert limiter.verdicts and all(limiter.verdicts)
    assert limiter.unavailable_marks == []
    assert res.stopped_unavailable is False
    assert res.error is None
    assert cache.known_missing() == {}
