"""Intraday safety: no partial session bar is ever stored (offline).

Fake clock + fake fetcher only. The A-share session is 09:15-15:00
Asia/Shanghai; a bar dated today is final only at/after 15:00. All datetimes
here are naive and interpreted as Asia/Shanghai (same convention as the
snapshot-append guard and the scheduler clock).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from src.market_scanner.sources.kline import (
    KlineRateLimiter,
    _latest_completed_date,
    fetch_benchmark,
    fetch_klines,
    warm_cache,
)
from src.market_scanner.sources.kline_cache import KlineCache


# ------------------------------------------------------------------ fakes ---
class _IntradayCalendar:
    """Trading days 10-13..10-16; latest-completed follows the 15:00 close."""

    DAYS = {"2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"}

    def is_trading_day(self, d: dt.date) -> bool:
        return d.isoformat() in self.DAYS

    def previous_trading_day(self, d: dt.date) -> dt.date:
        d -= dt.timedelta(days=1)
        while d.isoformat() not in self.DAYS:
            d -= dt.timedelta(days=1)
        return d

    def latest_completed_trading_day(self, now: dt.datetime) -> dt.date:
        day = now.date()
        if self.is_trading_day(day) and now.time() >= dt.time(15, 0):
            return day
        return self.previous_trading_day(day)


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


def _snapshot(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    recs = [{"code": c, "price": p, "settlement": s,
             "volume": 1000.0 * 100, "amount": p * 1000.0 * 100,
             "open": s, "high": max(p, s), "low": min(p, s)}
            for c, p, s in rows]
    return pd.DataFrame(recs).set_index("code")


def _limiter() -> KlineRateLimiter:
    return KlineRateLimiter(rate_per_sec=100.0)


# ------------------------------------------------------- cutoff fallback --
def test_latest_completed_weekday_fallback():
    assert _latest_completed_date(None, dt.datetime(2026, 10, 15, 10, 0)
                                  ) == dt.date(2026, 10, 14)
    assert _latest_completed_date(None, dt.datetime(2026, 10, 15, 15, 30)
                                  ) == dt.date(2026, 10, 15)
    assert _latest_completed_date(None, dt.datetime(2026, 10, 17, 10, 0)
                                  ) == dt.date(2026, 10, 16)  # Saturday
    assert _latest_completed_date(None, dt.datetime(2026, 10, 19, 9, 0)
                                  ) == dt.date(2026, 10, 16)  # Monday pre-open


def test_latest_completed_uses_calendar_when_available():
    cal = _IntradayCalendar()
    assert _latest_completed_date(cal, dt.datetime(2026, 10, 15, 10, 0)
                                  ) == dt.date(2026, 10, 14)
    assert _latest_completed_date(cal, dt.datetime(2026, 10, 15, 15, 0)
                                  ) == dt.date(2026, 10, 15)


# ------------------------------------------------------- (a) warm-up ------
def test_warmup_at_1030_stores_only_previous_session(tmp_path):
    """Hourly warm-up mid-session: today's partial bar is dropped."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    seen: list[str] = []

    def fetch(url, params):
        seen.append(params["param"])
        symbol = params["param"].split(",")[0]
        return _payload(symbol, ["2026-10-14", "2026-10-15"])

    res = warm_cache(
        ["sh600000"], dt.date(2026, 10, 14), fetch=fetch,
        calendar=_IntradayCalendar(), cache=cache, limiter=_limiter(),
        now_fn=lambda: dt.datetime(2026, 10, 15, 10, 30))
    assert res.filled == 1
    assert res.intraday_dropped == 1
    assert cache.get_frame("600000")["meta"]["last_date"] == "2026-10-14"


def test_warmup_only_today_bar_is_missing_without_block(tmp_path):
    """A brand-new listing with just today's partial bar: missing for now,
    retried next run (no 24 h known-missing block)."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")

    def fetch(url, params):
        symbol = params["param"].split(",")[0]
        return _payload(symbol, ["2026-10-15"])

    res = warm_cache(
        ["sh600001"], dt.date(2026, 10, 14), fetch=fetch,
        calendar=_IntradayCalendar(), cache=cache, limiter=_limiter(),
        now_fn=lambda: dt.datetime(2026, 10, 15, 10, 30))
    assert res.filled == 0 and res.failed == 1
    assert cache.known_missing() == {}
    assert cache.get_frame("600001") is None


# ------------------------------------------------------- (b)(c) gap-fill -
def _gap_fetch(dates):
    def fetch(url, params):
        symbol = params["param"].split(",")[0]
        return _payload(symbol, dates)
    return fetch


def test_gap_fill_at_1400_drops_today(tmp_path):
    """Gap-fill during the session: merge stops at the previous session."""
    code = "600002"
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    cache.put_frame(code, _bars(code, ["2026-10-12", "2026-10-13"]),
                    "tencent")
    snap = _snapshot([(code, 10.0, 10.0)])
    frames, st = fetch_klines(
        ["sh600002"], trade_date=dt.date(2026, 10, 15),
        now_fn=lambda: dt.datetime(2026, 10, 15, 14, 0),
        snapshot=snap, calendar=_IntradayCalendar(), cache=cache,
        workers=1, fetch=_gap_fetch(
            ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]),
        limiter=_limiter(), snapshot_source="sina")
    assert st.kline_reasons["counts"]["gap_filled"] == 1
    assert st.kline_reasons["counts"]["intraday_dropped"] == 1
    out = frames["600002"]
    assert out["date"].max().date() == dt.date(2026, 10, 14)
    assert cache.get_frame(code)["meta"]["last_date"] == "2026-10-14"


def test_gap_fill_at_1530_keeps_today(tmp_path):
    """After the 15:00 close the day's bar is final and may be stored."""
    code = "600003"
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    cache.put_frame(code, _bars(code, ["2026-10-12", "2026-10-13"]),
                    "tencent")
    snap = _snapshot([(code, 10.0, 10.0)])
    frames, st = fetch_klines(
        ["sh600003"], trade_date=dt.date(2026, 10, 15),
        now_fn=lambda: dt.datetime(2026, 10, 15, 15, 30),
        snapshot=snap, calendar=_IntradayCalendar(), cache=cache,
        workers=1, fetch=_gap_fetch(
            ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]),
        limiter=_limiter(), snapshot_source="sina")
    assert st.kline_reasons["counts"]["gap_filled"] == 1
    assert st.kline_reasons["counts"]["intraday_dropped"] == 0
    assert frames["600003"]["date"].max().date() == dt.date(2026, 10, 15)
    assert cache.get_frame(code)["meta"]["last_date"] == "2026-10-15"


# ------------------------------------------------------- (d) weekend ------
def test_weekend_warmup_keeps_latest_session(tmp_path):
    """Saturday: the Friday session's bar is final and stored as-is."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    res = warm_cache(
        ["sh600004"], dt.date(2026, 10, 16),
        fetch=lambda u, p: _payload(
            p["param"].split(",")[0], ["2026-10-15", "2026-10-16",
                                       "2026-10-17"]),
        calendar=_IntradayCalendar(), cache=cache, limiter=_limiter(),
        now_fn=lambda: dt.datetime(2026, 10, 17, 10, 0))
    assert res.filled == 1
    assert cache.get_frame("600004")["meta"]["last_date"] == "2026-10-16"


# ------------------------------------------------------- full refetch ----
def test_full_refetch_intraday_drops_today_without_block(tmp_path):
    """No cache + fetch returning only today's bar: missing, no mark."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    frames, st = fetch_klines(
        ["sh600005"], trade_date=dt.date(2026, 10, 15),
        now_fn=lambda: dt.datetime(2026, 10, 15, 10, 30),
        snapshot=None, calendar=_IntradayCalendar(), cache=cache,
        workers=1,
        fetch=lambda u, p: _payload(
            p["param"].split(",")[0], ["2026-10-15"]),
        limiter=_limiter())
    assert frames == {}
    assert st.status == "failed"
    assert "no completed kline bars" in (st.error or "")
    assert st.kline_reasons["counts"]["intraday_dropped"] == 1
    assert st.kline_reasons["counts"]["refetch_no_cache"] == 1
    assert st.kline_reasons["counts"]["transport_error"] == 0
    assert cache.known_missing() == {}


def test_benchmark_intraday_falls_back_to_stale(tmp_path):
    """Benchmark fetch with only today's bar: stale cache served instead."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    cache.put_frame("sh000300", _bars("sh000300", ["2026-10-12"]),
                    "tencent")
    df, st = fetch_benchmark(
        "sh000300", fetch=lambda u, p: _payload("sh000300", ["2026-10-15"]),
        cache=cache, limiter=_limiter(),
        now_fn=lambda: dt.datetime(2026, 10, 15, 10, 30))
    assert st.status == "degraded"
    assert df is not None and df["date"].max().date() == dt.date(2026, 10, 12)
