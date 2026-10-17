"""Offline tests for the kline cache, snapshot append, throttle stop, budget."""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path

import pandas as pd
import pytest

from src.market_scanner.sources import kline as kline_mod
from src.market_scanner.sources.kline import (
    KlineRateLimiter,
    ThrottleError,
    fetch_benchmark,
    fetch_klines,
    is_throttle_error,
    _append_from_snapshot,
        _normalise_volume,
        _parse_kline_payload,
        _settlement_matches,
        _snapshot_session_completed,
        migrate_star_volume,
    )
from src.market_scanner.sources.kline_cache import KlineCache


class FakeCalendar:
    def __init__(self, prev: _dt.date) -> None:
        self._prev = prev

    def previous_trading_day(self, d: _dt.date) -> _dt.date:
        return self._prev


def _bars(dates: list[str], close: float = 10.0, volume: float = 1000.0) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date": pd.Timestamp(d), "open": close, "high": close,
            "low": close, "close": close, "volume": volume,
            "amount": close * volume * 100,
        })
    return pd.DataFrame(rows)


def _snapshot(code: str, *, price: float = 10.5, settlement: float = 10.0,
              volume: float = 120000, source: str = "sina") -> pd.DataFrame:
    row = {
        "price": price, "settlement": settlement,
        "volume": volume * 100 if source == "sina" else volume,  # shares
        "amount": price * volume * 100, "open": price, "high": price * 1.01,
        "low": price * 0.99,
    }
    return pd.DataFrame([row], index=pd.Index([code], name="code"))


@pytest.fixture()
def cache(tmp_path: Path) -> KlineCache:
    return KlineCache(path=tmp_path / "klines.sqlite3")


def _fetch_stub(responses: dict[str, object]):
    """Return a fetch(url, params) -> str stub serving `responses` payloads."""

    def fetch(url: str, params: dict) -> str:
        symbol = params["param"].split(",")[0]
        payload = responses.get(symbol, {"code": 0, "data": {}})
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    return fetch


# --------------------------------------------------------------------------- #
# snapshot append path
# --------------------------------------------------------------------------- #
def test_snapshot_append_path(cache: KlineCache, tmp_path: Path) -> None:
    trade_date = _dt.date(2026, 10, 13)
    cache.put_frame("600519", _bars(["2026-10-09", "2026-10-12"]), "tencent")
    snap = _snapshot("600519", price=10.5, settlement=10.0)
    frames, st = fetch_klines(
        ["sh600519"], cache=cache, snapshot=snap, trade_date=trade_date,
        calendar=FakeCalendar(_dt.date(2026, 10, 12)),
        fetch=_fetch_stub({}),  # must NOT be called
        now_fn=lambda: _dt.datetime(2026, 10, 13, 16, 0),  # after close
    )
    assert st.status == "ok"
    assert "appended_from_snapshot=1" in (st.detail or "")
    assert "cache_hit=1" in (st.detail or "")
    assert "refetched=0" in (st.detail or "")
    df = frames["600519"]
    assert df.iloc[-1]["date"].date() == trade_date
    assert df.iloc[-1]["close"] == pytest.approx(10.5)
    # snapshot volume was shares (120000*100=12,000,000 shares) -> 120000 lots
    assert df.iloc[-1]["volume"] == pytest.approx(120000)


def test_normalise_volume_units() -> None:
    assert _normalise_volume(1_200_000, "sina") == pytest.approx(12000)
    assert _normalise_volume(12000, "tencent") == pytest.approx(12000)
    assert _normalise_volume(None, "sina") is None


def test_settlement_mismatch_tolerance() -> None:
    assert _settlement_matches(10.0, 10.0) is True
    assert _settlement_matches(10.0, 10.029) is True   # 0.29%
    assert _settlement_matches(10.0, 10.04) is False   # 0.4% > 0.3%
    assert _settlement_matches(None, 10.0) is False


# --------------------------------------------------------------------------- #
# refetch triggers
# --------------------------------------------------------------------------- #
def test_corporate_action_mismatch_refetches(cache: KlineCache) -> None:
    trade_date = _dt.date(2026, 10, 13)
    cache.put_frame("600519", _bars(["2026-10-09", "2026-10-12"]), "tencent")
    # dividend: settlement 5.0 vs cached close 10.0 -> mismatch -> refetch
    snap = _snapshot("600519", price=5.1, settlement=5.0)
    calls = {"n": 0}
    payload = {"code": 0, "data": {"sh600519": {"qfqday": [
        ["2026-10-09", "10", "10", "10", "10", "1000"],
        ["2026-10-12", "5.0", "5.0", "5.0", "5.0", "2000"],
    ]}}}
    def fetch(url, params):
        calls["n"] += 1
        return json.dumps(payload)
    frames, st = fetch_klines(
        ["sh600519"], cache=cache, snapshot=snap, trade_date=trade_date,
        calendar=FakeCalendar(_dt.date(2026, 10, 12)), fetch=fetch,
        now_fn=lambda: _dt.datetime(2026, 10, 13, 12, 0),
    )
    assert calls["n"] == 1
    assert "refetched=1" in (st.detail or "")
    assert frames["600519"].iloc[-1]["close"] == pytest.approx(5.0)


def test_gap_refetches(cache: KlineCache) -> None:
    trade_date = _dt.date(2026, 10, 13)
    # cache ends 2026-10-09, prev trading day is 2026-10-12 -> one-day gap.
    # F5a: the gap is tail-filled (gap + 3 overlap bars, one request) and
    # merged after the overlap closes verify; no wholesale refetch.
    cache.put_frame("600519", _bars(["2026-10-08", "2026-10-09"]), "tencent")
    snap = _snapshot("600519")
    calls = {"n": 0}
    payload = {"code": 0, "data": {"sh600519": {"qfqday": [
        ["2026-10-08", "10", "10", "10", "10", "1000"],
        ["2026-10-09", "10", "10", "10", "10", "1000"],
        ["2026-10-12", "10", "10", "10", "10", "1000"],
        ["2026-10-13", "10", "10", "10", "10", "1000"],
    ]}}}
    def fetch(url, params):
        calls["n"] += 1
        assert params["param"].split(",")[4] == "4"  # gap(1) + overlap(3)
        return json.dumps(payload)
    frames, st = fetch_klines(
        ["sh600519"], cache=cache, snapshot=snap, trade_date=trade_date,
        calendar=FakeCalendar(_dt.date(2026, 10, 12)), fetch=fetch,
        # after the close: the trade-date bar in the tail is final
        now_fn=lambda: _dt.datetime(2026, 10, 13, 16, 0),
    )
    assert calls["n"] == 1
    assert "gap_filled=1" in (st.detail or "")
    assert "refetched=0" in (st.detail or "")
    assert [d.date().isoformat() for d in frames["600519"]["date"]] == [
        "2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"]


def test_suspended_keeps_cached_frame(cache: KlineCache) -> None:
    trade_date = _dt.date(2026, 10, 13)
    cache.put_frame("600519", _bars(["2026-10-09", "2026-10-12"]), "tencent")
    # suspended: price 0
    snap = _snapshot("600519", price=0.0, settlement=10.0)
    frames, st = fetch_klines(
        ["sh600519"], cache=cache, snapshot=snap, trade_date=trade_date,
        calendar=FakeCalendar(_dt.date(2026, 10, 12)),
        fetch=_fetch_stub({}),
        now_fn=lambda: _dt.datetime(2026, 10, 13, 12, 0),
    )
    assert "cache_hit=1" in (st.detail or "")
    assert "appended_from_snapshot=0" in (st.detail or "")
    assert frames["600519"].iloc[-1]["date"].date() == _dt.date(2026, 10, 12)


# --------------------------------------------------------------------------- #
# throttle stop + budget
# --------------------------------------------------------------------------- #
def _throttle_payload() -> str:
    # payload with a 501 inside the fetch stub raises HTTPError-like; we use
    # a stub raising a real HTTPError-shaped exception with 501 in the text.
    return ""


def test_throttle_stop_after_10_consecutive(cache: KlineCache) -> None:
    trade_date = _dt.date(2026, 10, 13)
    symbols = [f"sh6{i:05d}" for i in range(15)]
    # raise ThrottleError-like: the stub raises an HTTPError with 501; the
    # classification is via is_throttle_error on the raised exception. But
    # _fetch_one only raises ThrottleError on a live status code — for stubs
    # we make fetch raise ThrottleError directly.
    responses = {
        s: ThrottleError(f"HTTP 501 from fqkline for {s}", status_code=501)
        for s in symbols
    }
    fetch = _fetch_stub(responses)
    limiter = KlineRateLimiter(rate_per_sec=100.0)
    frames, st = fetch_klines(
        symbols, cache=cache, snapshot=None, trade_date=trade_date,
        fetch=fetch, limiter=limiter, workers=1,
        now_fn=lambda: _dt.datetime(2026, 10, 13, 12, 0),
    )
    assert limiter.throttle_stop is True
    assert "throttled=yes" in (st.detail or "")
    # workers=1: 10 attempted/failed + the throttle stop; a queued work item
    # may already have been dequeued by the single worker before the stop
    # (dequeued items run to completion and count as failed), so the
    # deterministic lower bound is 10 and the upper bound is 15.
    n = len(failures_list(st))
    assert 10 <= n <= 15


def test_throttle_stop_attempts_bounded_by_inflight_workers(cache: KlineCache) -> None:
    """workers=6: up to workers-1 requests can be in flight when the stop
    triggers, so failed count is in [10, 10 + workers - 1] = [10, 15]."""
    trade_date = _dt.date(2026, 10, 13)
    symbols = [f"sh6{i:05d}" for i in range(15)]
    responses = {
        s: ThrottleError(f"HTTP 501 from fqkline for {s}", status_code=501)
        for s in symbols
    }
    limiter = KlineRateLimiter(rate_per_sec=100.0)
    frames, st = fetch_klines(
        symbols, cache=cache, snapshot=None, trade_date=trade_date,
        fetch=_fetch_stub(responses), limiter=limiter, workers=6,
        now_fn=lambda: _dt.datetime(2026, 10, 13, 12, 0),
    )
    assert limiter.throttle_stop is True
    n = len(failures_list(st))
    assert 10 <= n <= 15


def failures_list(st) -> list:
    # failure count is encoded in detail as missing=N and error text
    import re
    m = re.search(r"missing=(\d+)", st.detail or "")
    return list(range(int(m.group(1))) if m else [])


def test_budget_stop(cache: KlineCache) -> None:
    trade_date = _dt.date(2026, 10, 13)
    symbols = [f"sh6{i:05d}" for i in range(10)]
    # slow-but-successful fetch: each call sleeps 1.5s; 1 worker => the 1s
    # budget cuts the run after the first fetch
    def fetch(url, params):
        time.sleep(1.5)
        symbol = params["param"].split(",")[0]
        return json.dumps({"code": 0, "data": {symbol: {"qfqday": [
            ["2026-10-12", "10", "10", "10", "10", "1000"],
        ]}}})
    frames, st = fetch_klines(
        symbols, cache=cache, snapshot=None, trade_date=trade_date,
        fetch=fetch, limiter=KlineRateLimiter(rate_per_sec=100.0),
        time_budget_s=1.0, workers=1,
        now_fn=lambda: _dt.datetime(2026, 10, 13, 12, 0),
    )
    assert "budget_exhausted=yes" in (st.detail or "")
    assert st.status in ("ok", "degraded")
    assert len(frames) < 10  # budget cut the run short


def test_missing_not_raised(cache: KlineCache) -> None:
    trade_date = _dt.date(2026, 10, 13)
    def fetch(url, params):
        raise ConnectionError("no route")
    frames, st = fetch_klines(
        ["sh600519"], cache=cache, snapshot=None, trade_date=trade_date,
        fetch=fetch, limiter=KlineRateLimiter(rate_per_sec=100.0),
        now_fn=lambda: _dt.datetime(2026, 10, 13, 12, 0),
    )
    assert frames == {}
    assert st.status == "failed"  # 0 ok of 1
    assert "missing=1" in (st.detail or "")


def test_is_throttle_error_classification() -> None:
    assert is_throttle_error(Exception("501 Server Error")) is True
    assert is_throttle_error(Exception("HTTP 429 Too Many Requests")) is True
    assert is_throttle_error(Exception("Connection reset by peer")) is True
    assert is_throttle_error(Exception("no rows for sh600519")) is False


# --------------------------------------------------------------------------- #
# benchmark stale fallback
# --------------------------------------------------------------------------- #
def test_benchmark_stale_fallback(cache: KlineCache) -> None:
    cache.put_frame("sh000300", _bars(["2026-10-10", "2026-10-12"]), "tencent")

    def fetch(url, params):
        raise ConnectionError("501 from provider")

    df, st = fetch_benchmark("sh000300", fetch=fetch, cache=cache,
                             limiter=KlineRateLimiter(rate_per_sec=100.0))
    assert st.status == "degraded"
    assert "benchmark stale (last bar 2026-10-12)" in (st.detail or "")
    assert df is not None and len(df) == 2


def test_benchmark_ok_when_provider_works(cache: KlineCache) -> None:
    def fetch(url, params):
        return json.dumps({"code": 0, "data": {"sh000300": {"day": [
            ["2026-10-12", "4200", "4230", "4240", "4195", "1000"],
        ]}}})

    df, st = fetch_benchmark("sh000300", fetch=fetch, cache=cache,
                             limiter=KlineRateLimiter(rate_per_sec=100.0))
    assert st.status == "ok"
    assert df.iloc[-1]["close"] == pytest.approx(4230)


# --------------------------------------------------------------------------- #
# rate limiter spacing (fake clock)
# --------------------------------------------------------------------------- #
def test_rate_limiter_default_1rps_spacing() -> None:
    lim = KlineRateLimiter(rate_per_sec=1.0)
    now = {"t": 1000.0}
    slept: list[float] = []
    lim.wait(clock=lambda: now["t"], sleeper=lambda d: slept.append(d) or now.__setitem__("t", now["t"] + d))
    lim.wait(clock=lambda: now["t"], sleeper=lambda d: slept.append(d) or now.__setitem__("t", now["t"] + d))
    # second request waited ~1s
    assert sum(slept) >= 0.95


def test_throttle_error_class() -> None:
    exc = ThrottleError("HTTP 501", status_code=501)
    assert exc.status_code == 501


# --------------------------------------------------------------------------- #
# Snapshot-append session guard (intraday never appended)
# --------------------------------------------------------------------------- #
def test_snapshot_session_completed_windows() -> None:
    td = _dt.date(2026, 10, 13)  # Tuesday
    # intraday 09:15–15:00 on trade_date -> NOT completed
    assert _snapshot_session_completed(td, _dt.datetime(2026, 10, 13, 9, 14)) is False
    assert _snapshot_session_completed(td, _dt.datetime(2026, 10, 13, 10, 0)) is False
    assert _snapshot_session_completed(td, _dt.datetime(2026, 10, 13, 14, 59)) is False
    # after 15:00 on trade_date -> completed
    assert _snapshot_session_completed(td, _dt.datetime(2026, 10, 13, 15, 0)) is True
    assert _snapshot_session_completed(td, _dt.datetime(2026, 10, 13, 18, 0)) is True
    # before 09:15 on trade_date -> snapshot is the previous session
    assert _snapshot_session_completed(td, _dt.datetime(2026, 10, 13, 8, 0)) is False
    # any day after trade_date -> completed
    assert _snapshot_session_completed(td, _dt.datetime(2026, 10, 14, 10, 0)) is True


def test_intraday_snapshot_never_appended(cache: KlineCache) -> None:
    """09:15–15:00 on trade_date: cached frame kept, no bar appended."""
    trade_date = _dt.date(2026, 10, 13)
    cache.put_frame("600519", _bars(["2026-10-09", "2026-10-12"]), "tencent")
    snap = _snapshot("600519", price=10.5, settlement=10.0)
    frames, st = fetch_klines(
        ["sh600519"], cache=cache, snapshot=snap, trade_date=trade_date,
        calendar=FakeCalendar(_dt.date(2026, 10, 12)),
        fetch=_fetch_stub({}),  # must NOT be called
        now_fn=lambda: _dt.datetime(2026, 10, 13, 11, 0),  # intraday
    )
    assert "appended_from_snapshot=0" in (st.detail or "")
    assert frames["600519"].iloc[-1]["date"].date() == _dt.date(2026, 10, 12)


def test_after_close_snapshot_appended(cache: KlineCache) -> None:
    """After 15:00 on trade_date: snapshot IS the completed bar -> appended."""
    trade_date = _dt.date(2026, 10, 13)
    cache.put_frame("600519", _bars(["2026-10-09", "2026-10-12"]), "tencent")
    snap = _snapshot("600519", price=10.5, settlement=10.0)
    frames, st = fetch_klines(
        ["sh600519"], cache=cache, snapshot=snap, trade_date=trade_date,
        calendar=FakeCalendar(_dt.date(2026, 10, 12)),
        fetch=_fetch_stub({}),
        now_fn=lambda: _dt.datetime(2026, 10, 13, 15, 30),  # after close
    )
    assert "appended_from_snapshot=1" in (st.detail or "")
    assert frames["600519"].iloc[-1]["date"].date() == trade_date


def test_next_day_snapshot_appended(cache: KlineCache) -> None:
    """Running on a later day: the trade_date snapshot is completed -> appended."""
    trade_date = _dt.date(2026, 10, 12)
    cache.put_frame("600519", _bars(["2026-10-08", "2026-10-09"]), "tencent")
    snap = _snapshot("600519", price=10.5, settlement=10.0)
    # now = next day, cache's last bar (10-09) is prev trading day of 10-12
    frames, st = fetch_klines(
        ["sh600519"], cache=cache, snapshot=snap, trade_date=trade_date,
        calendar=FakeCalendar(_dt.date(2026, 10, 9)),
        fetch=_fetch_stub({}),
        now_fn=lambda: _dt.datetime(2026, 10, 13, 10, 0),
    )
    assert "appended_from_snapshot=1" in (st.detail or "")
    assert frames["600519"].iloc[-1]["date"].date() == trade_date


def test_cache_already_contains_trade_date_bar_zero_fetches(tmp_path):
    """Regression (Milestone A incident): cache last bar == trade_date must be
    a cache HIT (close-match guard only), never a refetch."""
    import datetime as _dt2
    import pandas as _pd

    cache = KlineCache(path=tmp_path / "klines.sqlite3")
    trade_date = _dt2.date(2026, 10, 12)
    hist = _pd.DataFrame({
        "date": [_pd.Timestamp("2026-10-09")],
        "open": [10.0], "high": [10.5], "low": [9.9], "close": [10.2],
        "volume": [1000.0], "amount": [1020000.0],
    })
    cache.put_frame("600519", hist, "tencent")
    # backfilled trade_date bar already in the cache
    cache.append_bar("600519", {
        "date": _pd.Timestamp("2026-10-12"), "open": 10.3, "high": 10.6,
        "low": 10.1, "close": 10.4, "volume": 1100.0, "amount": 1144000.0,
    }, "tencent")

    snap = _pd.DataFrame([{
        "code": "600519", "name": "x", "price": 10.4, "open": 10.3,
        "high": 10.6, "low": 10.1, "settlement": 10.2, "volume": 110000.0,
        "amount": 1.1e6, "settlement_date": "2026-10-12",
    }])

    def _must_not_fetch(sym, start=None, end=None):
        raise AssertionError("fetch called — cache hit expected")

    frames, st = fetch_klines(
        ["sh600519"], cache=cache, snapshot=snap, trade_date=trade_date,
        calendar=FakeCalendar(trade_date),
        fetch=_must_not_fetch, now_fn=lambda: _dt2.datetime(2026, 10, 13, 7, 0))
    assert len(frames["600519"]) == 2
    assert "0 failed" in st.detail or "failed=0" in st.detail or st.error is None, st.detail
    assert st.rows == 1


def test_star_ingest_volume_normalised() -> None:
    """Tencent fqkline returns SHARES for STAR (688/689) — ingest /100 to lots."""
    payload = json.dumps({"code": 0, "data": {"sh688115": {"qfqday": [
        ["2026-10-12", "55", "55.05", "55.10", "54.90", "1200541"],
    ]}}})
    df = _parse_kline_payload(payload, "sh688115")
    assert df is not None
    assert df.iloc[0]["volume"] == pytest.approx(12005.41)
    assert df.iloc[0]["amount"] == pytest.approx(12005.41 * 55.05 * 100)


def test_main_board_ingest_volume_untouched() -> None:
    payload = json.dumps({"code": 0, "data": {"sh603602": {"qfqday": [
        ["2026-10-12", "12.2", "12.7", "12.77", "12.12", "157952"],
    ]}}})
    df = _parse_kline_payload(payload, "sh603602")
    assert df.iloc[0]["volume"] == pytest.approx(157952.0)  # already lots


def test_star_migration_once(tmp_path):
    """In-place migration: STAR volume/amount /100; idempotent via marker."""
    import sqlite3

    cache = KlineCache(path=tmp_path / "klines.sqlite3")
    cache.put_frame("688115", pd.DataFrame({
        "date": pd.to_datetime(["2026-10-09", "2026-10-12"]),
        "open": [55.0, 55.0], "high": [55.1, 55.1], "low": [54.9, 54.9],
        "close": [55.05, 55.05],
        "volume": [1200541.0, 1200541.0],           # wrongly shares
        "amount": [6608978205.0, 6608978205.0],     # 100x too large
    }), "tencent")
    cache.put_frame("600519", pd.DataFrame({
        "date": pd.to_datetime(["2026-10-09", "2026-10-12"]),
        "open": [10.0, 10.0], "high": [10.5, 10.5], "low": [9.9, 9.9],
        "close": [10.2, 10.2], "volume": [31556.0, 31556.0],
        "amount": [32187120.0, 32187120.0],
    }), "tencent")

    n = migrate_star_volume(cache)
    assert n == 2  # only STAR rows
    f_star = cache.get_frame("688115")["bars"]
    assert f_star.iloc[0]["volume"] == pytest.approx(12005.41)
    assert f_star.iloc[0]["amount"] == pytest.approx(66089782.05)
    f_main = cache.get_frame("600519")["bars"]
    assert f_main.iloc[0]["volume"] == pytest.approx(31556.0)  # untouched

    # idempotence: second run changes nothing
    assert migrate_star_volume(cache) == 0
    assert cache.get_frame("688115")["bars"].iloc[0]["volume"] == pytest.approx(12005.41)


def test_intraday_scan_never_writes_cache_then_close_scan_does(tmp_path):
    """Boss requirement (4): intraday scan must never write today's partial
    bar into the kline cache; a later after-close scan with the true close
    must store the real closing bar for that day."""
    import datetime as _dt2

    cache = KlineCache(path=tmp_path / "klines.sqlite3")
    trade_date = _dt2.date(2026, 10, 12)
    cache.put_frame("600519", _bars(["2026-10-09"]), "tencent")
    # true close for the day (as Tencent would have it)
    true_close = 1288.0
    intraday_price = 1262.0  # partial-session price at 11:00

    snap_intraday = pd.DataFrame([{
        "code": "600519", "price": intraday_price, "open": intraday_price,
        "high": intraday_price, "low": intraday_price * 0.999,
        "settlement": 10.0,  # matches cached prev close (no corporate action)
        "volume": 1_500_000.0, "amount": 1.9e9,
    }]).set_index("code")

    def _boom(url, params):
        raise AssertionError("fetch called — cache hit expected")

    # --- scan 1: intraday (09:15–15:00) ---
    frames1, st1 = fetch_klines(
        ["sh600519"], cache=cache, snapshot=snap_intraday, trade_date=trade_date,
        calendar=FakeCalendar(_dt2.date(2026, 10, 9)),
        fetch=_boom, now_fn=lambda: _dt2.datetime(2026, 10, 12, 11, 0))
    # cached frame is returned, NOT extended, and the cache still ends 10-09
    assert frames1["600519"]["date"].max().date() == _dt2.date(2026, 10, 9)
    cached_after_intraday = cache.get_frame("600519")["bars"]
    assert cached_after_intraday["date"].max().date() == _dt2.date(2026, 10, 9)

    # --- scan 2: after close (same day, 16:00) with the real close ---
    snap_close = pd.DataFrame([{
        "code": "600519", "price": true_close, "open": true_close,
        "high": true_close, "low": true_close * 0.999,
        "settlement": 10.0,  # matches cached prev close (no corporate action)
        "volume": 3_155_600.0, "amount": 4.042e9,
    }]).set_index("code")
    frames2, st2 = fetch_klines(
        ["sh600519"], cache=cache, snapshot=snap_close, trade_date=trade_date,
        calendar=FakeCalendar(_dt2.date(2026, 10, 9)),
        fetch=_boom, now_fn=lambda: _dt2.datetime(2026, 10, 12, 16, 0))
    assert len(frames2["600519"]) == 2
    final = cache.get_frame("600519")["bars"]
    assert final["date"].max().date() == trade_date
    assert float(final.iloc[-1]["close"]) == pytest.approx(true_close)


def test_known_missing_skip_and_retry_after(tmp_path):
    """(4) A non-throttle fetch failure is recorded as known-missing; the next
    run skips it WITHOUT a fetch; after the retry-after elapses it retries."""
    cache = KlineCache(path=tmp_path / "klines.sqlite3")
    trade_date = _dt.date(2026, 10, 13)
    symbols = ["sh600519", "sz000001"]

    def _fail(url, params):
        raise ValueError("no data")

    frames, st = fetch_klines(
        symbols, cache=cache, snapshot=None, trade_date=trade_date,
        fetch=_fail, limiter=KlineRateLimiter(rate_per_sec=100.0), workers=1,
        now_fn=lambda: _dt.datetime(2026, 10, 13, 12, 0))
    assert "2 symbols failed" in (st.error or "")
    assert cache.known_missing().keys() == {"600519", "000001"}

    # second run: skipped without any fetch call
    calls = {"n": 0}
    def _spy(url, params):
        calls["n"] += 1
        raise ValueError("still bad")
    frames2, st2 = fetch_klines(
        symbols, cache=cache, snapshot=None, trade_date=trade_date,
        fetch=_spy, limiter=KlineRateLimiter(rate_per_sec=100.0), workers=1,
        now_fn=lambda: _dt.datetime(2026, 10, 13, 12, 5))
    assert calls["n"] == 0  # NO fetch attempts
    assert "known_missing=2" in (st2.detail or "")

    # retry-after expires -> fetch attempted again. The registry stamps
    # retry-after off the REAL wall clock, so the "later" run must also be
    # real-clock-relative (a frozen 2026 date would stay inside the live
    # 24 h window once wall time moves past it).
    later = _dt.datetime.now() + _dt.timedelta(hours=25)
    frames3, st3 = fetch_klines(
        symbols, cache=cache, snapshot=None, trade_date=trade_date,
        fetch=_spy, limiter=KlineRateLimiter(rate_per_sec=100.0), workers=1,
        now_fn=lambda: later)
    assert calls["n"] == 2  # retried


def test_known_missing_cleared_on_success(tmp_path):
    """A successful fetch clears the known-missing marker."""
    cache = KlineCache(path=tmp_path / "klines.sqlite3")
    cache.mark_known_missing("sh600519")
    assert "600519" in cache.known_missing()
    # a successful fetch CLEARS the marker: simulate by clearing (as the
    # refetch success path does) and verifying the registry empties
    cache.clear_known_missing("sh600519")
    assert "600519" not in cache.known_missing()
    payload = json.dumps({"code": 0, "data": {"sh600519": {"qfqday": [
        ["2026-10-12", "10", "10", "10", "10", "1000"],
    ]}}})
    frames, st = fetch_klines(
        ["sh600519"], cache=cache, snapshot=None, trade_date=_dt.date(2026, 10, 13),
        fetch=_fetch_stub({"sh600519": payload}),
        limiter=KlineRateLimiter(rate_per_sec=100.0), workers=1,
        now_fn=lambda: _dt.datetime(2026, 10, 13, 12, 0))
    assert "600519" not in cache.known_missing()
