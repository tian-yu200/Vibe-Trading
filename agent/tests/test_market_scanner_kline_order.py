"""F10: phase-2 refetch order — corporate-action mismatches first."""

from __future__ import annotations

import datetime as dt
import json
import time as _time
from pathlib import Path

import pandas as pd

from src.market_scanner.sources.kline import (
    KlineRateLimiter,
    fetch_klines,
)
from src.market_scanner.sources.kline_cache import KlineCache


TRADE_DATE = dt.date(2026, 10, 15)
AFTER_CLOSE = dt.datetime(2026, 10, 15, 20, 0)


class _FakeCalendar:
    def previous_trading_day(self, d):
        return dt.date(2026, 10, 14)

    def is_trading_day(self, d):
        return d.weekday() < 5

    def latest_completed_trading_day(self, now):
        return dt.date(2026, 10, 15)


def _bars(dates, close=10.0):
    return pd.DataFrame([{
        "date": pd.Timestamp(d), "open": close, "high": close,
        "low": close, "close": close, "volume": 1000.0,
        "amount": close * 1000.0 * 100,
    } for d in dates])


def _snapshot(rows):
    recs = [{"code": c, "price": p, "settlement": s,
             "volume": 1000.0 * 100, "amount": p * 1000.0 * 100,
             "open": s, "high": max(p, s), "low": min(p, s)}
            for c, p, s in rows]
    return pd.DataFrame(recs).set_index("code")


def test_mismatches_fetch_first_under_tiny_budget(tmp_path):
    """2 prev-close mismatches + 10 no-cache codes, network budget for ~2
    requests: the 2 mismatches (not the first-come missing codes) get their
    fetches; the rest count as budget_not_fetched."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    mis = ["600010", "600011"]
    for code in mis:
        cache.put_frame(code, _bars(["2026-10-13", "2026-10-14"]), "tencent")
    snap_rows = [(c, 10.0, 99.0) for c in mis]  # settlement mismatch
    nocache = [f"6000{i:02d}" for i in range(20, 30)]
    snap_rows += [(c, 10.0, 10.0) for c in nocache]
    snap = _snapshot(snap_rows)
    fetched: list[str] = []

    def fetch(url, params):
        fetched.append(params["param"].split(",")[0])
        _time.sleep(0.05)  # real 50 ms network work per request
        symbol = params["param"].split(",")[0]
        return json.dumps({"code": 0, "data": {symbol: {"qfqday": [
            ["2026-10-14", "10", "10", "10", "10", "1000"],
            ["2026-10-15", "10", "10", "10", "10", "1000"]]}}})

    frames, st = fetch_klines(
        [("sh" + c) for c in mis + nocache], trade_date=TRADE_DATE,
        now_fn=lambda: AFTER_CLOSE, snapshot=snap,
        calendar=_FakeCalendar(), cache=cache, time_budget_s=0.08,
        workers=1, fetch=fetch, limiter=KlineRateLimiter(rate_per_sec=100.0),
        snapshot_source="sina")
    assert fetched[:2] == ["sh600010", "sh600011"]
    assert st.kline_reasons["counts"]["refetch_prev_close_mismatch"] == 2
    # the bulk missing set never got its requests: budget stopped the run
    # (a couple of in-flight completions may land in the 2 s drain window
    # without being tallied — pre-existing drain semantics, unchanged).
    assert st.kline_reasons["counts"]["budget_not_fetched"] >= 9
    assert len(fetched) < 12
    assert "budget_exhausted=yes" in (st.detail or "")
