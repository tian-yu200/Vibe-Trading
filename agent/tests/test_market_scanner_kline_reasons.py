"""F7: kline_reasons dict shape, counts and example caps (offline)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from src.market_scanner.sources.kline import (
    KLINE_REASONS,
    KlineRateLimiter,
    fetch_klines,
)
from src.market_scanner.sources.kline_cache import KlineCache


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


TRADE_DATE = dt.date(2026, 10, 15)
AFTER_CLOSE = dt.datetime(2026, 10, 15, 20, 0)


class _FakeCalendar:
    def previous_trading_day(self, d: dt.date) -> dt.date:
        return dt.date(2026, 10, 14)

    def is_trading_day(self, d: dt.date) -> bool:
        return d.weekday() < 5


def test_reasons_shape_counts_and_example_cap(tmp_path):
    """25 known-missing codes: count 25, examples capped at 20, all keys."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    codes = [f"{600100 + i:06d}" for i in range(25)]
    for code in codes:
        cache.mark_known_missing(f"sh{code}", retry_after_s=86400)

    frames, st = fetch_klines(
        [f"sh{c}" for c in codes], trade_date=TRADE_DATE,
        now_fn=lambda: AFTER_CLOSE, snapshot=None,
        calendar=_FakeCalendar(), cache=cache, workers=1,
        fetch=lambda u, p: (_ for _ in ()).throw(
            AssertionError("no fetch expected")),
        limiter=KlineRateLimiter(rate_per_sec=100.0),
    )
    reasons = st.kline_reasons
    assert set(reasons["counts"]) == set(KLINE_REASONS)
    assert set(reasons["examples"]) == set(KLINE_REASONS)
    assert reasons["counts"]["known_missing"] == 25
    assert len(reasons["examples"]["known_missing"]) == 20
    assert reasons["examples"]["known_missing"][0] == codes[0]
    assert frames == {}


def test_reasons_cover_fetch_outcomes(tmp_path):
    """One code per outcome class lands in the right bucket (F7)."""
    hit = _bars("600010", ["2026-10-13", "2026-10-14"], close=10.0)
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    cache.put_frame("600010", hit, "tencent")

    def fetch(url, params):
        symbol = params["param"].split(",")[0]
        if symbol == "sh600011":
            raise TimeoutError("ReadTimeout: timed out")  # transport
        if symbol == "sh600012":
            raise ValueError("no kline rows")  # permanent -> known-missing
        raise AssertionError("unexpected fetch")

    frames, st = fetch_klines(
        ["sh600010", "sh600011", "sh600012"], trade_date=TRADE_DATE,
        now_fn=lambda: AFTER_CLOSE, snapshot=None,
        calendar=_FakeCalendar(), cache=cache, time_budget_s=600,
        workers=1, fetch=fetch,
        limiter=KlineRateLimiter(rate_per_sec=100.0),
    )
    counts = st.kline_reasons["counts"]
    # no snapshot row -> suspended keep == cache_hit
    assert counts["cache_hit"] == 1
    assert counts["transport_error"] == 1
    assert st.kline_reasons["examples"]["transport_error"] == ["600011"]
    # permanent failure is marked known-missing for the next run...
    assert cache.known_missing() != {}
    # ...and the 600012 outcome is a full-refetch attempt (no-cache)
    assert counts["refetch_no_cache"] == 2
