"""F10: every phase-2 submission lands in exactly one terminal bucket.

No symbol may vanish — including late completions that finish after a
budget stop (the drain path stores them instead of dropping them).
Offline: fake fetcher, tmp cache, real (fast) clock.
"""

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


def _snapshot(codes):
    recs = [{"code": c, "price": 10.0, "settlement": 10.0,
             "volume": 1000.0 * 100, "amount": 10.0 * 1000.0 * 100,
             "open": 10.0, "high": 10.0, "low": 10.0} for c in codes]
    return pd.DataFrame(recs).set_index("code")


TERMINAL = ("refetched", "gap_filled", "refetch_gap_mismatch",
            "transport_error", "permanent_error", "budget_not_fetched",
            "stopped_unavailable")


def test_phase2_every_symbol_lands_in_exactly_one_bucket(tmp_path):
    """2 gap + 4 full-ok + 2 transport + 2 permanent; the budget trips after
    the 8th completion, so one in-flight permanent drains into its bucket
    and one queued ok is budget-cancelled. Every submitted symbol lands in
    exactly one terminal bucket (no silent drops)."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    gap = ["600010", "600011"]
    for code in gap:
        cache.put_frame(code, _bars(["2026-10-12", "2026-10-13"]), "tencent")
    ok = ["600020", "600021", "600022", "600023"]
    trans = ["600030", "600031"]
    perm = ["600040", "600041"]
    # submission order (gaps first by rank, then in listed order) decides
    # which symbols complete before the budget trips
    symbols = (["sh" + c for c in gap]
               + ["sh" + c for c in ok + trans + perm])
    snap = _snapshot(gap + ok + trans + perm)

    def fetch(url, params):
        symbol = params["param"].split(",")[0]
        _time.sleep(0.12)  # paced network stand-in (deterministic order)
        if symbol in ("sh600030", "sh600031"):
            raise TimeoutError("ReadTimeout: timed out")
        if symbol in ("sh600040", "sh600041"):
            raise ValueError("no kline rows for symbol")
        dates = (["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]
                 if symbol in ("sh600010", "sh600011")
                 else ["2026-10-14", "2026-10-15"])
        rows = [[d, "10", "10", "10", "10", "1000"] for d in dates]
        return json.dumps({"code": 0, "data": {symbol: {"qfqday": rows}}})

    frames, st = fetch_klines(
        symbols, trade_date=TRADE_DATE, now_fn=lambda: AFTER_CLOSE,
        snapshot=snap, calendar=_FakeCalendar(), cache=cache,
        time_budget_s=0.9, workers=1,
        fetch=fetch, limiter=KlineRateLimiter(rate_per_sec=100.0))
    counts = st.kline_reasons["counts"]
    # terminal accounting: 2 gap + 4 refetched (still-running one drains)
    # + 2 transport + 1 permanent + 1 budget-cancelled == 10 submitted
    assert counts["gap_filled"] == 2
    assert counts["transport_error"] == 2
    assert counts["permanent_error"] == 1
    assert counts["refetched"] == 4
    assert counts["budget_not_fetched"] == 1
    assert sum(counts[k] for k in TERMINAL) == len(symbols), counts
    assert "budget_exhausted" in (st.detail or "")
    # the drained permanent failure is marked for retry-skip (transport
    # ones never are); the cancelled one is retried next run like the rest
    assert set(cache.known_missing()) == {"600040"}
    # stored frames match the terminal accounting
    assert len(frames) == counts["gap_filled"] + counts["refetched"]
    assert st.status == "degraded"
