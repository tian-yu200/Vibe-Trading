"""F5a: tail gap-fill (offline). Fake fetchers only, tmp caches."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from src.market_scanner.sources.kline import (
    KlineRateLimiter,
    fetch_klines,
)
from src.market_scanner.sources.kline_cache import KlineCache


# ---------------------------------------------------------------- helpers ---
def _bars(code: str, dates: list[str], close: float = 10.0,
          volume: float = 1000.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "code": code, "date": pd.Timestamp(d), "open": close,
        "high": close, "low": close, "close": close, "volume": volume,
        "amount": close * volume * 100,
    } for d in dates])


def _payload(symbol: str, dates: list[str], close: float = 10.0,
             volume: float = 1000.0) -> str:
    rows = [[d, str(close), str(close), str(close), str(close), str(volume)]
            for d in dates]
    return json.dumps({"code": 0, "data": {symbol: {"qfqday": rows}}})


def _cache(tmp_path: Path, code: str, frame: pd.DataFrame,
           fetched_days_ago: int = 0) -> KlineCache:
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    cache.put_frame(code, frame, "tencent")
    if fetched_days_ago:
        import datetime as _dt
        import sqlite3
        stale = (_dt.datetime.now() - _dt.timedelta(
            days=fetched_days_ago)).isoformat(timespec="seconds")
        with sqlite3.connect(str(tmp_path / "k.sqlite3")) as db:
            db.execute("UPDATE kline_meta SET fetched_at = ? WHERE code = ?",
                       (stale, code))
    return cache


def _snapshot(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    recs = [{"code": c, "price": p, "settlement": s, "volume": 1000.0 * 100,
             "amount": p * 1000.0 * 100, "open": s,
             "high": max(p, s), "low": min(p, s)}
            for c, p, s in rows]
    return pd.DataFrame(recs).set_index("code")


TRADE_DATE = dt.date(2026, 10, 15)  # Thursday
AFTER_CLOSE = dt.datetime(2026, 10, 15, 20, 0)
# trading days: ... 10-12 (Mon), 10-13, 10-14, 10-15


class _FakeCalendar:
    def previous_trading_day(self, d: dt.date) -> dt.date:
        return dt.date(2026, 10, 14)

    def is_trading_day(self, d: dt.date) -> bool:
        return d.weekday() < 5


def _limiter() -> KlineRateLimiter:
    return KlineRateLimiter(rate_per_sec=100.0)


# ------------------------------------------------------------------- gap ---
def test_two_day_gap_tail_fills_with_one_request(tmp_path):
    """Cache ends 10-13, trade 10-15: fetch 1 gap + 3 overlap bars, merge."""
    code = "600000"
    frame = _bars(code, ["2026-10-08", "2026-10-09", "2026-10-12",
                         "2026-10-13"], close=10.0)
    cache = _cache(tmp_path, code, frame)
    snap = _snapshot([(code, 10.0, 10.0)])
    seen: list[str] = []

    def fetch(url, params):
        seen.append(params["param"])
        return _payload("sh600000",
                        ["2026-10-12", "2026-10-13", "2026-10-14",
                         "2026-10-15"], close=10.0)

    frames, st = fetch_klines(
        ["sh600000"], trade_date=TRADE_DATE, now_fn=lambda: AFTER_CLOSE,
        snapshot=snap, calendar=_FakeCalendar(), cache=cache,
        time_budget_s=600, workers=1, fetch=fetch, limiter=_limiter(),
        snapshot_source="sina",
    )
    assert len(seen) == 1
    assert seen[0].split(",")[4] == "4"  # gap(1) + overlap(3)
    out = frames["600000"]
    assert [d.date().isoformat() for d in out["date"]] == [
        "2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13",
        "2026-10-14", "2026-10-15"]
    reasons = st.kline_reasons
    assert reasons["counts"]["gap_filled"] == 1
    assert reasons["examples"]["gap_filled"] == ["600000"]
    assert "gap_filled=1" in (st.detail or "")
    # persisted with the merged tail
    reread = cache.get_frame(code)
    assert reread["meta"]["last_date"] == "2026-10-15"


def test_overlap_mismatch_falls_back_to_full_refetch(tmp_path):
    """Corporate action inside the gap: tail closes disagree -> full fetch."""
    code = "600001"
    frame = _bars(code, ["2026-10-12", "2026-10-13"], close=10.0)
    cache = _cache(tmp_path, code, frame)
    snap = _snapshot([(code, 5.0, 10.0)])
    seen: list[str] = []

    def fetch(url, params):
        seen.append(params["param"])
        tail_close = 5.0 if len(seen) == 1 else 5.0
        dates = (["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]
                 if len(seen) == 1 else
                 ["2026-01-05", "2026-01-06", "2026-10-14", "2026-10-15"])
        return _payload("sh600001", dates, close=tail_close)

    frames, st = fetch_klines(
        ["sh600001"], trade_date=TRADE_DATE, now_fn=lambda: AFTER_CLOSE,
        snapshot=snap, calendar=_FakeCalendar(), cache=cache,
        time_budget_s=600, workers=1, fetch=fetch, limiter=_limiter(),
        snapshot_source="sina",
    )
    assert len(seen) == 2  # tail, then full refetch
    assert seen[1].split(",")[4] == "260"
    reasons = st.kline_reasons
    assert reasons["counts"]["refetch_gap_mismatch"] == 1
    assert reasons["counts"]["gap_filled"] == 0
    assert "600001" in frames


def test_star_volume_unit_mismatch_forces_full_refetch(tmp_path):
    """Pre-migration STAR bars (shares stored as lots, 100x) vs a correct
    lots tail: the ratio guard rejects the merge and refetches fully."""
    code = "688001"
    frame = _bars(code, ["2026-10-12", "2026-10-13"], close=20.0,
                  volume=10000.0)  # shares mis-stored as lots
    cache = _cache(tmp_path, code, frame)
    snap = _snapshot([(code, 20.0, 20.0)])
    seen: list[str] = []

    def fetch(url, params):
        seen.append(params["param"])
        # raw fqkline volume 10000 shares -> parsed to 100 lots; closes match
        return _payload("sh688001",
                        ["2026-10-12", "2026-10-13", "2026-10-14",
                         "2026-10-15"] if len(seen) == 1 else
                        ["2026-10-14", "2026-10-15"],
                        close=20.0, volume=10000.0)

    frames, st = fetch_klines(
        ["sh688001"], trade_date=TRADE_DATE, now_fn=lambda: AFTER_CLOSE,
        snapshot=snap, calendar=_FakeCalendar(), cache=cache,
        time_budget_s=600, workers=1, fetch=fetch, limiter=_limiter(),
        snapshot_source="sina",
    )
    assert len(seen) == 2
    assert st.kline_reasons["counts"]["refetch_gap_mismatch"] == 1


def test_stale_cache_still_refetches_fully(tmp_path):
    """fetched_at > 30 d wins over a small gap (conservative full fetch)."""
    code = "600002"
    frame = _bars(code, ["2026-10-12", "2026-10-13"], close=10.0)
    cache = _cache(tmp_path, code, frame, fetched_days_ago=40)
    snap = _snapshot([(code, 10.0, 10.0)])
    seen: list[str] = []

    def fetch(url, params):
        seen.append(params["param"])
        return _payload("sh600002", ["2026-10-14", "2026-10-15"], close=10.0)

    frames, st = fetch_klines(
        ["sh600002"], trade_date=TRADE_DATE, now_fn=lambda: AFTER_CLOSE,
        snapshot=snap, calendar=_FakeCalendar(), cache=cache,
        time_budget_s=600, workers=1, fetch=fetch, limiter=_limiter(),
        snapshot_source="sina",
    )
    assert len(seen) == 1
    assert seen[0].split(",")[4] == "260"
    assert st.kline_reasons["counts"]["refetch_no_cache"] == 1


def test_reason_vocabulary_for_decision_paths(tmp_path):
    """prev-close / last-bar mismatches and appends are labelled (F7)."""
    ok = _bars("600010", ["2026-10-12", "2026-10-13", "2026-10-14"],
               close=10.0)
    prev_bad = _bars("600011", ["2026-10-12", "2026-10-13", "2026-10-14"],
                     close=10.0)
    last_bad = _bars("600012", ["2026-10-13", "2026-10-14", "2026-10-15"],
                     close=10.0)
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    cache.put_frame("600010", ok, "tencent")
    cache.put_frame("600011", prev_bad, "tencent")
    cache.put_frame("600012", last_bad, "tencent")
    snap = _snapshot([(code, price, settlement) for code, price, settlement in [
        ("600010", 10.0, 10.0),   # append
        ("600011", 10.0, 99.0),   # settlement mismatch -> prev-close refetch
        ("600012", 99.0, 10.0),   # last-bar mismatch -> last-bar refetch
    ]])

    def fetch(url, params):
        symbol = params["param"].split(",")[0]
        return _payload(symbol, ["2026-10-14", "2026-10-15"], close=10.0)

    frames, st = fetch_klines(
        ["sh600010", "sh600011", "sh600012"], trade_date=TRADE_DATE,
        now_fn=lambda: AFTER_CLOSE, snapshot=snap, calendar=_FakeCalendar(),
        cache=cache, time_budget_s=600, workers=1, fetch=fetch,
        limiter=_limiter(), snapshot_source="sina",
    )
    counts = st.kline_reasons["counts"]
    assert counts["appended"] == 1
    assert counts["refetch_prev_close_mismatch"] == 1
    assert counts["refetch_last_bar_mismatch"] == 1
    assert counts["cache_hit"] == 0
