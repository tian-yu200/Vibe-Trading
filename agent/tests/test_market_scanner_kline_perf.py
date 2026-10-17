"""F10 perf: batched appends scale to a full-market cache (offline).

4000 cached codes appended in ONE run must finish in well under 2 min on a
realistic-size temp cache. The test also counts sqlite connections to pin
the batched I/O design (no per-symbol get_frame/append round-trips).
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import time as _time
from pathlib import Path

import pandas as pd
import pytest

from src.market_scanner.sources.kline import (
    KlineRateLimiter,
    fetch_klines,
)
from src.market_scanner.sources.kline_cache import KlineCache


TRADE_DATE = dt.date(2026, 10, 15)  # Thursday
AFTER_CLOSE = dt.datetime(2026, 10, 15, 20, 0)


class _FakeCalendar:
    DAYS = {"2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"}

    def previous_trading_day(self, d: dt.date) -> dt.date:
        if d.isoformat() == "2026-10-15":
            return dt.date(2026, 10, 14)
        raise AssertionError(d)

    def is_trading_day(self, d: dt.date) -> bool:
        return d.isoformat() in self.DAYS

    def latest_completed_trading_day(self, now: dt.datetime) -> dt.date:
        return dt.date(2026, 10, 15)


def _build_big_cache(path: Path, n_codes: int = 4000,
                     bars: int = 60) -> None:
    """Realistic-size cache via bulk inserts (fast to build)."""
    db = sqlite3.connect(str(path))
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS kline_bars (code TEXT, date TEXT, "
            "open REAL, high REAL, low REAL, close REAL, volume REAL, "
            "amount REAL, PRIMARY KEY (code, date))")
        db.execute(
            "CREATE TABLE IF NOT EXISTS kline_meta (code TEXT PRIMARY KEY, "
            "last_date TEXT, fetched_at TEXT, source TEXT)")
        meta_rows = []
        fetched_at = _dt_now_iso()
        for i in range(n_codes):
            code = f"{600000 + i:06d}"
            meta_rows.append((code, "2026-10-14", fetched_at, "tencent"))
        # last `bars` weekdays ending 2026-10-14
        import datetime as _d
        base = _d.date(2026, 10, 14)
        days = []
        cur = base
        while len(days) < bars:
            if cur.weekday() < 5:
                days.append(cur.isoformat())
            cur -= _d.timedelta(days=1)
        days.reverse()
        fixed = []
        for code_idx in range(n_codes):
            code = f"{600000 + code_idx:06d}"
            for d in days:
                fixed.append((code, d, 10.0, 10.0, 10.0, 10.0, 1000.0,
                              10.0 * 1000.0 * 100))
        db.executemany(
            "INSERT OR REPLACE INTO kline_bars VALUES (?,?,?,?,?,?,?,?)",
            fixed)
        db.executemany(
            "INSERT OR REPLACE INTO kline_meta VALUES (?,?,?,?)", meta_rows)
        db.commit()
    finally:
        db.close()


def _dt_now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat(
        timespec="seconds")


def _snapshot(codes: list[str]) -> pd.DataFrame:
    recs = [{"code": c, "price": 10.0, "settlement": 10.0,
             "volume": 1000.0 * 100, "amount": 10.0 * 1000.0 * 100,
             "open": 10.0, "high": 10.0, "low": 10.0} for c in codes]
    return pd.DataFrame(recs).set_index("code")


def test_4000_appends_under_two_minutes(tmp_path, monkeypatch):
    path = tmp_path / "big.sqlite3"
    _build_big_cache(path)
    cache = KlineCache(path=path)
    codes = [f"{600000 + i:06d}" for i in range(4000)]
    symbols = [("sh" if c.startswith("6") else "sz") + c for c in codes]
    snap = _snapshot(codes)

    opens = {"n": 0}
    orig_connect = cache._connect

    def counting_connect():
        opens["n"] += 1
        return orig_connect()

    monkeypatch.setattr(cache, "_connect", counting_connect)

    def boom(url, params):
        raise AssertionError("no network on the append path: %r" % (params,))

    t0 = _time.monotonic()
    frames, st = fetch_klines(
        symbols, trade_date=TRADE_DATE, now_fn=lambda: AFTER_CLOSE,
        snapshot=snap, calendar=_FakeCalendar(), cache=cache,
        time_budget_s=600.0, workers=6, fetch=boom,
        limiter=KlineRateLimiter(rate_per_sec=100.0),
        snapshot_source="sina",
    )
    elapsed = _time.monotonic() - t0
    assert st.kline_reasons["counts"]["appended"] == 4000, st.detail
    assert "refetched=0" in (st.detail or "")
    assert cache.get_frame(codes[0])["meta"]["last_date"] == "2026-10-15"
    assert cache.get_frame(codes[-1])["meta"]["last_date"] == "2026-10-15"
    # batched I/O: a handful of connections for the whole run, not ~8000+
    assert opens["n"] < 50, opens
    # wall-clock bound from the task (F10): must clear with wide margin
    assert elapsed < 120, elapsed
    print(f"\n4000 appends: {elapsed:.1f}s, {opens['n']} sqlite connections, "
          f"rows={st.rows}")
