"""Regression tests: next-day snapshot-append (2026-10-13 15:03 bug).

Reproduces the official after-close scan where the append path never fired
(scan_M1.txt: ``cache_hit=0 appended_from_snapshot=0 refetched=559
missing=3663 (budget stop)`` -> only 501 stocks scored).

Root cause: the scan pipeline called ``fetch_klines(snapshot=..., trade_date=...)``
WITHOUT ``calendar=``. ``prev_day`` stayed ``None``, so
``expected_last = prev_day or trade_date`` silently became ``trade_date``
itself: a cache ending on the previous trading day then looked like a gap and
every symbol was refetched from Tencent, blowing the time budget.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.market_scanner.sources.kline import (
    _settlement_matches,
    _snapshot_row,
    _snapshot_session_completed,
    fetch_klines,
)
from src.market_scanner.sources.kline_cache import KlineCache
from src.market_scanner.sources.status import SourceStatus


# ---------------------------------------------------------------- helpers ---
class _FakeCalendar:
    """Fixed previous trading day (avoids depending on the real one)."""

    def __init__(self, prev: dt.date = dt.date(2026, 10, 12)) -> None:
        self._prev = prev

    def previous_trading_day(self, d: dt.date) -> dt.date:
        return self._prev

    def is_trading_day(self, d: dt.date) -> bool:
        return True


class _ExplodingFetch:
    """Must never be reached on the append path."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str, params: dict) -> str:
        self.calls.append(str(params))
        raise AssertionError("fetch() must not be called on the snapshot path")

    def tencent_response(self, symbol: str, close: float) -> str:
        """A valid minimal qt.gtimg.cn payload for the refetch path."""
        day = "-".join([
            symbol[0],  # seconds-ish, unused
            str(close),
        ])
        return (
            day + "\n"
            + "~" * 30 + f"~{close}~{close}~{close}~{close}~"
        )


class _RecordingFetch:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[str] = []

    def __call__(self, url: str, params: dict) -> str:
        self.calls.append(str(params))
        raise AssertionError("frame-based stub: use _RecordingFrameFetch")


class _RecordingFrameFetch:
    """DataFrame-level fetch (bypasses HTTP parsing): returns bars directly."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[str] = []

    def __call__(self, symbol: str, bars: int, **kwargs) -> pd.DataFrame:
        self.calls.append(symbol)
        return self.frame


def _bars(code: str, dates: list[str], close: float = 10.0) -> pd.DataFrame:
    rows = []
    for d in dates:
        rows.append({
            "code": code, "date": pd.Timestamp(d),
            "open": close, "high": close * 1.02, "low": close * 0.98,
            "close": close, "volume": 1_000_000, "amount": close * 1_000_000 * 100,
            "pct": 0.0,
        })
    return pd.DataFrame(rows)


def _cache(tmp_path: Path, code: str, frame: pd.DataFrame) -> KlineCache:
    cache = KlineCache(path=tmp_path / "klines.sqlite3")
    cache.put_frame(code, frame, "tencent")
    return cache


def _snapshot(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """(code, price, settlement) -> universe frame indexed by code."""
    recs = []
    for code, price, settlement in rows:
        recs.append({
            "code": code, "price": price, "settlement": settlement,
            "volume": 1000.0 * 100, "amount": price * 1000.0 * 100,
            "open": settlement, "high": max(price, settlement),
            "low": min(price, settlement),
        })
    return pd.DataFrame(recs).set_index("code")


def _counters(result: tuple[dict[str, pd.DataFrame], SourceStatus]) -> dict[str, int]:
    """Parse the ``cache_hit=.. appended_from_snapshot=..`` summary string."""
    frames, st = result
    detail = str(st.detail or "")
    counters: dict[str, int] = {}
    for token in detail.replace(",", " ").split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        if not key or key in counters:
            continue
        try:
            counters[key] = int(value)
        except ValueError:
            continue
    counters["n_frames"] = len(frames)
    counters.setdefault("appended_from_snapshot", 0)
    counters.setdefault("cache_hit", 0)
    counters.setdefault("refetched", 0)
    counters.setdefault("missing", 0)
    return counters


TRADE_DATE = dt.date(2026, 10, 13)   # Tue: the day of the broken 15:03 scan
PREV_DAY = dt.date(2026, 10, 12)      # Mon: previous trading day
AFTER_CLOSE = dt.datetime(2026, 10, 13, 16, 0)  # naive local == Asia/Shanghai


# ------------------------------------------------------- the exact repro ----
def test_missing_calendar_appends_from_snapshot(tmp_path):
    """Bug repro: no calendar -> append one bar, zero fetches."""
    code = "600000"
    frame = _bars(code, ["2026-10-08", "2026-10-09", "2026-10-12"], close=10.0)
    cache = _cache(tmp_path, code, frame)
    snap = _snapshot([(code, 10.30, 10.0)])  # settlement == cached close
    exploding = _ExplodingFetch()

    result = fetch_klines(
        ["sh600000"],
        trade_date=TRADE_DATE,
        now_fn=lambda: AFTER_CLOSE,
        snapshot=snap,
        calendar=None,          # <-- exactly what the scan pipeline did
        cache=cache,
        time_budget_s=30,
        workers=1,
        fetch=exploding,
        snapshot_source="sina",
    )
    c = _counters(result)
    assert c["appended_from_snapshot"] == 1
    # append also increments cache_hit (no network per symbol); what matters is
    # that ZERO symbols needed a fetch.
    assert c["refetched"] == 0
    assert c["missing"] == 0
    assert exploding.calls == []

    # returned in memory AND persisted
    out_code = next(iter(result[0]))
    appended_row = result[0][out_code].iloc[-1]
    assert pd.Timestamp(appended_row["date"]).date() == TRADE_DATE
    assert float(appended_row["close"]) == pytest.approx(10.30)
    assert len(result[0][out_code]) == len(frame) + 1

    reread = cache.get_frame(code)
    assert reread is not None
    assert pd.Timestamp(reread["bars"].iloc[-1]["date"]).date() == TRADE_DATE


def test_second_call_after_append_is_pure_cache_hit(tmp_path):
    """Append must update meta: the next call on the same day is a HIT."""
    code = "600001"
    frame = _bars(code, ["2026-10-09", "2026-10-12"], close=5.0)
    cache = _cache(tmp_path, code, frame)
    snap = _snapshot([(code, 5.5, 5.0)])
    exploding = _ExplodingFetch()

    fetch_klines(
        ["sh600001"], trade_date=TRADE_DATE, now_fn=lambda: AFTER_CLOSE,
        snapshot=snap, calendar=None, cache=cache, time_budget_s=30,
        workers=1, fetch=exploding, snapshot_source="sina",
    )
    c1 = _counters(fetch_klines(
        ["sh600001"], trade_date=TRADE_DATE,
        now_fn=lambda: AFTER_CLOSE + dt.timedelta(hours=3),
        snapshot=snap, calendar=None, cache=cache, time_budget_s=30,
        workers=1, fetch=exploding, snapshot_source="sina",
    ))
    assert c1["cache_hit"] == 1
    assert c1["appended_from_snapshot"] == 0
    assert c1["refetched"] == 0
    assert exploding.calls == []


def test_bad_settlement_still_forces_refetch(tmp_path):
    """Corporate-action guard: snapshot settlement != cached close."""
    code = "600002"
    frame = _bars(code, ["2026-10-12"], close=10.0)
    cache = _cache(tmp_path, code, frame)
    snap = _snapshot([(code, 12.0, 11.0)])  # settlement 11.0 != cached 10.0

    result = fetch_klines(
        ["sh600002"], bars=3, trade_date=TRADE_DATE, now_fn=lambda: AFTER_CLOSE,
        snapshot=snap, calendar=None, cache=cache, time_budget_s=30,
        workers=1, fetch=lambda url, params: _tencent_payload("sh600002", [
            ("2026-10-09", 10.0), ("2026-10-12", 10.0), ("2026-10-13", 11.0),
        ]),
        snapshot_source="sina",
    )
    c = _counters(result)
    assert c["appended_from_snapshot"] == 0
    assert c["refetched"] == 1
    assert c["missing"] == 0


def test_intraday_snapshot_never_appends(tmp_path):
    """14:00 on trade_date -> intraday: keep cached frame, no fetch."""
    code = "600003"
    frame = _bars(code, ["2026-10-12"], close=8.0)
    cache = _cache(tmp_path, code, frame)
    snap = _snapshot([(code, 8.4, 8.0)])
    exploding = _ExplodingFetch()

    result = fetch_klines(
        ["sh600003"], trade_date=TRADE_DATE,
        now_fn=lambda: dt.datetime(2026, 10, 13, 14, 0),
        snapshot=snap, calendar=None, cache=cache, time_budget_s=30,
        workers=1, fetch=exploding, snapshot_source="sina",
    )
    c = _counters(result)
    assert c["appended_from_snapshot"] == 0
    assert c["cache_hit"] == 1
    assert c["refetched"] == 0
    assert exploding.calls == []
    reread = cache.get_frame(code)
    assert pd.Timestamp(reread["bars"].iloc[-1]["date"]).date() == PREV_DAY


def test_explicit_calendar_behaviour_unchanged(tmp_path):
    """An explicit calendar keeps working exactly as before the fix."""
    code = "600004"
    # cache ends Fri 2026-10-09; prev trading day given as 10-09 (holiday Mon)
    frame = _bars(code, ["2026-10-08", "2026-10-09"], close=3.0)
    cache = _cache(tmp_path, code, frame)
    snap = _snapshot([(code, 3.1, 3.0)])
    exploding = _ExplodingFetch()

    result = fetch_klines(
        ["sh600004"], trade_date=TRADE_DATE, now_fn=lambda: AFTER_CLOSE,
        snapshot=snap, calendar=_FakeCalendar(dt.date(2026, 10, 9)),
        cache=cache, time_budget_s=30, workers=1, fetch=exploding,
        snapshot_source="sina",
    )
    c = _counters(result)
    assert c["appended_from_snapshot"] == 1
    assert c["refetched"] == 0
    assert exploding.calls == []


def test_genuine_gap_cache_still_refetches(tmp_path):
    """Cache two trading days behind -> real gap -> refetch remains."""
    code = "600005"
    frame = _bars(code, ["2026-10-08", "2026-10-09"], close=3.0)
    cache = _cache(tmp_path, code, frame)
    snap = _snapshot([(code, 3.1, 3.0)])

    result = fetch_klines(
        ["sh600005"], bars=3, trade_date=TRADE_DATE, now_fn=lambda: AFTER_CLOSE,
        snapshot=snap, calendar=None, cache=cache, time_budget_s=30,
        workers=1, fetch=lambda url, params: _tencent_payload("sh600005", [
            ("2026-10-08", 3.0), ("2026-10-09", 3.0), ("2026-10-13", 3.1),
        ]),
        snapshot_source="sina",
    )
    c = _counters(result)
    assert c["refetched"] == 1
    assert c["appended_from_snapshot"] == 0


def test_gap_weekend_uses_weekday_fallback(tmp_path):
    """Friday cache on a Monday trade date: prev day = Friday (weekend skip)."""
    code = "600005"
    frame = _bars(code, ["2026-10-08", "2026-10-09"], close=3.0)  # Fri
    cache = _cache(tmp_path, code, frame)
    snap = _snapshot([(code, 3.1, 3.0)])
    exploding = _ExplodingFetch()

    result = fetch_klines(
        ["sh600005"], trade_date=dt.date(2026, 10, 12),  # Monday
        now_fn=lambda: dt.datetime(2026, 10, 12, 16, 0),
        snapshot=snap, calendar=None, cache=cache, time_budget_s=30,
        workers=1, fetch=exploding, snapshot_source="sina",
    )
    c = _counters(result)
    assert c["appended_from_snapshot"] == 1
    assert c["refetched"] == 0
    assert exploding.calls == []


def test_helpers_match_session_guard_intent():
    """Session guard + settlement comparison behave as documented."""
    # completed after 15:00 on trade_date, any later day, next morning <09:15
    assert _snapshot_session_completed(TRADE_DATE, dt.datetime(2026, 10, 13, 16, 0))
    assert _snapshot_session_completed(TRADE_DATE, dt.datetime(2026, 10, 13, 23, 59))
    assert _snapshot_session_completed(TRADE_DATE, dt.datetime(2026, 10, 14, 7, 0))
    assert _snapshot_session_completed(TRADE_DATE, dt.datetime(2026, 10, 15, 12, 0))
    # intraday 09:15-15:00 on trade_date -> NOT completed
    assert not _snapshot_session_completed(TRADE_DATE, dt.datetime(2026, 10, 13, 14, 0))
    assert not _snapshot_session_completed(TRADE_DATE, dt.datetime(2026, 10, 13, 11, 30))
    assert not _snapshot_session_completed(TRADE_DATE, dt.datetime(2026, 10, 13, 9, 20))

    assert _settlement_matches(10.0, 10.0)
    assert _settlement_matches(10.0, 10.0 + 1e-9)
    assert not _settlement_matches(10.0, 10.5)
    assert not _settlement_matches(10.0, None)
    assert not _settlement_matches(10.0, float("nan"))


def test_snapshot_row_lookup_and_missing():
    code = "600006"
    snap = _snapshot([(code, 9.9, 9.5)])
    row = _snapshot_row(snap, code)
    assert row is not None and float(row["price"]) == pytest.approx(9.9)
    assert _snapshot_row(snap, "600999") is None


def test_append_survives_nan_volume(tmp_path):
    """A snapshot row with NaN volume must still append, not crash."""
    code = "600007"
    frame = _bars(code, ["2026-10-12"], close=4.0)
    cache = _cache(tmp_path, code, frame)
    snap = pd.DataFrame(
        [{
            "price": 4.2, "settlement": 4.0,
            "volume": np.nan, "amount": np.nan,
            "open": 4.0, "high": 4.3, "low": 3.9,
        }],
        index=pd.Index([code], name="code"),
    )
    exploding = _ExplodingFetch()
    result = fetch_klines(
        ["sh600007"], trade_date=TRADE_DATE, now_fn=lambda: AFTER_CLOSE,
        snapshot=snap, calendar=None, cache=cache, time_budget_s=30,
        workers=1, fetch=exploding, snapshot_source="sina",
    )
    c = _counters(result)
    assert c["appended_from_snapshot"] == 1
    assert exploding.calls == []


def test_fifty_symbols_all_append_zero_fetches(tmp_path):
    """Scale check matching the 18:00 run: 50 symbols, ~1 universe fetch."""
    snaps: list[tuple[str, float, float]] = []
    frames: dict[str, pd.DataFrame] = {}
    for i in range(50):
        code = f"6001{i:02d}"
        frames[code] = _bars(code, ["2026-10-09", "2026-10-12"], close=10.0 + i)
        snaps.append((code, 10.0 + i + 0.3, 10.0 + i))
    cache = KlineCache(path=tmp_path / "klines.sqlite3")
    for code, frame in frames.items():
        cache.put_frame(code, frame, "tencent")
    snap = _snapshot(snaps)
    exploding = _ExplodingFetch()

    result = fetch_klines(
        [f"sh{c}" for c in frames], trade_date=TRADE_DATE,
        now_fn=lambda: AFTER_CLOSE, snapshot=snap, calendar=None,
        cache=cache, time_budget_s=60, workers=6, fetch=exploding,
        snapshot_source="sina",
    )
    c = _counters(result)
    assert c["appended_from_snapshot"] == 50
    # cache_hit also counts appended bars (both mean "no network for this
    # symbol"); the point of the style test is zero fetches, not zero cache.
    assert c["refetched"] == 0
    assert c["missing"] == 0
    assert exploding.calls == []
    assert c["n_frames"] == 50


# --------------------------------------------------------------------------- #
# The deeper root cause: the Sina snapshot never carried ``settlement``.
# --------------------------------------------------------------------------- #
def test_universe_row_carries_settlement_and_ohlc():
    """Sina's ``settlement`` (prev close) and OHLC must reach the universe row.

    Without them the append guard ``_settlement_matches(cached_close, None)``
    is always False, so even with a correct calendar every symbol is refetched
    (``_append_from_snapshot`` also needs open/high/low, and returns None
    without a settlement).
    """
    from src.market_scanner.sources.universe import (
        _UNIVERSE_COLUMNS, _row_to_record,
    )

    sina_row = {
        "code": "000001", "name": "平安银行", "trade": "11.57",
        "changepercent": "-1.364", "settlement": "11.73", "open": "11.80",
        "high": "11.85", "low": "11.50", "volume": "99493686",
        "amount": "1157375141", "per": "5.589", "pb": "0.479",
        "mktcap": "224526473.55086", "nmc": "224523775.34587",
        "turnoverratio": "0.5127",
    }
    rec = _row_to_record(sina_row)
    for col in ("settlement", "open", "high", "low"):
        assert col in rec, f"{col} missing from universe record"
        assert col in _UNIVERSE_COLUMNS, f"{col} missing from universe columns"
        assert rec[col] is not None, f"{col} parsed as None"
    assert rec["settlement"] == pytest.approx(11.73)
    # and the guard now actually passes against the cached last close
    from src.market_scanner.sources.kline import _settlement_matches

    assert _settlement_matches(11.73, rec["settlement"]) is True


# --------------------------------------------------------------------------- #
# Minimal Tencent payload builder for the refetch-path tests.
# --------------------------------------------------------------------------- #
def _tencent_payload(symbol: str, rows: list[tuple[str, float]]) -> str:
    """A syntactically valid ifzq.gtimg.cn fqkline response.

    ``rows`` is [(date, close)]; the parser is happy with a repeated close.
    """
    qfqday = [
        [d, str(c), str(c), str(c), str(c), "1000"] for d, c in rows
    ]
    return json.dumps({"code": 0, "data": {symbol: {"qfqday": qfqday}}})
