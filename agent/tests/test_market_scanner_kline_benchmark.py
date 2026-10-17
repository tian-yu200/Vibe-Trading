"""Benchmark append-at-close via injected quote (offline, no network)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from src.market_scanner.sources.kline import (
    KlineRateLimiter,
    fetch_benchmark,
)
from src.market_scanner.sources.kline_cache import KlineCache


TRADE_DATE = dt.date(2026, 10, 15)
AFTER_CLOSE = dt.datetime(2026, 10, 15, 20, 0)
MID_SESSION = dt.datetime(2026, 10, 15, 10, 0)


def _payload(symbol: str, dates: list[str], close: float = 4200.0) -> str:
    rows = [[d, str(close), str(close), str(close), str(close), "100000"]
            for d in dates]
    return json.dumps({"code": 0, "data": {symbol: {"day": rows}}})


def _bars(dates: list[str], close: float = 4200.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "date": pd.Timestamp(d), "open": close, "high": close,
        "low": close, "close": close, "volume": 100000.0,
        "amount": close * 100000.0 * 100,
    } for d in dates])


def _fetch(dates):
    def fetch(url, params):
        symbol = params["param"].split(",")[0]
        return _payload(symbol, dates)
    return fetch


def test_benchmark_appended_for_trade_date(tmp_path):
    """Fetched history ends 10-14; quote appends the completed 10-15 bar."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    quote = {"price": 4240.0, "open": 4230.0, "high": 4250.0,
             "low": 4220.0, "volume_lots": 100000.0}
    df, st = fetch_benchmark(
        "sh000300", fetch=_fetch(["2026-10-13", "2026-10-14"]),
        cache=cache, limiter=KlineRateLimiter(rate_per_sec=100.0),
        now_fn=lambda: AFTER_CLOSE, trade_date=TRADE_DATE,
        quote_fetch=lambda sym: dict(quote))
    assert st.status == "ok"
    assert df is not None and df.iloc[-1]["date"].date() == TRADE_DATE
    assert float(df.iloc[-1]["close"]) == 4240.0
    assert st.as_of == "2026-10-15"
    assert "quote-append" in (st.detail or "")
    assert cache.get_frame("sh000300")["meta"]["last_date"] == "2026-10-15"


def test_benchmark_intraday_never_appends(tmp_path):
    """Mid-session: today's bar is partial; quote must not even be asked."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")

    def no_quote(symbol):
        raise AssertionError("quote must not be fetched intraday")

    df, st = fetch_benchmark(
        "sh000300", fetch=_fetch(["2026-10-13", "2026-10-14"]),
        cache=cache, limiter=KlineRateLimiter(rate_per_sec=100.0),
        now_fn=lambda: MID_SESSION, trade_date=TRADE_DATE,
        quote_fetch=no_quote)
    assert st.status == "ok"
    assert df is not None and df.iloc[-1]["date"].date() == dt.date(2026, 10, 14)


def test_benchmark_bad_quote_falls_back(tmp_path):
    """Quote failing verification (50% away): keep fetched frame, note it."""
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    quote = {"price": 6400.0, "open": 6400.0, "high": 6400.0,
             "low": 6400.0, "volume_lots": 100000.0}
    df, st = fetch_benchmark(
        "sh000300", fetch=_fetch(["2026-10-13", "2026-10-14"]),
        cache=cache, limiter=KlineRateLimiter(rate_per_sec=100.0),
        now_fn=lambda: AFTER_CLOSE, trade_date=TRADE_DATE,
        quote_fetch=lambda sym: dict(quote))
    assert df is not None and df.iloc[-1]["date"].date() == dt.date(2026, 10, 14)
    assert "quote failed verification" in (st.detail or "")


def test_benchmark_no_quote_keeps_frame(tmp_path):
    cache = KlineCache(path=tmp_path / "k.sqlite3")
    df, st = fetch_benchmark(
        "sh000300", fetch=_fetch(["2026-10-13", "2026-10-14"]),
        cache=cache, limiter=KlineRateLimiter(rate_per_sec=100.0),
        now_fn=lambda: AFTER_CLOSE, trade_date=TRADE_DATE,
        quote_fetch=lambda sym: None)
    assert df is not None and df.iloc[-1]["date"].date() == dt.date(2026, 10, 14)
    assert "no quote" in (st.detail or "")


def test_benchmark_quote_goes_through_shared_sina_layer(tmp_path, monkeypatch):
    """The live quote path uses http_get_text(host_key='sina'): host
    throttle, run-meter accounting, no dead proxy. Offline via a stubbed
    session; the active meter must count exactly one sina request."""
    from src.market_scanner.sources import _common as common
    from src.market_scanner.sources.kline import _fetch_index_quote

    seen: dict = {}

    class _Resp:
        status_code = 200
        text = ('var hq_str_sh000300="CSI300,4200.00,4180.00,4240.00,'
                '4250.00,4190.00,0,0,120000000,52000000000,0,0,0,0,0,0,0,0,'
                '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2026-10-15,15:00:03,00";')

        def raise_for_status(self):
            pass

    class _Session:
        def get(self, url, params=None, headers=None, timeout=None):
            seen["url"] = url
            seen["headers"] = dict(headers or {})
            return _Resp()

    monkeypatch.setattr(common, "_session_for", lambda bucket: _Session())
    waits: list = []
    monkeypatch.setattr(
        common._THROTTLE, "wait",
        lambda bucket, interval: waits.append((bucket, interval)))
    meter = common.HostMeter()
    common.set_active_meter(meter)
    try:
        quote = _fetch_index_quote("sh000300", None)
    finally:
        common.set_active_meter(None)
    assert quote is not None
    assert quote["price"] == pytest.approx(4240.0)
    assert quote["volume_lots"] == pytest.approx(120000000 / 100)
    assert seen["url"] == "https://hq.sinajs.cn/list=sh000300"
    assert "User-Agent" in seen["headers"] and "Referer" in seen["headers"]
    # host bucket is the Sina one (shared per-host throttle applies)
    assert waits and waits[0][0] == "sina"
    stats = meter.to_dict()
    assert stats["sina"]["count"] == 1
    assert stats["sina"]["errors"] == 0


def test_benchmark_quote_failure_returns_none(tmp_path, monkeypatch):
    """A dead quote endpoint degrades to None (never raises)."""
    from src.market_scanner.sources import _common as common
    from src.market_scanner.sources.kline import _fetch_index_quote

    class _Boom:
        def get(self, *a, **k):
            raise ConnectionError("net down")

    monkeypatch.setattr(common, "_session_for", lambda bucket: _Boom())
    monkeypatch.setattr(common, "host_min_interval", lambda host_key: 0.0)
    meter = common.HostMeter()
    common.set_active_meter(meter)
    try:
        assert _fetch_index_quote("sh000300", None) is None
    finally:
        common.set_active_meter(None)
    # errors are metered even on the failing path
    assert meter.to_dict()["sina"]["errors"] >= 1
