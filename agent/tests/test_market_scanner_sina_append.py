"""Offline tests: Sina-appended bars match cached Tencent bars (30-stock
recorded fixture) + corporate-action mismatch refetches only that stock."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pandas as pd
import pytest

from src.market_scanner.sources.kline import (
    KlineRateLimiter,
    _append_from_snapshot,
    fetch_klines,
)
from src.market_scanner.sources.kline_cache import KlineCache

FIXTURES = Path(__file__).parent / "fixtures" / "market_scanner"
LOT = 100


def _load() -> dict:
    return json.loads((FIXTURES / "sina_append_pairs.json").read_text(encoding="utf-8"))


def _sina_row_from_tencent(t_bar: dict) -> dict:
    """Reconstruct the Sina snapshot row that _append_from_snapshot consumed.

    The fixture stores the Tencent bar; the Sina row is derived by inverting
    the verified unit normalisation (volume lots -> shares) so the append
    function reproduces the identical bar. amount_reported is set to the
    Sina-style turnover (within ±1.3% of the estimate) to exercise the
    homogeneous-amount rule.
    """
    return {
        "price": t_bar["close"],
        "settlement": t_bar["close"],  # prev close matches (verified live)
        "volume": t_bar["volume"] * LOT,  # sina reports shares
        "amount": t_bar["amount"] * 0.999,  # sina turnover differs slightly
        "open": t_bar["open"], "high": t_bar["high"], "low": t_bar["low"],
    }


@pytest.fixture()
def pairs() -> list[dict]:
    return _load()["pairs"]


def test_sina_appended_bars_match_tencent(tmp_path: Path, pairs: list[dict]) -> None:
    """OHLC ±0.01, volume exact after unit normalisation, amount homogeneous."""
    cache = KlineCache(path=tmp_path / "klines.sqlite3")
    for pair in pairs:
        code = pair["code"]
        hist = pair["history_dates"]
        close = pair["history_last_close"]
        cache.put_frame(code, pd.DataFrame([
            {"date": pd.Timestamp(d), "open": close, "high": close,
             "low": close, "close": close, "volume": 1000.0,
             "amount": close * 1000 * LOT}
            for d in hist
        ]), "tencent")
        t = pair["tencent_bar"]
        sina_row = _sina_row_from_tencent(t)
        appended = _append_from_snapshot(
            cache.get_frame(code)["bars"], pd.Series(sina_row),
            _dt.date.fromisoformat(_load()["trade_date"]), "sina")
        assert appended is not None, code
        a = appended.iloc[-1]
        t_bar = t
        for field in ("open", "high", "low", "close"):
            assert abs(float(a[field]) - t_bar[field]) <= 0.01, (code, field)
        # volume exact after normalisation (sina shares -> lots)
        assert float(a["volume"]) == pytest.approx(t_bar["volume"], abs=0.5), code
        # amount per the homogeneous rule: close*volume*100
        assert float(a["amount"]) == pytest.approx(
            t_bar["close"] * t_bar["volume"] * LOT, rel=1e-9), code
        # sina's reported turnover preserved for audit
        assert float(a["amount_reported"]) == pytest.approx(
            t_bar["amount"] * 0.999, rel=1e-6)


def test_corporate_action_refetches_only_that_stock(
    tmp_path: Path, pairs: list[dict],
) -> None:
    """Prev-close mismatch on ONE stock -> only that stock hits the fetcher."""
    subset = pairs[:4]
    cache = KlineCache(path=tmp_path / "klines.sqlite3")
    trade_date = _dt.date.fromisoformat(_load()["trade_date"])
    ca_code = subset[0]["code"]  # this one gets a prev-close mismatch
    # the fixture history ends 2026-10-09; prev trading day before 10-12
    prev_day = _dt.date.fromisoformat(subset[0]["history_dates"][-1])

    fetched: list[str] = []

    def fetch(url: str, params: dict) -> str:
        symbol = params["param"].split(",")[0]
        fetched.append(symbol)
        return json.dumps({"code": 0, "data": {symbol: {"qfqday": [
            ["2026-10-12", "10", "10", "10", "10", "1000"],
        ]}}})

    # seed caches from the fixture history
    snap_rows = []
    for i, pair in enumerate(subset):
        hist_close = pair["history_last_close"]
        cache.put_frame(pair["code"], pd.DataFrame([
            {"date": pd.Timestamp(d), "open": hist_close, "high": hist_close,
             "low": hist_close, "close": hist_close, "volume": 1000.0,
             "amount": hist_close * 1000 * LOT}
            for d in pair["history_dates"]
        ]), "tencent")
        sina_row = _sina_row_from_tencent(pair["tencent_bar"])
        sina_row["settlement"] = hist_close  # matches -> append path
        if pair["code"] == ca_code:
            sina_row["settlement"] = hist_close * 0.5  # dividend -> mismatch
        snap_rows.append(sina_row)
    snapshot = pd.DataFrame(snap_rows, index=pd.Index(
        [p["code"] for p in subset], name="code"))

    limiter = KlineRateLimiter(rate_per_sec=100.0)
    frames, st = fetch_klines(
        [f"sh{p['code']}" for p in subset],
        cache=cache, snapshot=snapshot, trade_date=trade_date,
        calendar=type("C", (), {"previous_trading_day":
                                staticmethod(lambda d: prev_day)}),
        fetch=fetch, limiter=limiter,
        now_fn=lambda: _dt.datetime(2026, 10, 13, 16, 0),
    )
    # only the mismatch stock was refetched from Tencent
    assert fetched == [f"sh{ca_code}"]
    # the others were appended from the snapshot (no fetch)
    appended_codes = {p["code"] for p in subset if p["code"] != ca_code}
    assert appended_codes <= set(frames)
    assert "refetched=1" in (st.detail or "")
