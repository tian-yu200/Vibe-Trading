"""Pipeline F2/F5/F7 wiring (offline): lock_check abort, budget scaling,
kline_reasons + home in the envelope, unavailable surfacing."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from src.market_scanner.config import ScanConfig
from src.market_scanner.pipeline import (
    SHANGHAI,
    SourceBundle,
    SourceStatus,
    _lock_abort_reason,
    run_scan,
)


@pytest.fixture(autouse=True)
def _no_default_llm(monkeypatch):
    """Keep the default-provider path offline and deterministic in tests."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no provider in tests")

    monkeypatch.setattr(
        "src.market_scanner.pipeline.build_default_llm_callable", _boom
    )

TRADE_DATE = date(2026, 3, 20)  # a Friday
NOW = datetime(2026, 3, 20, 10, 0, tzinfo=SHANGHAI)
LISTED = date(2023, 1, 1)
CAL_PREV = date(2026, 3, 19)  # Thursday


class _FakeCalendar:
    def previous_trading_day(self, d):
        return CAL_PREV

    def is_trading_day(self, d):
        return d.weekday() < 5

    def latest_completed_trading_day(self, now):
        return TRADE_DATE


def _ok(data, name, status="ok", **extra):
    rows = len(data) if hasattr(data, "__len__") else 0
    return data, SourceStatus(name=name, status=status, rows=rows, **extra)


def _universe(n=3):
    recs = []
    for i in range(n):
        code = f"{600001 + i:06d}"
        recs.append({
            "code": code, "symbol": f"sh{code}", "name": f"name{i}",
            "exchange": "SH", "board": "main_sh", "price": 10.0,
            "pct_chg": 0.5, "volume": 1e6, "amount": 2e8, "pe": 15.0,
            "pb": 2.0, "total_mv": 2e10, "float_mv": 1e10,
            "turnover_rate": 1.2, "is_st": False, "is_delisting": False,
            "is_suspended": False, "list_date": LISTED,
        })
    return pd.DataFrame.from_records(recs).set_index("code")


def _fundamentals(n=3):
    recs = []
    for i in range(n):
        code = f"{600001 + i:06d}"
        recs.append({"code": code, "report_period": "2025-12-31", "roe": 20.0,
                     "eps": 1.0, "bps": 8.0, "revenue_yoy": 10.0,
                     "profit_yoy": 10.0, "gross_margin": 30.0,
                     "ocf_per_share": 1.2, "debt_ratio": 40.0,
                     "industry": "ind"})
    return pd.DataFrame.from_records(recs).set_index("code")


def _klines_frame():
    closes = [10.0 * (1 + 0.001) ** i for i in range(130)]
    return pd.DataFrame({
        "date": pd.bdate_range(end=TRADE_DATE, periods=130),
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes,
        "volume": [1e5] * 130, "amount": [2e8] * 130,
    })


def _klines_map(n=3):
    return {f"{600001 + i:06d}": _klines_frame() for i in range(n)}


def _benchmark():
    closes = [3000.0 * (1 + 0.0002) ** i for i in range(130)]
    return pd.DataFrame(
        {"date": pd.bdate_range(end=TRADE_DATE, periods=130),
         "close": closes})


def _flow(n=3):
    recs = []
    for i in range(n):
        code = f"{600001 + i:06d}"
        recs.append({"code": code, "flow_1d": 1e8, "flow_3d": 2e8,
                     "flow_5d": 5e8, "pct_chg_3d": 1.0, "pct_chg_5d": 2.0})
    return pd.DataFrame.from_records(recs).set_index("code")


def _bundle(n=3, klines_fn=None):
    return SourceBundle(
        load_calendar=lambda: (_FakeCalendar(),
                               SourceStatus(name="calendar", status="ok")),
        fetch_universe=lambda: _ok(_universe(n), "universe"),
        fetch_listing_info=lambda: (None, SourceStatus(name="listing",
                                                      status="ok", rows=0)),
        fetch_fundamentals=lambda trade_date: _ok(_fundamentals(n),
                                                  "fundamentals"),
        fetch_klines=klines_fn or (lambda symbols, **kw: _ok(
            _klines_map(n), "klines")),
        fetch_benchmark=lambda **kw: _ok(_benchmark(), "benchmark"),
        fetch_fund_flow=lambda: _ok(_flow(n), "fund_flow"),
        fetch_dragon_tiger=lambda trade_date, days=30: (
            None, SourceStatus(name="dragon_tiger", status="skipped")),
        fetch_northbound_info=lambda: (
            None, SourceStatus(name="northbound", status="skipped")),
    )


def test_lock_check_ok_runs_and_is_called_at_stages():
    calls: list[str] = []

    def check():
        calls.append("check")

    result = run_scan(ScanConfig(min_coverage=0.0), now=NOW,
                      sources=_bundle(), llm=None, lock_check=check)
    assert result["status"] in ("ok", "partial")
    assert len(calls) >= 3  # klines, klines-done/exclusions, ai_review
    assert result["home"] is not None and "home" in result["home"]
    assert "kline_reasons" in result  # fake kline status -> None is fine


class _JobLockLost(RuntimeError):
    pass


def test_lock_lost_aborts_with_exact_reason():
    def check():
        raise _JobLockLost("job lock lost (deleted or taken over)")

    result = run_scan(ScanConfig(min_coverage=0.0), now=NOW,
                      sources=_bundle(), llm=None, lock_check=check)
    assert result["status"] == "failed"
    assert result["failure_reason"] == (
        "job lock lost (deleted or taken over) - aborted")
    assert result["picks"] == []


def test_lock_check_helper_matches_by_name():
    assert _lock_abort_reason(None, "klines") is None

    class Other(RuntimeError):
        pass

    reason = _lock_abort_reason(
        lambda: (_ for _ in ()).throw(
            _JobLockLost("job lock lost (deleted or taken over)")),
        "klines")
    assert reason == "job lock lost (deleted or taken over) - aborted"
    other = _lock_abort_reason(
        lambda: (_ for _ in ()).throw(Other("boom")), "ai_review")
    assert other is not None and "lock check failed at ai_review" in other


def test_kline_reasons_and_unavailable_surface():
    seen: dict = {}

    def klines(symbols, **kw):
        seen.update(kw)
        st = SourceStatus(name="klines", status="failed", rows=0,
                          error="price source unavailable (test outage)")
        st.unavailable = True
        st.lock_aborted = False
        st.kline_reasons = {"counts": {"transport_error": 5}, "examples": {}}
        return {}, st

    result = run_scan(ScanConfig(min_coverage=1.0), now=NOW,
                      sources=_bundle(n=1, klines_fn=klines), llm=None)
    assert result["kline_reasons"] == {
        "counts": {"transport_error": 5}, "examples": {}}
    assert result["status"] == "failed"
    assert "price source unavailable" in (result["failure_reason"] or "")
    assert "insufficient coverage" in (result["failure_reason"] or "")
    assert seen.get("limiter") is not None  # fresh per-scan limiter
    assert seen.get("calendar") is not None  # pipeline now passes it
    assert seen.get("lock_check") is None  # none given -> passthrough None


def test_budget_scales_with_shortfall_and_caps(tmp_path, monkeypatch):
    # Hermetic home: the shared sandbox default cache accumulates other
    # tests' fixtures (e.g. sh600519, which falls inside this test's code
    # range), so point the runtime AND pin lookup at tmp instead of relying
    # on whatever happens to be cached. (Found via a 699-vs-700 budget.)
    sandbox = tmp_path / "home"
    monkeypatch.setenv("VIBE_TRADING_HOME", str(sandbox))
    monkeypatch.setenv("VIBE_TRADING_HOME_PIN", str(tmp_path / "no-pin"))
    seen: dict = {}

    def klines(symbols, **kw):
        seen.update(kw)
        return {}, SourceStatus(name="klines", status="failed", rows=0)

    # 700 missing symbols in a fresh default cache -> est 700 s
    result = run_scan(ScanConfig(min_coverage=0.0), now=NOW,
                      sources=_bundle(n=700, klines_fn=klines), llm=None)
    assert seen.get("time_budget_s") == 700.0
    assert result["status"] == "partial"  # kline failed, 0.0 floor passes

    seen.clear()
    run_scan(ScanConfig(min_coverage=0.0,
                        kline_time_budget_s=600.0,
                        kline_time_budget_cap_s=650.0),
             now=NOW, sources=_bundle(n=700, klines_fn=klines), llm=None)
    assert seen.get("time_budget_s") == 650.0  # capped
