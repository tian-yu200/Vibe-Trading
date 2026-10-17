"""Offline tests for TODO_R2 items C and D.

* Item C (period comparability): like-for-like ROE (TTM, else annualised),
  the mandatory-period calendar, and the ``late_filer`` flag/penalty.
* Item D (one-off gains): recurring (扣非) profit growth when a whole-market
  source exists, else damping the reported profit-growth input when revenue
  YoY < -10%.

No network, no ``~/.vibe-trading``, no git state. All fixtures are built in
code or read from ``agent/tests/fixtures/market_scanner/``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from src.market_scanner.config import ScanConfig
from src.market_scanner.models import DEFAULT_WEIGHTS
from src.market_scanner.scoring import (
    LATE_FILER_PENALTY,
    PERIOD_ANNUALISE,
    REVENUE_DECLINE_DAMP_AT,
    composite_score,
    is_late_filer,
    latest_mandatory_period,
    score_universe,
    _period_label,
    _period_factor,
    _profit_growth_input,
    _roe_like_for_like,
)

TRADE_DATE = dt.date(2026, 3, 20)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_klines(bars: int = 130, start: float = 10.0, drift: float = 0.001) -> pd.DataFrame:
    dates = pd.bdate_range(end=TRADE_DATE, periods=bars)
    closes: list[float] = []
    price = start
    for _ in range(bars):
        price *= 1.0 + drift
        closes.append(price)
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1e5] * bars,
            "amount": [2e8] * bars,
        }
    )


def make_benchmark(bars: int = 130, start: float = 3000.0) -> pd.DataFrame:
    closes = [start * (1.0002) ** i for i in range(bars)]
    return pd.DataFrame(
        {"date": pd.bdate_range(end=TRADE_DATE, periods=bars), "close": closes}
    )


def _row(code: str, **fund) -> dict:
    base = {
        "code": code,
        "symbol": f"sh{code}",
        "name": f"股票{code[-3:]}",
        "exchange": "SH",
        "board": "main_sh",
        "price": 10.0,
        "pct_chg": 0.5,
        "volume": 1e6,
        "amount": 2e8,
        "pe": 12.0,
        "pb": 1.5,
        "total_mv": 2e10,
        "float_mv": 1e10,
        "turnover_rate": 1.2,
        "is_st": False,
        "is_delisting": False,
        "is_suspended": False,
    }
    return base


def _fund_row(code: str, **over) -> dict:
    row = {
        "code": code,
        "report_period": "2025-12-31",
        "roe": 12.0,
        "eps": 1.0,
        "bps": 8.0,
        "revenue_yoy": 5.0,
        "profit_yoy": 8.0,
        "gross_margin": 30.0,
        "ocf_per_share": 1.0,
        "debt_ratio": 40.0,
        "industry": "机械设备",
    }
    row.update(over)
    return row


def run_scan(
    codes: list[str],
    fund_rows: list[dict],
    *,
    as_of: dt.date | None = None,
    config: ScanConfig | None = None,
) -> dict:
    universe = pd.DataFrame.from_records([_row(c) for c in codes]).set_index("code")
    fundamentals = pd.DataFrame.from_records(fund_rows).set_index("code")
    klines = {c: make_klines() for c in codes}
    return score_universe(
        codes,
        universe_df=universe,
        fundamentals_df=fundamentals,
        klines=klines,
        benchmark_df=make_benchmark(),
        as_of=as_of,
        config=config or ScanConfig(),
    )


# --------------------------------------------------------------------------- #
# Item C — mandatory-period calendar
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "as_of,expected",
    [
        (dt.date(2026, 1, 15), "2025Q4"),   # FY of 2025 still mandatory
        (dt.date(2026, 4, 29), "2025Q4"),
        (dt.date(2026, 4, 30), "2025Q4"),
        (dt.date(2026, 5, 1), "2026Q1"),    # Q1 published by 04-30
        (dt.date(2026, 8, 31), "2026Q1"),
        (dt.date(2026, 9, 1), "2026Q2"),    # H1 published by 08-31
        (dt.date(2026, 10, 31), "2026Q2"),
        (dt.date(2026, 11, 1), "2026Q3"),   # Q3 published by 10-31
        (dt.date(2026, 12, 31), "2026Q3"),
        (dt.date(2027, 1, 5), "2026Q4"),    # FY published only next 04-30
        (dt.date(2027, 5, 5), "2027Q1"),
    ],
)
def test_latest_mandatory_period(as_of, expected) -> None:
    assert latest_mandatory_period(as_of) == expected


def test_mandatory_period_unknown_without_as_of() -> None:
    assert latest_mandatory_period(None) is None


@pytest.mark.parametrize(
    "period,as_of,expected",
    [
        ("2025Q3", dt.date(2026, 10, 12), True),   # two periods behind
        ("2026Q1", dt.date(2026, 10, 12), True),
        ("2026Q2", dt.date(2026, 10, 12), False),  # on time
        ("2026Q3", dt.date(2026, 10, 12), False),
        ("2025Q4", dt.date(2026, 3, 1), False),
        ("2025Q3", dt.date(2026, 3, 1), True),
    ],
)
def test_is_late_filer_predicate(period, as_of, expected) -> None:
    assert is_late_filer(period, as_of) is expected


def test_late_filer_disabled_without_as_of() -> None:
    assert is_late_filer("2020Q1", None) is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026Q2", "2026Q2"),
        ("2026-06-30", "2026Q2"),
        ("20260630", "2026Q2"),
        ("2026-03-31", "2026Q1"),
        ("2026-09-30", "2026Q3"),
        ("2026-12-31", "2026Q4"),
        (None, None),
        ("nonsense", None),
    ],
)
def test_period_label_normalisation(raw, expected) -> None:
    assert _period_label(raw) == expected


def test_period_annualisation_factors() -> None:
    assert _period_factor("2026Q1") == pytest.approx(4.0)
    assert _period_factor("2026Q2") == pytest.approx(2.0)
    assert _period_factor("2026Q3") == pytest.approx(4.0 / 3.0)
    assert _period_factor("2026Q4") == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Item C — like-for-like ROE
# --------------------------------------------------------------------------- #
def test_roe_ttm_preferred_when_present() -> None:
    value, basis = _roe_like_for_like(
        {"roe": 6.0, "roe_ttm": 12.0, "report_period": "2026Q1"}
    )
    assert value == pytest.approx(12.0)
    assert basis == "ttm"


def test_roe_annualised_by_period_without_ttm() -> None:
    # Q1 6.0% -> 24.0% annualised; H1 6.0% -> 12.0%: a Q1 reporter and an H1
    # reporter with the SAME reported figure must rank differently.
    q1, _ = _roe_like_for_like({"roe": 6.0, "report_period": "2026Q1"})
    h1, _ = _roe_like_for_like({"roe": 6.0, "report_period": "2026Q2"})
    assert q1 == pytest.approx(24.0)
    assert h1 == pytest.approx(12.0)
    assert q1 > h1


def test_roe_never_ranked_as_a_bare_period_figure() -> None:
    row = {"roe": 10.0, "report_period": "2026Q3"}
    value, basis = _roe_like_for_like(row)
    assert basis == "annualised"
    assert value == pytest.approx(10.0 * 4.0 / 3.0)


def test_roe_missing_stays_missing() -> None:
    value, basis = _roe_like_for_like({"roe": None, "report_period": "2026Q2"})
    assert value is None
    assert basis == "reported"


def test_quarterly_reporter_does_not_outrank_annual_on_same_reported_roe() -> None:
    """Same reported ROE 8.0%: the half-year reporter is the stronger stock."""
    codes = ["600001", "600002"]
    funds = [
        _fund_row("600001", roe=8.0, report_period="2026-06-30"),  # 16.0% annualised
        _fund_row("600002", roe=8.0, report_period="2026-03-31"),  # 32.0% annualised
    ]
    results = run_scan(codes, funds, as_of=dt.date(2026, 10, 12))
    assert results["600001"].metrics["components"]["roe_basis"] == "annualised"
    assert results["600001"].metrics["roe_annualised"] == pytest.approx(16.0)
    assert results["600002"].metrics["roe_annualised"] == pytest.approx(32.0)
    # The Q1 reporter ranks higher on quality, hence on fundamental.
    assert results["600002"].fundamental > results["600001"].fundamental
    assert any("annualise" in r for r in results["600002"].reasons)


def test_eps_and_ocf_per_share_are_annualised_in_metrics() -> None:
    results = run_scan(
        ["600001"],
        [_fund_row("600001", eps=0.5, report_period="2026-06-30", ocf_per_share=0.4)],
    )
    metrics = results["600001"].metrics
    assert metrics["eps_annualised"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Item C — late filer: flag + reason + penalty
# --------------------------------------------------------------------------- #
def test_late_filer_gets_flag_reason_and_penalty() -> None:
    codes = ["600001", "600002", "600003", "600004"]
    # Two pairs with IDENTICAL fundamentals; only the report period differs.
    funds = [
        _fund_row("600001", report_period="2026-06-30"),  # on time
        _fund_row("600002", report_period="2026-03-31"),  # one period late
        _fund_row("600003", report_period="2026-06-30"),
        _fund_row("600004", report_period="2026-03-31"),
    ]
    as_of = dt.date(2026, 10, 12)
    results = run_scan(codes, funds, as_of=as_of)
    late = results["600002"]
    assert "late_filer" in late.flags
    assert "late_filer" not in results["600001"].flags
    assert any("late filer" in r for r in late.reasons)
    # The penalty is a fixed raw-score subtraction, visible in the components.
    # 600004 is the same stock with the same (late) period, so it is an exact
    # twin: identical inputs, identical percentiles, identical penalty.
    twin = results["600004"]
    assert late.metrics["components"]["quality"] == pytest.approx(
        twin.metrics["components"]["quality"]
    )
    assert late.metrics["components"]["fundamental_raw"] == pytest.approx(
        twin.metrics["components"]["fundamental_raw"]
    )
    # On-time 600001 reports a HALF-YEAR, so its annualised ROE (2x) is lower
    # than the Q1 twin's (4x): ranking on annualised values shifts 600001 down
    # vs the Q1 reporters even though its report is on time — that is the
    # like-for-like effect, and the penalty is on top of it, not instead of it.
    assert late.metrics["components"]["roe_basis"] == "annualised"
    assert late.metrics["roe_annualised"] == 12.0 * 4
    assert (
        late.metrics["components"]["fundamental_raw"]
        - twin.metrics["components"]["fundamental_raw"]
        == pytest.approx(0.0)
    )


def test_late_filer_penalty_is_not_applied_without_as_of() -> None:
    results = run_scan(
        ["600001", "600002"],
        [
            _fund_row("600001", report_period="2026-06-30"),
            _fund_row("600002", report_period="2026-03-31"),
        ],
        as_of=None,
    )
    assert "late_filer" not in results["600002"].flags


def test_late_filer_does_not_punish_the_report_itself() -> None:
    """Only the penalty is subtracted; the quality inputs still score.

    Two identical stocks, one late: the late one keeps the same quality
    percentile (its REPORT is untouched) and only loses the penalty.
    """
    codes = ["600001", "600002", "600004"]
    funds = [
        _fund_row("600001", roe=25.0, report_period="2026-06-30"),
        _fund_row("600002", roe=25.0, report_period="2026-03-31"),
        _fund_row("600004", roe=25.0, report_period="2026-03-31"),
    ]
    as_of = dt.date(2026, 10, 12)
    results = run_scan(codes, funds, as_of=as_of)
    on_time, late = results["600001"], results["600002"]
    # The like-for-like effect is real and visible: a Q1 report annualises x4
    # while an H1 report annualises x2, so both get annualised ROE values that
    # are comparable; only the LATE one additionally carries the penalty.
    assert on_time.metrics["components"]["roe_basis"] == "annualised"
    assert late.metrics["components"]["roe_basis"] == "annualised"
    assert on_time.metrics["roe_annualised"] == 25.0 * 2
    assert late.metrics["roe_annualised"] == 25.0 * 4
    assert "late_filer" in late.flags
    assert "late_filer" not in on_time.flags
    assert late.metrics["components"]["late_filer"] == 1.0
    assert on_time.metrics["components"]["late_filer"] == 0.0


# --------------------------------------------------------------------------- #
# Item D — recurring profit growth
# --------------------------------------------------------------------------- #
def test_profit_growth_input_prefers_recurring_when_available() -> None:
    value, basis = _profit_growth_input(
        {"profit_yoy": 96.03, "deduct_profit_yoy": -32.64, "revenue_yoy": -22.36}
    )
    assert basis == "recurring"
    assert value == pytest.approx(-32.64)


def test_profit_growth_input_falls_back_to_damping() -> None:
    """Revenue -22%: reported +96% must NOT score as growth (item D fallback)."""
    value, basis = _profit_growth_input(
        {"profit_yoy": 96.03, "revenue_yoy": -22.36, "report_period": "2026Q2"}
    )
    assert basis == "damped"
    assert value == pytest.approx(max(-22.36, 0.0))


def test_profit_growth_input_reports_missing_when_no_profit() -> None:
    assert _profit_growth_input({"revenue_yoy": -22.4}) == (None, "reported")


def test_no_damping_when_revenue_holds_up() -> None:
    value, basis = _profit_growth_input(
        {"profit_yoy": 96.03, "revenue_yoy": -5.0, "report_period": "2026Q2"}
    )
    assert basis == "reported"
    assert value == pytest.approx(96.03)


def test_damp_threshold_is_minus_ten_percent() -> None:
    assert REVENUE_DECLINE_DAMP_AT == pytest.approx(-10.0)
    value, basis = _profit_growth_input(
        {"profit_yoy": 30.0, "revenue_yoy": -10.0, "report_period": "2026Q2"}
    )
    assert basis == "reported"


def test_recurring_beats_damped_in_the_score() -> None:
    """A stock with the same reported profit YoY drops when 扣非 is negative."""
    codes = ["600001", "600002", "600003"]
    funds = [
        _fund_row("600001", report_period="2026-06-30"),
        _fund_row("600002", report_period="2026-06-30", deduct_profit_yoy=-40.0),
        _fund_row("600003", report_period="2026-06-30", deduct_profit_yoy=-40.0),
    ]
    results = run_scan(codes, funds, as_of=dt.date(2026, 10, 12))
    assert results["600001"].metrics["deduct_profit_yoy"] is None
    assert results["600002"].metrics["deduct_profit_yoy"] == pytest.approx(-40.0)
    assert (
        results["600002"].metrics["components"]["profit_growth_basis"] == "recurring"
    )
    assert results["600002"].fundamental < results["600001"].fundamental
    assert any("recurring (扣非) profit YoY" in r for r in results["600002"].reasons)


def test_damped_path_flags_and_reason() -> None:
    codes = ["600001", "600002", "600003"]
    funds = [
        _fund_row("600001", report_period="2026-06-30"),
        _fund_row("600002", report_period="2026-06-30", revenue_yoy=-22.4, profit_yoy=96.0),
        _fund_row("600003", report_period="2026-06-30"),
    ]
    results = run_scan(codes, funds, as_of=dt.date(2026, 10, 12))
    stock = results["600002"]
    assert "profit_growth_damped" in stock.flags
    assert any("not supported by revenue" in r for r in stock.reasons)
    assert stock.metrics["profit_growth_input"] == pytest.approx(0.0)


def test_damping_only_depends_on_revenue_not_absolute_profit() -> None:
    """The sign of revenue YoY drives damping; profit size does not matter."""
    good = _profit_growth_input(
        {"profit_yoy": -50.0, "revenue_yoy": 5.0, "report_period": "2026Q2"}
    )
    bad = _profit_growth_input(
        {"profit_yoy": 5.0, "revenue_yoy": -20.0, "report_period": "2026Q2"}
    )
    assert good == (pytest.approx(-50.0), "reported")
    assert bad == (pytest.approx(0.0), "damped")


def test_weights_never_renormalise_with_new_inputs() -> None:
    """Items C/D change inputs, not the composite arithmetic."""
    score = composite_score(
        {"fundamental": 100.0, "trend": 100.0, "flow": None}, DEFAULT_WEIGHTS
    )
    assert score == pytest.approx(87.5)
