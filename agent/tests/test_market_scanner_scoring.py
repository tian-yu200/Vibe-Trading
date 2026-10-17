"""Unit tests for cross-sectional scoring (CONTRACT section 6). Offline only."""

from __future__ import annotations

import re
from datetime import date

import pandas as pd
import pytest

from src.market_scanner.config import ScanConfig
from src.market_scanner.models import DEFAULT_WEIGHTS
from src.market_scanner.scoring import (
    EXTREME_VOL_PENALTY,
    FLOW_HORIZON_WEIGHTS,
    FUNDAMENTAL_COMPONENT_WEIGHTS,
    NEUTRAL,
    OVERHEAT_PENALTY,
    TREND_COMPONENT_WEIGHTS,
    composite_score,
    score_universe,
    _trend_raw_batch,
)

TRADE_DATE = date(2026, 3, 20)

# Six strictly-ordered stocks: 600001 is best on every input, 600006 worst.
_DOMINANT_SET = [
    # code, roe, gm, rev, profit, eps, ocf, pe, pb, debt, drift, flow_5d
    ("600001", 30.0, 60.0, 50.0, 60.0, 2.0, 6.0, 8.0, 1.0, 10.0, 0.004, 8e8),
    ("600002", 24.0, 52.0, 35.0, 40.0, 1.6, 4.0, 12.0, 1.6, 25.0, 0.002, 4e8),
    ("600003", 18.0, 44.0, 20.0, 22.0, 1.2, 2.5, 18.0, 2.2, 35.0, 0.001, 1e8),
    ("600004", 12.0, 36.0, 8.0, 5.0, 0.8, 1.2, 26.0, 3.0, 45.0, 0.0005, 0.0),
    ("600005", 6.0, 28.0, -5.0, -12.0, 0.3, 0.2, 45.0, 4.0, 60.0, -0.0005, -2e8),
    ("600006", 2.0, 20.0, -15.0, -40.0, 0.05, -0.1, 90.0, 6.0, 75.0, -0.002, -5e8),
]


def make_klines(
    *,
    bars: int = 130,
    start: float = 10.0,
    drift: float = 0.001,
    zigzag: float = 0.0,
    last20_mult: float = 0.0,
    volume: float = 1e5,
    recent_volume_mult: float = 1.0,
) -> pd.DataFrame:
    """Deterministic ascending klines ending on TRADE_DATE."""
    dates = pd.bdate_range(end=TRADE_DATE, periods=bars)
    closes: list[float] = []
    price = start
    for i in range(bars):
        step = drift + (zigzag if i % 2 == 0 else -zigzag)
        price *= 1.0 + step
        if i >= bars - 20:
            price *= 1.0 + last20_mult
        closes.append(price)
    volumes = [volume] * bars
    if recent_volume_mult != 1.0:
        for i in range(bars - 20, bars):
            volumes[i] = volume * recent_volume_mult
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": volumes,
            "amount": [2e8] * bars,
        }
    )


def make_benchmark(bars: int = 130, start: float = 3000.0, drift: float = 0.0002) -> pd.DataFrame:
    """Flat-ish CSI 300 stand-in."""
    closes = [start * (1 + drift) ** i for i in range(bars)]
    return pd.DataFrame(
        {
            "date": pd.bdate_range(end=TRADE_DATE, periods=bars),
            "close": closes,
        }
    )


def dominant_frames(skip_flow: set[str] | None = None):
    """Build (codes, universe, fundamentals, klines, flow) for _DOMINANT_SET."""
    skip_flow = skip_flow or set()
    codes, uni_rows, fund_rows, klines, flow_rows = [], [], [], {}, []
    for code, roe, gm, rev, profit, eps, ocf, pe, pb, debt, drift, flow5 in _DOMINANT_SET:
        codes.append(code)
        uni_rows.append(
            {
                "code": code,
                "symbol": f"sh{code}",
                "name": f"股票{code[-3:]}",
                "exchange": "SH",
                "board": "main_sh",
                "price": 10.0,
                "pct_chg": 0.5,
                "volume": 1e6,
                "amount": 2e8,
                "pe": pe,
                "pb": pb,
                "total_mv": 2e10,
                "float_mv": 1e10,
                "turnover_rate": 1.2,
                "is_st": False,
                "is_delisting": False,
                "is_suspended": False,
            }
        )
        fund_rows.append(
            {
                "code": code,
                "report_period": "2025-12-31",
                "roe": roe,
                "eps": eps,
                "bps": 8.0,
                "revenue_yoy": rev,
                "profit_yoy": profit,
                "gross_margin": gm,
                "ocf_per_share": ocf,
                "debt_ratio": debt,
                "industry": "大消费",
            }
        )
        klines[code] = make_klines(drift=drift)
        if code not in skip_flow:
            flow_rows.append(
                {
                    "code": code,
                    "flow_1d": flow5 / 5,
                    "flow_3d": flow5 / 2,
                    "flow_5d": flow5,
                    "pct_chg_3d": 1.0,
                    "pct_chg_5d": 2.0,
                }
            )
    universe = pd.DataFrame.from_records(uni_rows).set_index("code")
    fundamentals = pd.DataFrame.from_records(fund_rows).set_index("code")
    flow = (
        pd.DataFrame.from_records(flow_rows).set_index("code") if flow_rows else None
    )
    return codes, universe, fundamentals, klines, flow


def run_dominant(skip_flow: set[str] | None = None, config: ScanConfig | None = None):
    codes, universe, fundamentals, klines, flow = dominant_frames(skip_flow)
    return score_universe(
        codes,
        universe_df=universe,
        fundamentals_df=fundamentals,
        klines=klines,
        benchmark_df=make_benchmark(),
        flow_df=flow,
        config=config or ScanConfig(),
    )


# --------------------------------------------------------------------------- #
# Component weight bookkeeping
# --------------------------------------------------------------------------- #


def test_component_weights_sum_to_one():
    assert sum(FUNDAMENTAL_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(TREND_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(FLOW_HORIZON_WEIGHTS.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# HARD RULES
# --------------------------------------------------------------------------- #


def test_composite_never_renormalises_weights():
    # The pure arithmetic, isolated: missing flow keeps its 0.25 weight and
    # contributes the neutral 50, so 100/100/missing is 87.5 — NOT 100.
    score = composite_score(
        {"fundamental": 100.0, "trend": 100.0, "flow": None}, DEFAULT_WEIGHTS
    )
    assert score == 87.5
    renormalised = (0.40 * 100 + 0.35 * 100) / (0.40 + 0.35)
    assert score != renormalised


def test_missing_flow_gets_50_flag_and_reason_without_renormalising():
    results = run_dominant(skip_flow={"600001"})
    top = results["600001"]
    assert top.flow == NEUTRAL
    assert "flow_missing" in top.flags
    assert any("flow" in reason.lower() for reason in top.reasons)
    assert any("neutral flow score 50" in reason for reason in top.reasons)
    # Best on fundamentals and trend => both percentiles 100; composite must be
    # 0.40*100 + 0.35*100 + 0.25*50 = 87.5, i.e. weights untouched.
    assert top.fundamental == 100.0
    assert top.trend == 100.0
    assert top.score == 87.5
    assert top.score != 100.0


def test_missing_fundamentals_or_klines_is_not_scored():
    codes, universe, fundamentals, klines, flow = dominant_frames()
    # 600001 has no fundamentals row; 600002 has klines below min_kline_bars.
    fundamentals = fundamentals.drop("600001")
    klines["600002"] = make_klines(bars=50)
    results = score_universe(
        codes,
        universe_df=universe,
        fundamentals_df=fundamentals,
        klines=klines,
        benchmark_df=make_benchmark(),
        flow_df=flow,
    )
    assert "600001" not in results
    assert "600002" not in results
    assert set(results) == {"600003", "600004", "600005", "600006"}


def test_northbound_is_info_only_never_scored():
    codes, universe, fundamentals, klines, flow = dominant_frames()
    north_a = pd.DataFrame(
        {"nb_hold_ratio": [0.5], "nb_hold_mv": [1e6], "nb_report_date": ["2026-03-19"]},
        index=pd.Index(["600001"], name="code"),
    )
    north_b = pd.DataFrame(
        {"nb_hold_ratio": [88.0], "nb_hold_mv": [9e9], "nb_report_date": ["2026-03-19"]},
        index=pd.Index(["600001"], name="code"),
    )
    common = dict(
        universe_df=universe,
        fundamentals_df=fundamentals,
        klines=klines,
        benchmark_df=make_benchmark(),
        flow_df=flow,
    )
    with_a = score_universe(codes, northbound_df=north_a, **common)
    with_b = score_universe(codes, northbound_df=north_b, **common)
    assert with_a["600001"].score == with_b["600001"].score
    assert with_a["600001"].flow == with_b["600001"].flow
    assert with_a["600001"].fundamental == with_b["600001"].fundamental
    # The data still shows up as info in metrics + reasons.
    assert with_a["600001"].metrics["northbound_hold_ratio"] == 0.5
    assert any("northbound" in r and "not scored" in r for r in with_a["600001"].reasons)


# --------------------------------------------------------------------------- #
# Percentile behaviour
# --------------------------------------------------------------------------- #


def test_sub_scores_and_composite_stay_within_0_100():
    results = run_dominant()
    for item in results.values():
        assert 0.0 <= item.fundamental <= 100.0
        assert 0.0 <= item.trend <= 100.0
        assert 0.0 <= item.flow <= 100.0
        assert 0.0 <= item.score <= 100.0
    # Extreme outlier must still clamp to the ends, not past them.
    assert results["600001"].fundamental == 100.0
    assert results["600006"].fundamental == 0.0


def test_strictly_ordered_inputs_produce_ordered_composites():
    results = run_dominant()
    ordered = [results[c].score for c, *_ in _DOMINANT_SET]
    assert ordered == sorted(ordered, reverse=True)


# --------------------------------------------------------------------------- #
# Fundamental design
# --------------------------------------------------------------------------- #


def test_loss_maker_gets_low_valuation_score():
    codes = ["600001", "600002"]
    universe = pd.DataFrame(
        [
            {"code": "600001", "pe": 12.0, "pb": 2.0, "float_mv": 1e10, "total_mv": 2e10},
            {"code": "600002", "pe": -5.0, "pb": 2.0, "float_mv": 1e10, "total_mv": 2e10},
        ]
    ).set_index("code")
    fundamentals = pd.DataFrame(
        [
            {"code": "600001", "roe": 15.0, "eps": 1.0, "revenue_yoy": 10.0,
             "profit_yoy": 10.0, "gross_margin": 40.0, "ocf_per_share": 1.0,
             "debt_ratio": 40.0, "industry": "机械"},
            {"code": "600002", "roe": 15.0, "eps": 1.0, "revenue_yoy": 10.0,
             "profit_yoy": 10.0, "gross_margin": 40.0, "ocf_per_share": 1.0,
             "debt_ratio": 40.0, "industry": "机械"},
        ]
    ).set_index("code")
    klines = {code: make_klines() for code in codes}
    results = score_universe(
        codes,
        universe_df=universe,
        fundamentals_df=fundamentals,
        klines=klines,
        benchmark_df=make_benchmark(),
    )
    winner, loser = results["600001"], results["600002"]
    assert winner.fundamental > loser.fundamental
    assert "loss_maker" in loser.flags
    assert any("loss-making" in r for r in loser.reasons)


def test_financial_firms_are_not_penalised_for_leverage():
    codes = ["600001", "600002"]
    universe = pd.DataFrame(
        [
            {"code": "600001", "pe": 6.0, "pb": 0.6, "float_mv": 1e11, "total_mv": 2e11},
            {"code": "600002", "pe": 15.0, "pb": 2.5, "float_mv": 1e10, "total_mv": 2e10},
        ]
    ).set_index("code")
    fundamentals = pd.DataFrame(
        [
            {"code": "600001", "roe": 11.0, "eps": 1.0, "revenue_yoy": 5.0,
             "profit_yoy": 5.0, "gross_margin": 40.0, "ocf_per_share": 1.0,
             "debt_ratio": 92.0, "industry": "银行"},
            {"code": "600002", "roe": 11.0, "eps": 1.0, "revenue_yoy": 5.0,
             "profit_yoy": 5.0, "gross_margin": 40.0, "ocf_per_share": 1.0,
             "debt_ratio": 30.0, "industry": "机械设备"},
        ]
    ).set_index("code")
    results = score_universe(
        codes,
        universe_df=universe,
        fundamentals_df=fundamentals,
        klines={code: make_klines() for code in codes},
        benchmark_df=make_benchmark(),
    )
    bank, industrial = results["600001"], results["600002"]
    assert bank.metrics["components"]["leverage"] == NEUTRAL
    assert "leverage_neutral_financial" in bank.flags
    assert any("not penalised" in r for r in bank.reasons)
    # The industrial firm is ranked on its (low) debt instead.
    assert industrial.metrics["components"]["leverage"] > NEUTRAL


def test_valuation_falls_back_to_whole_set_for_small_industry_groups():
    # Two industries: 机械 has 5 members (industry-relative PE), 有色 has 1.
    codes = ["600001", "600002", "600003", "600004", "600005", "600009"]
    pes = {"600001": 10.0, "600002": 12.0, "600003": 14.0,
           "600004": 16.0, "600005": 18.0, "600009": 100.0}
    industries = {c: ("机械" if c != "600009" else "有色金属") for c in codes}
    universe = pd.DataFrame(
        [
            {"code": c, "pe": pes[c], "pb": 2.0, "float_mv": 1e10, "total_mv": 2e10}
            for c in codes
        ]
    ).set_index("code")
    fundamentals = pd.DataFrame(
        [
            {"code": c, "roe": 10.0, "eps": 1.0, "revenue_yoy": 5.0, "profit_yoy": 5.0,
             "gross_margin": 30.0, "ocf_per_share": 1.0, "debt_ratio": 40.0,
             "industry": industries[c]}
            for c in codes
        ]
    ).set_index("code")
    results = score_universe(
        codes,
        universe_df=universe,
        fundamentals_df=fundamentals,
        klines={c: make_klines() for c in codes},
        benchmark_df=make_benchmark(),
    )
    # Inside the 5-member group, PE ranks within the industry: cheapest -> 100,
    # dearest -> 0; the equal PBs tie at the average rank (50).
    assert results["600001"].metrics["components"]["valuation"] == pytest.approx(75.0)
    assert results["600005"].metrics["components"]["valuation"] == pytest.approx(25.0)
    assert results["600001"].metrics["components"]["valuation_industry_fallback"] == 0.0
    # The singleton industry is too small -> whole-set fallback + flag value.
    assert results["600009"].metrics["components"]["valuation_industry_fallback"] == 1.0


# --------------------------------------------------------------------------- #
# Trend design
# --------------------------------------------------------------------------- #


def test_overheated_move_is_penalised_in_trend():
    codes = ["600001", "600002"]
    universe = pd.DataFrame(
        [
            {"code": c, "pe": 15.0, "pb": 2.0, "float_mv": 1e10, "total_mv": 2e10}
            for c in codes
        ]
    ).set_index("code")
    fundamentals = pd.DataFrame(
        [
            {"code": c, "roe": 12.0, "eps": 1.0, "revenue_yoy": 8.0, "profit_yoy": 8.0,
             "gross_margin": 35.0, "ocf_per_share": 1.0, "debt_ratio": 40.0,
             "industry": "电子"}
            for c in codes
        ]
    ).set_index("code")
    klines = {
        "600001": make_klines(drift=0.001, last20_mult=0.02),  # +~48% in 20d
        "600002": make_klines(drift=0.001),
    }
    results = score_universe(
        codes,
        universe_df=universe,
        fundamentals_df=fundamentals,
        klines=klines,
        benchmark_df=make_benchmark(),
    )
    hot, calm = results["600001"], results["600002"]
    assert "overheated_move" in hot.flags
    assert "overheated_move" not in calm.flags
    assert any("overheated" in r for r in hot.reasons)


def test_overheat_and_vol_penalties_subtract_exactly_20_points():
    # Isolate the penalty arithmetic: identical trend inputs, single-member
    # sets so every percentile input ties at NEUTRAL, so the only difference
    # between the two raw scores is the penalty itself.
    def _fake_trend(**overrides: object) -> dict:
        base: dict = {
            "ma_score": 80.0,
            "dist_to_high_250": 0.9,
            "rel_60d": 5.0,
            "rel_120d": 8.0,
            "vol_expansion": 1.2,
            "ret_20d": 10.0,
            "flags": [],
        }
        base.update(overrides)
        return base

    hot = _trend_raw_batch({"600001": _fake_trend(flags=["overheated_move"])})
    wild = _trend_raw_batch({"600002": _fake_trend(flags=["extreme_volatility"])})
    calm = _trend_raw_batch({"600003": _fake_trend()})
    assert hot["600001"] == pytest.approx(calm["600003"] - OVERHEAT_PENALTY)
    assert wild["600002"] == pytest.approx(calm["600003"] - EXTREME_VOL_PENALTY)


def test_extreme_volatility_is_flagged():
    codes = ["600001", "600002"]
    universe = pd.DataFrame(
        [
            {"code": c, "pe": 15.0, "pb": 2.0, "float_mv": 1e10, "total_mv": 2e10}
            for c in codes
        ]
    ).set_index("code")
    fundamentals = pd.DataFrame(
        [
            {"code": c, "roe": 12.0, "eps": 1.0, "revenue_yoy": 8.0, "profit_yoy": 8.0,
             "gross_margin": 35.0, "ocf_per_share": 1.0, "debt_ratio": 40.0,
             "industry": "电子"}
            for c in codes
        ]
    ).set_index("code")
    results = score_universe(
        codes,
        universe_df=universe,
        fundamentals_df=fundamentals,
        klines={
            "600001": make_klines(drift=0.001, zigzag=0.04),
            "600002": make_klines(drift=0.001),
        },
        benchmark_df=make_benchmark(),
    )
    assert "extreme_volatility" in results["600001"].flags


# --------------------------------------------------------------------------- #
# Flow design
# --------------------------------------------------------------------------- #


def test_dragon_tiger_bonus_adds_flag_and_reason():
    results = run_dominant()
    lhb = pd.DataFrame(
        {
            "lhb_count": [3],
            "lhb_net_buy": [5e7],
            "lhb_last_date": ["2026-03-10"],
            "lhb_reasons": ["日涨幅偏离值达7%"],
        },
        index=pd.Index(["600002"], name="code"),
    )
    codes, universe, fundamentals, klines, flow = dominant_frames()
    boosted = score_universe(
        codes,
        universe_df=universe,
        fundamentals_df=fundamentals,
        klines=klines,
        benchmark_df=make_benchmark(),
        flow_df=flow,
        lhb_df=lhb,
    )
    assert "dragon_tiger_buy" in boosted["600002"].flags
    assert boosted["600002"].flow >= results["600002"].flow
    assert any("Dragon-Tiger" in r for r in boosted["600002"].reasons)


# --------------------------------------------------------------------------- #
# Reasons
# --------------------------------------------------------------------------- #


def test_reasons_carry_real_numbers():
    results = run_dominant()
    reasons = results["600001"].reasons
    assert any(re.search(r"ROE \d+\.\d+%", r) for r in reasons)
    assert any(re.search(r"THS net inflow \+\d", r) for r in reasons)
    assert any(re.search(r"of float cap", r) for r in reasons)
    assert any(re.search(r"120d return", r) for r in reasons)
    # Every reason line mentions at least one digit (contract: real numbers).
    assert all(re.search(r"\d", r) for r in reasons)


def test_flow_reason_mentions_cap_share():
    results = run_dominant()
    # 8e8 net 5d inflow on a 1e10 float cap -> 8.00% of float cap.
    assert any("8.00% of float cap" in r for r in results["600001"].reasons)
