"""Unit tests for the ordered exclusion funnel (CONTRACT section 5). Offline."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.market_scanner.config import ScanConfig
from src.market_scanner.exclusions import (
    LIQUIDITY_SNAPSHOT_FLAG,
    STAGE_INSUFFICIENT_LIQUIDITY,
    apply_early_rules,
    run_exclusions,
)
from src.market_scanner.models import STAGE_UNIVERSE, board_from_code

TRADE_DATE = date(2026, 3, 20)  # a Friday
LISTED_LONG_AGO = TRADE_DATE - timedelta(days=800)
RECENT_LISTING = TRADE_DATE - timedelta(days=100)


def make_universe(rows: list[dict]) -> pd.DataFrame:
    """Build a universe frame indexed by code with sensible defaults."""
    base = {
        "name": "测试",
        "exchange": "SH",
        "board": "main_sh",
        "price": 10.0,
        "pct_chg": 0.5,
        "volume": 1e6,
        "amount": 1e8,
        "pe": 20.0,
        "pb": 3.0,
        "total_mv": 5e9,
        "float_mv": 4e9,
        "turnover_rate": 1.5,
        "is_st": False,
        "is_delisting": False,
        "is_suspended": False,
        # Old enough to clear the new_listing rule by default; the
        # new_listing tests override or drop this.
        "list_date": LISTED_LONG_AGO,
    }
    records = []
    for row in rows:
        rec = dict(base)
        rec.update(row)
        code = str(rec.pop("code")).zfill(6)
        if "board" not in row:
            rec["board"] = board_from_code(code)
        if rec["board"] == "main_sh":
            rec["exchange"] = "SH"
        elif rec["board"] == "main_sz":
            rec["exchange"] = "SZ"
        elif rec["board"] == "bse":
            rec["exchange"] = "BJ"
        rec["code"] = code
        records.append(rec)
    return pd.DataFrame.from_records(records).set_index("code")


def make_klines(bars: int = 130, amount: float = 2e8) -> pd.DataFrame:
    """Ascending synthetic klines ending on TRADE_DATE."""
    dates = pd.bdate_range(end=TRADE_DATE, periods=bars)
    close = [10.0 + 0.01 * i for i in range(bars)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
            "close": close,
            "volume": [1e5] * bars,
            "amount": [amount] * bars,
        }
    )


def make_fundamentals(rows: list[dict] | None = None) -> pd.DataFrame:
    """Fundamentals frame; each row overrides the defaults."""
    base = {
        "report_period": "2025-12-31",
        "roe": 12.0,
        "eps": 1.0,
        "bps": 8.0,
        "revenue_yoy": 10.0,
        "profit_yoy": 12.0,
        "gross_margin": 30.0,
        "ocf_per_share": 1.2,
        "debt_ratio": 40.0,
        "industry": "食品饮料",
    }
    records = []
    for row in rows or [{"code": "600001"}]:
        rec = dict(base)
        rec.update(row)
        rec["code"] = str(rec.pop("code")).zfill(6)
        records.append(rec)
    return pd.DataFrame.from_records(records).set_index("code")


def _run(universe_rows, *, klines=None, fund_rows=None, config=None, listing=None):
    uni = make_universe(universe_rows)
    funds = make_fundamentals(fund_rows) if fund_rows is not None else make_fundamentals(
        [{"code": code} for code in uni.index]
    )
    if klines is None:
        klines = {code: make_klines() for code in uni.index}
    return run_exclusions(
        uni,
        config=config or ScanConfig(),
        listing_df=listing,
        klines=klines,
        fundamentals_df=funds,
        trade_date=TRADE_DATE,
    )


# --------------------------------------------------------------------------- #
# Order and counting
# --------------------------------------------------------------------------- #


def test_funnel_stage_order_matches_contract():
    result = _run([{"code": "600001"}])
    stages = [entry["stage"] for entry in result.funnel]
    assert stages == [
        STAGE_UNIVERSE,
        "excluded:bse_unsupported",
        "excluded:st",
        "excluded:delisting",
        "excluded:suspended",
        "excluded:new_listing",
        "excluded:board_disabled",
        "excluded:micro_cap",
        "excluded:illiquid",
        # Fail-closed liquidity: the no-signal stage sits right after
        # excluded:illiquid, before fundamentals.
        STAGE_INSUFFICIENT_LIQUIDITY,
        "insufficient_data:fundamentals",
        "excluded:shrinking",
        "insufficient_data:trend",
    ]
    assert STAGE_INSUFFICIENT_LIQUIDITY == "insufficient_data:liquidity"


def test_each_stock_counted_once_at_first_failing_rule():
    # ST + micro-cap + BSE all true: only the first rule (bse) may count it.
    result = _run(
        [
            {"code": "920001", "board": "bse", "exchange": "BJ",
             "is_st": True, "total_mv": 1e8, "is_suspended": True},
        ]
    )
    assert result.reason_for("920001").startswith("excluded:bse_unsupported")
    counts = {e["stage"]: e for e in result.funnel}
    assert counts["excluded:bse_unsupported"]["removed"] == 1
    assert counts["excluded:st"]["removed"] == 0
    assert counts["excluded:micro_cap"]["removed"] == 0
    assert counts["excluded:suspended"]["removed"] == 0


def test_funnel_counts_are_consistent():
    result = _run(
        [
            {"code": "920001", "board": "bse", "exchange": "BJ"},
            {"code": "600001"},
            {"code": "600002", "is_st": True},
            {"code": "600003", "is_suspended": True},
        ]
    )
    assert result.funnel[0] == {"stage": "universe", "count": 4, "removed": 0, "note": None}
    previous = 4
    for entry in result.funnel[1:]:
        assert entry["count"] == previous - entry["removed"], entry
        previous = entry["count"]
    assert result.passed == ["600001"]
    assert result.reason_for("600001") is None


# --------------------------------------------------------------------------- #
# Individual rules
# --------------------------------------------------------------------------- #


def test_bse_unsupported():
    result = _run([{"code": "920001", "board": "bse", "exchange": "BJ"}])
    assert result.reason_for("920001").startswith("excluded:bse_unsupported")
    assert result.passed == []


def test_st_rule():
    result = _run([{"code": "600001", "is_st": True}])
    assert result.reason_for("600001").startswith("excluded:st")


def test_delisting_rule():
    result = _run([{"code": "600001", "is_delisting": True}])
    assert result.reason_for("600001").startswith("excluded:delisting")


def test_suspended_rule():
    result = _run([{"code": "600001", "is_suspended": True}])
    assert result.reason_for("600001").startswith("excluded:suspended")


def test_new_listing_rule_with_recent_list_date():
    result = _run(
        [{"code": "600001", "list_date": RECENT_LISTING}],
    )
    reason = result.reason_for("600001")
    assert reason.startswith("excluded:new_listing")
    assert "100 days" in reason


def test_new_listing_rule_passes_old_listing():
    result = _run([{"code": "600001", "list_date": LISTED_LONG_AGO}])
    assert result.reason_for("600001") is None


def test_new_listing_unknown_list_date_falls_back_to_kline_history():
    """A missing list_date must NOT mean new_listing when K-lines prove age."""
    uni = make_universe([{"code": "600001"}]).drop(columns=["list_date"])
    funds = make_fundamentals([{"code": "600001"}])  # has fundamentals
    result = run_exclusions(
        uni,
        config=ScanConfig(),
        klines={"600001": make_klines(bars=300)},  # >= 250 bars: > 1 year
        fundamentals_df=funds,
        trade_date=TRADE_DATE,
    )
    assert result.reason_for("600001") is None
    assert "600001" in result.new_listing_kline_fallback
    # The stage carries a note counting how many were decided by the fallback.
    stage = next(e for e in result.funnel if e["stage"] == "excluded:new_listing")
    assert stage["note"] is not None
    assert "kline_fallback" in stage["note"]
    assert "1 stocks" in stage["note"]


def test_new_listing_unknown_date_and_short_history_is_insufficient_not_new():
    """Both signals unknown: insufficient_data (cannot determine age), NOT new_listing."""
    uni = make_universe([{"code": "600001"}]).drop(columns=["list_date"])
    funds = make_fundamentals([{"code": "600001"}])
    result = run_exclusions(
        uni,
        config=ScanConfig(),
        klines={"600001": make_klines(bars=50)},  # < 250: fallback fails
        fundamentals_df=funds,
        trade_date=TRADE_DATE,
    )
    reason = result.reason_for("600001")
    assert not reason.startswith("excluded:new_listing")
    assert reason.startswith("insufficient_data:trend")
    assert "600001" not in result.new_listing_kline_fallback


def test_new_listing_reads_list_date_from_listing_frame():
    uni = make_universe([{"code": "600001"}]).drop(columns=["list_date"])
    listing = pd.DataFrame(
        {"list_date": [RECENT_LISTING], "industry": ["化工"]},
        index=pd.Index(["600001"], name="code"),
    )
    result = run_exclusions(
        uni, config=ScanConfig(), listing_df=listing, trade_date=TRADE_DATE
    )
    assert result.reason_for("600001").startswith("excluded:new_listing")


def test_board_disabled_star_and_chinext():
    config_off_star = ScanConfig(include_star=False)
    result = _run(
        [{"code": "688001"}, {"code": "600001"}], config=config_off_star
    )
    assert result.reason_for("688001").startswith("excluded:board_disabled")
    assert result.reason_for("600001") is None

    config_off_chinext = ScanConfig(include_chinext=False)
    result = _run(
        [{"code": "300001", "board": "chinext", "exchange": "SZ"},
         {"code": "600001"}],
        config=config_off_chinext,
    )
    assert result.reason_for("300001").startswith("excluded:board_disabled")


def test_board_prefix_overrides_mislabeled_vendor_board():
    config = ScanConfig(include_star=False, include_chinext=False)
    result = _run(
        [
            {"code": "301507", "board": "main_sz", "exchange": "SZ"},
            {"code": "688001", "board": "main_sh", "exchange": "SH"},
        ],
        config=config,
    )
    assert result.reason_for("301507").startswith("excluded:board_disabled")
    assert result.reason_for("688001").startswith("excluded:board_disabled")


def test_star_and_chinext_pass_when_enabled():
    result = _run([{"code": "688001"}, {"code": "300001", "board": "chinext", "exchange": "SZ"}])
    assert result.reason_for("688001") is None
    assert result.reason_for("300001") is None


def test_micro_cap_rule_and_boundary():
    result = _run(
        [
            {"code": "600001", "total_mv": 2e9},       # below 3e9
            {"code": "600002", "total_mv": 3e9},       # exactly at threshold
            {"code": "600003", "total_mv": None},      # missing cap
        ]
    )
    assert result.reason_for("600001").startswith("excluded:micro_cap")
    assert result.reason_for("600002") is None
    assert result.reason_for("600003").startswith("excluded:micro_cap")


def test_illiquid_uses_kline_amount_not_spot_amount():
    uni_rows = [
        # Huge spot amount but thin klines -> illiquid by the contract rule.
        {"code": "600001", "amount": 5e8},
        # Tiny spot amount but heavy klines -> passes liquidity.
        {"code": "600002", "amount": 1e5},
    ]
    klines = {
        "600001": make_klines(amount=1e7),  # 20d avg = 1e7 < 5e7
        "600002": make_klines(amount=2e8),
    }
    result = _run(uni_rows, klines=klines)
    reason = result.reason_for("600001")
    assert reason.startswith("excluded:illiquid")
    assert "20d avg amount" in reason
    assert result.reason_for("600002") is None


def test_liquidity_uses_20d_average_at_exactly_20_usable_bars():
    """Rule (a) boundary: exactly 20 usable bars -> 20d average decides."""
    frame = make_klines(bars=130, amount=1e7)
    frame.loc[frame.index[:-20], "amount"] = float("nan")  # only last 20 usable
    result = _run([{"code": "600001", "amount": 5e8}], klines={"600001": frame})
    reason = result.reason_for("600001")
    assert reason.startswith("excluded:illiquid")
    assert "20d avg amount" in reason  # not the snapshot proxy


def test_fewer_than_20_usable_bars_falls_back_to_snapshot():
    """Rule (b): 19 usable thin bars must NOT exclude when the proxy passes."""
    frame = make_klines(bars=130, amount=1e7)
    frame.loc[frame.index[:-19], "amount"] = float("nan")  # 19 usable
    result = _run(
        [{"code": "600001", "amount": 2e8}], klines={"600001": frame}
    )
    assert result.reason_for("600001") is None
    assert "600001" in result.liquidity_snapshot_fallback


def test_no_klines_with_big_snapshot_passes_liquidity_falls_to_trend():
    """No klines + snapshot above the bar: snapshot pass, trend cut catches it."""
    result = _run([{"code": "600001"}], klines={})
    assert result.reason_for("600001").startswith("insufficient_data:trend")
    assert "600001" in result.liquidity_snapshot_fallback
    illiquid_stage = next(
        e for e in result.funnel if e["stage"] == "excluded:illiquid"
    )
    assert illiquid_stage["note"] is not None
    assert "snapshot_fallback" in illiquid_stage["note"]
    assert "1 decided" in illiquid_stage["note"]


def test_snapshot_below_bar_excludes_with_snapshot_reason():
    """Rule (b) fail side: thin snapshot, no 20d history -> excluded:illiquid."""
    result = _run([{"code": "600001", "amount": 1e7}], klines={})
    reason = result.reason_for("600001")
    assert reason.startswith("excluded:illiquid")
    assert "snapshot day amount, no 20d history" in reason
    assert "600001" not in result.liquidity_snapshot_fallback


def test_snapshot_at_bar_boundary_passes_with_flag():
    """Rule (b) boundary: snapshot exactly at the bar passes (flagged)."""
    frame = make_klines().drop(columns=["amount"])  # no kline signal...
    result = _run(
        [{"code": "600001", "amount": 5e7}],  # ...but snapshot exactly 5e7
        klines={"600001": frame},
    )
    assert result.reason_for("600001") is None
    assert "600001" in result.liquidity_snapshot_fallback
    assert LIQUIDITY_SNAPSHOT_FLAG == "liquidity_from_snapshot"


def test_snapshot_just_below_bar_excludes():
    frame = make_klines().drop(columns=["amount"])
    result = _run(
        [{"code": "600001", "amount": 49999999}],
        klines={"600001": frame},
    )
    assert result.reason_for("600001").startswith("excluded:illiquid")


def test_unusable_amount_with_big_snapshot_is_decided_not_silent():
    """Frame without an amount column + big snapshot: proxy pass, counted."""
    klines = {"600001": make_klines().drop(columns=["amount"])}
    result = _run([{"code": "600001"}], klines=klines)
    assert result.reason_for("600001") is None
    assert "600001" in result.liquidity_snapshot_fallback
    illiquid_stage = next(
        e for e in result.funnel if e["stage"] == "excluded:illiquid"
    )
    assert illiquid_stage["note"] is not None
    assert illiquid_stage["note"].startswith("snapshot_fallback")
    assert "600001" not in illiquid_stage["note"]  # codes are counted, not named


def test_unusable_amount_column_all_nan_with_thin_snapshot_excludes():
    frame = make_klines()
    frame["amount"] = float("nan")
    result = _run([{"code": "600001", "amount": 1e7}], klines={"600001": frame})
    reason = result.reason_for("600001")
    assert reason.startswith("excluded:illiquid")
    assert "snapshot day amount, no 20d history" in reason


def test_no_signal_at_all_is_insufficient_liquidity_not_a_pass():
    """Rule (c): no kline amount AND no snapshot amount -> new stage."""
    uni = make_universe([{"code": "600001"}]).drop(columns=["amount"])
    funds = make_fundamentals([{"code": "600001"}])
    result = run_exclusions(
        uni,
        config=ScanConfig(),
        klines={},
        fundamentals_df=funds,
        trade_date=TRADE_DATE,
    )
    reason = result.reason_for("600001")
    assert reason.startswith("insufficient_data:liquidity")
    assert "600001" not in result.passed
    assert "600001" not in result.liquidity_snapshot_fallback


def test_no_signal_stage_sits_between_illiquid_and_fundamentals():
    """Rule (c) placement: right after excluded:illiquid, one-count rule holds."""
    result = _run(
        [
            {"code": "600001", "amount": 5e8},  # 20d thin -> illiquid (20d)
            {"code": "600002", "amount": 1e7},  # snapshot thin -> illiquid (proxy)
            {"code": "600003", "amount": 2e8},  # snapshot ok, no klines -> trend
            {"code": "600004", "amount": None},  # no signal at all -> liquidity
            {"code": "600005", "amount": 2e8},  # full pass
        ],
        klines={
            "600001": make_klines(amount=1e7),
            "600005": make_klines(amount=2e8),
        },
    )
    assert result.reason_for("600001").startswith("excluded:illiquid")
    assert "20d avg amount" in result.reason_for("600001")
    assert "snapshot day amount, no 20d history" in result.reason_for("600002")
    assert result.reason_for("600003").startswith("insufficient_data:trend")
    assert result.reason_for("600004").startswith("insufficient_data:liquidity")
    assert result.reason_for("600005") is None
    assert result.passed == ["600005"]
    assert result.liquidity_snapshot_fallback == ["600003"]

    stages = [e["stage"] for e in result.funnel]
    assert stages.index(STAGE_INSUFFICIENT_LIQUIDITY) == stages.index("excluded:illiquid") + 1
    assert stages.index("insufficient_data:fundamentals") == stages.index(STAGE_INSUFFICIENT_LIQUIDITY) + 1
    by_stage = {e["stage"]: e for e in result.funnel}
    assert by_stage["excluded:illiquid"]["removed"] == 2
    assert by_stage[STAGE_INSUFFICIENT_LIQUIDITY]["removed"] == 1
    assert by_stage["insufficient_data:fundamentals"]["removed"] == 0
    assert by_stage["insufficient_data:trend"]["removed"] == 1
    # One-count-per-stock: every removal plus the survivors equals the universe.
    removed_total = sum(
        e["removed"] for e in result.funnel if e["stage"] != STAGE_UNIVERSE
    )
    assert removed_total + len(result.passed) == 5
    # Funnel note shows how many were decided via the snapshot fallback
    # (600002 excluded + 600003 passed = 2 decided).
    note = by_stage["excluded:illiquid"]["note"]
    assert note is not None and "snapshot_fallback" in note
    assert "2 decided" in note


def test_illiquid_note_empty_when_no_snapshot_decisions():
    """Nobody decided via the proxy -> no snapshot note on the stage."""
    result = _run(
        [{"code": "600001"}, {"code": "600002"}],
        klines={
            "600001": make_klines(amount=2e8),
            "600002": make_klines(amount=1e7),
        },
    )
    assert result.reason_for("600001") is None
    assert result.reason_for("600002").startswith("excluded:illiquid")
    assert result.liquidity_snapshot_fallback == []
    illiquid_stage = next(
        e for e in result.funnel if e["stage"] == "excluded:illiquid"
    )
    assert illiquid_stage["note"] is None


def test_insufficient_fundamentals_when_row_missing():
    uni = make_universe([{"code": "600001"}, {"code": "600002"}])
    funds = make_fundamentals([{"code": "600001"}])  # 600002 absent
    result = run_exclusions(
        uni,
        config=ScanConfig(),
        klines={code: make_klines() for code in uni.index},
        fundamentals_df=funds,
        trade_date=TRADE_DATE,
    )
    assert result.reason_for("600002").startswith("insufficient_data:fundamentals")
    assert result.reason_for("600001") is None


def test_insufficient_fundamentals_when_all_fields_nan():
    uni = make_universe([{"code": "600001"}])
    funds = make_fundamentals(
        [{"code": "600001", "roe": None, "gross_margin": None,
          "revenue_yoy": None, "profit_yoy": None, "debt_ratio": None}]
    )
    result = run_exclusions(
        uni,
        config=ScanConfig(),
        klines={"600001": make_klines()},
        fundamentals_df=funds,
        trade_date=TRADE_DATE,
    )
    assert result.reason_for("600001").startswith("insufficient_data:fundamentals")


def test_partial_fundamentals_still_eligible():
    uni = make_universe([{"code": "600001"}])
    funds = make_fundamentals(
        [{"code": "600001", "roe": 9.0, "gross_margin": None,
          "revenue_yoy": None, "profit_yoy": None, "debt_ratio": None}]
    )
    result = run_exclusions(
        uni,
        config=ScanConfig(),
        klines={"600001": make_klines()},
        fundamentals_df=funds,
        trade_date=TRADE_DATE,
    )
    assert result.reason_for("600001") is None


def test_shrinking_gate_uses_recurring_profit_and_reports_both_numbers():
    result = _run(
        [{"code": "600001"}],
        fund_rows=[{
            "code": "600001",
            "revenue_yoy": -12.5,
            "profit_yoy": 8.0,
            "deduct_profit_yoy": -7.25,
        }],
    )
    reason = result.reason_for("600001")
    assert reason.startswith("excluded:shrinking")
    assert "-12.5%" in reason
    assert "-7.2%" in reason
    assert "deduct_profit_yoy" in reason
    assert result.passed == []


def test_shrinking_gate_falls_back_to_reported_profit_and_notes_it():
    result = _run(
        [{"code": "600001"}],
        fund_rows=[{
            "code": "600001",
            "revenue_yoy": -3.0,
            "profit_yoy": -4.5,
            "deduct_profit_yoy": None,
        }],
    )
    reason = result.reason_for("600001")
    assert reason.startswith("excluded:shrinking")
    assert "-3.0%" in reason
    assert "-4.5%" in reason
    assert "reported profit YoY fallback" in reason


def test_shrinking_gate_disabled_and_sign_boundaries_pass():
    rows = [
        {"code": "600001", "revenue_yoy": -1.0, "deduct_profit_yoy": -1.0},
        {"code": "600002", "revenue_yoy": 0.0, "deduct_profit_yoy": -1.0},
        {"code": "600003", "revenue_yoy": -1.0, "deduct_profit_yoy": 0.0},
    ]
    result = _run(
        rows,
        fund_rows=rows,
        config=ScanConfig(exclude_shrinking=False),
    )
    assert all(
        result.reason_for(code) is None
        for code in ("600001", "600002", "600003")
    )
    result = _run(rows, fund_rows=rows)
    assert result.reason_for("600001").startswith("excluded:shrinking")
    assert result.reason_for("600002") is None
    assert result.reason_for("600003") is None


def test_shrinking_is_counted_as_exclusion_not_insufficient_data():
    result = _run(
        [{"code": "600001"}, {"code": "600002"}],
        fund_rows=[
            {"code": "600001", "revenue_yoy": -5.0, "deduct_profit_yoy": -6.0},
            {"code": "600002", "revenue_yoy": 5.0, "deduct_profit_yoy": 6.0},
        ],
    )
    by_stage = {entry["stage"]: entry for entry in result.funnel}
    assert by_stage["excluded:shrinking"]["removed"] == 1
    assert by_stage["excluded:shrinking"]["count"] == 1
    assert by_stage["insufficient_data:fundamentals"]["removed"] == 0
    assert by_stage["insufficient_data:trend"]["removed"] == 0
    assert result.passed == ["600002"]


def test_insufficient_trend_when_too_short():
    result = _run([{"code": "600001"}], klines={"600001": make_klines(bars=50)})
    reason = result.reason_for("600001")
    assert reason.startswith("insufficient_data:trend")
    assert "50 bars" in reason


def test_trend_passes_at_min_kline_bars_boundary():
    result = _run([{"code": "600001"}], klines={"600001": make_klines(bars=120)})
    assert result.reason_for("600001") is None


def test_apply_early_rules_only_covers_stages_2_to_8():
    uni = make_universe([{"code": "600001", "is_st": True}])
    reasons = apply_early_rules(uni, ScanConfig(), trade_date=TRADE_DATE)
    assert reasons["600001"].startswith("excluded:st")
    # A stock with good fundamentals but no klines passes the early rules.
    healthy = make_universe([{"code": "600002", "list_date": LISTED_LONG_AGO}])
    healthy_reasons = apply_early_rules(healthy, ScanConfig(), trade_date=TRADE_DATE)
    assert healthy_reasons["600002"] is None


def test_reasons_are_unique_per_stock_and_stage_prefixed():
    result = _run(
        [
            {"code": "920001", "board": "bse", "exchange": "BJ"},
            {"code": "600001", "is_st": True},
            {"code": "600002", "total_mv": 1e9},
        ]
    )
    for code in ("920001", "600001", "600002"):
        reason = result.reason_for(code)
        assert reason.split(":")[0] in {"excluded", "insufficient_data"}
    assert len(set(result.reason_for(c) for c in ("920001", "600001", "600002"))) == 3
