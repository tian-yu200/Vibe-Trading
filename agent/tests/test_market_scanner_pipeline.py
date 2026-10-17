"""End-to-end tests for run_scan (CONTRACT section 7). Offline fake sources."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from src.market_scanner.config import ScanConfig
from src.market_scanner.models import (
    AI_STATUS_PENDING,
    FULL_FUNNEL_STAGES,
    board_from_code,
)
from src.market_scanner.pipeline import (
    SESSION_CLOSE,
    SESSION_OPEN,
    SHANGHAI,
    SourceBundle,
    SourceStatus,
    _resolve_trade_date,
    run_scan,
)

TRADE_DATE = date(2026, 3, 20)  # a Friday
NOW = datetime(2026, 3, 20, 10, 0, tzinfo=SHANGHAI)
LISTED = date(2023, 1, 1)


@pytest.fixture(autouse=True)
def _no_default_llm(monkeypatch):
    """Keep the default-provider path offline and deterministic in tests.

    Without an injected ``llm``, run_scan tries to construct the app's real
    ChatLLM provider; that must never succeed (or hit the network) here.
    """

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no provider in tests")

    monkeypatch.setattr(
        "src.market_scanner.pipeline.build_default_llm_callable", _boom
    )

# code -> (name, industry, roe, pe, drift, flow_5d or None, extra universe flags)
_STOCKS: list[tuple[str, str, str, float, float, float, float | None, dict]] = [
    ("600001", "贵州测试", "食品饮料", 25.0, 10.0, 0.003, 8e8, {}),
    ("600002", "饮料测试", "食品饮料", 15.0, 18.0, 0.001, None, {}),  # flow missing
    ("600003", "电子测试", "电子", 18.0, 22.0, 0.002, 2e8, {}),
    ("600004", "芯片测试", "半导体", 12.0, 30.0, 0.0015, 1e8, {"board": "star"}),
    ("300001", "医药测试", "医药", 10.0, 28.0, 0.0008, 5e7, {"board": "chinext"}),
    ("600005", "ST测试", "综合", 5.0, 40.0, 0.0, 1e7, {"is_st": True}),
    ("600006", "无K线", "机械", 8.0, 35.0, 0.0005, 1e7, {}),  # klines withheld
    ("920001", "北交测试", "综合", 5.0, 20.0, 0.0, 1e7, {"board": "bse"}),
]


def _ok(data, name: str, status: str = "ok", **extra):
    rows = len(data) if hasattr(data, "__len__") else 0
    return data, SourceStatus(name=name, status=status, rows=rows, **extra)


def make_universe() -> pd.DataFrame:
    records = []
    for code, name, _ind, _roe, pe, _drift, _flow, extra in _STOCKS:
        board = extra.get("board") or board_from_code(code)
        exchange = board.split("_")[-1].upper()
        records.append(
            {
                "code": code,
                "symbol": f"{exchange.lower()}{code}",
                "name": name,
                "exchange": exchange,
                "board": board,
                "price": 10.0,
                "pct_chg": 0.5,
                "volume": 1e6,
                "amount": 2e8,
                "pe": pe,
                "pb": 2.0,
                "total_mv": 2e10,
                "float_mv": 1e10,
                "turnover_rate": 1.2,
                "is_st": bool(extra.get("is_st", False)),
                "is_delisting": False,
                "is_suspended": False,
                "list_date": LISTED,
            }
        )
    return pd.DataFrame.from_records(records).set_index("code")


def make_fundamentals() -> pd.DataFrame:
    records = []
    for code, _name, industry, roe, _pe, _drift, _flow, _extra in _STOCKS:
        records.append(
            {
                "code": code,
                "report_period": "2025-12-31",
                "roe": roe,
                "eps": 1.0,
                "bps": 8.0,
                "revenue_yoy": roe / 2,
                "profit_yoy": roe / 2,
                "gross_margin": 30.0,
                "ocf_per_share": 1.2,
                "debt_ratio": 40.0,
                "industry": industry,
            }
        )
    return pd.DataFrame.from_records(records).set_index("code")


def make_klines(bars: int = 130, drift: float = 0.001) -> pd.DataFrame:
    closes = [10.0 * (1 + drift) ** i for i in range(bars)]
    return pd.DataFrame(
        {
            "date": pd.bdate_range(end=TRADE_DATE, periods=bars),
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1e5] * bars,
            "amount": [2e8] * bars,
        }
    )


def make_klines_map(*, keyed_by_symbol: bool = False) -> dict:
    out = {}
    for code, _name, _ind, _roe, _pe, drift, _flow, _extra in _STOCKS:
        if code in {"600005", "600006", "920001"}:
            continue  # ST/BSE fail earlier; 600006 has no klines on purpose
        key = f"sh{code}" if keyed_by_symbol else code
        out[key] = make_klines(drift=drift)
    return out


def make_benchmark() -> pd.DataFrame:
    closes = [3000.0 * (1 + 0.0002) ** i for i in range(130)]
    return pd.DataFrame(
        {"date": pd.bdate_range(end=TRADE_DATE, periods=130), "close": closes}
    )


def make_flow() -> pd.DataFrame:
    records = []
    for code, _name, _ind, _roe, _pe, _drift, flow5, _extra in _STOCKS:
        if flow5 is None:
            continue
        records.append(
            {
                "code": code,
                "flow_1d": flow5 / 5,
                "flow_3d": flow5 / 2,
                "flow_5d": flow5,
                "pct_chg_3d": 1.0,
                "pct_chg_5d": 2.0,
            }
        )
    return pd.DataFrame.from_records(records).set_index("code")


def make_sources(**overrides) -> SourceBundle:
    """A full fake source bundle; override any member per test."""
    bundle = SourceBundle(
        load_calendar=lambda: (
            None,
            SourceStatus(name="calendar", status="ok", rows=0),
        ),
        fetch_universe=lambda: _ok(make_universe(), "universe"),
        fetch_listing_info=lambda: (
            None,
            SourceStatus(name="listing", status="ok", rows=0),
        ),
        fetch_fundamentals=lambda trade_date: _ok(make_fundamentals(), "fundamentals"),
        fetch_klines=lambda symbols, bars=None, workers=None, progress=None, **_kw: _ok(
            make_klines_map(), "klines"
        ),
        fetch_benchmark=lambda **kw: _ok(make_benchmark(), "benchmark"),
        fetch_fund_flow=lambda: _ok(make_flow(), "fund_flow"),
        fetch_dragon_tiger=lambda trade_date, days=30: (
            None,
            SourceStatus(name="dragon_tiger", status="skipped"),
        ),
        fetch_northbound_info=lambda: (
            None,
            SourceStatus(name="northbound", status="skipped"),
        ),
    )
    for name, value in overrides.items():
        setattr(bundle, name, value)
    return bundle


def run(**kwargs) -> dict:
    # The default fixture universe has 1/6 insufficient trend (83.3%); most
    # tests target ranking/funnel mechanics, so the default run clears a
    # floor of 0.8. The coverage guard gets its own tests below.
    kwargs.setdefault("config", ScanConfig(min_coverage=0.8))
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("sources", make_sources())
    return run_scan(**kwargs)


# --------------------------------------------------------------------------- #
# Envelope + happy path
# --------------------------------------------------------------------------- #


def test_envelope_has_contract_fields_and_ok_status():
    result = run()
    assert result["status"] == "ok"
    assert result["trade_date"] == "2026-03-20"
    assert result["intraday"] is True  # Friday 10:00 < 15:00
    assert result["trigger"] == "cli"
    assert result["ai_status"] == AI_STATUS_PENDING
    assert result["ai_note"] == (
        "no LLM provider configured (RuntimeError: no provider in tests)"
    )
    for key in (
        "run_id", "started_at", "finished_at", "duration_s", "config",
        "funnel", "sources", "picks", "ai_demoted", "requalified",
        "pinned", "scores", "failure_reason",
    ):
        assert key in result
    assert result["failure_reason"] is None
    assert result["config"] == ScanConfig(min_coverage=0.8).to_dict()


def test_picks_are_ranked_with_contract_shape():
    result = run()
    picks = result["picks"]
    assert picks, "expected at least one pick from the fake universe"
    assert [p["rank"] for p in picks] == list(range(1, len(picks) + 1))
    assert all(p["rule_rank"] >= 1 for p in picks)
    scores = [p["score"] for p in picks]
    assert scores == sorted(scores, reverse=True)
    for pick in picks:
        assert set(pick) == {
            "code", "name", "board", "industry", "rank", "rule_rank", "score",
            "sub_scores", "metrics", "reasons", "flags", "ai",
        }
        assert set(pick["sub_scores"]) == {"fundamental", "trend", "flow"}
        assert pick["ai"]["status"] == AI_STATUS_PENDING


def test_top_stock_is_the_strongest_one():
    result = run()
    assert result["picks"][0]["code"] == "600001"
    assert result["picks"][0]["industry"] == "食品饮料"


def test_missing_flow_pick_gets_neutral_50_and_flag():
    result = run()
    pick = next(p for p in result["picks"] if p["code"] == "600002")
    assert pick["sub_scores"]["flow"] == 50.0
    assert "flow_missing" in pick["flags"]


def test_liquidity_snapshot_fallback_gets_flag_on_pick():
    """opencode2 rule (b): liquidity judged from the Sina snapshot day amount
    (no usable 20d K-line amount history) passes and flags the item
    ``liquidity_from_snapshot`` on the pick."""
    from src.market_scanner.exclusions import LIQUIDITY_SNAPSHOT_FLAG

    base = make_klines_map()

    def _klines_no_amount(*_args, **_kw):
        out = {code: frame.copy() for code, frame in base.items()}
        out["600001"]["amount"] = float("nan")  # 0 usable bars -> snapshot
        return _ok(out, "klines")

    result = run(sources=make_sources(fetch_klines=_klines_no_amount))
    pick = next(p for p in result["picks"] if p["code"] == "600001")
    assert LIQUIDITY_SNAPSHOT_FLAG in pick["flags"]
    # A stock with full K-line amount history keeps the bar-based verdict.
    other = next(p for p in result["picks"] if p["code"] == "600002")
    assert LIQUIDITY_SNAPSHOT_FLAG not in other["flags"]
    # 600001 was decided via the proxy, not cut at insufficient_data:liquidity.
    liquidity = next(
        e for e in result["funnel"] if e["stage"] == "insufficient_data:liquidity"
    )
    assert liquidity["removed"] == 0


def test_funnel_lists_every_stage_in_order():
    result = run()
    stages = [entry["stage"] for entry in result["funnel"]]
    assert stages == list(FULL_FUNNEL_STAGES)
    by_stage = {e["stage"]: e for e in result["funnel"]}
    assert by_stage["universe"]["count"] == len(_STOCKS)
    assert by_stage["excluded:st"]["removed"] == 1
    assert by_stage["excluded:bse_unsupported"]["removed"] == 1
    assert by_stage["insufficient_data:trend"]["removed"] == 1
    assert by_stage["ai_demoted"]["count"] == 0
    assert by_stage["ai_demoted"]["note"] is None


def test_scores_cover_the_whole_universe():
    result = run()
    scored_rows = [r for r in result["scores"] if r["score"] is not None]
    excluded_rows = [r for r in result["scores"] if r["exclusion_reason"]]
    assert len(result["scores"]) == len(_STOCKS)
    assert len(scored_rows) == 5  # 600001-4 + 300001
    reasons = {r["code"]: r["exclusion_reason"] for r in excluded_rows}
    assert reasons["600005"].startswith("excluded:st")
    assert reasons["920001"].startswith("excluded:bse_unsupported")
    assert reasons["600006"].startswith("insufficient_data:trend")


# --------------------------------------------------------------------------- #
# Coverage + timings (CONTRACT §7 diagnostic block)
# --------------------------------------------------------------------------- #


def test_coverage_block_matches_funnel_and_scored():
    result = run()
    cov = result["coverage"]
    # 8 stocks; ST + BSE + new_listing/board/micro/illiquid all land in excluded:*,
    # leaving the 5 scored + 600006 (insufficient trend). 600002 has flow_missing.
    assert cov["scored"] == 5
    assert cov["trend_scored"] == 5
    assert cov["flow_neutral"] == 1  # 600002
    assert cov["flow_real"] == 4
    assert cov["insufficient"] == 1  # 600006
    assert cov["passed_exclusions"] == 6  # 5 scored + 600006
    assert cov["insufficient_by_key"] == {"insufficient_data:trend": 1}
    assert cov["trend_coverage"] == pytest.approx(5 / 6, abs=1e-4)
    assert cov["fundamentals_coverage"] == 1.0
    assert cov["min_coverage"] == 0.8
    assert result["coverage_note"] is None


def test_coverage_invariant_holds():
    from src.market_scanner.pipeline import _coverage_invariant_holds

    result = run()
    cov = result["coverage"]
    assert _coverage_invariant_holds(cov)
    assert (
        cov["passed_exclusions"] - cov["insufficient"]
        == cov["scored"]
        == cov["trend_scored"]
        == cov["flow_real"] + cov["flow_neutral"]
    )


def test_shrinking_gate_is_an_exclusion_before_scoring():
    funds = make_fundamentals()
    funds.loc["600001", "revenue_yoy"] = -8.0
    funds.loc["600001", "deduct_profit_yoy"] = -9.0
    result = run(
        config=ScanConfig(min_coverage=0.0),
        sources=make_sources(
            fetch_fundamentals=lambda trade_date: _ok(funds, "fundamentals")
        ),
    )
    growth = next(e for e in result["funnel"] if e["stage"] == "excluded:shrinking")
    assert growth["removed"] == 1
    assert "600001" not in {
        row["code"] for row in result["scores"] if row["score"] is not None
    }
    scored_row = next(row for row in result["scores"] if row["code"] == "600001")
    assert scored_row["exclusion_reason"].startswith("excluded:shrinking")
    assert result["coverage"]["passed_exclusions"] == 5
    assert result["coverage"]["insufficient"] == 1
    assert result["coverage"]["scored"] == 4
    assert result["coverage"]["trend_scored"] == 4
    assert result["coverage"]["fundamentals_coverage"] == 1.0
    assert result["failure_reason"] is None


def test_timings_block_has_every_stage_and_total():
    result = run()
    timings = result["timings"]
    for stage in (
        "universe", "listing", "fundamentals", "klines", "flows",
        "exclusions", "scoring", "news_red_flags", "ai_review", "total",
    ):
        assert stage in timings, f"missing timing for {stage}"
        assert timings[stage] >= 0.0
    assert timings["total"] == result["duration_s"]


def test_failed_run_has_empty_coverage_and_timings():
    from src.market_scanner.pipeline import SourceStatus as _SS

    empty = pd.DataFrame().rename_axis("code")
    bundle = SourceBundle(
        load_calendar=make_sources().load_calendar,
        fetch_universe=lambda: (empty, _SS(name="universe", status="failed")),
    )
    result = run(sources=bundle)
    assert result["status"] == "failed"
    assert result["coverage"] == {
        "passed_exclusions": 0,
        "insufficient": 0,
        "insufficient_by_key": {},
        "scored": 0,
        "trend_scored": 0,
        "flow_real": 0,
        "flow_neutral": 0,
        "trend_coverage": 1.0,
        "fundamentals_coverage": 1.0,
        "min_coverage": 0.8,
    }
    assert result["coverage_note"] is None
    assert result["timings"]["total"] >= 0.0


# --------------------------------------------------------------------------- #
# Coverage guard (min_coverage floor -> failed run)
# --------------------------------------------------------------------------- #


def test_coverage_guard_fails_low_trend_coverage():
    """trend 5/6 = 83.3% < 90% -> failed with the standard reason."""
    result = run(config=ScanConfig(min_coverage=0.90))
    assert result["status"] == "failed"
    assert result["failure_reason"] == (
        "insufficient coverage (trend 83.3% < 90%)"
    )
    assert result["coverage"]["trend_coverage"] == pytest.approx(5 / 6, abs=1e-4)
    assert result["coverage"]["min_coverage"] == 0.90
    # Picks/funnel still computed (kept for history), only the status flips.
    assert result["picks"]


def test_coverage_guard_passes_at_or_above_floor():
    result = run(config=ScanConfig(min_coverage=0.8))
    assert result["status"] == "ok"
    assert result["failure_reason"] is None


def test_coverage_guard_helper_reasons():
    from src.market_scanner.pipeline import _coverage_guard_failure

    trend_low = {
        "passed_exclusions": 4166,
        "trend_scored": 501,
        "insufficient_by_key": {},
    }
    assert _coverage_guard_failure(trend_low, 0.90) == (
        "insufficient coverage (trend 12.0% < 90%)"
    )
    fund_low = {
        "passed_exclusions": 10,
        "trend_scored": 10,
        "insufficient_by_key": {"insufficient_data:fundamentals": 3},
    }
    assert _coverage_guard_failure(fund_low, 0.90) == (
        "insufficient coverage (fundamentals 70.0% < 90%)"
    )
    both_low = {
        "passed_exclusions": 10,
        "trend_scored": 5,
        "insufficient_by_key": {"insufficient_data:fundamentals": 3},
    }
    assert _coverage_guard_failure(both_low, 0.90) == (
        "insufficient coverage (trend 50.0% < 90%)"
    )
    ok = {"passed_exclusions": 10, "trend_scored": 10, "insufficient_by_key": {}}
    assert _coverage_guard_failure(ok, 0.90) is None
    empty = {"passed_exclusions": 0, "trend_scored": 0, "insufficient_by_key": {}}
    assert _coverage_guard_failure(empty, 0.90) is None


# --------------------------------------------------------------------------- #
# Industry cap / suppressed / pinned
# --------------------------------------------------------------------------- #


def test_industry_cap_limits_picks_per_industry():
    result = run(config=ScanConfig(top_n=10, max_per_industry=1))
    picks = result["picks"]
    industries = [p["industry"] for p in picks]
    assert len(industries) == len(set(industries))
    # 食品饮料 has two scored members; only the better one survives the cap.
    food = [p["code"] for p in picks if p["industry"] == "食品饮料"]
    assert food == ["600001"]


def test_suppressed_pick_is_requalified_not_picked():
    result = run(suppressed={"600001"})
    assert "600001" not in {p["code"] for p in result["picks"]}
    assert [p["code"] for p in result["requalified"]] == ["600001"]


def test_pinned_low_scorer_bypasses_top_n_and_gets_flagged():
    result = run(config=ScanConfig(top_n=1), pinned={"600003"})
    codes = [p["code"] for p in result["picks"]]
    assert codes[0] == "600001"
    assert "600003" in codes
    pinned_pick = next(p for p in result["picks"] if p["code"] == "600003")
    assert "pinned" in pinned_pick["flags"]
    assert [p["code"] for p in result["pinned"]] == ["600003"]


def test_pinned_already_selected_is_not_duplicated():
    result = run(pinned={"600001"})
    assert [p["code"] for p in result["picks"]].count("600001") == 1
    pick = next(p for p in result["picks"] if p["code"] == "600001")
    assert "pinned" in pick["flags"]
    assert [p["code"] for p in result["pinned"]] == ["600001"]


def test_pinned_beats_suppression():
    result = run(suppressed={"600001"}, pinned={"600001"})
    assert "600001" in {p["code"] for p in result["picks"]}
    assert result["requalified"] == []


def test_pinned_excluded_stock_reports_exclusion_reason():
    result = run(pinned={"600005"})
    entry = result["pinned"][0]
    assert entry["code"] == "600005"
    assert entry["exclusion_reason"].startswith("excluded:st")
    assert entry["name"] == "ST测试"
    assert "600005" not in {p["code"] for p in result["picks"]}


def test_pinned_unknown_code_reports_absence_from_universe():
    result = run(pinned={"999999"})
    entry = result["pinned"][0]
    assert entry["exclusion_reason"] == "not in the current universe snapshot"


# --------------------------------------------------------------------------- #
# Source failure handling
# --------------------------------------------------------------------------- #


def test_empty_universe_is_a_failed_run():
    result = run(
        sources=make_sources(fetch_universe=lambda: _ok(pd.DataFrame(), "universe"))
    )
    assert result["status"] == "failed"
    assert result["picks"] == []
    assert result["scores"] == []
    assert result["funnel"] == [
        {"stage": "universe", "count": 0, "removed": 0, "note": "universe fetch returned no rows"}
    ]
    assert result["ai_note"] == "universe fetch returned no rows"


def test_raising_universe_is_a_failed_run():
    def _boom():
        raise RuntimeError("feed down")

    result = run(sources=make_sources(fetch_universe=_boom))
    assert result["status"] == "failed"
    universe_status = next(s for s in result["sources"] if s["name"] == "universe")
    assert universe_status["status"] == "failed"
    assert "RuntimeError: feed down" in universe_status["error"]


def test_failing_flow_source_degrades_run_to_partial():
    def _boom():
        raise RuntimeError("flow endpoint 503")

    result = run(sources=make_sources(fetch_fund_flow=_boom))
    assert result["status"] == "partial"
    assert result["picks"], "scan must still produce picks without flow data"
    flow_status = next(s for s in result["sources"] if s["name"] == "flows")
    assert flow_status["status"] == "failed"
    assert "flow endpoint 503" in flow_status["error"]
    # Without flow data every pick falls back to the neutral 50.
    assert all(p["sub_scores"]["flow"] == 50.0 for p in result["picks"])
    assert all("flow_missing" in p["flags"] for p in result["picks"])


def test_absent_sources_are_skipped_not_failed():
    bundle = SourceBundle(
        load_calendar=make_sources().load_calendar,
        fetch_universe=make_sources().fetch_universe,
    )
    # min_coverage=0 isolates SOURCE-status semantics: with every data source
    # absent the coverage guard (tested above) would legitimately fail the run.
    result = run(sources=bundle, config=ScanConfig(min_coverage=0.0))
    assert result["status"] == "ok"
    by_name = {s["name"]: s["status"] for s in result["sources"]}
    assert by_name["universe"] == "ok"
    assert by_name["klines"] == "skipped"
    assert by_name["flows"] == "skipped"
    assert result["picks"] == []  # no fundamentals -> everyone insufficient_data
    funds_stage = next(
        e for e in result["funnel"] if e["stage"] == "insufficient_data:fundamentals"
    )
    assert funds_stage["removed"] == 6  # early survivors, minus st/bse already cut


def test_bad_return_shape_is_recorded_as_failed_source():
    result = run(sources=make_sources(fetch_benchmark=lambda **kw: {"not": "a tuple"}))
    assert result["status"] == "partial"
    bench = next(s for s in result["sources"] if s["name"] == "benchmark")
    assert bench["status"] == "failed"
    assert "bad return shape" in bench["error"]


# --------------------------------------------------------------------------- #
# Clock / progress / robustness
# --------------------------------------------------------------------------- #


def test_weekend_run_is_not_intraday_and_backdates_trade_date():
    result = run(now=datetime(2026, 3, 21, 10, 0, tzinfo=SHANGHAI))
    assert result["intraday"] is False
    assert result["trade_date"] == "2026-03-20"  # rolls back to Friday


class _ClockCalendar:
    """Calendar seam for deterministic session-boundary tests."""

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5

    def previous_trading_day(self, day: date) -> date:
        day -= timedelta(days=1)
        while day.weekday() >= 5:
            day -= pd.Timedelta(days=1)
        return day

    def latest_completed_trading_day(self, current: datetime) -> date:
        if self.is_trading_day(current.date()) and current.time() >= SESSION_CLOSE:
            return current.date()
        return self.previous_trading_day(current.date())


@pytest.mark.parametrize(
    ("clock", "expected_date", "expected_intraday"),
    [
        (datetime(2026, 3, 20, 0, 47, tzinfo=SHANGHAI), date(2026, 3, 19), False),
        (datetime(2026, 3, 20, 9, 30, tzinfo=SHANGHAI), date(2026, 3, 20), True),
        (datetime(2026, 3, 20, 15, 30, tzinfo=SHANGHAI), date(2026, 3, 20), False),
        (datetime(2026, 3, 21, 10, 0, tzinfo=SHANGHAI), date(2026, 3, 20), False),
    ],
)
def test_trade_date_session_windows_use_calendar(clock, expected_date, expected_intraday):
    assert _resolve_trade_date(_ClockCalendar(), clock) == (
        expected_date,
        expected_intraday,
    )


def test_naive_now_is_treated_as_shanghai():
    result = run(now=datetime(2026, 3, 20, 14, 0))
    assert result["intraday"] is True


def test_progress_callback_starts_at_start_and_ends_at_done():
    calls: list[tuple[str, float]] = []
    result = run(progress=lambda stage, fraction: calls.append((stage, fraction)))
    assert calls[0] == ("start", 0.0)
    assert calls[-1] == ("done", 1.0)
    assert all(0.0 <= fraction <= 1.0 for _, fraction in calls)


def test_raising_progress_callback_does_not_break_the_scan():
    def _bad_progress(stage: str, fraction: float) -> None:
        raise RuntimeError("ui gone")

    result = run(progress=_bad_progress)
    assert result["status"] == "ok"
    assert result["picks"]


def test_symbol_keyed_klines_are_normalized():
    bundle = make_sources(
        fetch_klines=lambda symbols, bars=None, workers=None, progress=None, **_kw: _ok(
            make_klines_map(keyed_by_symbol=True), "klines"
        )
    )
    result = run(sources=bundle)
    codes = [r["code"] for r in result["scores"] if r["score"] is not None]
    assert "600001" in codes


def test_injected_llm_is_reviewed_and_picks_get_ai_blocks():
    calls: list[str] = []

    def _llm(messages):
        user = messages[-1]["content"]
        code = user.split("Review stock ", 1)[1].split(" ", 1)[0]
        calls.append(code)
        return json.dumps(
            {
                "code": code,
                "verdict": "watch",
                "confidence": 0.9,
                "summary": "solid candidate",
                "reasons": [{"text": "ROE strong", "cites": ["M:roe"]}],
                "red_flags": [],
            }
        )

    result = run(llm=_llm)
    assert result["ai_status"] == "reviewed"
    assert result["picks"], "expected picks with a reviewed ai block"
    for pick in result["picks"]:
        assert pick["ai"]["status"] == "reviewed"
        assert pick["ai"]["verdict"] == "watch"
    # every pool stock (all 5 scored) is reviewed exactly once
    assert sorted(calls) == ["300001", "600001", "600002", "600003", "600004"]
    assert all(row["ai"]["status"] == "reviewed" for row in result["ai_pool"])


def test_ai_disabled_skips_review_entirely():
    called: list = []

    def _llm(messages):
        called.append(messages)
        return "{}"

    cfg = ScanConfig(ai_enabled=False)
    result = run(config=cfg, llm=_llm)
    assert called == []
    assert result["ai_status"] == "not_reviewed"
    assert result["ai_note"] == "AI review disabled by config"
    assert all(p["ai"]["status"] == "not_reviewed" for p in result["picks"])


def test_chatllm_instance_is_adapted_to_callable_not_called_directly():
    """Regression for run 246483edc829: a raw ChatLLM object passed as the
    injected ``llm`` must be bound to ``.chat`` (which takes
    ``messages, tools, timeout`` and returns an ``LLMResponse``), not invoked
    as ``llm(messages)`` — that raised ``TypeError: 'ChatLLM' object is not
    callable`` on every pick."""
    calls: list[list] = []

    class _FakeResponse:
        content_filter_triggered = False

        def __init__(self, text: str):
            self.content = text

    class _FakeChatLLM:
        """Mirrors the real ChatLLM: NOT callable, has a .chat method."""

        def chat(self, messages, tools=None, timeout=None):
            calls.append(messages)
            user = messages[-1]["content"]
            code = user.split("Review stock ", 1)[1].split(" ", 1)[0]
            return _FakeResponse(
                json.dumps(
                    {
                        "code": code,
                        "verdict": "watch",
                        "confidence": 0.9,
                        "summary": "ok",
                        "reasons": [{"text": "ROE strong", "cites": ["M:roe"]}],
                        "red_flags": [],
                    }
                )
            )

    assert not callable(_FakeChatLLM())  # mirrors the real client

    result = run(llm=_FakeChatLLM())
    assert result["ai_status"] == "reviewed"
    assert result["picks"], "expected picks with a reviewed ai block"
    for pick in result["picks"]:
        assert pick["ai"]["status"] == "reviewed"
    assert calls, "the .chat method should have been invoked (not the object)"


def test_unusable_llm_input_skips_review_without_typerror():
    class _NotCallableAtAll:
        pass

    result = run(llm=_NotCallableAtAll())
    assert result["ai_status"] == "not_reviewed"
    assert "llm input not usable" in result["ai_note"]
    assert all(p["ai"]["status"] == "not_reviewed" for p in result["picks"])


def test_kline_full_history_coverage_only_counts_passed_exclusions():
    """coverage['kline_full_history'] must count only stocks that passed the
    exclusion funnel AND have >= min_kline_bars bars — not every frame in the
    klines dict (which still holds stocks later cut by other rules)."""
    # 600006 is in the universe but has NO klines; it is cut by
    # insufficient_data:trend, so it must not inflate kline_full_history.
    # Every other passed stock has a full-history frame.
    result = run()
    passed = result["coverage"]["passed_exclusions"]
    kfh = result["coverage"]["kline_full_history"]
    assert kfh <= passed
    # 600006 (no klines) is excluded by insufficient_data:trend, so it is
    # not in passed_exclusions at all.
    assert result["coverage"]["insufficient"] >= 1


def test_funnel_removed_is_always_count_delta():
    """Every *cut* stage's removed must equal count[i-1] - count[i].

    The exclusion funnel (universe through insufficient_data:trend) is a
    single ordered pass where each stock is removed once; the tail stages
    (scored/finalists/ai_demoted/watchlist) report pool/pick metrics rather
    than a count chain, so the invariant is scoped to the cut stages only.
    """
    result = run()
    funnel = result["funnel"]
    cut_stages = {
        "universe", "excluded:bse_unsupported", "excluded:st",
        "excluded:delisting", "excluded:suspended", "excluded:new_listing",
        "excluded:board_disabled", "excluded:micro_cap", "excluded:illiquid",
        "insufficient_data:liquidity",
        "insufficient_data:fundamentals", "insufficient_data:trend",
        "excluded:shrinking",
    }
    cut = [e for e in funnel if e["stage"] in cut_stages]
    assert cut[0]["stage"] == "universe"
    assert cut[0]["removed"] == 0
    previous = cut[0]["count"]
    for entry in cut[1:]:
        assert entry["removed"] == previous - entry["count"], entry
        previous = entry["count"]


def test_scored_stage_removed_is_zero_not_universe_scale():
    """Regression: the old code reported removed=len(uni)-len(scored) for the
    scored stage, which showed as a huge nonsensical number. It must be 0."""
    result = run()
    scored_stage = next(e for e in result["funnel"] if e["stage"] == "scored")
    assert scored_stage["removed"] == 0


def test_trigger_is_passed_through():
    result = run(trigger="schedule")
    assert result["trigger"] == "schedule"


def test_run_id_is_unique_per_scan():
    first, second = run(), run()
    assert first["run_id"] != second["run_id"]
