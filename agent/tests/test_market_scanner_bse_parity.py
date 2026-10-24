"""Parity: the refactored BSE predicate matches the HEAD funnel exactly.

The warm-up filters its symbols with the scan's own board exclusion
(FIX2 phase (e), Q2). That refactor touches the scan funnel, so this test
runs the exclusion step twice on the same offline fixture: once with the
HEAD (commit 8dfddb1) module snapshot and once with the current one, then
requires identical per-code reasons AND identical funnel counts.

Offline: synthetic universe rows, synthetic klines and fundamentals, a
local ScanConfig. No network, no live data, no personal data.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
AGENT = Path(__file__).resolve().parents[1]
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))

HEAD_PY = FIXTURES / "_head_exclusions_snapshot.py"
UNIVERSE_JSON = FIXTURES / "bse_parity_universe.json"
GOLDEN_JSON = FIXTURES / "bse_parity_golden.json"
TRADE_DATE = date(2026, 10, 16)

# Thresholds above these keep the fixture away from the liquidity and
# micro-cap cuts, so the funnel reaches the later stages as well.
AMOUNT = 200000000.0

BASE = {
    "name": "synthetic",
    "exchange": "SH",
    "price": 10.0,
    "pct_chg": 0.5,
    "volume": 1000000.0,
    "amount": AMOUNT,
    "pe": 20.0,
    "pb": 3.0,
    "total_mv": 5e9,
    "float_mv": 4e9,
    "turnover_rate": 1.5,
    "is_st": False,
    "is_delisting": False,
    "is_suspended": False,
    "settlement": 10.0,
    "open": 10.0,
    "high": 10.0,
    "low": 10.0,
}


def _load_head():
    """Import the HEAD snapshot as a package submodule (relative imports)."""
    alias = "src.market_scanner._head_exclusions"
    if alias not in sys.modules:
        pkg = types.ModuleType(alias)
        pkg.__path__ = []
        sys.modules[alias] = pkg
        spec = importlib.util.spec_from_file_location(alias, HEAD_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    return sys.modules[alias]


def _apply_flags(row, flags):
    rec = dict(row)
    if "st" in flags:
        rec["name"] = "ST synthetic"
        rec["is_st"] = True
    if "delisting" in flags:
        rec["name"] = "synthetic TUI"
        rec["is_delisting"] = True
    if "suspended" in flags:
        rec["price"] = 0.0
        rec["is_suspended"] = True
    return rec


def _make_universe(rows):
    records = []
    for row in rows:
        rec = dict(BASE)
        rec.update(row)
        rec["code"] = str(rec.pop("code")).zfill(6)
        records.append(rec)
    return pd.DataFrame.from_records(records).set_index("code")


def _make_klines(n=250):
    dates = pd.bdate_range(end="2026-10-16", periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": 10.0,
        "high": 10.0,
        "low": 10.0,
        "close": 10.0,
        "volume": 1000000.0,
        "amount": AMOUNT,
    })


def _run(exclusions, universe):
    codes = list(universe.index)
    funds = pd.DataFrame.from_records([
        {"code": c, "list_date": "2000-01-01", "revenue_yoy": 5.0,
         "profit_yoy": 5.0, "debt_ratio": 40.0} for c in codes
    ]).set_index("code")
    klines = {c: _make_klines() for c in codes}
    return exclusions.run_exclusions(
        universe, config=exclusions.ScanConfig(), listing_df=None,
        klines=klines, fundamentals_df=funds, trade_date=TRADE_DATE)


def _run_funnel(module):
    """Run one module's funnel over the offline fixture (no network)."""
    rows = json.loads(UNIVERSE_JSON.read_text(encoding="utf-8"))
    uni = _make_universe(
        [_apply_flags(r, r.get("flags") or []) for r in rows])
    funds = pd.DataFrame.from_records([
        {"code": c, "list_date": "2000-01-01",
         "revenue_yoy": 5.0, "profit_yoy": 5.0, "debt_ratio": 40.0}
        for c in uni.index
    ]).set_index("code")
    klines = {c: _make_klines() for c in uni.index}
    return module.run_exclusions(
        uni, config=module.ScanConfig(), listing_df=None,
        klines=klines, fundamentals_df=funds,
        trade_date=TRADE_DATE)


def test_refactored_funnel_matches_head_snapshot():
    """HEAD (8dfddb1) vs now: same per-code reasons, same funnel counts."""
    from src.market_scanner import exclusions as current

    head = _load_head()
    head_res = _run_funnel(head)
    cur_res = _run_funnel(current)

    assert set(head_res.reasons) == set(cur_res.reasons)
    diffs = {
        code: (head_res.reasons[code], cur_res.reasons[code])
        for code in sorted(head_res.reasons)
        if head_res.reasons[code] != cur_res.reasons[code]
    }
    assert not diffs, f"reason diffs: {diffs}"
    assert list(head_res.passed) == list(cur_res.passed)
    head_funnel = [dict(e) for e in head_res.funnel]
    cur_funnel = [dict(e) for e in cur_res.funnel]
    assert head_funnel == cur_funnel, (
        f"funnel mismatch\nHEAD: {head_funnel}\nNOW : {cur_funnel}")


def test_current_funnel_matches_committed_golden():
    """The golden file was produced once from HEAD; now must reproduce it.

    Bounds the fixture: if a later change moves the funnel, this fails with
    the exact stage/field, instead of silently drifting.
    """
    from src.market_scanner import exclusions as current

    res = _run_funnel(current)
    golden = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    assert golden["head"] == "8dfddb1d2b5e1d83a8343889c599db0fd79f5c1e"
    assert sorted(res.reasons) == sorted(golden["codes"])
    assert {c: res.reasons[c] for c in golden["codes"]} == {
        c: golden["reasons"][c] for c in golden["codes"]}
    assert [dict(e) for e in res.funnel] == golden["funnel"]
    assert list(res.passed) == golden["passed"]


@pytest.mark.parametrize(
    "code,board,expected",
    [
        ("920002", "main_sh", True),     # bse prefix wins over the board
        ("430047", None, True),         # old-system bse prefix
        ("430300", "", True),           # ... with a blank board
        ("400001", "bse", True),        # plain bse code
        ("600000", "bse", True),      # vendor board bse, prefix not determined
        ("600000", "BSE", False),       # case-sensitive: not the bse marker
        ("300001", "unknown", False),
        ("600zz6", "", False),         # non-numeric code
    ],
)
def test_is_bse_code_is_the_scan_predicate(code, board, expected):
    """One shared predicate: same verdict as stage 2 of the funnel."""
    from src.market_scanner.exclusions import is_bse_code
    from src.market_scanner.sources._common import resolve_board

    assert is_bse_code(code, board) is expected
    # ... and it really is the funnel's own rule, not a copy of it
    assert expected == (str(resolve_board(code, board)) == "bse")
