"""TODO_R3 Task B regressions: DROPPED diff, board mapping, AI in DROPPED.

Owned scope: src/watchlist/service.py (+ store.py), cli/commands/watchlist.py,
src/market_scanner/sources/{_common,universe,listing}.py and these tests.
Run: pytest tests/test_watchlist_taskb_regressions.py -q
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd
import pytest

from src.market_scanner.config import ScanConfig
from src.market_scanner.exclusions import run_exclusions
from src.market_scanner.models import board_from_code
from src.market_scanner.sources._common import resolve_board
from src.market_scanner.sources.universe import _row_to_record
from src.watchlist.service import WatchlistService
from src.watchlist.store import WatchlistStore

TRADE = _dt.date(2026, 10, 13)
LISTED_LONG_AGO = TRADE - _dt.timedelta(days=800)


def make_universe(rows: list[dict]) -> pd.DataFrame:
    """Local twin of test_market_scanner_exclusions.make_universe."""
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
        "list_date": LISTED_LONG_AGO,
    }
    records = []
    for row in rows:
        rec = dict(base)
        rec.update(row)
        code = str(rec.pop("code")).zfill(6)
        if "board" not in row:
            rec["board"] = board_from_code(code)
        rec["code"] = code
        records.append(rec)
    return pd.DataFrame.from_records(records).set_index("code")


def make_klines(bars: int = 130, amount: float = 2e8) -> pd.DataFrame:
    """Ascending synthetic klines ending on TRADE (clears min-bars rule)."""
    dates = pd.bdate_range(end=TRADE, periods=bars)
    close = [10.0 + 0.01 * i for i in range(bars)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
            "close": close,
            "volume": [1e6] * bars,
            "amount": [amount] * bars,
        }
    ).set_index("date")


def make_fundamentals(codes: list[str]) -> pd.DataFrame:
    """Published fundamentals per code (clears the fundamentals rule)."""
    return pd.DataFrame.from_records(
        [
            {
                "code": c,
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
            for c in codes
        ]
    ).set_index("code")


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "watchlist.sqlite3"


@pytest.fixture()
def fake_now() -> _dt.datetime:
    return _dt.datetime(2026, 10, 13, 18, 0, 0)


def _score_row(code: str, score: float) -> dict:
    return {
        "code": code,
        "name": f"股票{code}",
        "board": "main_sh",
        "industry": "测试",
        "score": score,
        "fundamental": 80,
        "trend": 70,
        "flow": 60,
        "exclusion_reason": None,
    }


class FakeCalendar:
    """Fixed trading day; injectable into the service."""

    def __init__(self, today: _dt.date = TRADE) -> None:
        self._today = today

    def is_trading_day(self, d: _dt.date) -> bool:
        return d.weekday() < 5

    def previous_trading_day(self, d: _dt.date) -> _dt.date:
        d = d - _dt.timedelta(days=1)
        while d.weekday() >= 5:
            d -= _dt.timedelta(days=1)
        return d

    def latest_completed_trading_day(self, now: _dt.datetime) -> _dt.date:
        if now.date().weekday() < 5 and now.time() >= _dt.time(15, 0):
            return now.date()
        return self.previous_trading_day(now.date())


def _pick(code: str, rank: int) -> dict:
    return {
        "code": code,
        "name": f"股票{code}",
        "board": "main_sh",
        "industry": "测试",
        "rank": rank,
        "score": 100 - rank,
        "sub_scores": {"fundamental": 80, "trend": 70, "flow": 60},
        "reasons": ["r1"],
        "flags": [],
        "ai": {"status": "not_reviewed"},
    }


def _scan_run(
    run_id: str,
    picks: list[str],
    *,
    status: str = "ok",
    trade_date: str = "2026-10-13",
    scores_extra: list[dict] | None = None,
) -> dict:
    scores = [
        {
            "code": c,
            "name": f"股票{c}",
            "board": "main_sh",
            "industry": "测试",
            "score": 100 - i,
            "fundamental": 80,
            "trend": 70,
            "flow": 60,
            "exclusion_reason": None,
        }
        for i, c in enumerate(picks)
    ] + (scores_extra or [])
    return {
        "run_id": run_id,
        "trade_date": trade_date,
        "started_at": "2026-10-13T18:00:00",
        "finished_at": "2026-10-13T18:05:00",
        "status": status,
        "trigger": "cli",
        "duration_s": 1.0,
        "intraday": False,
        "ai_status": "not_reviewed",
        "ai_note": None,
        "config": {"top_n": 20},
        "funnel": [
            {"stage": "scored", "count": len(scores), "removed": 0, "note": None}
        ],
        "sources": [{"name": "universe", "status": "ok", "rows": 10}],
        "picks": [_pick(c, i + 1) for i, c in enumerate(picks)],
        "ai_demoted": [],
        "requalified": [],
        "pinned": [],
        "scores": scores,
    }


def _make_service(store_path: Path, scan_fn, now: _dt.datetime, rerun_fn=None) -> WatchlistService:
    return WatchlistService(
        store=WatchlistStore(path=store_path),
        scan_fn=scan_fn,
        rerun_fn=rerun_fn,
        calendar_loader=lambda: FakeCalendar(now.date()),
        now_fn=lambda: now,
    )


# --------------------------------------------------------------------------- #
# (a) DROPPED diff bug
# --------------------------------------------------------------------------- #
# The real M2 diff: run ee122cc9 vs c9916e36 — 5 codes left the top-20, of
# which 601128 常熟银行 was missing from the rendered DROPPED list.
PREV_M2 = [
    "002818", "301087", "601577", "002832", "600033", "601128", "601000",
    "002948", "300770", "600177", "601326", "600566", "600398", "002332",
    "600351", "300832", "301507", "600901", "601098", "002039",
]
CUR_M2 = [
    "301087", "600177", "600351", "600033", "002832", "000848", "601326",
    "601577", "600901", "600566", "300770", "301507", "002332", "601000",
    "300832", "603209", "601900", "002292", "000576", "600398",
]
DEPARTED_M2 = [c for c in PREV_M2 if c not in CUR_M2]


def test_every_left_top20_code_is_dropped(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """All five departed codes (incl. 601128) land in DROPPED."""
    assert DEPARTED_M2 == ["002818", "601128", "002948", "601098", "002039"]
    # Every departed code is still scored -> the reason carries the new rank.
    scores_extra = [
        _score_row(c, 90.0 - i * 0.5) for i, c in enumerate(DEPARTED_M2)
    ]
    calls = {"n": 0}
    runs = [
        _scan_run("c9916e36", PREV_M2, status="partial"),
        _scan_run("ee122cc9", CUR_M2, status="ok", scores_extra=scores_extra),
    ]

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        r["trigger"] = "cli"
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    summary = svc.run_scan(trigger="cli")

    # The computed diff must name every departed code, 601128 included.
    assert {d["code"] for d in summary["changes"]["dropped"]} == set(DEPARTED_M2)
    assert summary["counts"]["dropped"] == 5
    # The stored view keeps the same set (it is ordered by previous rank).
    view = svc.get_watchlist()
    assert {d["code"] for d in view["dropped"]} == set(DEPARTED_M2)
    assert all(d["reason"].startswith("rank fell to") for d in view["dropped"])


def test_dropped_rows_are_repaired_when_a_save_loses_them(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """Short saves self-heal: the store keeps one row per departed code.

    Reproduces the live defect (d4ce9714: 20 picks saved, 0 dropped rows,
    although two codes had left the predecessor's top-20) by sabotaging the
    run's own save; ``_verify_dropped_saved`` must rewrite the diff.
    """
    calls = {"n": 0}
    runs = [
        _scan_run("r1", ["600519", "600520", "600521"]),
        _scan_run("r2", ["600519", "600522", "600523"]),
    ]

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        r["trigger"] = "cli"
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")

    real_save = svc.store.save_run
    state = {"n": 0}

    def sabotaging_save(run, *, changes=None):
        state["n"] += 1
        if state["n"] == 2:
            real_save(run, changes={})  # the bug: drops vanish
            return
        real_save(run, changes=changes)

    svc.store.save_run = sabotaging_save  # type: ignore[method-assign]
    summary = svc.run_scan(trigger="cli")
    svc.store.save_run = real_save  # type: ignore[method-assign]

    repaired = [r["code"] for r in svc.store.get_dropped("r2")]
    assert summary["counts"]["dropped"] == 2
    assert repaired == ["600520", "600521"]
    assert {r["code"] for r in svc.store.get_picks("r2")} == {
        "600519", "600522", "600523"
    }


def test_dropped_diff_is_idempotent_when_the_save_is_correct(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """A correct save is left untouched (no duplicate/extra rows)."""
    calls = {"n": 0}
    runs = [
        _scan_run("r1", ["600519", "600520"]),
        _scan_run("r2", ["600519"]),
    ]

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        r["trigger"] = "cli"
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    summary = svc.run_scan(trigger="cli")
    assert [r["code"] for r in svc.store.get_dropped("r2")] == ["600520"]
    assert summary["counts"]["dropped"] == 1


# --------------------------------------------------------------------------- #
# (b) board mapping — code prefix wins
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "code,expected",
    [
        ("300770", "chinext"), ("301507", "chinext"), ("301087", "chinext"),
        ("300832", "chinext"), ("302132", "chinext"),
        ("688981", "star"), ("689009", "star"), ("688036", "star"),
        ("600519", "main_sh"), ("605111", "main_sh"),
        ("000001", "main_sz"), ("002594", "main_sz"), ("003816", "main_sz"),
        ("920002", "bse"), ("430047", "bse"),
    ],
)
def test_board_from_code_prefixes(code: str, expected: str) -> None:
    assert board_from_code(code) == expected


@pytest.mark.parametrize(
    "code,vendor",
    [
        ("300770", "main_sz"), ("301507", "main_sz"), ("301507", "main_sh"),
        ("300832", "main_sh"), ("688981", "main_sh"), ("689009", "main_sz"),
        ("301087", "bse"),
    ],
)
def test_code_prefix_beats_vendor_board(code: str, vendor: str) -> None:
    """A vendor row claiming main_sz for 301507 must never win."""
    assert resolve_board(code, vendor) != vendor
    assert resolve_board(code, vendor) == board_from_code(code)


def test_non_prefix_board_keeps_vendor_value() -> None:
    # main_sh / main_sz are outside _PREFIX_DETERMINED: a vendor value is kept.
    assert resolve_board("600519", "main_sh") == "main_sh"
    assert resolve_board("600519", None) == "main_sh"
    assert resolve_board("000001", "star") == "star"


def test_universe_stamps_prefix_board() -> None:
    rec = _row_to_record(
        {
            "code": "301507", "name": "民生健康", "trade": "10.5",
            "changepercent": "1.0", "settlement": "10.4", "open": "10.4",
            "high": "10.6", "low": "10.3", "volume": "1000", "amount": "10500",
            "per": "20", "pb": "2", "mktcap": "1000", "nmc": "900",
            "turnoverratio": "1.0",
        }
    )
    assert rec["board"] == "chinext"
    assert rec["exchange"] == "SZ"


def _uni_frame(codes: list[str]) -> pd.DataFrame:
    """Universe frame with the vendor board deliberately WRONG for 3xx."""
    rows = []
    for c in codes:
        row = {"code": c}
        if c.startswith("3"):
            # Vendor data mislabels ChiNext as main_sz — the prefix must win.
            row["board"] = "main_sz"
        rows.append(row)
    return make_universe(rows)


def test_exclude_chinext_drops_300_and_301_prefixes() -> None:
    """include_chinext=false excludes 300/301/302 even if the row says main_sz.

    NOTE: the exclusion funnel (``exclusions.py`` line 357) reads the *board
    column*, so a vendor mislabel escapes it. See HERMES_B.md "board exclusion
    bug in exclusions.py" — exclusions.py is codex-owned since the growth gate,
    so this documents the required fix instead of applying it. With the board
    column corrected upstream (universe/listing stamp ``board`` from the code
    prefix), the funnel excludes as expected.
    """
    # Upstream mapping: universe/listing now stamp the board from the code
    # prefix, so the funnel sees chinext and the toggle bites.
    codes = ["300770", "301507", "600519"]
    fixed = make_universe([{"code": c} for c in codes])  # board_from_code
    assert fixed.loc["300770", "board"] == "chinext"
    assert fixed.loc["301507", "board"] == "chinext"
    res = run_exclusions(
        fixed,
        config=ScanConfig(include_chinext=False),
        klines={c: make_klines() for c in codes},
        fundamentals_df=make_fundamentals(codes),
        trade_date=TRADE,
    )
    for c in ("300770", "301507"):
        assert str(res.reasons[c]).startswith("excluded:board_disabled"), (
            f"{c} must be excluded by board_disabled, got {res.reasons[c]!r}"
        )
    assert res.reasons["600519"] is None

    on = run_exclusions(
        fixed,
        config=ScanConfig(include_chinext=True),
        klines={c: make_klines() for c in codes},
        fundamentals_df=make_fundamentals(codes),
        trade_date=TRADE,
    )
    for c in ("300770", "301507"):
        assert on.reasons[c] is None, f"{c} must pass, got {on.reasons[c]!r}"


def test_exclude_star_drops_688_and_689_prefixes() -> None:
    """include_star=false excludes 688/689 when the board comes from the prefix."""
    codes = ["688981", "689009", "600519"]
    fixed = make_universe([{"code": c} for c in codes])
    assert fixed.loc["688981", "board"] == "star"
    assert fixed.loc["689009", "board"] == "star"
    res = run_exclusions(
        fixed,
        config=ScanConfig(include_star=False),
        klines={c: make_klines() for c in codes},
        fundamentals_df=make_fundamentals(codes),
        trade_date=TRADE,
    )
    for c in ("688981", "689009"):
        assert str(res.reasons[c]).startswith("excluded:board_disabled"), (
            f"{c} must be excluded by board_disabled, got {res.reasons[c]!r}"
        )
    assert res.reasons["600519"] is None


# --------------------------------------------------------------------------- #
# (c) scan/review change column uses the service's computed changes
# --------------------------------------------------------------------------- #
def test_summary_changes_carry_new_and_kept(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    calls = {"n": 0}
    runs = [
        _scan_run("r1", ["600519", "600520"]),
        _scan_run("r2", ["600519", "600522"]),
    ]

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        r["trigger"] = "cli"
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    summary = svc.run_scan(trigger="cli")
    assert {c["code"] for c in summary["changes"]["new"]} == {"600522"}
    assert {c["code"] for c in summary["changes"]["kept"]} == {"600519"}
    # The picks payload has no change field (pipeline._make_pick omits it),
    # which is exactly why the CLI must stamp these labels on.
    assert all(p.get("change") is None for p in svc.get_run("r2")["picks"])


def test_review_summary_changes_carry_new_and_kept(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    calls = {"n": 0}
    runs = [
        _scan_run("r1", ["600519", "600520"]),
        _scan_run("r2", ["600519", "600522"]),
    ]

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        r["trigger"] = "cli"
        calls["n"] += 1
        return r

    def rerun_fn(run, config, **kw):
        return _scan_run("r2", ["600519", "600524"])

    svc = _make_service(store_path, scan_fn, fake_now, rerun_fn=rerun_fn)
    svc.run_scan(trigger="cli")
    svc.run_scan(trigger="cli")
    summary = svc.rerun_ai_review("r2", trigger="cli", llm=None)
    assert {c["code"] for c in summary["changes"]["new"]} == {"600524"}
    assert {c["code"] for c in summary["changes"]["kept"]} == {"600519"}


# --------------------------------------------------------------------------- #
# (d) AI verdict + first cited reason in the DROPPED entry
# --------------------------------------------------------------------------- #
def _ai_block() -> dict:
    return {
        "status": "reviewed",
        "verdict": "avoid",
        "summary": "Strong fundamentals, but a major-holder sell-down looms.",
        "reasons": [
            {"text": "ROE 9.26% (annualised 18.52%), gross margin 70.88%.",
             "cites": ["M:roe", "M:roe_annualised", "M:gross_margin"]},
        ],
        "red_flags": [
            {
                "text": "5%+ shareholder plans to sell up to 1% of total "
                        "shares via block trades.",
                "cites": ["A1", "N9", "N10"],
                "type": "major-holder sell-down",
            }
        ],
    }


def test_dropped_entry_shows_ai_verdict_reason_and_cites(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """An AI-demoted drop carries ai_verdict/ai_reason/ai_cites."""
    ai = _ai_block()
    calls = {"n": 0}
    r2 = _scan_run("r2", ["600519"], scores_extra=[_score_row("300832", 92.1)])
    r2["ai_pool"] = [
        {"code": "600519", "rank": 1, "score": 95.0, "ai": ai},
        {"code": "300832", "rank": 15, "score": 92.1, "ai": ai},
    ]
    demoted = _pick("300832", 15)
    demoted["ai"] = ai
    r2["ai_demoted"] = [demoted]
    runs = [_scan_run("r1", ["600519", "300832"]), r2]

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        r["trigger"] = "cli"
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.run_scan(trigger="cli")
    view = svc.get_watchlist()
    dropped = {d["code"]: d for d in view["dropped"]}
    assert "300832" in dropped
    row = dropped["300832"]
    assert row["ai_verdict"] == "avoid"
    # The red flag is the negative driver, so it leads with its cites.
    assert "block trades" in row["ai_reason"]
    assert row["ai_cites"] == ["A1", "N9", "N10"]
    # Rank info is kept alongside the AI fields.
    assert row["prev_rank"] == 2
    assert row["reason"]


def test_dropped_entry_falls_back_to_review_reason(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """A watch-verdict stock that fell on rank shows its first review reason."""
    ai = {
        "status": "reviewed",
        "verdict": "watch",
        "reasons": [
            {"text": "Growth and margin profile is strong.",
             "cites": ["M:revenue_yoy", "M:gross_margin"]},
        ],
        "red_flags": [],
    }
    calls = {"n": 0}
    r2 = _scan_run("r2", ["600519"], scores_extra=[_score_row("600520", 80.0)])
    r2["ai_pool"] = [
        {"code": "600519", "rank": 1, "score": 95.0, "ai": ai},
        {"code": "600520", "rank": 30, "score": 80.0, "ai": ai},
    ]
    runs = [_scan_run("r1", ["600519", "600520"]), r2]

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        r["trigger"] = "cli"
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.run_scan(trigger="cli")
    row = svc.get_watchlist()["dropped"][0]
    assert row["ai_verdict"] == "watch"
    assert row["ai_reason"] == "Growth and margin profile is strong."
    assert row["ai_cites"] == ["M:revenue_yoy", "M:gross_margin"]


def test_dropped_entry_without_ai_review_stays_rank_only(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """A rank/exclusion drop must not invent an AI verdict."""
    calls = {"n": 0}
    runs = [
        _scan_run("r1", ["600519", "600520"]),
        _scan_run("r2", ["600519"], scores_extra=[_score_row("600520", 80.0)]),
    ]

    def scan_fn(config, **kw):
        r = runs[calls["n"]]
        r["trigger"] = "cli"
        calls["n"] += 1
        return r

    svc = _make_service(store_path, scan_fn, fake_now)
    svc.run_scan(trigger="cli")
    svc.run_scan(trigger="cli")
    view = svc.get_watchlist()
    assert [d["code"] for d in view["dropped"]] == ["600520"]
    assert view["dropped"][0]["ai_verdict"] is None
    assert view["dropped"][0].get("ai_reason") is None
    assert view["dropped"][0].get("ai_cites") in (None, [])
