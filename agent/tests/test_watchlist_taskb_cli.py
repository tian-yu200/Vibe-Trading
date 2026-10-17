"""TODO_R3 Task B CLI regression tests.

Covers (c) the scan/review change column must render the service-computed
changes instead of the "kept" fallback, and (d) the DROPPED entry must show
the AI verdict + first cited reason + rank fall.

Run: pytest tests/test_watchlist_taskb_cli.py -q
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import pytest

from cli.commands import watchlist as wl


def _run(run_id: str = "r2") -> dict:
    picks = [
        {"rank": 1, "code": "600519", "name": "贵州茅台", "board": "main_sh",
         "industry": "食品饮料", "score": 95.0,
         "sub_scores": {"fundamental": 90, "trend": 80, "flow": 70},
         "reasons": [], "flags": []},
        # 600522 is NEW in this run: picks carry no 'change' field, so the
        # label must come from the service's diff.
        {"rank": 2, "code": "600522", "name": "新增股", "board": "main_sh",
         "industry": "银行", "score": 93.0,
         "sub_scores": {"fundamental": 88, "trend": 75, "flow": 65},
         "reasons": [], "flags": []},
    ]
    return {
        "run_id": run_id,
        "trade_date": "2026-10-13",
        "status": "ok",
        "ai_status": "reviewed",
        "duration_s": 12.0,
        "intraday": False,
        "funnel": [],
        "sources": [],
        "requests": [],
        "picks": picks,
        "ai_demoted": [],
        "requalified": [],
        "pinned": [],
        "scores": [],
        "review_revision": 1,
        "reviewed_at": "2026-10-13T18:06:00",
    }


def _changes(new_code: str = "600522", kept_code: str = "600519") -> dict:
    return {
        "new": [(new_code, 2, 93.0, {})],
        "kept": [(kept_code, 1, 95.0, {})],
        "dropped": [],
        "requalified": [],
        "manual": [],
    }


# --------------------------------------------------------------------------- #
# (c) change column uses the service's computed changes
# --------------------------------------------------------------------------- #
def test_picks_render_new_and_kept_from_service_changes() -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        wl._print_picks(wl._apply_changes_to_picks(_run(), _changes()))
    text = out.getvalue()
    assert "600522" in text and "new" in text
    assert "600519" in text and "kept" in text
    # The bug this regression guards: every pick printed as kept.
    lines = [ln for ln in text.splitlines() if "600522" in ln]
    assert lines and "new" in lines[0]


def test_no_changes_leaves_labels_alone() -> None:
    """--json / older runs: absent changes must not fabricate labels."""
    run = _run()
    merged = wl._apply_changes_to_picks(run, None)
    assert merged is run  # untouched, no copy


def test_change_word_fallback_still_kept_without_changes() -> None:
    """A pick with no change field and no service diff reads 'kept'."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        wl._print_picks(_run())
    assert "kept" in out.getvalue()


# --------------------------------------------------------------------------- #
# (d) AI verdict + first cited reason in the DROPPED entry
# --------------------------------------------------------------------------- #
def _dropped_view() -> dict:
    return {
        "picks": [],
        "items": [],
        "dropped": [
            {
                "code": "300832",
                "name": "新产业",
                "prev_rank": 15,
                "reason": "AI demoted: avoid (avoid); rank fell to 15 (score 92.1)",
                "ai_verdict": "avoid",
                "ai_reason": "5%+ shareholder plans to sell up to 1% of "
                             "total shares via block trades.",
                "ai_cites": ["A1", "N9", "N10"],
            },
            {
                "code": "000576",
                "name": "甘化科工",
                "prev_rank": 19,
                "reason": "rank fell to 27 (score 91.1)",
                "ai_verdict": None,
                "ai_reason": None,
                "ai_cites": [],
            },
        ],
        "requalified": [],
        "ai_demoted": [],
        "counts": {},
        "latest_ok": {"run_id": "ee122cc9"},
        "latest_run": {"run_id": "ee122cc9", "trade_date": "2026-10-13",
                       "status": "ok"},
        "manually_added": [],
        "manual_only": [],
        "stale": {"is_stale": False, "latest_trading_day": "2026-10-13"},
    }


class FakeService:
    """Only get_watchlist() is exercised by _cmd_list's DROPPED block."""

    def get_watchlist(self) -> dict:
        return _dropped_view()


def test_show_renders_ai_reason_and_rank(capsys: pytest.CaptureFixture) -> None:
    args = argparse.Namespace(json=False)
    assert wl._cmd_list(args, FakeService()) == 0
    text = capsys.readouterr().out
    assert "DROPPED (vs previous run)" in text
    # The AI-demoted drop shows verdict + reason + cites.
    assert "AI: avoid - 5%+ shareholder plans to sell" in text
    assert "[A1, N9, N10]" in text
    # ... and still carries the rank information.
    assert "rank fell to 15" in text
    # A rank/exclusion drop prints no AI block at all.
    assert "prev_rank=19 reason=rank fell to 27 (score 91.1)" in text
    ai_lines = [ln for ln in text.splitlines() if "甘化科工" in ln]
    assert ai_lines and "AI:" not in ai_lines[0]
