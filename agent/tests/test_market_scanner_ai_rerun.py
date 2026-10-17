"""Focused regression tests for selective AI re-review and pool persistence."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

from src.market_scanner.config import ScanConfig
from src.market_scanner.models import ai_pending
from src.market_scanner.pipeline import rerun_ai_review, run_scan

_spec = importlib.util.spec_from_file_location(
    "_msp_pipeline_tests",
    Path(__file__).with_name("test_market_scanner_pipeline.py"),
)
_msp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_msp)
NOW = _msp.NOW
make_sources = _msp.make_sources


def _llm(counter: list[str]):
    def complete(messages):
        content = next(
            message["content"]
            for message in reversed(messages)
            if message.get("role") == "user"
            and "Review stock " in message.get("content", "")
        )
        code = content.split("Review stock ", 1)[1].split(" ", 1)[0]
        counter.append(code)
        return json.dumps(
            {
                "code": code,
                "verdict": "watch",
                "confidence": 0.9,
                "summary": "reviewed",
                "reasons": [{"text": "ROE is solid", "cites": ["M:roe"]}],
                "red_flags": [],
            }
        )

    return complete


def _scan(counter: list[str], *, top_n: int = 2) -> dict:
    return run_scan(
        config=ScanConfig(top_n=top_n, min_coverage=0.8),
        now=NOW,
        sources=make_sources(),
        llm=_llm(counter),
    )


def test_rerun_skips_every_already_reviewed_pool_row():
    initial_calls: list[str] = []
    run = _scan(initial_calls)
    rerun_calls: list[str] = []

    updated = rerun_ai_review(
        run,
        ScanConfig(top_n=2, min_coverage=0.8),
        llm=_llm(rerun_calls),
        sources=make_sources(),
        now=NOW,
    )

    assert len(initial_calls) == len(run["ai_pool"])
    assert rerun_calls == []
    assert "skipped already reviewed: 5" in updated["ai_note"]
    assert all(row["ai"]["status"] == "reviewed" for row in updated["ai_pool"])
    assert updated["ai_stats"]["count"] == 5


def test_rerun_repairs_stale_pool_and_calls_only_failed_or_pending_rows():
    initial_calls: list[str] = []
    run = _scan(initial_calls)
    stale = copy.deepcopy(run)

    failed_pick = stale["picks"][0]
    failed_pick["ai"]["status"] = "failed"
    failed_pick["ai"]["outcome"] = "error"
    failed_pick["ai"]["failure_kind"] = "error"
    for row in stale["ai_pool"]:
        row["ai"] = ai_pending()

    rerun_calls: list[str] = []
    updated = rerun_ai_review(
        stale,
        ScanConfig(top_n=2, min_coverage=0.8),
        llm=_llm(rerun_calls),
        sources=make_sources(),
        now=NOW,
    )

    pool_codes = [row["code"] for row in stale["ai_pool"]]
    reviewed_pick_code = stale["picks"][1]["code"]
    expected_codes = [
        code for code in pool_codes if code != reviewed_pick_code
    ]
    assert sorted(rerun_calls) == sorted(expected_codes)
    assert "skipped already reviewed: 1" in updated["ai_note"]
    assert all(row["ai"]["status"] == "reviewed" for row in updated["ai_pool"])
    assert updated["ai_stats"]["count"] == len(updated["ai_pool"])
