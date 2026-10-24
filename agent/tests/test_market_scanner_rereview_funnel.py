"""An AI re-review must not rewrite the funnel's 'scored' count (RELFIX T2).

Bug seen live on 2026-10-23: after a web re-review, run 6b5005b7's funnel
showed ``scored`` = 30 (the review-pool size) instead of 2815, because
``_rebuild_ai_run`` rebuilt the funnel tail from the pool only.

Fixtures are shrunk from a byte copy of the live DB (runs 094fc2df, healthy,
and 6b5005b7, already damaged): funnel / coverage / config / ai_pool / picks
only, with no paths, URLs or model names. No network and no LLM: every call
passes an empty ``SourceBundle`` (news / announcements / red flags skipped)
and an LLM stub that either raises or returns a fixed verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.market_scanner.config import ScanConfig
from src.market_scanner.models import (
    STAGE_AI_DEMOTED,
    STAGE_FINALISTS,
    STAGE_SCORED,
    STAGE_WATCHLIST,
)
from src.market_scanner.pipeline import SourceBundle, rerun_ai_review

FIXTURES = Path(__file__).with_name("fixtures")
HEALTHY = "rereview_run_094fc2df.json"  # scored 2698, intact funnel
DAMAGED = "rereview_run_6b5005b7.json"  # funnel scored 30, coverage 2815


@pytest.fixture(autouse=True)
def _no_provider(monkeypatch):
    """Never let the real LLM provider path run in these tests."""

    def _boom(*_a, **_k):
        raise RuntimeError("no provider in tests")

    monkeypatch.setattr(
        "src.market_scanner.pipeline.build_default_llm_callable", _boom)


def load_run(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def rereview(run: dict, llm) -> dict:
    return rerun_ai_review(
        run,
        ScanConfig.from_dict(run["config"]),
        llm=llm,
        sources=SourceBundle(),  # every review input skipped: no network
    )


def boom_llm(*_a, **_k):
    raise AssertionError("the LLM must not be called by this re-review")


def _demote_llm(demote_code: str):
    """'avoid' for one code, 'watch' for every other reviewed code."""

    def _llm(messages):
        content = next(
            m["content"]
            for m in reversed(messages)
            if m.get("role") == "user" and "Review stock " in m.get("content", "")
        )
        code = content.split("Review stock ", 1)[1].split(" ", 1)[0]
        return json.dumps({
            "code": code,
            "verdict": "avoid" if code == demote_code else "watch",
            "confidence": 0.85,
            "summary": f"{code} stub",
            "reasons": [{"text": "stub", "cites": ["M:roe"]}],
            "red_flags": [],
        })

    return _llm


def pairs(funnel):
    return [(e["stage"], e["count"], e["removed"]) for e in funnel]


def stage(funnel, name):
    found = [e for e in funnel if e.get("stage") == name]
    assert len(found) == 1, f"{name} appears {len(found)} times"
    return found[0]


def assert_cut_invariant(funnel):
    """removed = previous count - count on every cut stage."""
    prev = None
    for entry in funnel:
        if prev is not None and entry["stage"] != STAGE_AI_DEMOTED:
            assert entry["removed"] == max(0, prev - entry["count"]), entry
        prev = entry["count"]


def test_no_call_rereview_keeps_the_stored_funnel():
    run = load_run(HEALTHY)
    stored = pairs(run["funnel"])
    assert stage(run["funnel"], STAGE_SCORED)["count"] == 2698

    out = rereview(run, boom_llm)

    assert pairs(out["funnel"]) == stored
    assert [p["code"] for p in out["picks"]] == [p["code"] for p in run["picks"]]
    assert out["review_revision"] == 1
    assert pairs(run["funnel"]) == stored  # input not mutated


def test_demotion_keeps_real_scored_and_recounts_the_tail():
    run = load_run(HEALTHY)
    for row in run["ai_pool"]:
        row["ai"] = {"status": "not_reviewed", "verdict": None, "reason": None}
    for pick in run["picks"]:
        pick["ai"] = {"status": "not_reviewed", "verdict": None}
    demote = run["picks"][0]["code"]

    out = rereview(run, _demote_llm(demote))

    assert [d["code"] for d in out["ai_demoted"]] == [demote]
    assert demote not in {p["code"] for p in out["picks"]}
    scored = stage(out["funnel"], STAGE_SCORED)
    finalists = stage(out["funnel"], STAGE_FINALISTS)
    assert (scored["count"], scored["removed"]) == (2698, 0)
    assert finalists["removed"] == 2698 - finalists["count"]
    assert stage(out["funnel"], STAGE_AI_DEMOTED)["count"] == 1
    assert stage(out["funnel"], STAGE_WATCHLIST)["count"] == len(out["picks"])
    assert_cut_invariant(out["funnel"])


def test_rereview_heals_a_funnel_damaged_by_the_old_code():
    run = load_run(DAMAGED)
    assert stage(run["funnel"], STAGE_SCORED)["count"] == 30
    assert run["coverage"]["scored"] == 2815

    out = rereview(run, boom_llm)

    scored = stage(out["funnel"], STAGE_SCORED)
    finalists = stage(out["funnel"], STAGE_FINALISTS)
    assert (scored["count"], scored["removed"]) == (2815, 0)
    assert (finalists["count"], finalists["removed"]) == (20, 2795)
    # Every stage before 'scored' is untouched.
    head = [p for p in pairs(run["funnel"])]
    cut = [p[0] for p in head].index(STAGE_SCORED)
    assert pairs(out["funnel"])[:cut] == head[:cut]
    assert_cut_invariant(out["funnel"])


def test_scored_falls_back_to_score_rows_then_funnel_then_pool():
    # No coverage: count rows with a score and no exclusion reason.
    run = load_run(DAMAGED)
    run.pop("coverage")
    run["scores"] = (
        [{"code": f"{i:06d}", "score": 50.0, "exclusion_reason": None}
         for i in range(2815)]
        + [{"code": "900001", "score": None, "exclusion_reason": "st"},
           {"code": "900002", "score": 40.0, "exclusion_reason": "illiquid"}]
    )
    out = rereview(run, boom_llm)
    assert stage(out["funnel"], STAGE_SCORED)["count"] == 2815

    # No coverage, no scores: the stored funnel's count (healthy run).
    run = load_run(HEALTHY)
    run.pop("coverage")
    out = rereview(run, boom_llm)
    assert stage(out["funnel"], STAGE_SCORED)["count"] == 2698

    # Coverage below the pool size is not trusted either.
    run = load_run(HEALTHY)
    run["coverage"]["scored"] = 3
    out = rereview(run, boom_llm)
    assert stage(out["funnel"], STAGE_SCORED)["count"] == 2698


def test_old_run_without_any_count_falls_back_to_the_pool():
    run = load_run(HEALTHY)
    run.pop("coverage")
    run["funnel"] = [e for e in run["funnel"] if e["stage"] != STAGE_SCORED]

    out = rereview(run, boom_llm)

    assert stage(out["funnel"], STAGE_SCORED)["count"] == len(run["ai_pool"])
    assert_cut_invariant(out["funnel"])


def test_stored_tail_of_a_different_shape_is_cut_at_scored():
    # An older run stored no ai_demoted stage: cutting by the new tail's
    # length used to eat a real exclusion stage and keep a stale 'scored'.
    run = load_run(HEALTHY)
    run["funnel"] = [e for e in run["funnel"] if e["stage"] != STAGE_AI_DEMOTED]
    head = [e["stage"] for e in run["funnel"]]
    head = head[: head.index(STAGE_SCORED)]

    out = rereview(run, boom_llm)

    stages = [e["stage"] for e in out["funnel"]]
    assert stages == head + [
        STAGE_SCORED, STAGE_FINALISTS, STAGE_AI_DEMOTED, STAGE_WATCHLIST]
    assert stage(out["funnel"], STAGE_SCORED)["count"] == 2698
