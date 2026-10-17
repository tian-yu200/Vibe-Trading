"""Tests for the AI review layer (CONTRACT section 8). Fully offline.

The fake LLM is a plain ``callable(messages) -> str`` injected through
``run_scan(llm=...)``; the fake source world is re-used from the pipeline
tests. The autouse fixture keeps the default-provider path from ever
constructing a real ChatLLM.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

from src.market_scanner.config import ScanConfig
from src.market_scanner.pipeline import SourceStatus, run_scan
from src.market_scanner.review import (
    NEUTRAL_RANK_PENALTY,
    build_data_pack,
    extract_json_object,
    review_one,
    validate_review,
)

# Re-use the offline fake source world from the pipeline tests. Sibling test
# modules are not importable as top-level names, so load the file directly.
_spec = importlib.util.spec_from_file_location(
    "_msp_pipeline_tests",
    Path(__file__).with_name("test_market_scanner_pipeline.py"),
)
_msp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_msp)
NOW = _msp.NOW
make_sources = _msp.make_sources


@pytest.fixture(autouse=True)
def _no_default_llm(monkeypatch):
    """Block the real provider path; tests inject their own llm callable."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no provider in tests")

    monkeypatch.setattr(
        "src.market_scanner.pipeline.build_default_llm_callable", _boom
    )


def run(**kwargs) -> dict:
    kwargs.setdefault("config", ScanConfig())
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("sources", make_sources())
    return run_scan(**kwargs)


def _code_from(messages: list[dict[str, str]]) -> str:
    # Find the user message carrying the pack; the retry path appends an
    # assistant reply + a corrective user message after it.
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        if "Review stock " in content:
            return content.split("Review stock ", 1)[1].split(" ", 1)[0]
    raise AssertionError("no user message carrying a stock code")


def _payload(code: str, verdict: str = "watch", **overrides) -> dict:
    base = {
        "code": code,
        "verdict": verdict,
        "confidence": 0.85,
        "summary": f"{code} looks fine",
        "reasons": [{"text": "ROE solid", "cites": ["M:roe"]}],
        "red_flags": [],
    }
    base.update(overrides)
    return base


def _json_llm(verdicts: dict[str, str] | None = None):
    """Completion callable emitting one valid JSON object per stock."""
    verdicts = verdicts or {}

    def _llm(messages: list[dict[str, str]]) -> str:
        code = _code_from(messages)
        return json.dumps(
            _payload(code, verdicts.get(code, "watch")), ensure_ascii=False
        )

    return _llm


# --------------------------------------------------------------------------- #
# Happy path: valid JSON, fenced JSON
# --------------------------------------------------------------------------- #


def test_valid_json_verdict_is_applied_to_picks():
    result = run(llm=_json_llm())
    assert result["ai_status"] == "reviewed"
    assert "5 reviewed" in result["ai_note"]
    for pick in result["picks"]:
        ai = pick["ai"]
        assert ai["status"] == "reviewed"
        assert ai["verdict"] == "watch"
        assert ai["summary"]
        assert ai["reasons"]


def test_fenced_json_reply_is_accepted():
    def _llm(messages):
        code = _code_from(messages)
        return "```json\n" + json.dumps(_payload(code)) + "\n```"

    result = run(llm=_llm)
    assert result["ai_status"] == "reviewed"
    assert all(p["ai"]["status"] == "reviewed" for p in result["picks"])


def test_run_contains_ai_stats_and_pool_has_real_ai_blocks():
    result = run(llm=_json_llm())
    assert result["ai_stats"]["count"] == len(result["ai_pool"])
    assert result["ai_stats"]["outcomes"]["ok"] == len(result["ai_pool"])
    assert result["ai_stats"]["p50_latency_s"] >= 0.0
    assert result["ai_stats"]["p90_latency_s"] >= result["ai_stats"]["p50_latency_s"]
    assert result["ai_stats"]["max_latency_s"] >= result["ai_stats"]["p90_latency_s"]
    assert all(row["ai"]["status"] == "reviewed" for row in result["ai_pool"])


# --------------------------------------------------------------------------- #
# Retry / failure isolation
# --------------------------------------------------------------------------- #


def test_malformed_first_reply_retries_once_then_succeeds():
    attempts: dict[str, int] = {}

    def _llm(messages):
        code = _code_from(messages)
        attempts[code] = attempts.get(code, 0) + 1
        if attempts[code] == 1:
            return "not json at all"
        return json.dumps(_payload(code))

    result = run(llm=_llm)
    assert result["ai_status"] == "reviewed"
    assert attempts and all(v == 2 for v in attempts.values())
    assert all(p["ai"]["status"] == "reviewed" for p in result["picks"])


def test_malformed_twice_marks_failed_but_stock_stays_listed():
    result = run(llm=lambda messages: "garbage")
    assert result["ai_status"] == "not_reviewed"
    assert result["picks"], "failed reviews must not drop picks"
    for pick in result["picks"]:
        assert pick["ai"]["status"] == "failed"
        assert pick["ai"]["note"].startswith("AI review failed")


def test_llm_exception_marks_failed_but_stock_stays_listed():
    def _boom(_messages):
        raise RuntimeError("gateway down")

    result = run(llm=_boom)
    assert result["picks"], "a raising llm must not drop picks"
    for pick in result["picks"]:
        assert pick["ai"]["status"] == "failed"
        assert "RuntimeError: gateway down" in pick["ai"]["note"]


def test_review_one_timeout_is_reported_as_failed():
    pack = build_data_pack("600001", metrics={"roe": 25.0})

    def _slow(_messages):
        time.sleep(0.3)
        return "{}"

    outcome = review_one(_slow, pack, timeout=0.05)
    assert outcome.ai["status"] == "failed"
    assert "timed out" in outcome.ai["note"]
    assert outcome.ai["attempts"] == 1
    assert outcome.ai["outcome"] == "timeout"
    assert outcome.ai["latency_s"] >= 0.0


def test_review_one_uses_configured_call_timeout(monkeypatch):
    seen: list[float] = []
    pack = build_data_pack("600001", metrics={"roe": 25.0})

    def _call(_llm, _messages, *, timeout):
        seen.append(timeout)
        return json.dumps(_payload("600001"))

    monkeypatch.setattr("src.market_scanner.review._call_llm", _call)
    outcome = review_one(
        lambda _messages: "",
        pack,
        config=ScanConfig(ai_call_timeout_s=12.5),
    )
    assert outcome.ai["outcome"] == "ok"
    assert outcome.ai["attempts"] == 1
    assert seen == [12.5]


# --------------------------------------------------------------------------- #
# Citation hygiene
# --------------------------------------------------------------------------- #


def test_unknown_cites_are_dropped_from_reasons():
    def _llm(messages):
        code = _code_from(messages)
        payload = _payload(
            code,
            reasons=[
                {"text": "good reason", "cites": ["M:roe"]},
                {"text": "hallucinated news", "cites": ["N99"]},
            ],
        )
        return json.dumps(payload, ensure_ascii=False)

    result = run(llm=_llm)
    for pick in result["picks"]:
        texts = [r["text"] for r in pick["ai"]["reasons"]]
        assert texts == ["good reason"]


# --------------------------------------------------------------------------- #
# Verdict effects: avoid demotion + backfill
# --------------------------------------------------------------------------- #


def test_avoid_verdict_demotes_and_backfills():
    baseline = run()
    order = [p["code"] for p in baseline["picks"]]
    assert order[0] == "600001"

    result = run(config=ScanConfig(top_n=1), llm=_json_llm({"600001": "avoid"}))
    assert [p["code"] for p in result["picks"]] == [order[1]]
    demoted = result["ai_demoted"]
    assert [d["code"] for d in demoted] == ["600001"]
    assert demoted[0]["ai"]["verdict"] == "avoid"
    funnel_ai = next(e for e in result["funnel"] if e["stage"] == "ai_demoted")
    assert funnel_ai["count"] == 1
    assert funnel_ai["note"] == "1 demoted by AI review"


def _red_flags_source(flags: dict[str, list]):
    def _fetch(_codes, *, today=None, **_kwargs):
        return (
            flags,
            SourceStatus(name="red_flags", status="ok", rows=len(flags)),
        )

    return _fetch


def test_uncited_material_red_flag_forces_avoid():
    bundle = make_sources(
        fetch_red_flags=_red_flags_source(
            {"600001": [{"type": "regulatory", "text": "监管立案调查"}]}
        )
    )
    result = run(sources=bundle, llm=_json_llm())
    assert [d["code"] for d in result["ai_demoted"]] == ["600001"]
    ai = result["ai_demoted"][0]["ai"]
    assert ai["verdict"] == "avoid"
    assert "red flag R1 not addressed by model" in ai["note"]
    assert "600001" not in [p["code"] for p in result["picks"]]


def test_cited_red_flag_counter_argument_keeps_watch_verdict():
    bundle = make_sources(
        fetch_red_flags=_red_flags_source(
            {"600001": [{"type": "regulatory", "text": "监管立案调查"}]}
        )
    )

    def _llm(messages):
        code = _code_from(messages)
        payload = _payload(
            code,
            reasons=[{"text": "公司公告案件已结案，无实质影响", "cites": ["R1"]}],
        )
        return json.dumps(payload, ensure_ascii=False)

    result = run(sources=bundle, llm=_llm)
    assert result["ai_demoted"] == []
    top = next(p for p in result["picks"] if p["code"] == "600001")
    assert top["ai"]["verdict"] == "watch"


# --------------------------------------------------------------------------- #
# No provider / no news
# --------------------------------------------------------------------------- #


def test_no_llm_provider_leaves_everything_not_reviewed():
    result = run()
    assert result["ai_status"] == "not_reviewed"
    assert result["ai_note"].startswith("no LLM provider configured")
    assert all(p["ai"]["status"] == "not_reviewed" for p in result["picks"])


def test_news_unavailable_marks_no_news_checked():
    """News gateway down -> news_checked False + 'no news checked' note.

    Every source is an injected fake (news raises; announcements/red flags
    return empty; the rest come from the offline pipeline fixture), so the
    test is deterministic and offline. ``min_coverage=0.8`` matches the
    pipeline fixture default so the fixture's 5/6 trend coverage does not
    trip the coverage guard — this test isolates news-unavailable semantics.
    """

    def _news_fail(_code, days=30, *, today=None, **_kwargs):
        raise RuntimeError("news gateway down")

    def _ann_empty(_code, days=30, *, today=None, **_kwargs):
        return [], SourceStatus(name="announcements", status="ok", rows=0)

    def _red_empty(_codes, *, today=None, **_kwargs):
        return {}, SourceStatus(name="red_flags", status="ok", rows=0)

    bundle = make_sources(
        fetch_news=_news_fail,
        fetch_announcements=_ann_empty,
        fetch_red_flags=_red_empty,
    )
    result = run(
        sources=bundle, llm=_json_llm(), config=ScanConfig(min_coverage=0.8)
    )
    for pick in result["picks"]:
        ai = pick["ai"]
        assert ai["status"] == "reviewed"
        assert ai["news_checked"] is False
        assert "no news checked" in (ai["note"] or "")
    news_status = next(s for s in result["sources"] if s["name"] == "news")
    assert news_status["status"] == "failed"
    assert result["status"] == "partial"


# --------------------------------------------------------------------------- #
# Neutral ordering penalty
# --------------------------------------------------------------------------- #


def test_neutral_verdict_penalises_ordering_only(monkeypatch):
    monkeypatch.setattr(
        "src.market_scanner.pipeline.NEUTRAL_RANK_PENALTY", 1000.0
    )
    result = run(llm=_json_llm({"600001": "neutral"}))
    codes = [p["code"] for p in result["picks"]]
    assert "600001" in codes
    assert codes[-1] == "600001"  # pushed to the bottom of the order
    top = next(p for p in result["picks"] if p["code"] == "600001")
    assert top["ai"]["verdict"] == "neutral"
    baseline = next(p for p in run()["picks"] if p["code"] == "600001")
    assert top["score"] == baseline["score"]  # score untouched
    assert top["rule_rank"] == 1
    assert NEUTRAL_RANK_PENALTY == 5.0


# --------------------------------------------------------------------------- #
# Small unit tests for the parse/validate layer
# --------------------------------------------------------------------------- #


def test_extract_json_object_handles_noise_and_fences():
    payload = {"code": "600001", "verdict": "watch"}
    assert extract_json_object(json.dumps(payload)) == payload
    assert (
        extract_json_object("```json\n" + json.dumps(payload) + "\n```")
        == payload
    )
    assert extract_json_object("Sure! " + json.dumps(payload)) == payload
    with pytest.raises(ValueError):
        extract_json_object("no json here")
    with pytest.raises(ValueError):
        extract_json_object("[1, 2]")


def test_validate_review_rejects_bad_verdict_and_confidence():
    pack = build_data_pack("600001", metrics={"roe": 25.0})
    with pytest.raises(ValueError, match="verdict"):
        validate_review(pack, {"verdict": "buy", "confidence": 0.5})
    with pytest.raises(ValueError, match="confidence"):
        validate_review(pack, {"verdict": "watch", "confidence": 1.5})
    with pytest.raises(ValueError, match="confidence"):
        validate_review(pack, {"verdict": "watch", "confidence": "high"})
    block, _notes = validate_review(
        pack,
        {
            "verdict": "watch",
            "confidence": 0.88888,
            "summary": "s",
            "reasons": [],
            "red_flags": [],
        },
    )
    assert block["confidence"] == 0.8889
