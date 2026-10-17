"""Offline tests for the SEVERE red-flag rule and the flow wording (TODO_R2 E).

Mock LLM callables only: no network, no real provider (the autouse fixture
keeps the default-provider path from ever constructing a real ChatLLM).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.market_scanner.config import ScanConfig
from src.market_scanner.pipeline import SourceStatus, run_scan
from src.market_scanner.review import (
    FLOW_LABEL,
    SEVERE_RED_FLAG_TYPES,
    _finalise_ai_block,
    _has_explicit_resolution,
    build_data_pack,
    build_messages,
    review_one,
)

# Re-use the offline fake source world from the pipeline tests (same pattern
# as test_market_scanner_review.py).
_spec = importlib.util.spec_from_file_location(
    "_msp_pipeline_tests_severe",
    Path(__file__).with_name("test_market_scanner_pipeline.py"),
)
_msp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_msp)
NOW = _msp.NOW
make_sources = _msp.make_sources

FLOW_WORDING = (
    "THS (同花顺) net fund flow - not Eastmoney main-force, "
    "not northbound/institutional"
)


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
        "confidence": 0.9,
        "summary": "strong fundamentals and trend",
        "reasons": [{"text": "ROE solid", "cites": ["M:roe"]}],
        "red_flags": [],
    }
    base.update(overrides)
    return base


def _llm_returning(payload: dict):
    """Mock LLM: always the same validated JSON payload."""

    def _llm(_messages):
        return json.dumps(payload, ensure_ascii=False)

    return _llm


def _llm_for(payload_fn):
    """Mock LLM building the payload from the stock code in the messages."""

    def _llm(messages):
        return json.dumps(payload_fn(_code_from(messages)), ensure_ascii=False)

    return _llm


def _red_flags_source(flags: dict[str, list]):
    def _fetch(_codes, *, today=None, **_kwargs):
        return (
            flags,
            SourceStatus(name="red_flags", status="ok", rows=len(flags)),
        )

    return _fetch


def _severe_pack(*, red_flags: list[dict], announcements: list[dict]) -> dict:
    return build_data_pack(
        "600001",
        name="贵州测试",
        board="main_sh",
        industry="食品饮料",
        score=92.4,
        sub_scores={"fundamental": 94.0, "trend": 91.0, "flow": 90.0},
        metrics={"roe": 25.0, "flow_3d": 1.2e8},
        rule_reasons=["ROE 25.0% (top 5%)"],
        news=[{"date": "2026-03-18", "title": "公司经营正常", "source": "sina"}],
        announcements=announcements,
        red_flags=red_flags,
    )


# --------------------------------------------------------------------------- #
# Flow wording in prompt + pack (TODO_R2 E)
# --------------------------------------------------------------------------- #


def test_flow_wording_in_prompt_and_pack():
    pack = build_data_pack(
        "600001",
        name="贵州测试",
        sub_scores={"fundamental": 94.0, "trend": 91.0, "flow": 90.0},
        metrics={"flow_3d": 1.2e8},
        rule_reasons=["3-day THS net inflow +1.2亿"],
    )
    assert pack["flow_label"] == FLOW_LABEL == FLOW_WORDING

    messages = build_messages(pack)
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert FLOW_WORDING in system  # the prompt states it explicitly
    assert FLOW_WORDING in user  # the pack JSON carries the label too
    assert "main-force" in system and "northbound/institutional" in system


def test_prompt_states_the_severe_rule():
    system = build_messages(build_data_pack("600001"))[0]["content"]
    assert "SEVERE red flags require verdict avoid" in system
    assert "unresolved severe red flag is forced to avoid" in system
    for ftype in sorted(SEVERE_RED_FLAG_TYPES):
        assert ftype in system
    assert {"csrc_investigation", "audit_opinion", "penalty_fraud",
            "delisting_risk"} == set(SEVERE_RED_FLAG_TYPES)


# --------------------------------------------------------------------------- #
# Code rule: watch + unresolved severe -> forced avoid
# --------------------------------------------------------------------------- #


def test_scan_watch_with_unresolved_severe_flag_forces_avoid():
    bundle = make_sources(
        fetch_red_flags=_red_flags_source(
            {
                "600001": [
                    {
                        "type": "csrc_investigation",
                        "severity": "severe",
                        "text": "关于收到中国证监会立案告知书的公告",
                        "date": "2026-03-10",
                        "value": None,
                        "source": "announcement AN1",
                    }
                ]
            }
        )
    )
    result = run(sources=bundle, llm=_llm_for(lambda code: _payload(code)))
    assert [d["code"] for d in result["ai_demoted"]] == ["600001"]
    ai = result["ai_demoted"][0]["ai"]
    assert ai["verdict"] == "avoid"
    assert "severe red flag R1 (csrc_investigation) not resolved" in ai["note"]
    assert "600001" not in [p["code"] for p in result["picks"]]
    funnel_ai = next(e for e in result["funnel"] if e["stage"] == "ai_demoted")
    assert funnel_ai["count"] == 1


def test_severe_flag_cited_without_resolution_forces_avoid():
    """Citing the flag is not enough: the entry must state a resolution."""
    pack = _severe_pack(
        red_flags=[
            {
                "type": "audit_opinion",
                "severity": "severe",
                "text": "2025年度审计报告（保留意见）",
                "date": "2026-03-17",
            }
        ],
        announcements=[
            {"date": "2026-03-17", "title": "2025年度审计报告（保留意见）"}
        ],
    )
    payload = _payload(
        "600001",
        reasons=[{"text": "审计意见为保留意见，需继续观察", "cites": ["R1"]}],
    )
    outcome = review_one(_llm_returning(payload), pack)
    assert outcome.forced is True
    assert outcome.ai["status"] == "reviewed"
    assert outcome.ai["verdict"] == "avoid"
    assert "severe red flag R1 (audit_opinion) not resolved" in (
        outcome.ai["note"] or ""
    )


def test_severe_flag_cited_with_resolution_keeps_model_verdict():
    pack = _severe_pack(
        red_flags=[
            {
                "type": "csrc_investigation",
                "severity": "severe",
                "text": "关于收到中国证监会立案告知书的公告",
                "date": "2026-03-10",
            }
        ],
        announcements=[
            {"date": "2026-03-10", "title": "关于收到中国证监会立案告知书的公告"},
            {"date": "2026-03-15", "title": "关于立案事项结案的公告"},
        ],
    )
    payload = _payload(
        "600001",
        reasons=[
            {"text": "立案事项已于2026-03-15结案，调查终止，无后续影响",
             "cites": ["R1"]},
        ],
    )
    outcome = review_one(_llm_returning(payload), pack)
    assert outcome.forced is False
    assert outcome.ai["verdict"] == "watch"
    assert outcome.ai["status"] == "reviewed"
    assert "not resolved" not in (outcome.ai["note"] or "")


def test_severity_field_marks_any_type_severe():
    """severity='severe' wins even for a normally material type."""
    pack = build_data_pack(
        "600001",
        metrics={"roe": 25.0},
        news=[{"date": "2026-03-18", "title": "公司经营正常", "source": "sina"}],
        red_flags=[
            {"type": "regulatory", "severity": "severe", "text": "监管立案"}
        ],
    )
    skeleton = {
        "verdict": "watch",
        "summary": "fine",
        "reasons": [{"text": "ROE solid", "cites": ["M:roe"]}],
        "red_flags": [],
    }
    ai, forced, note = _finalise_ai_block(pack, skeleton)
    assert forced is True
    assert ai["verdict"] == "avoid"
    assert "severe red flag R1 (regulatory) not resolved" in (note or "")


def test_material_rule_unchanged_for_non_severe_flags():
    pack = build_data_pack(
        "600001",
        metrics={"roe": 25.0},
        news=[{"date": "2026-03-18", "title": "公司经营正常", "source": "sina"}],
        red_flags=[{"type": "regulatory", "text": "收到监管函"}],
    )
    skeleton = {
        "verdict": "watch",
        "summary": "fine",
        "reasons": [{"text": "ROE solid", "cites": ["M:roe"]}],
        "red_flags": [],
    }
    ai, forced, note = _finalise_ai_block(pack, skeleton)
    assert forced is True
    assert ai["verdict"] == "avoid"
    assert "red flag R1 not addressed by model" in (note or "")
    assert "severe" not in (note or "")


# --------------------------------------------------------------------------- #
# Resolution phrase detection
# --------------------------------------------------------------------------- #


def test_resolution_phrases_and_negation():
    assert _has_explicit_resolution("案件已结案，影响消除")
    assert _has_explicit_resolution("监管已撤销立案，风险解除")
    assert _has_explicit_resolution("decision was withdrawn by the regulator")
    assert _has_explicit_resolution("case closed and cleared")
    assert _has_explicit_resolution("后续审计报告为无保留意见")
    assert not _has_explicit_resolution("尚未结案，调查仍在进行")
    assert not _has_explicit_resolution("相关风险尚未解除")
    assert not _has_explicit_resolution("整改未完成，处罚决定仍然有效")
    assert not _has_explicit_resolution("ROE 25% and rising")
    assert not _has_explicit_resolution("")
