"""Offline tests for the AI-review evidence resolution (audit gap fix).

Every cite id in an ai block's reasons/red_flags resolves to a human-readable
evidence row from that stock's data pack.
"""

from __future__ import annotations

import pytest

from src.market_scanner.review import (
    build_data_pack,
    resolve_evidence,
)


def _pack() -> dict:
    return build_data_pack(
        "600519",
        name="贵州茅台",
        metrics={"roe": 30.5, "eps": 52.03, "debt_ratio": None},  # None dropped
        news=[
            {"date": "2026-10-10", "title": "茅台三季报超预期", "source": "新浪财经",
             "url": "https://example.com/n1"},
            {"date": "2026-10-11", "title": "白酒板块回调", "source": "东财",
             "url": "https://example.com/n2"},
        ],
        announcements=[
            {"date": "2026-10-09", "title": "股东减持计划公告", "source": "上交所",
             "url": "https://example.com/a1"},
        ],
        red_flags=[
            {"type": "holder_selldown", "date": "2026-10-09",
             "text": "大股东拟减持1%", "source": "上交所", "url": None},
        ],
    )


def test_resolve_evidence_news_metric() -> None:
    pack = _pack()
    evidence = resolve_evidence(pack, {"N1", "M:roe"})
    by_id = {e["id"]: e for e in evidence}
    assert set(by_id) == {"N1", "M:roe"}
    n1 = by_id["N1"]
    assert n1["kind"] == "news"
    assert n1["date"] == "2026-10-10"
    assert n1["title_or_text"] == "茅台三季报超预期"
    assert n1["source"] == "新浪财经"
    assert n1["url"] == "https://example.com/n1"
    m = by_id["M:roe"]
    assert m["kind"] == "metric"
    assert m["name"] == "roe"
    assert m["value"] == 30.5
    assert "url" not in m and "title_or_text" not in m


def test_resolve_evidence_red_flag_and_announcement() -> None:
    pack = _pack()
    evidence = resolve_evidence(pack, {"A1", "R1"})
    by_id = {e["id"]: e for e in evidence}
    assert by_id["A1"]["kind"] == "announcement"
    assert by_id["A1"]["title_or_text"] == "股东减持计划公告"
    assert by_id["R1"]["kind"] == "red_flag"
    assert by_id["R1"]["title_or_text"] == "大股东拟减持1%"


def test_resolve_evidence_unknown_ids_skipped() -> None:
    pack = _pack()
    evidence = resolve_evidence(pack, {"N1", "N99", "M:nope", "X1"})
    ids = {e["id"] for e in evidence}
    assert ids == {"N1"}  # unknown ids silently dropped


def test_resolve_evidence_sorted_by_id() -> None:
    pack = _pack()
    evidence = resolve_evidence(pack, {"R1", "A1", "N2", "N1", "M:roe", "M:eps"})
    ids = [e["id"] for e in evidence]
    assert ids == sorted(ids)
    assert ids == ["A1", "M:eps", "M:roe", "N1", "N2", "R1"]


def test_resolve_evidence_empty_pack() -> None:
    pack = build_data_pack("600519")
    assert resolve_evidence(pack, {"N1", "M:roe"}) == []


def test_finalise_ai_block_includes_evidence() -> None:
    """_finalise_ai_block attaches evidence for the skeleton's cites."""
    from src.market_scanner.review import _finalise_ai_block

    pack = _pack()
    skeleton = {
        "verdict": "watch",
        "summary": "基本面强",
        "reasons": [{"text": "ROE 30.5% 领先", "cites": ["M:roe", "N1"]}],
        "red_flags": [{"type": "holder_selldown", "text": "减持1%",
                       "cites": ["R1"]}],
    }
    ai, forced, note = _finalise_ai_block(pack, skeleton)
    assert forced is False
    ev_ids = {e["id"] for e in ai["evidence"]}
    assert ev_ids == {"M:roe", "N1", "R1"}  # cited ids resolved, nothing else


def test_finalise_ai_block_forced_avoid_cites_evidence() -> None:
    """A forced avoid (uncited material red flag) shows in note; evidence only
    covers ids the model actually cited."""
    from src.market_scanner.review import _finalise_ai_block

    pack = _pack()
    skeleton = {
        "verdict": "watch",
        "summary": "强",
        "reasons": [{"text": "ROE高", "cites": ["M:roe"]}],
        "red_flags": [],  # R1 not addressed -> forced avoid
    }
    ai, forced, note = _finalise_ai_block(pack, skeleton)
    assert forced is True
    assert ai["verdict"] == "avoid"
    assert "R1 not addressed" in (note or "")
    assert {e["id"] for e in ai["evidence"]} == {"M:roe"}
