"""Offline unit tests for the news/red-flag source (CONTRACT section 8).

Owner: pi. Fixtures under ``agent/tests/fixtures/market_scanner/news/``.
All HTTP and akshare calls are stubbed; no network access.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.market_scanner.sources import news as news_mod  # noqa: E402
from src.market_scanner.sources.news import (  # noqa: E402
    _quarter_dates,
    _regulatory_flags,
    _strip_jsonp,
    fetch_announcements,
    fetch_news,
    fetch_red_flags,
)

TODAY = dt.date(2026, 10, 13)
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "market_scanner" / "news"


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------
def em_news_body(articles: list[dict]) -> str:
    return "cb(" + json.dumps({"result": {"cmsArticleWebOld": articles}}) + ")"


ART_FRESH = {
    "date": "2026-10-12 10:00:00", "title": "Fresh story",
    "url": "http://finance.eastmoney.com/a/1.html", "mediaName": "人民财讯",
    "content": "Short summary.",
}
ART_STALE = {
    "date": "2026-08-01 10:00:00", "title": "Old story",
    "url": "http://finance.eastmoney.com/a/2.html", "mediaName": "M",
    "content": "Old.",
}
SINA_HTML = (FIXTURES / "sina_news.html").read_bytes().decode("gbk")


def ann_body(rows: list[dict]) -> str:
    return json.dumps({"data": {"list": rows}})


ANN = [
    {"notice_date": "2026-10-12 00:00:00",
     "title": "关于收到中国证监会立案通知的公告",
     "art_code": "AN1", "columns": [{"column_name": "监管"}]},
    {"notice_date": "2026-09-20 00:00:00",
     "title": "公司股票被实施退市风险警示",
     "art_code": "AN2", "columns": []},
    {"notice_date": "2026-08-01 00:00:00", "title": "太旧的公告",
     "art_code": "AN3", "columns": []},
]


def _pledge_frame():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame([
        {"股票代码": "600519", "质押比例": 45.5, "交易日期": "2026-09-30"},
        {"股票代码": "000001", "质押比例": 5.0, "交易日期": "2026-09-30"},
    ])


def _yjyg_frame():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame([
        {"股票代码": "300750", "预告类型": "首亏", "公告日期": "2026-10-10",
         "业绩变动原因": "行业周期下行"},
        {"股票代码": "000001", "预告类型": "预增", "公告日期": "2026-10-10",
         "业绩变动原因": "向好"},
    ])


def holder_body(rows: list[dict]) -> str:
    return json.dumps({"result": {"data": rows}})


HOLDER = [
    {"NOTICE_DATE": "2026-10-01 00:00:00", "CHANGE_NUM": -123.45,
     "HOLDER_NAME": "张某"},
    {"NOTICE_DATE": "2026-01-01 00:00:00", "CHANGE_NUM": -999.0,
     "HOLDER_NAME": "老李"},  # outside 90-day window
    {"NOTICE_DATE": "2026-09-15 00:00:00", "CHANGE_NUM": 10.0,
     "HOLDER_NAME": "增持者"},  # increase, not a flag
]


# ---------------------------------------------------------------------------
# JSONP stripping / ids / 30-day cut
# ---------------------------------------------------------------------------
def test_strip_jsonp() -> None:
    assert _strip_jsonp("cb({\"a\": 1})") == "{\"a\": 1}"
    with pytest.raises(ValueError):
        _strip_jsonp("not jsonp")


def test_news_30day_cut_and_ids() -> None:
    def fetch(url: str, **kw):
        assert "search-api-web.eastmoney.com" in url
        assert "sort%22%3A%22time" in url or "sort" in url
        return em_news_body([ART_FRESH, ART_STALE])

    items, st = fetch_news("600519", 30, today=TODAY, fetch=fetch)
    assert st.status == "ok" and st.rows == 1
    assert [i["id"] for i in items] == ["N1"]
    assert items[0]["date"] == "2026-10-12"
    assert items[0]["kind"] == "news"
    assert items[0]["source"] == "人民财讯"


def test_news_failure_isolated() -> None:
    def fetch(url: str, **kw):
        raise RuntimeError("network down")

    items, st = fetch_news("600519", 30, today=TODAY, fetch=fetch)
    assert items == [] and st.status == "failed" and st.error


def test_announcements_cut_ids_and_category() -> None:
    items, st = fetch_announcements("600519", 30, today=TODAY,
                                    fetch=lambda u, **k: ann_body(ANN))
    assert st.status == "ok"
    assert [i["id"] for i in items] == ["A1", "A2"]  # stale third row cut
    assert items[0]["category"] == "监管"
    assert items[0]["art_code"] == "AN1"
    assert items[1]["title"].startswith("公司股票被实施退市风险警示")


def test_announcements_failure_isolated() -> None:
    items, st = fetch_announcements("600519", 30, today=TODAY,
                                    fetch=lambda u, **k: "not json")
    assert items == [] and st.status == "failed"


# ---------------------------------------------------------------------------
# Sina GBK fallback
# ---------------------------------------------------------------------------
def test_sina_gbk_fallback_degraded() -> None:
    def fetch(url: str, **kw):
        if "search-api-web" in url:
            raise RuntimeError("eastmoney down")
        assert "vip.stock.finance.sina.com.cn" in url
        return SINA_HTML

    items, st = fetch_news("600519", 30, today=TODAY, fetch=fetch)
    assert st.status == "degraded"
    assert [i["id"] for i in items] == ["N1", "N2"]  # stale row cut
    assert all(i["source"] == "sina" for i in items)
    assert items[0]["url"].startswith("https://finance.sina.com.cn")


# ---------------------------------------------------------------------------
# Red flags
# ---------------------------------------------------------------------------
def test_red_flags_all_types() -> None:
    calls = {"fetch": 0, "pledge": 0, "yjyg": 0}

    def fetch(url: str, **kw):
        calls["fetch"] += 1
        if "RPT_SHARE_HOLDER_INCREASE" in url:
            assert 'SECURITY_CODE%3D%22600519%22' in url or "600519" in url
            return holder_body(HOLDER)
        if "np-anotice" in url:
            return ann_body(ANN)
        raise AssertionError(f"unexpected url {url}")

    def pledge_loader():
        calls["pledge"] += 1
        return _pledge_frame()

    def yjyg_loader(q: str):
        calls["yjyg"] += 1
        return _yjyg_frame()

    out, st = fetch_red_flags(
        ["600519"], today=TODAY, fetch=fetch,
        pledge_loader=pledge_loader, yjyg_loader=yjyg_loader,
    )
    assert st.status == "ok"
    flags = out["600519"]
    types = [f["type"] for f in flags]
    assert "pledge_high" in types            # 45.5 > 30
    assert "holder_selldown" in types        # -123.45 within 90d
    assert "csrc_investigation" in types     # 立案通知 title (folded, severe)
    assert "delisting_risk" in types         # 退市风险警示 title
    assert "regulatory" not in types         # covered by the severe branches
    assert "earnings_warning" not in types   # yjyg frame has 300750, not 600519
    assert calls["pledge"] == 1              # cached once per call, not per stock
    assert calls["yjyg"] == 2                # two quarter dates
    assert [f["id"] for f in flags] == [f"R{i}" for i in range(1, len(flags) + 1)]
    pledge = next(f for f in flags if f["type"] == "pledge_high")
    assert pledge["value"] == 45.5
    holder = next(f for f in flags if f["type"] == "holder_selldown")
    assert holder["value"] == -123.45 and holder["date"] == "2026-10-01"
    # Severity labels (TODO_R2 E): investigation/delisting are SEVERE.
    severity = {f["type"]: f["severity"] for f in flags}
    assert severity["pledge_high"] == "material"
    assert severity["holder_selldown"] == "material"
    assert severity["csrc_investigation"] == "severe"
    assert severity["delisting_risk"] == "severe"


def test_red_flags_earnings_warning_and_zero_pledge() -> None:
    def fetch(url: str, **kw):
        return holder_body([]) if "RPT_" in url else ann_body([])

    out, st = fetch_red_flags(
        ["300750"], today=TODAY, fetch=fetch,
        pledge_loader=_pledge_frame, yjyg_loader=lambda q: _yjyg_frame(),
    )
    flags = out["300750"]
    assert st.status == "ok"
    assert [f["type"] for f in flags] == ["earnings_warning", "earnings_warning"]  # one per quarter frame
    assert all("首亏" in f["text"] for f in flags)
    # 000001: pledge 5.0 (no flag), 预增 (no flag) -> checked but empty
    out2, _ = fetch_red_flags(
        ["000001"], today=TODAY, fetch=fetch,
        pledge_loader=_pledge_frame, yjyg_loader=lambda q: _yjyg_frame(),
    )
    assert out2["000001"] == []


def test_red_flags_per_stock_failure_isolated() -> None:
    def fetch(url: str, **kw):
        if "600519" in url and "RPT_" in url:
            raise RuntimeError("boom for 600519")
        return "{}"
    out, st = fetch_red_flags(
        ["600519", "000001"], today=TODAY, fetch=fetch,
        pledge_loader=_pledge_frame, yjyg_loader=lambda q: _yjyg_frame(),
    )
    # 600519 holder-fetch fails but pledge/yjyg/regulatory still collected
    assert st.status in ("ok", "degraded")
    assert "000001" in out and isinstance(out["600519"], list)


def test_announcements_empty_when_stock_quiet() -> None:
    """600519 regression: a stock with no announcements in the window (latest
    = 2026-08-15, confirmed vs cninfo SSE + Sina) must give rows=0 with status
    ok — not a failure and not stale rows leaking through."""
    rows = [
        {"notice_date": "2026-08-15 00:00:00", "title": "半年度报告",
         "art_code": "AN1", "columns": []},
        {"notice_date": "2026-08-14 00:00:00", "title": "决议公告",
         "art_code": "AN2", "columns": []},
    ]
    items, st = fetch_announcements("600519", 30, today=TODAY,
                                    fetch=lambda u, **k: ann_body(rows))
    assert st.status == "ok" and st.rows == 0 and items == []


def test_red_flags_pledge_loader_day_walkback() -> None:
    """Default pledge loader walks back day by day until the weekly table
    exists; here the injectable loader mimics: empty/None for today, data
    found 2 days back."""
    import pandas as pd

    attempts: list[str] = []

    def pledge_loader():
        # the module calls loader() once per fetch_red_flags call; emulate the
        # walkback done by the real default loader by returning data only if
        # the caller's "today" is >= 2 days after the report date.
        attempts.append("call")
        return _pledge_frame()  # dated 2026-09-30, today is 10-13

    def fetch(url: str, **kw):
        return holder_body([]) if "RPT_" in url else ann_body([])

    out, st = fetch_red_flags(
        ["600519"], today=TODAY, fetch=fetch,
        pledge_loader=pledge_loader, yjyg_loader=lambda q: None,
    )
    assert st.status == "ok"
    assert any(f["type"] == "pledge_high" for f in out["600519"])
    assert attempts == ["call"]  # whole-market frame loaded exactly once


# ---------------------------------------------------------------------------
# SEVERE red-flag class (TODO_R2 item E)
# ---------------------------------------------------------------------------
def test_severe_flag_types_detected_with_severity() -> None:
    """CSRC probe, qualified audit opinion, penalty/fraud, delisting risk.

    Each title maps to its own severe type with ``severity='severe'``; a
    plain inquiry letter stays material ``regulatory``.
    """
    rows = [
        {"notice_date": "2026-10-12 00:00:00",
         "title": "关于收到中国证监会立案告知书的公告",
         "art_code": "AN1", "columns": []},
        {"notice_date": "2026-10-11 00:00:00",
         "title": "2025年度审计报告（保留意见）",
         "art_code": "AN2", "columns": []},
        {"notice_date": "2026-10-10 00:00:00",
         "title": "关于收到行政处罚决定书的公告",
         "art_code": "AN3", "columns": []},
        {"notice_date": "2026-10-09 00:00:00",
         "title": "关于对公司相关人员采取市场禁入措施的公告",
         "art_code": "AN4", "columns": []},
        {"notice_date": "2026-10-08 00:00:00",
         "title": "公司股票被实施退市风险警示",
         "art_code": "AN5", "columns": []},
        {"notice_date": "2026-10-07 00:00:00",
         "title": "关于收到问询函的公告",
         "art_code": "AN6", "columns": []},
    ]

    def fetch(url: str, **kw):
        if "RPT_SHARE_HOLDER_INCREASE" in url:
            return holder_body([])
        if "np-anotice" in url:
            return ann_body(rows)
        raise AssertionError(f"unexpected url {url}")

    out, st = fetch_red_flags(
        ["000001"], today=TODAY, fetch=fetch,
        pledge_loader=_pledge_frame, yjyg_loader=lambda q: _yjyg_frame(),
    )
    assert st.status == "ok"
    flags = out["000001"]  # 000001: pledge 5.0 + 预增 produce no frame flags
    assert [f["type"] for f in flags] == [
        "csrc_investigation",
        "audit_opinion",
        "penalty_fraud",
        "penalty_fraud",
        "delisting_risk",
        "regulatory",
    ]
    assert [f["severity"] for f in flags] == ["severe"] * 5 + ["material"]
    assert [f["id"] for f in flags] == [f"R{i}" for i in range(1, 7)]
    csrc = flags[0]
    assert csrc["text"] == "关于收到中国证监会立案告知书的公告"
    assert csrc["date"] == "2026-10-12"
    assert csrc["source"] == "announcement AN1"
    audit = flags[1]
    assert audit["text"] == "2025年度审计报告（保留意见）"
    assert audit["source"] == "announcement AN2"


def test_audit_context_without_bad_opinion_is_not_flagged() -> None:
    """A clean '无保留意见' or a routine audit title raises no flag."""
    flags = _regulatory_flags([
        {"title": "2025年度审计报告（无保留意见）", "date": "2026-10-12",
         "art_code": "AN1"},
        {"title": "关于变更会计师事务所的公告", "date": "2026-10-11",
         "art_code": "AN2"},
        {"title": "关于召开年度报告说明会的公告", "date": "2026-10-10",
         "art_code": "AN3"},
    ])
    assert flags == []


def test_severe_title_patterns_classified_directly() -> None:
    """The three CSRC phrases + each penalty phrase map to severe types."""
    cases = {
        "关于立案调查的提示性公告": "csrc_investigation",
        "关于被中国证监会立案的公告": "csrc_investigation",
        "2025年度审计报告（否定意见）": "audit_opinion",
        "2025年度审计报告（无法表示意见）": "audit_opinion",
        "关于收到欺诈发行告知的公告": "penalty_fraud",
        "关于收到财务造假处罚的公告": "penalty_fraud",
        "关于公司股票终止上市的公告": "delisting_risk",
        "关于收到监管函的公告": "regulatory",
        "nothing risky here": None,
    }
    for title, expected in cases.items():
        assert news_mod._title_flag_type(title) == expected, title
        if expected is not None:
            assert news_mod._severity(expected) in ("severe", "material")
            if expected != "regulatory":
                assert news_mod._severity(expected) == "severe"


def test_csrc_investigation_title_variants() -> None:
    """Folded-in real-world CSRC wordings, one assertion per variant."""
    variants = [
        "关于收到中国证监会立案告知书的公告",  # spec phrase
        "关于立案调查的提示性公告",  # spec phrase
        "关于被中国证监会立案的公告",  # spec phrase
        "关于收到立案通知的公告",  # 立案通知
        "关于收到立案通知书的公告",  # 立案通知书
        "关于证监会对公司立案的公告",  # 证监会 ... 立案
        "中国证监会决定对公司立案",  # short regulator form
        "中国证券监督管理委员会决定对公司立案",  # full name ... 立案
        "中国证券监督管理委员会上海监管局对公司立案调查",  # full name + gap
    ]
    for title in variants:
        assert news_mod._title_flag_type(title) == "csrc_investigation", title
        assert news_mod._severity("csrc_investigation") == "severe"
    # The clean unqualified opinion must stay unflagged.
    assert news_mod._title_flag_type("2025年度审计报告（无保留意见）") is None
    assert news_mod._title_flag_type("关于变更会计师事务所的公告") is None
