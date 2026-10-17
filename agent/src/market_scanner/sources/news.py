"""News, announcements and red flags for the AI review (CONTRACT section 8).

Owner: pi. Probe evidence: ``E:/vibe-trading/_probe/news/NEWS_SUMMARY.md``.

Sources (all verified working from this machine, 2026-10-13):
* News: Eastmoney search-api-web (JSONP, sort=time) with Sina vCB_AllNewsStock
  (GBK HTML) as fallback.
* Announcements: Eastmoney np-anotice per-stock feed.
* Red flags: whole-market pledge ratio + earnings-forecast frames cached once
  per call, per-stock holder-change API, and a title scan over the 30-day
  announcement titles.

Announcement titles are classified into a SEVERE class (``csrc_investigation``
立案告知书/立案调查/立案通知[/书]/证监会…立案/中国证券监督管理委员会…立案,
``audit_opinion`` 保留意见/否定意见/无法表示意见 in audit-report titles (a clean
无保留意见 is never flagged), ``penalty_fraud`` 行政处罚决定书/市场禁入/
欺诈发行/财务造假, ``delisting_risk`` 退市风险/终止上市) carrying
``severity='severe'``, and a material class (``regulatory`` plus the frame-based
flags) carrying ``severity='material'``. The AI review forces ``avoid`` for an
unresolved severe flag (``market_scanner.review``).

All fetchers are injectable for offline tests. Nothing raises; failures come
back as ``SourceStatus(status="failed")``. Item ids are ``N1..``, ``A1..``,
``R1..`` — stable and citable within one stock's pack.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
from datetime import date, timedelta
from typing import Any, Callable

# Single source of truth for the SEVERE class lives in the review layer
# (the module that enforces it); news.py only stamps the label.
from src.market_scanner.review import SEVERE_RED_FLAG_TYPES
from src.market_scanner.sources._common import http_get_text, meter_akshare_call
from src.market_scanner.sources.status import SourceStatus, fail_status

logger = logging.getLogger(__name__)

FetchText = Callable[[str, Any], str]


def _assign_ids(items: list[dict], prefix: str) -> list[dict]:
    """Stable citable ids: prefix + 1-based index (N1.., A1.., R1..)."""
    for i, item in enumerate(items, 1):
        item["id"] = f"{prefix}{i}"
    return items

# Host keys for the shared per-host throttle (<=3 req/s => 0.34s min interval).
_H_EM_SEARCH = "search-api-web.eastmoney.com"
_H_EM_NOTICE = "np-anotice-stock.eastmoney.com"
_H_EM_DATA = "datacenter-web.eastmoney.com"
_H_SINA = "vip.stock.finance.sina.com.cn"

_MIN_INTERVAL = 0.34

_NEWS_URL = "https://search-api-web.eastmoney.com/search/jsonp"
_NOTICE_URL = (
    "https://np-anotice-stock.eastmoney.com/api/security/ann"
    "?sr=-1&page_size=50&page_index={page}&ann_type=A&client_source=web"
    "&stock_list={code}"
)
_HOLDER_URL = (
    "https://datacenter-web.eastmoney.com/api/data/v1/get"
    "?reportName=RPT_SHARE_HOLDER_INCREASE"
    "&columns=SECURITY_CODE,SECURITY_NAME_ABBR,NOTICE_DATE,CHANGE_NUM,"
    "CHANGE_RATE,HOLDER_NAME&pageSize=50&pageNumber=1"
    "&sortColumns=NOTICE_DATE&sortTypes=-1&filter=(SECURITY_CODE=%22{code}%22)"
)
_SINA_NEWS_URL = (
    "https://vip.stock.finance.sina.com.cn/corp/go.php/"
    "vCB_AllNewsStock/symbol/{sym}.phtml"
)

_SINA_DATE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})&nbsp;\d{2}:\d{2}&nbsp;&nbsp;"
    r"<a target='_blank' href='([^']+)'>([^<]+)</a>"
)

# Title patterns. Order matters: SEVERE classes are checked first, then the
# material `regulatory` catch-all (its regex also matches 处罚/立案 etc.).
# CSRC investigation: the spec phrases plus the most common real wordings —
# 立案通知[/书] and 证监会 / 中国证券监督管理委员会 ... 立案.
_CSRC_RE = re.compile(
    r"立案告知书|立案调查|立案通知|"
    r"(?:中国证券监督管理委员会|证监会).{0,20}立案"
)
_AUDIT_CONTEXT_RE = re.compile(r"审计|会计师事务所|会计事务所|年度报告|财务报告")
# (?<!无) keeps a clean "无保留意见" (unqualified opinion) from flagging.
_AUDIT_OPINION_RE = re.compile(r"(?<!无)保留意见|否定意见|无法表示意见")
_PENALTY_RE = re.compile(r"行政处罚决定书|市场禁入|欺诈发行|财务造假")
_DELISTING_RE = re.compile(r"退市风险|终止上市")
_REGULATORY_RE = re.compile(r"立案|调查|问询函|监管函|警示|处罚|违规")

BadTypes = ("预减", "首亏", "续亏", "略减", "增亏")


def _severity(ftype: str) -> str:
    """``'severe'`` for the SEVERE class, ``'material'`` otherwise."""
    return "severe" if ftype in SEVERE_RED_FLAG_TYPES else "material"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _symbol(code: str) -> str:
    """Sina symbol prefix from the 6-digit code."""
    return ("sh" if code[0] in "69" else "sz") + code


def _iso(value: str) -> str:
    """Normalise '2026-10-13 15:14:00' / '2026-10-13 00:00:00' to ISO date."""
    return str(value)[:10]


def _strip_jsonp(text: str) -> str:
    """Extract the JSON payload from a ``cb({...})`` JSONP wrapper."""
    start = text.find("(")
    end = text.rfind(")")
    if start < 0 or end <= start:
        raise ValueError("body is not JSONP")
    return text[start + 1:end]


def _news_source_name(media: Any, url: str) -> str:
    return str(media) if media else ("eastmoney" if "eastmoney" in url else "sina")


def _days_back(today: date, days: int) -> date:
    return today - timedelta(days=days)

# ---------------------------------------------------------------------------
# News (Eastmoney search-api, Sina GBK fallback)
# ---------------------------------------------------------------------------
def _em_news_page(code: str, page: int, page_size: int, fetch: FetchText) -> list[dict]:
    param = {
        "uid": "", "keyword": code, "type": ["cmsArticleWebOld"],
        "client": "web", "clientType": "web", "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {
            "searchScope": "default", "sort": "time", "pageIndex": page,
            "pageSize": page_size, "preTag": "", "postTag": "",
        }},
    }
    url = _NEWS_URL + "?cb=x&param=" + urllib.parse.quote(
        json.dumps(param, separators=(",", ":")))
    text = fetch(url, host_key=_H_EM_SEARCH, min_interval=_MIN_INTERVAL)
    data = json.loads(_strip_jsonp(text))
    arts = (data.get("result") or {}).get("cmsArticleWebOld") or []
    return list(arts)


def _sina_news(code: str, fetch: FetchText) -> list[dict]:
    html = fetch(
        _SINA_NEWS_URL.format(sym=_symbol(code)),
        host_key=_H_SINA, min_interval=_MIN_INTERVAL,
    )
    rows = []
    for m in _SINA_DATE_RE.finditer(html):
        rows.append({
            "date": m.group(1),
            "url": urllib.parse.urljoin("https://vip.stock.finance.sina.com.cn", m.group(2)),
            "title": m.group(3).strip(),
            "mediaName": "sina",
            "content": "",
        })
    return rows


def _news_items(
    code: str,
    days: int,
    today: date,
    fetch: FetchText,
) -> list[dict]:
    cutoff = _iso(_days_back(today, days))
    arts: list[dict] = []
    em_failed = False
    for page in (1, 2, 3):  # CONTRACT: up to 3 pages
        try:
            batch = _em_news_page(code, page, 50, fetch)
        except Exception:
            em_failed = True
            break
        if not batch:
            break
        arts.extend(batch)
        if _iso(batch[-1].get("date", "")) < cutoff:
            break
    if not arts or em_failed:
        arts = _sina_news(code, fetch)  # GBK fallback
    items = []
    for a in arts:
        d = _iso(a.get("date", ""))
        if d < cutoff:
            continue
        items.append({
            "kind": "news",
            "date": d,
            "title": str(a.get("title", "")).strip(),
            "source": _news_source_name(a.get("mediaName"), str(a.get("url", ""))),
            "url": str(a.get("url", "")),
            "summary": str(a.get("content", "")).strip(),
        })
    return items


def fetch_news(
    code: str,
    days: int = 30,
    *,
    today: date | None = None,
    fetch: FetchText | None = None,
) -> tuple[list[dict], SourceStatus]:
    """Per-stock news for the last ``days`` days (CONTRACT section 8).

    Never raises; on total failure returns ``([], failed_status)``.
    """
    fetch = fetch or http_get_text
    today = today or date.today()
    t0 = time.monotonic()
    try:
        items = _news_items(code, days, today, fetch)
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        return [], fail_status("news", exc, t0)
    for i, item in enumerate(items, 1):
        item["id"] = f"N{i}"
    used_sina = bool(items) and all(i.get("source") == "sina" for i in items)
    latest = max((i.get("date") for i in items), default=None)
    return items, SourceStatus(
        name="news", status="degraded" if used_sina else "ok", rows=len(items),
        latency_s=time.monotonic() - t0,
        detail=f"code={code} days={days} source="
               + ("sina_fallback" if used_sina else "eastmoney_search"),
        as_of=str(latest)[:10] if latest else None,
    )

# ---------------------------------------------------------------------------
# Announcements (Eastmoney np-anotice)
# ---------------------------------------------------------------------------
def _ann_items(code: str, days: int, today: date, fetch: FetchText) -> list[dict]:
    cutoff = _iso(_days_back(today, days))
    url = (
        _NOTICE_URL.replace("{page}", "1").replace("{code}", code)
    )
    text = fetch(url, host_key=_H_EM_NOTICE, min_interval=_MIN_INTERVAL)
    rows = (json.loads(text).get("data") or {}).get("list") or []
    items = []
    for r in rows:
        d = _iso(r.get("notice_date", ""))
        if d < cutoff:
            continue
        cols = r.get("columns") or []
        items.append({
            "kind": "announcement",
            "date": d,
            "title": str(r.get("title", "")).strip(),
            "category": (cols[0].get("column_name") if cols else "") or "",
            "source": "Eastmoney announcements",
            "url": (
                "https://data.eastmoney.com/notices/detail/"
                + code + "/" + str(r.get("art_code", "")) + ".html"
            ),
            "art_code": str(r.get("art_code", "")),
        })
    return items




def fetch_announcements(
    code: str,
    days: int = 30,
    *,
    today: date | None = None,
    fetch: FetchText | None = None,
) -> tuple[list[dict], SourceStatus]:
    """Per-stock announcements for the last ``days`` days (CONTRACT section 8).

    Source: Eastmoney np-anotice per-stock feed (30 items, filtered to the
    window). Never raises.
    """
    fetch = fetch or http_get_text
    today = today or date.today()
    t0 = time.monotonic()
    try:
        items = _ann_items(code, days, today, fetch)
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        return [], fail_status("announcements", exc, t0)
    for i, item in enumerate(items, 1):
        item["id"] = f"A{i}"
    latest = max((i.get("date") for i in items), default=None)
    return items, SourceStatus(
        name="announcements", status="ok", rows=len(items),
        latency_s=time.monotonic() - t0,
        detail=f"code={code} days={days} source=eastmoney_np-anotice",
        as_of=str(latest)[:10] if latest else None,
    )


# ---------------------------------------------------------------------------
# Red flags
# ---------------------------------------------------------------------------
_PLEDGE_WARN = 30.0
_HOLDER_WINDOW_DAYS = 90


def _flags_frame(
    loader: Callable[[], Any],
    *,
    today: date | None,
) -> Any:
    """Run one akshare loader; None on failure (frame cached by caller)."""
    del today  # quarter dates are computed by the caller
    try:
        return loader()
    except Exception as exc:  # noqa: BLE001 — flag source failing is not fatal
        logger.info("red-flag loader failed: %s", exc)
        return None


def _quarter_dates(today: date) -> list[str]:
    """The two most recent quarter-end dates (YYYYMMDD) for yjyg, newest first."""
    out = []
    y, m = today.year, today.month
    # current quarter end (or previous if we're on the boundary month+2)
    ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
    cur = None
    for em, ed in reversed(ends):
        d = date(y, em, ed)
        if d <= today:
            cur = d
            break
    if cur is None:
        cur = date(y - 1, 12, 31)
    out.append(cur.strftime("%Y%m%d"))
    # previous quarter end
    py, pm = (cur.year - 1, 12) if cur.month <= 3 else (cur.year, ((cur.month - 1) // 3) * 3 + 3 - 3)
    prev_idx = ends.index((cur.month, cur.day)) - 1
    if prev_idx < 0:
        out.append(date(cur.year - 1, 12, 31).strftime("%Y%m%d"))
    else:
        em, ed = ends[prev_idx]
        out.append(date(cur.year, em, ed).strftime("%Y%m%d"))
    return out


def _holder_flags(code: str, cut: str, fetch: FetchText) -> list[dict]:
    """Major-holder sell-downs: CHANGE_NUM < 0 with NOTICE_DATE in window."""
    params = {
        "reportName": "RPT_SHARE_HOLDER_INCREASE",
        "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,NOTICE_DATE,CHANGE_NUM,CHANGE_RATE,HOLDER_NAME",
        "pageSize": "50",
        "pageNumber": "1",
        "sortColumns": "NOTICE_DATE",
        "sortTypes": "-1",
        "filter": f'(SECURITY_CODE="{code}")',
    }
    text = fetch(_HOLDER_URL + urllib.parse.urlencode(params),
                 host_key=_H_EM_DATA, min_interval=_MIN_INTERVAL)
    rows = ((json.loads(text).get("result") or {}).get("data")) or []
    flags: list[dict] = []
    for r in rows:
        d = _iso(str(r.get("NOTICE_DATE", "")))[:10]
        num = r.get("CHANGE_NUM")
        if d < cut or num is None:
            continue
        try:
            num_f = float(num)
        except (TypeError, ValueError):
            continue
        if num_f >= 0:
            continue  # increase, not a sell-down
        holder = str(r.get("HOLDER_NAME", "")).strip()
        rate = r.get("CHANGE_RATE")
        flags.append({
            "type": "holder_selldown",
            "severity": _severity("holder_selldown"),
            "text": f"{holder} reduced holding by {abs(num_f):.2f}万股 on {d}",
            "date": d,
            "value": num_f,
            "source": "eastmoney datacenter RPT_SHARE_HOLDER_INCREASE",
        })
    return flags


def _title_flag_type(title: str) -> str | None:
    """Classify one announcement title; None when it carries no red flag.

    SEVERE first (``csrc_investigation`` → ``audit_opinion`` →
    ``penalty_fraud`` → ``delisting_risk``), then the material
    ``regulatory`` catch-all.
    """
    if _CSRC_RE.search(title):
        return "csrc_investigation"
    if _AUDIT_CONTEXT_RE.search(title) and _AUDIT_OPINION_RE.search(title):
        return "audit_opinion"
    if _PENALTY_RE.search(title):
        return "penalty_fraud"
    if _DELISTING_RE.search(title):
        return "delisting_risk"
    if _REGULATORY_RE.search(title):
        return "regulatory"
    return None


def _regulatory_flags(anns: list[dict]) -> list[dict]:
    """Severe + material red flags from announcement titles (CONTRACT §8).

    Every flag carries ``severity``: ``'severe'`` for the SEVERE class the
    AI review forces to ``avoid`` unless resolved, ``'material'`` for the
    rest.
    """
    flags: list[dict] = []
    for a in anns:
        title = str(a.get("title", ""))
        ftype = _title_flag_type(title)
        if ftype is None:
            continue
        flags.append({
            "type": ftype,
            "severity": _severity(ftype),
            "text": title,
            "date": a.get("date", ""),
            "value": None,
            "source": f"announcement {a.get('art_code', '')}",
        })
    return flags


def fetch_red_flags(
    codes: list[str],
    *,
    today: date | None = None,
    fetch: FetchText | None = None,
    pledge_loader: Callable[[], Any] | None = None,
    yjyg_loader: Callable[[str], Any] | None = None,
) -> tuple[dict[str, list[dict]], SourceStatus]:
    """Structured red flags for a batch of stocks (CONTRACT section 8).

    The whole-market pledge and earnings-forecast frames are loaded once per
    call (not once per stock). Per-stock failures are isolated; nothing raises.
    Returns ``{code: [flag, ...]}`` with ids ``R1..Rn`` per stock; every flag
    carries ``severity`` (``severe`` / ``material``).
    """
    import akshare as ak  # deferred: heavy import, not needed for news/anns

    fetch = fetch or http_get_text
    today = today or date.today()
    t0 = time.monotonic()
    failed = 0
    out: dict[str, list[dict]] = {code: [] for code in codes}

    cut_holder = _iso(today - timedelta(days=_HOLDER_WINDOW_DAYS))

    # Whole-market frames, loaded once per call.
    def _default_pledge_loader():
        # Weekly report: today's date often has no table yet, walk back day by day.
        for back in range(10):
            try:
                df = meter_akshare_call(
                    ak.stock_gpzy_pledge_ratio_em,
                    date=(today - timedelta(days=back)).strftime("%Y%m%d"),
                    host="eastmoney")
            except Exception:
                continue
            if df is not None and len(df):
                return df
        return None
    pledge_loader = pledge_loader or _default_pledge_loader
    yjyg_loader = yjyg_loader or (lambda q: meter_akshare_call(
        ak.stock_yjyg_em, date=q, host="eastmoney"))
    pledge_df = _flags_frame(pledge_loader, today=today)
    quarters = _quarter_dates(today)
    yjyg_frames = [(q, _flags_frame(lambda q=q: yjyg_loader(q), today=today))
                   for q in quarters]

    def _failed(name: str, exc: Exception) -> SourceStatus:
        return SourceStatus(
            name=name, status="failed", rows=0,
            latency_s=time.monotonic() - t0,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )

    for code in codes:
        flags: list[dict] = []
        # 1) Pledge ratio (whole-market frame indexed by 股票代码).
        if pledge_df is not None:
            try:
                rows = pledge_df[pledge_df["股票代码"].astype(str).str.zfill(6) == code]
                for _, r in rows.iterrows():
                    ratio = r.get("质押比例")
                    ratio_f = float(ratio) if ratio is not None else None
                    if ratio_f is not None and ratio_f > _PLEDGE_WARN:
                        flags.append({
                            "type": "pledge_high",
                            "severity": _severity("pledge_high"),
                            "text": f"质押比例 {ratio_f:.1f}% (> {_PLEDGE_WARN:.0f}%)",
                            "date": _iso(str(r.get("交易日期", "")))[:10],
                            "value": ratio_f,
                            "source": "akshare stock_gpzy_pledge_ratio_em",
                        })
            except Exception as exc:  # noqa: BLE001 — isolate per-flag-source
                logger.info("pledge flag failed for %s: %s", code, exc)
        # 2) Earnings warnings (whole-market frames, one per quarter).
        for q, ydf in yjyg_frames:
            if ydf is None:
                continue
            try:
                rows = ydf[ydf["股票代码"].astype(str).str.zfill(6) == code]
                for _, r in rows.iterrows():
                    if str(r.get("预告类型", "")) not in BadTypes:
                        continue
                    flags.append({
                        "type": "earnings_warning",
                        "severity": _severity("earnings_warning"),
                        "text": (f"{r.get('预告类型', '')}: "
                                 f"{str(r.get('业绩变动原因', ''))[:80]}"),
                        "date": _iso(str(r.get("公告日期", "")))[:10],
                        "value": None,
                        "source": "akshare stock_yjyg_em",
                    })
            except Exception as exc:  # noqa: BLE001
                logger.info("yjyg flag failed for %s: %s", code, exc)
        # 3) Holder sell-down (per-stock HTTP).
        try:
            flags.extend(_holder_flags(code, cut_holder, fetch))
        except Exception as exc:  # noqa: BLE001 — per-stock isolation
            logger.info("holder flags failed for %s: %s", code, exc)
            failed += 1
        # 4) Severe + material title flags from this stock's 30-day
        #    announcements (CSRC probe / audit opinion / penalty / delisting
        #    / regulatory — see _title_flag_type).
        try:
            anns, _st = fetch_announcements(code, 30, today=today, fetch=fetch)
            flags.extend(_regulatory_flags(anns))
        except Exception as exc:  # noqa: BLE001 — per-stock isolation
            logger.info("announcement flags failed for %s: %s", code, exc)
            failed += 1
        out[code] = _assign_ids(flags, "R")

    ok = failed < len(codes)
    # as_of: newest flag date across all stocks (data date).
    flag_dates = [
        str(f.get("date"))[:10]
        for v in out.values() for f in v if f.get("date")
    ]
    st = SourceStatus(
        name="red_flags",
        status="ok" if ok else "failed",
        rows=sum(len(v) for v in out.values()),
        latency_s=time.monotonic() - t0,
        error=None if ok else "all stocks failed",
        detail=f"codes={len(codes)} stocks_with_partial_failure={failed}"
               f" pledge={'ok' if pledge_df is not None else 'failed'}",
        as_of=max(flag_dates) if flag_dates else None,
    )
    return out, st
