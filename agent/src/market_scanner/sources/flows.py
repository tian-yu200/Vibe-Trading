"""Money-flow (THS), dragon-tiger (EM datacenter), northbound semi-annual info."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from src.market_scanner.sources._common import _to_float, meter_akshare_call, parse_ths_money
from src.market_scanner.sources.status import SourceStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# G1: THS fund flow (akshare stock_fund_flow_individual) — the only working
# whole-market main-fund source on this network (push2 blocked).
# ---------------------------------------------------------------------------
def _ths_number(value) -> float | None:
    """THS flow cell: '亿/万' text or already-numeric."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    return parse_ths_money(str(value))


def fetch_fund_flow(
    fetchers: dict[str, Callable] | None = None,
) -> tuple[pd.DataFrame, SourceStatus]:
    """Whole-market fund flow (THS net inflow; NOT Eastmoney main-force).

    Args:
        fetchers: Injectable mapping with keys ``即时`` / ``3日排行`` /
            ``5日排行`` mapping to zero-arg callables returning DataFrames
            (tests pass fixtures). Defaults to
            ``ak.stock_fund_flow_individual(symbol=...)``.

    Returns:
        ``(df, st)`` — indexed by ``code`` with columns
        ``flow_1d, flow_3d, flow_5d, pct_chg_3d, pct_chg_5d`` (CNY / percent).
        ``st.detail`` always notes "THS net flow, not Eastmoney main-force".
    """
    import akshare as ak

    t0 = time.monotonic()
    fetchers = fetchers or {}
    errors: list[str] = []
    frames: dict[str, pd.DataFrame] = {}

    for key, cols in (
        ("即时", {"net": "净额", "pct": None}),
        ("3日排行", {"net": "资金流入净额", "pct": "阶段涨跌幅"}),
        ("5日排行", {"net": "资金流入净额", "pct": "阶段涨跌幅"}),
    ):
        fn = fetchers.get(key) or (lambda k=key: ak.stock_fund_flow_individual(symbol=k))
        try:
            df = meter_akshare_call(fn, host="ths")
        except Exception as exc:  # noqa: BLE001 — degraded, not raised
            errors.append(f"{key}: {type(exc).__name__}: {exc}"[:150])
            continue
        if df is None or df.empty:
            errors.append(f"{key}: empty")
            continue
        code_col = next((c for c in ("股票代码", "证券代码") if c in df.columns), None)
        if code_col is None:
            errors.append(f"{key}: no code column")
            continue
        out = pd.DataFrame({"code": df[code_col].astype(str).str.zfill(6)})
        if cols["net"] in df.columns:
            out["flow"] = df[cols["net"]].map(_ths_number)
        else:
            out["flow"] = None
            errors.append(f"{key}: no {cols['net']} column")
        if cols["pct"] and cols["pct"] in df.columns:
            out["pct"] = df[cols["pct"]].map(
                lambda v: _to_float(str(v).replace("%", "")) if v is not None else None
            )
        else:
            out["pct"] = None
        frames[key] = out.drop_duplicates(subset="code")

    inst = frames.get("即时")
    d3 = frames.get("3日排行")
    d5 = frames.get("5日排行")

    merged: pd.DataFrame
    if inst is not None:
        merged = inst.rename(columns={"flow": "flow_1d", "pct": "pct_chg_1d"})
    else:
        merged = pd.DataFrame(columns=["code", "flow_1d", "pct_chg_1d"])
    if d3 is not None:
        merged = merged.merge(
            d3.rename(columns={"flow": "flow_3d", "pct": "pct_chg_3d"}),
            on="code", how="outer",
        )
    else:
        merged["flow_3d"] = None
        merged["pct_chg_3d"] = None
    if d5 is not None:
        merged = merged.merge(
            d5.rename(columns={"flow": "flow_5d", "pct": "pct_chg_5d"}),
            on="code", how="outer",
        )
    else:
        merged["flow_5d"] = None
        merged["pct_chg_5d"] = None

    merged = merged.set_index("code").drop_duplicates()

    latency = time.monotonic() - t0
    if merged.empty:
        st = SourceStatus(
            name="fund_flow", status="failed", rows=0, latency_s=latency,
            error="; ".join(errors)[:500] or "no THS flow data",
            detail="THS net flow, not Eastmoney main-force",
        )
        return merged, st
    import datetime as _dt

    now = _dt.datetime.now()
    intraday = now.weekday() < 5 and _dt.time(9, 15) <= now.time() <= _dt.time(15, 0)
    ths_date = (now.date() if (now.time() >= _dt.time(17, 30) or now.weekday() == 0)
                else _prev_weekday(now.date()))
    st = SourceStatus(
        name="fund_flow",
        status="degraded" if errors else "ok",
        rows=len(merged),
        latency_s=latency,
        error="; ".join(errors)[:500] or None,
        detail="THS net flow, not Eastmoney main-force (push2 blocked on this network)",
        as_of=f"intraday {ths_date.isoformat()}" if intraday else ths_date.isoformat(),
    )
    return merged, st


def _prev_weekday(d):
    import datetime as _dt
    d = d - _dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= _dt.timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# G3: dragon-tiger list (EM datacenter via akshare, works here)
# ---------------------------------------------------------------------------
_LHB_REASON_MAP = {"解读": "reason"}


def fetch_dragon_tiger(
    end: date,
    days: int = 30,
    lhb_fn: Callable | None = None,
) -> tuple[pd.DataFrame, SourceStatus]:
    """Dragon-tiger aggregates per stock over the trailing window.

    Args:
        end: Window end date (inclusive).
        days: Window length in calendar days.
        lhb_fn: Injectable for ``ak.stock_lhb_detail_em``.

    Returns:
        ``(df, st)`` — indexed by ``code`` with columns
        ``lhb_count, lhb_net_buy, lhb_last_date, lhb_reasons``.
    """
    import akshare as ak

    t0 = time.monotonic()
    start = end - timedelta(days=days)
    lhb_fn = lhb_fn or ak.stock_lhb_detail_em
    try:
        raw = meter_akshare_call(
            lhb_fn, start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"), host="eastmoney")
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        st = SourceStatus(
            name="dragon_tiger", status="failed", rows=0,
            latency_s=time.monotonic() - t0,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )
        return pd.DataFrame(), st
    if raw is None or raw.empty:
        st = SourceStatus(
            name="dragon_tiger", status="degraded", rows=0,
            latency_s=time.monotonic() - t0, error="empty LHB frame",
        )
        return pd.DataFrame(), st

    code_col = next((c for c in ("代码", "股票代码") if c in raw.columns), None)
    date_col = next((c for c in ("上榜日", "交易日") if c in raw.columns), None)
    net_col = next((c for c in ("龙虎榜净买额", "龙虎榜净额") if c in raw.columns), None)
    reason_col = "解读" if "解读" in raw.columns else None
    if code_col is None:
        st = SourceStatus(
            name="dragon_tiger", status="failed", rows=0,
            latency_s=time.monotonic() - t0, error=f"unexpected columns: {list(raw.columns)[:8]}",
        )
        return pd.DataFrame(), st

    records: dict[str, dict] = {}
    for _, row in raw.iterrows():
        code = str(row[code_col]).zfill(6)
        rec = records.setdefault(code, {"count": 0, "net": 0.0, "last": None, "reasons": set()})
        rec["count"] += 1
        if net_col:
            val = _to_float(row.get(net_col))
            if val is not None:
                rec["net"] += val
        d = row.get(date_col) if date_col else None
        if d is not None:
            d = str(d)[:10]
            if rec["last"] is None or d > rec["last"]:
                rec["last"] = d
        if reason_col and row.get(reason_col):
            rec["reasons"].add(str(row[reason_col])[:60])

    df = pd.DataFrame.from_records(
        [
            {
                "code": code,
                "lhb_count": rec["count"],
                "lhb_net_buy": rec["net"],
                "lhb_last_date": rec["last"],
                "lhb_reasons": "; ".join(sorted(rec["reasons"])) or None,
            }
            for code, rec in records.items()
        ]
    ).set_index("code")

    # as_of: latest list date actually present in the window; note when
    # today's list is not out yet (published ~17:30).
    import datetime as _dt
    lasts = [v.get("last") for v in records.values() if v.get("last")]
    latest = max(lasts) if lasts else None
    note = ""
    if end >= _dt.date.today() and _dt.datetime.now().time() < _dt.time(17, 30):
        note = " (today's list not out yet, before ~17:30)"
    st = SourceStatus(
        name="dragon_tiger", status="ok", rows=len(df),
        latency_s=time.monotonic() - t0,
        detail=f"window {start}..{end} source=EM datacenter LHB",
        as_of=f"{latest}{note}" if latest else None,
    )
    return df, st


# ---------------------------------------------------------------------------
# Northbound: daily per-stock disclosure stopped 2024-08; only semi-annual
# RPT_MUTUAL_HOLDSTOCKNORTH_STA via datacenter-web works on this network.
# ---------------------------------------------------------------------------
_NB_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def fetch_northbound_info(
    trade_date: str | None = None,
    fetch: Callable | None = None,
) -> tuple[pd.DataFrame, SourceStatus]:
    """Semi-annual northbound per-stock holdings (info only, never scored).

    Args:
        trade_date: Report date ``YYYY-MM-DD``; None picks the latest known
            semi-annual date (2026-06-30 at probe time).
        fetch: Injectable ``(url, params) -> dict`` returning parsed JSON.

    Returns:
        ``(df, st)`` — indexed by ``code`` with columns
        ``nb_hold_ratio, nb_hold_mv, nb_report_date``.
    """
    from src.market_scanner.sources._common import http_get_json

    t0 = time.monotonic()
    trade_date = trade_date or "2026-06-30"
    params = {
        "reportName": "RPT_MUTUAL_HOLDSTOCKNORTH_STA",
        "columns": "ALL",
        "pageSize": 500,
        "pageNumber": 1,
        "sortColumns": "HOLD_MARKET_CAP",
        "sortTypes": -1,
        "filter": f"(TRADE_DATE='{trade_date}')",
        "source": "WEB",
        "client": "WEB",
    }
    try:
        if fetch is None:
            payload = http_get_json(_NB_URL, host_key="datacenter-eastmoney", params=params)
        else:
            payload = fetch(_NB_URL, params)
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        st = SourceStatus(
            name="northbound", status="failed", rows=0,
            latency_s=time.monotonic() - t0,
            error=f"{type(exc).__name__}: {exc}"[:500],
            detail="daily per-stock northbound disclosure stopped 2024-08; semi-annual only",
        )
        return pd.DataFrame(), st

    result = payload.get("result") or {}
    rows = result.get("data") or []
    records = []
    for row in rows:
        secu = row.get("SECUCODE") or ""
        code = secu.split(".")[0].zfill(6)
        if not code:
            continue
        records.append({
            "code": code,
            "nb_hold_ratio": _to_float(row.get("HOLD_SHARES_RATIO")),
            "nb_hold_mv": _to_float(row.get("HOLD_MARKET_CAP")),
            "nb_report_date": trade_date,
        })

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.set_index("code").drop_duplicates()
    st = SourceStatus(
        name="northbound",
        status="ok" if not df.empty else "failed",
        rows=len(df),
        latency_s=time.monotonic() - t0,
        error=None if not df.empty else f"datacenter reply missing rows: {payload.get('message', '')[:120]}",
        detail=f"semi-annual report {trade_date}; info only, never scored",
        as_of=trade_date,
    )
    return df, st
