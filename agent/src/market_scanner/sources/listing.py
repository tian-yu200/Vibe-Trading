"""Listing date + industry per stock from the akshare exchange name lists."""

from __future__ import annotations

import datetime as _dt
import logging
import time
from typing import Any, Callable

import pandas as pd

from src.market_scanner.sources._common import (
    exchange_from_code, meter_akshare_call, resolve_board, symbol_from_code
)
from src.market_scanner.sources.status import SourceStatus

logger = logging.getLogger(__name__)

_NAME = "listing"


def _safe_call(fn: Callable, host: str = "eastmoney", **kwargs: Any) -> tuple[pd.DataFrame | None, str | None]:
    """Run one akshare list call; return (frame, error)."""
    try:
        df = meter_akshare_call(fn, host=host, **kwargs)
        if df is None:
            return None, f"{fn.__name__}: returned None"
        return df, None
    except Exception as exc:  # noqa: BLE001 — converted to degraded/failed
        return None, f"{fn.__name__}: {type(exc).__name__}: {exc}"[:200]


def _coerce_date(value: Any):
    """Parse a listing date cell (datetime.date | str) to datetime.date or None."""
    import datetime as _dt

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    try:
        return _dt.datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def fetch_listing_info(
    fetchers: dict[str, Callable] | None = None,
) -> tuple[pd.DataFrame, SourceStatus]:
    """Listing date + industry for every listed A-share.

    Args:
        fetchers: Injectable mapping ``{"sh": fn, "sz": fn, "bj": fn}`` where
            each ``fn`` is a zero-arg callable returning an akshare-shaped
            DataFrame (tests use fixtures). Defaults to the real akshare calls:
            ``stock_info_sh_name_code`` (主板A股 + 科创板A股),
            ``stock_info_sz_name_code`` (A股列表), ``stock_info_bj_name_code``.

    Returns:
        ``(df, st)`` — frame indexed by ``code`` with columns
        ``list_date, industry, exchange``; ``list_date`` is ``datetime.date``.
    """
    import akshare as ak

    t0 = time.monotonic()
    fetchers = fetchers or {}
    errors: list[str] = []
    records: list[dict] = []

    # --- SH: main board A + STAR ---
    sh_fn = fetchers.get("sh") or (
        lambda: pd.concat(
            [
                ak.stock_info_sh_name_code(symbol="主板A股"),
                ak.stock_info_sh_name_code(symbol="科创板"),
            ],
            ignore_index=True,
        )
    )
    df, err = _safe_call(sh_fn, host="eastmoney")
    if err:
        errors.append(err)
    elif df is not None:
        code_col = next((c for c in ("证券代码", "A股代码") if c in df.columns), None)
        date_col = "上市日期" if "上市日期" in df.columns else None
        ind_col = "所属行业" if "所属行业" in df.columns else None
        if code_col and date_col:
            for _, row in df.iterrows():
                code = str(row[code_col]).zfill(6)
                records.append({
                    "code": code,
                    "list_date": _coerce_date(row[date_col]),
                    # SH name list carries no industry column -> None there;
                    # pipeline backfills SH industries from yjbb/Sina mapping.
                    "industry": (str(row[ind_col]).strip() if ind_col and pd.notna(row[ind_col]) else None),
                    "exchange": exchange_from_code(code),
                    # Board identity from the code prefix: the name lists only
                    # prove an exchange, and a mis-split STAR/ChiNext list
                    # must never relabel a 300/301/302 or 688/689 code.
                    "board": resolve_board(code),
                })
        else:
            errors.append("sh: missing expected columns")

    # --- SZ ---
    sz_fn = fetchers.get("sz") or (lambda: ak.stock_info_sz_name_code(symbol="A股列表"))
    df, err = _safe_call(sz_fn, host="eastmoney")
    if err:
        errors.append(err)
    elif df is not None:
        code_col = next((c for c in ("A股代码", "证券代码") if c in df.columns), None)
        date_col = next((c for c in df.columns if "上市日期" in c), None)
        if code_col and date_col:
            for _, row in df.iterrows():
                code = str(row[code_col]).zfill(6)
                ind = row.get("所属行业")
                records.append({
                    "code": code,
                    "list_date": _coerce_date(row[date_col]),
                    "industry": str(ind).strip() if ind is not None and pd.notna(ind) else None,
                    "exchange": exchange_from_code(code),
                    "board": resolve_board(code),
                })
        else:
            errors.append("sz: missing expected columns")

    # --- BJ ---
    bj_fn = fetchers.get("bj") or (lambda: ak.stock_info_bj_name_code())
    df, err = _safe_call(bj_fn, host="eastmoney")
    if err:
        errors.append(err)
    elif df is not None:
        code_col = next((c for c in ("证券代码", "A股代码") if c in df.columns), None)
        date_col = next((c for c in df.columns if "上市日期" in c), None)
        if code_col and date_col:
            for _, row in df.iterrows():
                code = str(row[code_col]).zfill(6)
                ind = row.get("所属行业")
                records.append({
                    "code": code,
                    "list_date": _coerce_date(row[date_col]),
                    "industry": str(ind).strip() if ind is not None and pd.notna(ind) else None,
                    "exchange": exchange_from_code(code),
                    "board": resolve_board(code),
                })
        else:
            errors.append("bj: missing expected columns")

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        # No dedup: every code from the exchange name lists is unique (live
        # 5574 codes). Deduping on (list_date, industry, exchange) silently
        # collapsed ~1700 stocks that share a list_date+exchange because
        # ``industry`` is None on the SH name list.
        df = df.set_index("code")

    latency = time.monotonic() - t0
    if not records:
        st = SourceStatus(
            name=_NAME, status="failed", rows=0, latency_s=latency,
            error="; ".join(errors)[:500] or "no rows",
        )
        return df, st
    # Coverage gate: any failed list, or universe coverage < 98%, is degraded
    # with the per-list reason (a silent short list hides delistings/new
    # listings and corrupts the exclusions).
    per_list = {
        "sh": sum(1 for r in records if str(r["code"]).startswith(("60", "68"))),
        "sz": sum(1 for r in records if str(r["code"]).startswith(("00", "30"))),
        "bj": sum(1 for r in records if str(r["code"]).startswith(("43", "83", "87", "92"))),
    }
    # hs_a coverage gate (CONTRACT): the universe's hs_a codes must be covered
    # by a list_date; any gap is a stock whose new_listing age cannot be proven
    # (it falls back to K-line bars, which is slower and less precise). <98% ->
    # degraded with the uncovered count.
    hsa_records = [r for r in records if not str(r["code"]).startswith(("43", "83", "87", "92"))]
    hsa_universe = len(hsa_records)
    covered_hsa = sum(1 for r in hsa_records if r["list_date"] is not None)
    uncovered_hsa = hsa_universe - covered_hsa
    coverage_pct = (covered_hsa / hsa_universe * 100) if hsa_universe else 100.0

    missing = [k for k, v in per_list.items() if v == 0 and len(records) > 0]
    counts_txt = ", ".join(f"{k}={v}" for k, v in per_list.items())
    coverage_txt = (f"hs_a_coverage {coverage_pct:.2f}% "
                    f"({covered_hsa}/{hsa_universe}, uncovered {uncovered_hsa})")
    status = "ok"
    if errors or missing or coverage_pct < 98.0:
        status = "degraded"
    st = SourceStatus(
        name=_NAME,
        status=status,
        rows=len(df),
        latency_s=latency,
        error="; ".join(errors)[:500] or None,
        detail=("sources: sh+star/sz/bj akshare name lists; "
                f"per-list {counts_txt}"
                + (f"; missing lists: {', '.join(missing)}" if missing else "")
                + f"; {coverage_txt}"),
        as_of=_dt.date.today().isoformat(),
    )
    return df, st
