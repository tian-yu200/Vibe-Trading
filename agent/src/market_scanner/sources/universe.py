"""Whole-market universe + spot snapshot: Sina hs_a + hs_bjs (CONTRACT section 3)."""

from __future__ import annotations

import datetime as _dt
import logging
import time
from typing import Callable

import pandas as pd

from src.market_scanner.sources._common import (
    _sina_name_flags,
    _to_float,
    resolve_board,
    exchange_from_code,
    sina_count,
    sina_fetch_page,
    symbol_from_code,
)
from src.market_scanner.sources.status import SourceStatus

logger = logging.getLogger(__name__)

_NAME = "universe"


def _row_to_record(row: dict) -> dict | None:
    """Convert one Sina Market_Center row to the contract universe schema."""
    code = str(row.get("code", "")).zfill(6)
    if not code or code == "None":
        return None
    name = str(row.get("name", ""))
    is_st, is_delisting = _sina_name_flags(name)
    price = _to_float(row.get("trade"))
    return {
        "code": code,
        "symbol": symbol_from_code(code),
        "name": name,
        "exchange": exchange_from_code(code),
        # Code prefix is authoritative: Sina's hs_a list carries no board and
        # never overrides it (300/301/302 -> chinext, 688/689 -> star).
        "board": resolve_board(code),
        "price": price,
        "pct_chg": _to_float(row.get("changepercent")),
        # Sina's ``settlement`` IS the previous close (前收盘价). The K-line
        # append path (kline._append_from_snapshot) needs it both as the
        # corporate-action guard (cached last close vs settlement) and for
        # open/high/low. Drop it and every snapshot append is refused.
        "settlement": _to_float(row.get("settlement")),
        "open": _to_float(row.get("open")),
        "high": _to_float(row.get("high")),
        "low": _to_float(row.get("low")),
        "volume": _to_float(row.get("volume")),
        "amount": _to_float(row.get("amount")),
        "pe": _to_float(row.get("per")),
        "pb": _to_float(row.get("pb")),
        # Sina cap units are 10k CNY -> CNY (CONTRACT section 2).
        "total_mv": None if _to_float(row.get("mktcap")) is None else _to_float(row.get("mktcap")) * 1e4,
        "float_mv": None if _to_float(row.get("nmc")) is None else _to_float(row.get("nmc")) * 1e4,
        "turnover_rate": _to_float(row.get("turnoverratio")),
        "is_st": is_st,
        "is_delisting": is_delisting,
        # Sina serves suspended rows with price 0.000; volume 0 corroborates.
        "is_suspended": price == 0.0,
    }


_UNIVERSE_COLUMNS = [
    "symbol", "name", "exchange", "board", "price", "pct_chg", "volume",
    "amount", "pe", "pb", "total_mv", "float_mv", "turnover_rate",
    "is_st", "is_delisting", "is_suspended", "settlement", "open", "high", "low",
]


def fetch_universe(fetch: Callable | None = None, max_pages: int = 200) -> tuple[pd.DataFrame, SourceStatus]:
    """Whole-market A-share spot snapshot (SH+SZ via hs_a, BSE via hs_bjs).

    Args:
        fetch: Injectable ``(url, params) -> str`` returning the JSON body
            (tests pass a fixture-backed stub).
        max_pages: Safety cap on pages per node.

    Returns:
        ``(df, st)`` — frame indexed by ``code`` with the contract universe
        columns; status ``ok`` / ``degraded`` (one node failed) / ``failed``.
    """
    t0 = time.monotonic()
    records: list[dict] = []
    errors: list[str] = []

    for node in ("hs_a", "hs_bjs"):
        node_rows = 0
        try:
            total = sina_count(node, fetch=fetch)
        except Exception as exc:  # noqa: BLE001 — converted to degraded below
            total = 0
            errors.append(f"{node}:count:{type(exc).__name__}: {exc}"[:200])
            logger.warning("sina count %s failed: %s", node, exc)
        pages = max(1, min(max_pages, (total + 99) // 100) if total else max_pages)
        page = 1
        while page <= pages:
            try:
                rows = sina_fetch_page(node, page, fetch=fetch)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{node}:page{page}:{type(exc).__name__}: {exc}"[:200])
                logger.warning("sina page %s/%s failed: %s", node, page, exc)
                break
            if not rows:
                break
            for row in rows:
                rec = _row_to_record(row)
                if rec is not None:
                    records.append(rec)
                    node_rows += 1
            if len(rows) < 100:
                break
            page += 1
        if node_rows:
            logger.info("universe node %s: %d rows", node, node_rows)

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.set_index("code").drop_duplicates()
        df = df[_UNIVERSE_COLUMNS]

    latency = time.monotonic() - t0
    if not records:
        st = SourceStatus(
            name=_NAME, status="failed", rows=0, latency_s=latency,
            error="; ".join(errors)[:500] or "no rows returned",
        )
        return df, st
    # as_of: the snapshot's session date — 'intraday' when 09:15–15:00 on a
    # trading day (CN time); before 09:15 / after 15:00 it is the completed
    # session the snapshot holds (prev session before 09:15, today after 15:00).
    now = _dt.datetime.now()
    if now.weekday() < 5 and _dt.time(9, 15) <= now.time() <= _dt.time(15, 0):
        as_of = f"intraday {now.date().isoformat()}"
    elif now.time() < _dt.time(9, 15):
        as_of = (now.date() - _dt.timedelta(days=1)).isoformat()
    else:
        as_of = now.date().isoformat()
    st = SourceStatus(
        name=_NAME,
        status="degraded" if errors else "ok",
        rows=len(df),
        latency_s=latency,
        error="; ".join(errors)[:500] or None,
        detail=f"nodes hs_a+hs_bjs; BSE rows={int((df['board'] == 'bse').sum()) if not df.empty else 0}",
        as_of=as_of,
    )
    return df, st
