"""Bulk fundamentals: EM datacenter quarterly boards (yjbb + zcfz + xjll),
latest published period per stock with per-stock fallback to earlier quarters.

Speed design (CONTRACT: whole scan well under 30 min; cold <= 150 s, warm <= 10 s):
* Only report periods whose period-end date is <= ``as_of`` are requested,
  newest first — future periods are never fetched.
* Each period's tables are cached as pickles under
  ``get_runtime_root()/market_scanner/cache/fundamentals/``: closed past
  periods get a long TTL (30 days — closed reports don't change), the newest
  period gets ~12 h so filings landing today are picked up.
* An empty period board (a not-yet-published quarter) is detected cheaply
  (zero rows from a completed fetch) and simply skipped — it does NOT degrade
  the status. ``degraded`` is reserved for a needed table that actually failed
  (network/parse error).
* Earlier periods are fetched only while stocks are still missing from the
  newest non-empty period's coverage.

TODO_R2 item D — recurring (扣非) profit growth. The ``mainfin`` table
(EM datacenter report ``RPT_F10_FINANCE_MAINFINADATA``, reached through
``http_get_json`` on datacenter-web — **never** push2, which is blocked on
this network) carries ``DJD_DEDUCTDPNP_YOY``, the YoY of recurring
deduct-parent net profit, for the whole market at one report period
(verified 2026-10-13 for 2026-06-30: 13,470 rows covering 100% of the
11,448 codes the yjbb board returned). It is fetched once per period, cached
like the other boards, and merged into the result's ``deduct_profit_yoy``
column. Scoring (``market_scanner.scoring._profit_growth_input``) prefers
that column over reported net-profit YoY because reported profit is
contaminated by one-off items (fair-value and investment gains, asset
disposals); when the column is missing it damps the growth input instead.

A mainfin failure is **not** an error for the whole board: the frame carries
no roe/eps, so per-stock fallback still works from yjbb, and the status only
notes the table miss.
"""

from __future__ import annotations

import datetime as _dt
import logging
import pickle
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.market_scanner.sources._common import meter_akshare_call
from src.market_scanner.sources.status import SourceStatus

logger = logging.getLogger(__name__)

_NAME = "fundamentals"

_QUARTER_ENDS = ["0331", "0630", "0930", "1231"]

_CACHE_SUBDIR = Path("market_scanner") / "cache" / "fundamentals"
_CLOSED_TTL_S = 30 * 24 * 3600.0   # closed past periods: 30 days
_CURRENT_TTL_S = 12 * 3600.0       # newest (possibly still-filing) period: 12 h


def _periods_back(as_of: date, depth: int) -> list[str]:
    """Report periods whose period-end is <= as_of, newest first (YYYYMMDD).

    Only *complete* quarters are eligible: a quarter ending after ``as_of``
    can't have been reported, and the quarter that just ended this week is
    also skipped until reports actually publish (the boards return empty
    until then, which the fetch treats as a cheap skip — but we don't even
    request a quarter that ended within the last 45 days, the typical filing
    window, to avoid wasted calls).
    """
    out: list[str] = []
    y = as_of.year
    for offset in range(0, depth + 2):
        for q in reversed(_QUARTER_ENDS):
            period = f"{y - offset}{q}"
            end = _dt.datetime.strptime(period, "%Y%m%d").date()
            # 45-day filing window: a quarter that ended very recently is
            # mostly unreported; skipping it avoids an expensive near-empty
            # board fetch. Older periods are eligible immediately.
            if end <= as_of and (as_of - end).days >= 45:
                out.append(period)
            if len(out) >= depth:
                return out
    return out


def _is_newest_period(period: str, as_of: date) -> bool:
    """True when this is the newest quarter-end not yet in the past."""
    return period == _periods_back(as_of, depth=1)[0]


def _cache_path(period: str, table: str) -> Path:
    """Pickle path for one (period, table)."""
    from src.config.paths import get_runtime_root

    directory = get_runtime_root() / _CACHE_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{period}_{table}.pkl"


def _cache_get(period: str, table: str, as_of: date) -> pd.DataFrame | None | "MISS":
    """Cached frame, None (cached-empty marker), or MISS when absent/stale."""
    path = _cache_path(period, table)
    try:
        if not path.exists():
            return _MISS
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        saved_at, frame = payload["saved_at"], payload["frame"]
        age = time.time() - saved_at
        ttl = _CURRENT_TTL_S if _is_newest_period(period, as_of) else _CLOSED_TTL_S
        if age > ttl:
            return _MISS
        return frame  # may legitimately be None (cached empty period)
    except Exception:  # noqa: BLE001 — cache is best-effort
        return _MISS


class _Miss:
    """Sentinel distinguishing 'no cache entry' from a cached-empty period."""


_MISS = _Miss()


def _cache_put(period: str, table: str, frame: pd.DataFrame | None) -> None:
    """Best-effort cache write; failures are logged, never raised."""
    try:
        path = _cache_path(period, table)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"saved_at": time.time(), "frame": frame}, fh)
    except OSError:
        logger.exception("fundamentals cache write failed: %s/%s", period, table)


def _fetch_board(
    fn: Callable, period: str, table: str, as_of: date, errors: list[str],
) -> pd.DataFrame | None:
    """One board fetch with cache; returns None on failure (never raises).

    A completed-but-empty fetch (unpublished period) is cached as ``None``
    and returned as ``None`` WITHOUT recording an error.
    """
    cached = _cache_get(period, table, as_of)
    if cached is not _MISS:
        return cached
    try:
        df = meter_akshare_call(fn, date=period, host="eastmoney")
    except Exception as exc:  # noqa: BLE001 — a real fetch failure is degraded
        errors.append(f"{period}/{table}: {type(exc).__name__}: {exc}"[:160])
        logger.info("fundamentals board %s/%s failed: %s", period, table, exc)
        return None
    if df is None or df.empty:
        # Not published (yet) — cheap detection, cached, not an error.
        _cache_put(period, table, None)
        return None
    _cache_put(period, table, df)
    return df


def _normalize_period_code(period: str) -> str:
    """'20260630' -> '2026Q2' label for the report_period column."""
    q = {"0331": "Q1", "0630": "Q2", "0930": "Q3", "1231": "Q4"}.get(period[4:])
    return f"{period[:4]}{q}" if q else period


def _normalize_period_iso(period: str) -> str:
    """'20260630' -> '2026-06-30' (what the mainfin report filters on)."""
    return f"{period[:4]}-{period[4:6]}-{period[6:8]}"


def _fetch_mainfin(
    period: str, as_of: date, errors: list[str], get_json: Callable | None = None,
) -> pd.DataFrame | None:
    """Whole-market recurring (扣非) net-profit YoY for one report period.

    TODO_R2 item D — see the module docstring. This is a *side table*: it
    only fills the ``deduct_profit_yoy`` field and never decides which
    report period a stock's row comes from, so a failure here costs the
    growth input only and scoring damps instead.

    Args:
        period: ``YYYYMMDD`` report period-end.
        as_of: Scan reference date (drives the cache TTL).
        errors: Collector for the human-readable error list.
        get_json: Injectable JSON GET for tests (default: throttled HTTP).

    Returns:
        Frame indexed by ``code`` with a ``deduct_profit_yoy`` column, or
        ``None`` when the table is unavailable.
    """
    cached = _cache_get(period, "mainfin", as_of)
    if cached is not _MISS:
        return _mainfin_frame(cached)
    get_json = get_json or _default_get_json
    base_filter = (
        '(SECURITY_TYPE_CODE in ("058001001","058001008"))'
        f"(REPORT_DATE='{_normalize_period_iso(period)}')"
    )
    rows: list[dict] = []
    page = 1
    try:
        while True:
            payload = get_json(
                _MAINFIN_URL,
                params={
                    "sortColumns": "SECURITY_CODE",
                    "sortTypes": "1",
                    "pageSize": "5000",
                    "pageNumber": str(page),
                    "reportName": _MAINFIN_REPORT,
                    "columns": "SECURITY_CODE,DJD_DEDUCTDPNP_YOY",
                    "filter": base_filter,
                },
            )
            if not payload.get("success") or not payload.get("result"):
                raise RuntimeError(str(payload.get("message"))[:120])
            result = payload["result"]
            rows.extend(result.get("data") or [])
            pages = int(result.get("pages") or 1)
            if page >= pages:
                break
            page += 1
    except Exception as exc:  # noqa: BLE001 — never raise out of a source
        errors.append(f"{period}/mainfin: {type(exc).__name__}: {exc}"[:160])
        logger.info("fundamentals mainfin %s failed: %s", period, exc)
        return None
    if not rows:
        # Not published (yet) — cheap detection, cached, not an error.
        _cache_put(period, "mainfin", None)
        return None
    frame = pd.DataFrame.from_records(rows)
    _cache_put(period, "mainfin", frame)
    return _mainfin_frame(frame)


def _mainfin_frame(
    raw: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Raw mainfin rows -> ``code`` / ``deduct_profit_yoy`` frame (indexed)."""
    if raw is None or raw.empty:
        return None
    if "SECURITY_CODE" not in raw.columns or "DJD_DEDUCTDPNP_YOY" not in raw.columns:
        return None
    out = pd.DataFrame(
        {
            "code": raw["SECURITY_CODE"].astype(str).str.zfill(6),
            "deduct_profit_yoy": pd.to_numeric(
                raw["DJD_DEDUCTDPNP_YOY"], errors="coerce"
            ),
        }
    )
    out = out.dropna(subset=["deduct_profit_yoy"]).drop_duplicates("code")
    return out.set_index("code") if not out.empty else None


#: Where item D's recurring-profit data comes from (datacenter-web, NOT push2 —
#: push2 is blocked on this network; see the module docstring).
_MAINFIN_REPORT = "RPT_F10_FINANCE_MAINFINADATA"
_MAINFIN_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def _default_get_json(url: str, *, params: dict[str, Any]) -> Any:
    """Throttled datacenter JSON GET (injectable for tests)."""
    from src.market_scanner.sources._common import http_get_json

    return http_get_json(url, host_key="eastmoney", params=params)


def _fetch_mainfin_for(
    fetch: Callable[[str, date, list[str]], pd.DataFrame | None],
    period: str,
    as_of: date,
    errors: list[str],
) -> pd.DataFrame | None:
    """Call the item-D board fetcher; a failure is noted, never raised."""
    try:
        return fetch(period, as_of, errors)
    except Exception as exc:  # noqa: BLE001 — a side table must not kill a board
        errors.append(f"{period}/mainfin: {type(exc).__name__}: {exc}"[:160])
        logger.info("fundamentals mainfin %s raised: %s", period, exc)
        return None


_FUND_COLUMNS = [
    "report_period", "roe", "eps", "bps", "revenue_yoy", "profit_yoy",
    "gross_margin", "ocf_per_share", "debt_ratio", "industry",
    "deduct_profit_yoy",
]


def fetch_fundamentals(
    as_of: date,
    yjbb_fn: Callable | None = None,
    zcfz_fn: Callable | None = None,
    xjll_fn: Callable | None = None,
    max_periods: int = 4,
    mainfin_fn: Callable | None = None,
) -> tuple[pd.DataFrame, SourceStatus]:
    """Latest published fundamentals per stock, falling back per stock.

    Args:
        as_of: Reference date; only periods with period-end <= as_of are used.
        yjbb_fn: Injectable for ``ak.stock_yjbb_em`` (tests use fixtures).
        zcfz_fn: Injectable for ``ak.stock_zcfz_em``.
        xjll_fn: Injectable for ``ak.stock_xjll_em``.
        max_periods: How many published periods may be walked back at most.
        mainfin_fn: Injectable for the item-D recurring-profit board
            (TODO_R2 item D): a callable ``(period, as_of, errors)`` returning
            a ``code`` / ``deduct_profit_yoy`` frame or ``None``. Defaults to
            :func:`_fetch_mainfin`. Pass ``False`` to skip the board entirely
            (offline tests that supply their own growth columns).

    Returns:
        ``(df, st)`` — frame indexed by ``code`` with the contract columns.
        ``report_period`` is a ``YYYYQn`` string. Each stock appears once,
        holding its newest period that actually returned data. Status is
        ``degraded`` only when a needed table actually failed — an
        unpublished/empty period is normal and stays ``ok``.
    """
    import akshare as ak

    t0 = time.monotonic()
    yjbb_fn = yjbb_fn or ak.stock_yjbb_em
    zcfz_fn = zcfz_fn or ak.stock_zcfz_em
    xjll_fn = xjll_fn or ak.stock_xjll_em
    if mainfin_fn is False:
        _mainfin_fetch = None
    else:
        _mainfin_fetch = mainfin_fn or _fetch_mainfin

    periods = _periods_back(as_of, depth=max_periods)
    errors: list[str] = []

    # period -> {code: partial record}
    per_period: dict[str, dict[str, dict]] = {}
    fetched_any = False

    def _key(df: pd.DataFrame) -> str:
        for c in ("股票代码", "证券代码"):
            if c in df.columns:
                return c
        return df.columns[1] if len(df.columns) > 1 else df.columns[0]

    def _num(df: pd.DataFrame, row: pd.Series, *names: str) -> float | None:
        for n in names:
            if n in df.columns:
                v = row[n]
                try:
                    if pd.isna(v):
                        return None
                    return float(str(v).replace(",", "").replace("%", ""))
                except (ValueError, TypeError):
                    return None
        return None

    remaining_codes: set[str] | None = None  # codes still missing coverage
    covered_count = 0
    # The earliest period is "done" once every income-statement stock in the
    # newest period is accounted for. We don't know the true universe size
    # inside this function, so we stop walking back when an earlier period
    # adds zero NEW income-statement stocks (fallback exhausted).
    for period in periods:
        if remaining_codes is not None and covered_count == 0:
            break

        yj = _fetch_board(yjbb_fn, period, "yjbb", as_of, errors)
        zc = _fetch_board(zcfz_fn, period, "zcfz", as_of, errors)
        xj = _fetch_board(xjll_fn, period, "xjll", as_of, errors)
        if yj is None and zc is None and xj is None:
            # Unpublished period or total failure: no new coverage — keep
            # covered_count at 0 so the next iteration breaks if the previous
            # real period also added nothing.
            covered_count = 0
            continue
        fetched_any = True
        bucket: dict[str, dict] = {}

        if yj is not None and not yj.empty:
            kc = _key(yj)
            for _, row in yj.iterrows():
                code = str(row[kc]).zfill(6)
                bucket[code] = {
                    "roe": _num(yj, row, "净资产收益率"),
                    "eps": _num(yj, row, "每股收益"),
                    "bps": _num(yj, row, "每股净资产"),
                    "revenue_yoy": _num(yj, row, "营业总收入-同比增长"),
                    "profit_yoy": _num(yj, row, "净利润-同比增长"),
                    "gross_margin": _num(yj, row, "销售毛利率"),
                    "industry": (str(row["所处行业"]).strip() if "所处行业" in yj.columns and pd.notna(row.get("所处行业")) else None),
                }
        if zc is not None and not zc.empty:
            kc = _key(zc)
            for _, row in zc.iterrows():
                code = str(row[kc]).zfill(6)
                rec = bucket.setdefault(code, {})
                rec["debt_ratio"] = _num(zc, row, "资产负债率")
        if xj is not None and not xj.empty:
            kc = _key(xj)
            for _, row in xj.iterrows():
                code = str(row[kc]).zfill(6)
                rec = bucket.setdefault(code, {})
                rec["ocf_per_share"] = _num(xj, row, "每股经营现金流量")

        per_period[period] = bucket

        # TODO_R2 item D: recurring (扣非) profit YoY for this period. It is a
        # side table keyed by code, so it only ever fills the
        # ``deduct_profit_yoy`` field — never the roe/eps that decide which
        # period a stock's row comes from.
        if _mainfin_fetch is not None:
            mf = _fetch_mainfin_for(
                _mainfin_fetch, period, as_of, errors
            )
            if mf is not None:
                for code, value in mf["deduct_profit_yoy"].items():
                    rec = bucket.setdefault(str(code), {})
                    rec.setdefault("deduct_profit_yoy", float(value))

        # Count how many stocks this period newly covered with a real
        # income-statement row; when an earlier period adds none, the
        # per-stock fallback is exhausted and we stop walking back.
        new_covered = {
            c for c, rec in bucket.items()
            if rec.get("roe") is not None or rec.get("eps") is not None
        } - (remaining_codes or set())
        if remaining_codes is None:
            remaining_codes = set()
        remaining_codes |= new_covered
        covered_count = len(new_covered)

    # Per-stock: newest period that has REAL data for that code. A period
    # bucket counts only when the income-statement board (yjbb) carries the
    # stock — balance/cash-only buckets would otherwise satisfy a stock whose
    # actual latest published report is an earlier quarter.
    seen: set[str] = set()
    records: list[dict] = []
    for period in periods:
        bucket = per_period.get(period, {})
        for code, rec in bucket.items():
            if code in seen or not rec or rec.get("roe") is None and rec.get("eps") is None:
                continue
            seen.add(code)
            out = {"code": code, "report_period": _normalize_period_code(period)}
            out.update({c: rec.get(c) for c in _FUND_COLUMNS if c != "report_period"})
            records.append(out)

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.set_index("code").drop_duplicates()[_FUND_COLUMNS]

    latency = time.monotonic() - t0
    if not fetched_any or df.empty:
        st = SourceStatus(
            name=_NAME, status="failed", rows=len(df), latency_s=latency,
            error="; ".join(errors)[:500] or "no fundamental data for any published period",
        )
        return df, st
    # as_of: report period(s) used with row counts,
    # e.g. "2026-06-30 (11433), 2026-03-31 (88)" (data dates, not fetch time).
    def _period_to_iso(yyyymmdd: str) -> str:
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"

    counts = {p: len(bucket) for p, bucket in per_period.items()}
    as_of = ", ".join(
        f"{_period_to_iso(p)} ({counts[p]})"
        for p in sorted(counts, reverse=True)
    ) or None
    st = SourceStatus(
        name=_NAME,
        status="degraded" if errors else "ok",
        rows=len(df),
        latency_s=latency,
        error="; ".join(errors)[:500] or None,
        detail=(
            "EM datacenter yjbb+zcfz+xjll; per-stock latest published period; "
            f"periods={','.join(per_period) or 'none'}"
        ),
        as_of=as_of,
    )
    return df, st
