"""Ordered exclusion funnel for the A-share scanner (CONTRACT section 5).

Each stock is counted **once**, at the first rule it fails. The funnel order is
fixed by the contract and must not be reordered:

1. ``universe``
2. ``excluded:bse_unsupported``
3. ``excluded:st``
4. ``excluded:delisting``
5. ``excluded:suspended``
6. ``excluded:new_listing``
7. ``excluded:board_disabled``
8. ``excluded:micro_cap``
9. ``excluded:illiquid`` (20-day average amount from K-lines, else the Sina
   snapshot day amount as a proxy; proxy passes carry
   ``liquidity_from_snapshot``)
9b. ``insufficient_data:liquidity`` (no usable K-line amount AND no snapshot
    day amount — the liquidity cut is fail-closed, never passed by default)
10. ``insufficient_data:fundamentals``
11. ``excluded:shrinking`` (negative revenue and recurring-profit growth)
12. ``insufficient_data:trend`` (K-line missing, failed or too short)

``scored`` / ``finalists`` / ``ai_demoted`` / ``watchlist`` are appended by
:mod:`src.market_scanner.pipeline`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from .config import ScanConfig
from .models import (
    EXCLUSION_STAGES,
    STAGE_UNIVERSE,
    fmt_cny,
    fmt_pct,
    make_funnel_entry,
    normalize_code,
)
from .sources._common import resolve_board

logger = logging.getLogger(__name__)

#: Reason string for a BSE (北交所) listing, stage 2 of the funnel. Kept as a
#: module constant because the same verdict is produced by :func:`is_bse_code`
#: and by stage 2 of the scan funnel; both must stay in lock-step.
BSE_UNSUPPORTED_REASON = "excluded:bse_unsupported: BSE (北交所) listing not supported yet"


def is_bse_code(code: Any, board: Any = None,
                   resolved: Any = None) -> bool:
    """True for a BSE (北交所) listing, which the pipeline cannot price.

    The single shared predicate for stage 2 of the exclusion funnel
    (``excluded:bse_unsupported``) and for the warm-up's universe filter:
    the scan drops BSE codes, so warming their K-lines would spend the rate
    budget on data the scan never reads.

    Args:
        code: 6-digit code (prefix wins over ``board``).
        board: Optional vendor-supplied board string (Sina carries none).
        resolved: Optional already-resolved board (skips the second call,
            used by :func:`_early_reason` which needs ``board`` later).

    Returns:
        True when :func:`resolve_board` reports ``"bse"``.
    """
    if resolved is None:
        resolved = resolve_board(code, board)
    return str(resolved) == "bse"


#: Bars of K-line history that prove a stock has been listed for more than
#: one trading year (~250 trading days), used as the ``new_listing`` age
#: fallback when ``list_date`` is unknown.
_KLINE_AGE_FALLBACK_BARS = 250

#: New funnel stage for stocks whose liquidity cannot be judged from any
#: source (fewer than 20 usable K-line amount bars AND no Sina snapshot day
#: amount). Emitted between ``excluded:illiquid`` and
#: ``insufficient_data:fundamentals``. The pipeline counts it into
#: ``coverage["insufficient"]`` by its ``insufficient_data:`` prefix, so no
#: pipeline change is needed for the coverage math.
STAGE_INSUFFICIENT_LIQUIDITY = "insufficient_data:liquidity"

#: Item flag for stocks that passed the liquidity rule via the Sina snapshot
#: day-amount proxy (codes in
#: :attr:`ExclusionResult.liquidity_snapshot_fallback`). The pipeline assigns
#: this flag; this module only reports the codes.
LIQUIDITY_SNAPSHOT_FLAG = "liquidity_from_snapshot"

#: K-line bars with usable ``amount`` needed to judge liquidity from history
#: (rule (a) below); fewer usable bars fall back to the Sina snapshot day
#: amount as a proxy.
_LIQUIDITY_MIN_HISTORY_BARS = 20

#: Fundamental fields that must be *present* (at least one non-NaN) for a
#: stock to count as having fundamentals. Missing fundamentals is an
#: eligibility cut (CONTRACT §6), not a neutral-score situation.
FUNDAMENTAL_REQUIRED_FIELDS: tuple[str, ...] = (
    "roe",
    "gross_margin",
    "revenue_yoy",
    "profit_yoy",
    "debt_ratio",
)


@dataclass
class ExclusionResult:
    """Outcome of one funnel pass over the universe."""

    #: ``code -> exclusion_reason``; ``None`` for stocks that passed.
    reasons: dict[str, str | None] = field(default_factory=dict)
    #: Funnel entries for the exclusion stages (universe .. insufficient_data:trend).
    funnel: list[dict[str, Any]] = field(default_factory=list)
    #: Codes that passed every rule, in universe order.
    passed: list[str] = field(default_factory=list)
    #: Codes whose ``new_listing`` verdict came from the kline-history
    #: fallback (list_date unknown, but bar-count proved age). Surfaced on
    #: the funnel stage so the fallback volume is auditable.
    new_listing_kline_fallback: list[str] = field(default_factory=list)
    #: Codes that passed the liquidity rule via the Sina snapshot day-amount
    #: proxy (< 20 usable K-line amount bars). The pipeline turns these into
    #: :data:`LIQUIDITY_SNAPSHOT_FLAG` item flags; the ``excluded:illiquid``
    #: funnel note counts how many stocks were decided via the fallback.
    liquidity_snapshot_fallback: list[str] = field(default_factory=list)

    def reason_for(self, code: str) -> str | None:
        """Return the exclusion reason for ``code`` (None when it passed)."""
        return self.reasons.get(normalize_code(code))


def ensure_code_index(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return ``df`` indexed by the 6-digit ``code`` string, or ``None``."""
    if df is None:
        return None
    if df.empty:
        return df.copy()
    frame = df.copy()
    if frame.index.name == "code":
        frame.index = frame.index.map(normalize_code)
        return frame
    if "code" in frame.columns:
        frame["code"] = frame["code"].map(normalize_code)
        return frame.set_index("code")
    frame.index = frame.index.map(normalize_code)
    return frame


def apply_early_rules(
    universe_df: pd.DataFrame,
    config: ScanConfig,
    listing_df: pd.DataFrame | None = None,
    trade_date: date | None = None,
    klines: dict[str, pd.DataFrame] | None = None,
    kline_fallback_codes: list[str] | None = None,
) -> pd.Series:
    """Apply the data-free rules (stages 2-8) to every universe row.

    Used by the pipeline to shrink the kline/fundamentals fetch set, and by
    :func:`run_exclusions` so the two cannot drift apart. Returns a Series
    indexed by code: the exclusion reason string, or ``None`` when the rule
    passes.

    ``new_listing`` semantics (CONTRACT section 5): a missing ``list_date``
    must NOT mean "new listing". When the listing date is unknown, fall back
    to the cached K-line history length (``>= 250`` bars proves the stock
    has been listed for more than a trading year); only when BOTH the date
    and the history are unknown/short does the stock fail — and that failure
    is classified as ``insufficient_data`` (cannot determine age), not
    ``new_listing``. ``kline_fallback_codes`` (when provided) collects the
    codes whose age was proven by this fallback, for funnel reporting.
    """
    uni = ensure_code_index(universe_df)
    listing = ensure_code_index(listing_df)
    as_of = trade_date or date.today()
    klines = klines or {}
    reasons: dict[str, str | None] = {}

    for code in uni.index:
        row = uni.loc[code]
        reason = _early_reason(
            row,
            code,
            config,
            listing,
            as_of,
            klines,
            kline_fallback_codes,
        )
        reasons[code] = reason
    return pd.Series(reasons, dtype=object)


def run_exclusions(
    universe_df: pd.DataFrame,
    *,
    config: ScanConfig,
    listing_df: pd.DataFrame | None = None,
    klines: dict[str, pd.DataFrame] | None = None,
    fundamentals_df: pd.DataFrame | None = None,
    trade_date: date | None = None,
) -> ExclusionResult:
    """Run the full ordered funnel (CONTRACT section 5).

    Args:
        universe_df: Spot snapshot indexed by ``code``.
        config: Scan settings (thresholds, board toggles, min kline bars).
        listing_df: ``list_date`` / ``industry`` per code (may be partial).
        klines: ``code -> ohlcv frame`` (ascending, forward-adjusted).
        fundamentals_df: Latest published fundamentals per code.
        trade_date: As-of date for the ``new_listing`` rule; defaults to today.

    Returns:
        :class:`ExclusionResult` with a per-stock reason (None when passed),
        funnel entries and the ordered list of surviving codes.
    """
    uni = ensure_code_index(universe_df)
    listing = ensure_code_index(listing_df)
    funds = ensure_code_index(fundamentals_df)
    as_of = trade_date or date.today()
    klines = klines or {}

    codes = list(uni.index)
    result = ExclusionResult(reasons={code: None for code in codes})
    remaining = list(codes)
    removed_total = 0
    result.funnel.append(make_funnel_entry(STAGE_UNIVERSE, len(codes), 0))

    def _cut(stage: str, fail: dict[str, str], note: str | None = None) -> None:
        nonlocal remaining, removed_total
        for code, reason in fail.items():
            result.reasons[code] = reason
        removed_total += len(fail)
        remaining = [c for c in remaining if c not in fail]
        result.funnel.append(
            make_funnel_entry(stage, len(remaining), len(fail), note)
        )

    # --- Stages 2-5: pure data-free rules (bse, st, delisting, suspended) - #
    early = apply_early_rules(uni, config, listing_df=listing, trade_date=as_of)
    for stage in EXCLUSION_STAGES[:4]:  # bse .. suspended
        fail = {
            code: early[code]
            for code in remaining
            if early.get(code) is not None and early[code].startswith(stage)
        }
        _cut(stage, fail)

    # --- Stage 6: new_listing (list_date, with K-line-history fallback) ---- #
    # A missing list_date must NOT mean "new listing". Prove age from the
    # cached K-line history (>= _KLINE_AGE_FALLBACK_BARS bars); only a
    # genuinely young listing fails here. Stocks whose age is still unknown
    # (short/absent K-lines) are left for insufficient_data:fundamentals or
    # insufficient_data:trend below, never labelled new_listing.
    kline_fallback_codes: list[str] = []
    new_listing_fail: dict[str, str] = {}
    for code in remaining:
        reason, proven_by_kline = _new_listing_reason(
            uni.loc[code], code, config, listing, as_of, klines
        )
        if reason is None:
            if proven_by_kline:
                kline_fallback_codes.append(code)
            continue
        if proven_by_kline:
            kline_fallback_codes.append(code)
        new_listing_fail[code] = reason
    fallback_note = (
        f"kline_fallback: {len(kline_fallback_codes)} stocks' age proven by "
        f">= {_KLINE_AGE_FALLBACK_BARS} bars (list_date unknown)"
        if kline_fallback_codes
        else None
    )
    _cut(EXCLUSION_STAGES[4], new_listing_fail, fallback_note)
    result.new_listing_kline_fallback = list(kline_fallback_codes)

    # --- Stages 7-8: board_disabled, micro_cap ----------------------------- #
    for stage in EXCLUSION_STAGES[5:7]:  # board_disabled .. micro_cap
        fail = {
            code: early[code]
            for code in remaining
            if early.get(code) is not None and early[code].startswith(stage)
        }
        _cut(stage, fail)

    # --- Stage 9: illiquid (fail-closed, snapshot fallback) ---------------- #
    # Boss rule: the liquidity cut must never pass a stock by default. Each
    # remaining stock gets exactly one verdict —
    #   (a) >= 20 K-line bars with usable amount -> 20d average vs the bar;
    #   (b) else the Sina snapshot day amount     -> proxy decision, flagged;
    #   (c) else no signal at all                 -> insufficient_data:liquidity.
    illiquid_fail: dict[str, str] = {}
    liquidity_unknown: dict[str, str] = {}
    snapshot_decided: list[str] = []
    snapshot_passed: list[str] = []
    for code in remaining:
        verdict = _liquidity_verdict(
            klines.get(code),
            _finite(_cell(uni.loc[code], "amount")),
            config,
        )
        if verdict.decision == "exclude":
            illiquid_fail[code] = verdict.reason
        elif verdict.decision == "unknown":
            liquidity_unknown[code] = verdict.reason
        elif verdict.decision == "snapshot_pass":
            snapshot_passed.append(code)
        # "pass" (20d average clears the bar): nothing to record.
        if verdict.via_snapshot:
            snapshot_decided.append(code)
    note = (
        f"snapshot_fallback: {len(snapshot_decided)} decided via Sina "
        f"snapshot day amount (no 20d history)"
        if snapshot_decided
        else None
    )
    _cut(EXCLUSION_STAGES[7], illiquid_fail, note)
    result.liquidity_snapshot_fallback = list(snapshot_passed)

    # --- New stage: insufficient_data:liquidity (no signal at all) --------- #
    # Placed right after excluded:illiquid, before fundamentals. Stocks land
    # here only when BOTH the K-line amount (< 20 usable bars) and the Sina
    # snapshot day amount are missing — fail-closed, never passed by default.
    _cut(STAGE_INSUFFICIENT_LIQUIDITY, liquidity_unknown)

    # --- Stage 10: insufficient fundamentals ------------------------------ #
    # NOTE: stage names below are spelled out (not EXCLUSION_STAGES[8]/[9])
    # so the emission order stays correct whether or not
    # models.EXCLUSION_STAGES gains STAGE_INSUFFICIENT_LIQUIDITY.
    fund_fail: dict[str, str] = {}
    for code in remaining:
        if not _has_fundamentals(funds, code):
            fund_fail[code] = (
                "insufficient_data:fundamentals: no published fundamental "
                "fields for the latest reports"
            )
    _cut("insufficient_data:fundamentals", fund_fail)

    # --- Stage 11: shrinking growth gate ---------------------------------- #
    shrinking_fail: dict[str, str] = {}
    if config.exclude_shrinking:
        for code in remaining:
            reason = _shrinking_reason(funds, code)
            if reason is not None:
                shrinking_fail[code] = reason
    _cut("excluded:shrinking", shrinking_fail)

    # --- Stage 12: insufficient trend ------------------------------------- #
    trend_fail: dict[str, str] = {}
    for code in remaining:
        frame = klines.get(code)
        bars = 0 if frame is None or frame.empty else len(frame)
        if bars < config.min_kline_bars:
            detail = "klines missing or fetch failed" if bars == 0 else f"only {bars} bars"
            trend_fail[code] = (
                f"insufficient_data:trend: {detail}, "
                f"need >= {config.min_kline_bars} bars"
            )
    _cut("insufficient_data:trend", trend_fail)

    result.passed = remaining
    return result


# --------------------------------------------------------------------------- #
# Rule helpers
# --------------------------------------------------------------------------- #


def _early_reason(
    row: pd.Series,
    code: str,
    config: ScanConfig,
    listing: pd.DataFrame | None,
    as_of: date,
    klines: dict[str, pd.DataFrame] | None = None,
    kline_fallback_codes: list[str] | None = None,
) -> str | None:
    """Evaluate stages 2-8 against one universe row; None means pass."""
    board = resolve_board(code, _cell(row, "board"))
    if is_bse_code(code, _cell(row, "board"), resolved=board):
        return BSE_UNSUPPORTED_REASON

    if bool(_cell(row, "is_st")):
        return "excluded:st: ST / risk-warning designation"

    if bool(_cell(row, "is_delisting")):
        return "excluded:delisting: delisting-risk flag on the name"

    if bool(_cell(row, "is_suspended")):
        return "excluded:suspended: trading suspended (spot price 0)"

    new_listing_reason, proven_by_kline = _new_listing_reason(
        row, code, config, listing, as_of, klines or {}
    )
    if proven_by_kline and kline_fallback_codes is not None:
        kline_fallback_codes.append(code)
    if new_listing_reason is not None:
        return new_listing_reason

    if str(board) == "star" and not config.include_star:
        return "excluded:board_disabled: STAR board disabled by config"
    if str(board) == "chinext" and not config.include_chinext:
        return "excluded:board_disabled: ChiNext board disabled by config"

    total_mv = _finite(_cell(row, "total_mv"))
    if total_mv is None or total_mv < config.min_total_mv:
        shown = fmt_cny(total_mv) if total_mv is not None else "unavailable"
        return (
            f"excluded:micro_cap: total market cap {shown} "
            f"< required {fmt_cny(config.min_total_mv)}"
        )
    return None


def _new_listing_reason(
    row: pd.Series,
    code: str,
    config: ScanConfig,
    listing: pd.DataFrame | None,
    as_of: date,
    klines: dict[str, pd.DataFrame],
) -> tuple[str | None, bool]:
    """Stage-6 ``new_listing`` decision for one code.

    Returns ``(reason, proven_by_kline)``. ``reason`` is ``None`` when the
    stock passes the stage (either proven old enough, or age still unknown
    and left for a later insufficient_data stage). ``proven_by_kline`` is
    True when the age was decided from the cached K-line history rather than
    a known ``list_date`` — the caller surfaces that count on the funnel.
    """
    list_date = _cell(row, "list_date")
    if list_date is None and listing is not None and code in listing.index:
        list_date = _cell(listing.loc[code], "list_date")
    listed = _as_date(list_date)
    if listed is not None:
        days_listed = (as_of - listed).days
        if days_listed < config.min_listing_days:
            return (
                f"excluded:new_listing: listed {days_listed} days ago "
                f"< {config.min_listing_days} required"
            ), False
        return None, False

    # list_date unknown: fall back to the K-line history length. A stock
    # with >= 250 bars has been trading for more than a year — not new.
    frame = klines.get(code)
    bars = 0 if frame is None or frame.empty else len(frame)
    if bars >= _KLINE_AGE_FALLBACK_BARS:
        return None, True

    # Both signals unknown/short: age cannot be determined. Do NOT claim
    # new_listing; let insufficient_data:fundamentals / insufficient_data:trend
    # further down handle the missing-data case.
    return None, False


@dataclass
class _LiquidityVerdict:
    """One stock's fail-closed liquidity decision.

    ``decision`` is ``"pass"`` (20d average clears the bar), ``"exclude"``
    (cut at ``excluded:illiquid``), ``"snapshot_pass"`` (cleared via the
    Sina snapshot proxy — the caller records the code for
    :data:`LIQUIDITY_SNAPSHOT_FLAG`) or ``"unknown"`` (no signal at all —
    the caller cuts ``insufficient_data:liquidity``). ``reason`` is set for
    ``"exclude"`` / ``"unknown"``; ``via_snapshot`` marks decisions taken
    from the proxy so the funnel note can count them.
    """

    decision: str
    reason: str | None = None
    via_snapshot: bool = False


def _usable_amount_values(frame: pd.DataFrame | None) -> pd.Series:
    """Finite K-line ``amount`` values, oldest first; empty when unusable."""
    if frame is None or frame.empty or "amount" not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame["amount"], errors="coerce").dropna()


def _liquidity_verdict(
    frame: pd.DataFrame | None,
    snapshot_amount: float | None,
    config: ScanConfig,
) -> _LiquidityVerdict:
    """Fail-closed liquidity decision for one stock (boss rule).

    (a) ``>= 20`` K-line bars with usable amount: 20d average vs the bar.
    (b) Else the Sina snapshot day amount (CNY) as a proxy: below the bar
        -> ``excluded:illiquid``; at/above -> pass flagged for the pipeline.
    (c) Neither available -> ``unknown`` (no default pass).

    ``snapshot_amount`` must already be a finite float or ``None``.
    """
    usable = _usable_amount_values(frame)
    if len(usable) >= _LIQUIDITY_MIN_HISTORY_BARS:
        avg = float(usable.tail(_LIQUIDITY_MIN_HISTORY_BARS).mean())
        if avg < config.min_avg_amount_20d:
            return _LiquidityVerdict(
                "exclude",
                f"excluded:illiquid: 20d avg amount {fmt_cny(avg)} "
                f"< required {fmt_cny(config.min_avg_amount_20d)}",
            )
        return _LiquidityVerdict("pass")
    if snapshot_amount is not None:
        if snapshot_amount < config.min_avg_amount_20d:
            return _LiquidityVerdict(
                "exclude",
                "excluded:illiquid: snapshot day amount, no 20d history: "
                f"{fmt_cny(snapshot_amount)} < required "
                f"{fmt_cny(config.min_avg_amount_20d)}",
                via_snapshot=True,
            )
        return _LiquidityVerdict("snapshot_pass", via_snapshot=True)
    return _LiquidityVerdict(
        "unknown",
        "insufficient_data:liquidity: no usable K-line amount "
        "(< 20 bars) and no Sina snapshot day amount",
    )


def _has_fundamentals(funds: pd.DataFrame | None, code: str) -> bool:
    """True when the stock has at least one published fundamental field."""
    if funds is None or funds.empty or code not in funds.index:
        return False
    row = funds.loc[code]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return any(_finite(_cell(row, f)) is not None for f in FUNDAMENTAL_REQUIRED_FIELDS)


def _shrinking_reason(funds: pd.DataFrame | None, code: str) -> str | None:
    """Return the growth-gate reason when both required YoY values are negative."""
    if funds is None or funds.empty or code not in funds.index:
        return None
    row = funds.loc[code]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    revenue_yoy = _finite(_cell(row, "revenue_yoy"))
    if revenue_yoy is None:
        return None
    deduct_profit_yoy = _finite(_cell(row, "deduct_profit_yoy"))
    fallback = deduct_profit_yoy is None
    profit_yoy = (
        _finite(_cell(row, "profit_yoy"))
        if fallback
        else deduct_profit_yoy
    )
    if profit_yoy is None or revenue_yoy >= 0 or profit_yoy >= 0:
        return None
    basis = (
        "reported profit YoY fallback; deduct_profit_yoy unavailable"
        if fallback
        else "deduct_profit_yoy"
    )
    return (
        "excluded:shrinking: revenue YoY "
        f"{fmt_pct(revenue_yoy, signed=True)} and recurring-profit YoY "
        f"{fmt_pct(profit_yoy, signed=True)} < 0 ({basis})"
    )


def _cell(row: pd.Series | None, key: str) -> Any:
    if row is None:
        return None
    try:
        value = row[key]
    except Exception:  # noqa: BLE001 — missing label on a partial row
        return None
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.date()


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number
