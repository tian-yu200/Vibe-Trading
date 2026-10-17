"""Cross-sectional percentile scoring for the A-share scanner (CONTRACT §6).

Design (MEDIUM-TERM horizon, one snapshot per scan):

* Every sub-score is a **0-100 percentile within the scored set**.
* ``score`` is the weighted sum of the three sub-scores with the configured
  weights. **Weights are never renormalised** — a missing flow input keeps its
  share of the weight and contributes the neutral 50 instead. The composite is
  computed in :func:`composite_score`, the only place that arithmetic happens.
* Missing fundamentals or missing/short klines make a stock *not eligible*;
  that cut lives in :mod:`src.market_scanner.exclusions`. Scoring skips such
  codes defensively.
* Northbound holding data lands in metrics and reasons as **info only** and
  never enters any score.

Sub-score component design (weights inside each sub-score sum to 1.0):

``fundamental`` = 0.35·quality + 0.25·growth + 0.25·valuation + 0.15·leverage

* quality    — ROE, gross margin, cash-flow vs earnings (OCF/EPS, clipped to
               [0, 3]); each input percentileed, then averaged.
* growth     — revenue YoY, recurring (扣非) profit YoY, both winsorised at
               set p5/p95, percentileed.
* valuation  — inverse PE (industry-relative when the industry group has at
               least ``MIN_INDUSTRY_GROUP`` scored members, else whole set)
               and inverse PB (whole set). Loss-makers take the bottom slot.
* leverage   — inverse debt ratio (lower debt ranks higher); banks / insurers
               / brokers are NOT penalised for leverage and get neutral 50
               plus the ``leverage_neutral_financial`` flag.

Period comparability (TODO_R2 item C) — ROE, EPS and OCF/share are
*period-level*: a Q1 figure and a half-year figure are not comparable, so
mixing them in one cross-section silently favours short-period reporters.
The rules, in order of preference, are applied in :func:`_quality_inputs`:

1. **TTM** — ``roe_ttm`` when the fundamentals frame carries it (computed in
   :mod:`src.market_scanner.sources.fundamentals` from four quarters' EPS over
   average book value per share), the preferred like-for-like basis.
2. **Annualised** — the reported ROE × the period factor (Q1×4, H1×2,
   Q3×4/3, FY×1). This is what this implementation uses whenever TTM is not
   available.
3. Ranking happens on the *annualised/TTM* values only, and a stock whose
   latest report is older than the latest mandatory period for ``as_of`` gets
   the ``late_filer`` flag, a reason line, and ``LATE_FILER_PENALTY`` points
   off its fundamental raw score (a penalty, not info-only — a stale report is
   a real comparability cost, and the size keeps it a tiebreaker).

One-off gains (TODO_R2 item D) — reported net profit is distorted by
non-recurring items (fair-value and investment gains, asset disposals), so
the growth input prefers **recurring (扣非) net-profit YoY** when the source
provides it whole-market. When it does not, the reported profit-growth input
is *damped*: with revenue YoY below ``REVENUE_DECLINE_DAMP_AT`` the profit
growth used for scoring is capped at ``max(revenue_yoy, 0)``, flagged
``profit_growth_damped``, and explained with a reason line carrying the real
revenue number. Reported profit YoY stays visible in metrics and reasons.

The blended fundamental raw is percentileed once more across the set so the
published sub-score is a true cross-sectional percentile.

``trend`` = 0.30·MA alignment + 0.20·distance-to-250d-high + 0.15·60d relative
return vs CSI 300 + 0.15·120d relative return + 0.10·volume expansion
(20d/60d) + 0.10·20d momentum — percentile inputs where cross-sectional, minus
an overheated-move penalty and an extreme-volatility penalty, then
percentileed across the set.

``flow`` = percentile of the float-cap-scaled THS net-flow blend
0.2·flow_1d + 0.3·flow_3d + 0.5·flow_5d (horizon weights re-balanced over the
horizons actually present) plus a capped Dragon-Tiger net-buy bonus. Missing
flow → 50, flag ``flow_missing``, and a reason line saying so.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import ScanConfig
from .exclusions import ensure_code_index
from .models import fmt_cny, fmt_pct, fmt_ratio, normalize_code

logger = logging.getLogger(__name__)

#: Neutral sub-score / component value for missing-but-optional inputs.
NEUTRAL = 50.0

#: Minimum scored-set size of an industry group before PE is ranked inside the
#: industry instead of the whole set.
MIN_INDUSTRY_GROUP = 5

#: Component weights inside each sub-score (sum to 1.0).
FUNDAMENTAL_COMPONENT_WEIGHTS: dict[str, float] = {
    "quality": 0.35,
    "growth": 0.25,
    "valuation": 0.25,
    "leverage": 0.15,
}
TREND_COMPONENT_WEIGHTS: dict[str, float] = {
    "ma_alignment": 0.30,
    "breakout": 0.20,
    "rel_60d": 0.15,
    "rel_120d": 0.15,
    "volume_expansion": 0.10,
    "momentum_20d": 0.10,
}
FLOW_HORIZON_WEIGHTS: dict[str, float] = {
    "flow_1d": 0.20,
    "flow_3d": 0.30,
    "flow_5d": 0.50,
}

#: 20-day return (percent) above which a stock is flagged overheated and
#: penalised in the trend raw score.
OVERHEAT_RET_20D_PCT = 40.0
OVERHEAT_PENALTY = 20.0

#: 60-bar daily-return standard deviation (percent) above which a stock is
#: flagged extremely volatile and penalised.
EXTREME_VOL_DAILY_PCT = 3.5
EXTREME_VOL_PENALTY = 20.0

#: Dragon-Tiger bonus cap, in blend units (fraction of float market cap).
LHB_BONUS_CAP = 0.02

#: Industry-name fragments that mark banks / insurers / brokers, whose leverage
#: is a business-model feature, not a risk signal.
FINANCIAL_INDUSTRY_KEYWORDS: tuple[str, ...] = (
    "银行",
    "保险",
    "证券",
    "券商",
    "多元金融",
    "bank",
    "insur",
    "broker",
    "securit",
)

# ---------------------------------------------------------------------------
# Period comparability (TODO_R2 item C)
# ---------------------------------------------------------------------------
#: Latest mandatory report period for a given ``as_of`` date, per the filing
#: calendar (TODO_R2 item C): Q1 by 04-30, H1 by 08-31, Q3 by 10-31, FY by
#: 04-30 of the following year. A FY period therefore stays the mandatory one
#: from its 12-31 period end until the next 04-30.
#: ``"Qn"`` (quarter label) -> ``(month, day)`` of the deadline, plus whether
#: the deadline falls in the following year.
MANDATORY_DEADLINES: dict[str, tuple[tuple[int, int], bool]] = {
    "Q1": ((4, 30), False),
    "Q2": ((8, 31), False),
    "Q3": ((10, 31), False),
    "Q4": ((4, 30), True),
}

#: Period-level multipliers: a quarterly figure is NOT an annual figure.
#: Annualising makes the cross-section like-for-like (TODO_R2 item C).
PERIOD_ANNUALISE: dict[str, float] = {
    "Q1": 4.0,
    "Q2": 2.0,
    "Q3": 4.0 / 3.0,
    "Q4": 1.0,
}

#: Fundamental sub-score raw points taken off a stock whose latest report is
#: older than the latest mandatory period. A penalty rather than info-only: a
#: stale report costs comparability, and the size keeps it a tiebreaker.
LATE_FILER_PENALTY = 5.0

# ---------------------------------------------------------------------------
# One-off gains (TODO_R2 item D)
# ---------------------------------------------------------------------------
#: Revenue YoY below this makes the reported profit-growth input untrustworthy
#: (profit growth not supported by revenue), so the profit-growth input is
#: damped to ``max(revenue_yoy, 0)``.
REVENUE_DECLINE_DAMP_AT = -10.0


@dataclass
class ScoredStock:
    """One scored stock: sub-scores, composite, metrics, reasons, flags."""

    code: str
    fundamental: float
    trend: float
    flow: float
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (used by the pipeline envelope builders)."""
        return {
            "code": self.code,
            "fundamental": self.fundamental,
            "trend": self.trend,
            "flow": self.flow,
            "score": self.score,
            "metrics": dict(self.metrics),
            "reasons": list(self.reasons),
            "flags": list(self.flags),
        }


def composite_score(
    sub_scores: dict[str, float | None],
    weights: dict[str, float],
) -> float:
    """Weighted sum of the three sub-scores.

    **Never renormalises** (CONTRACT §6): a ``None`` sub-score contributes
    ``NEUTRAL`` (50) at its configured weight — exactly what a missing flow
    does. Single place the composite arithmetic lives, so the rule is
    testable in isolation.
    """
    total = 0.0
    for key, weight in weights.items():
        value = sub_scores.get(key)
        total += float(weight) * (NEUTRAL if value is None else float(value))
    return round(total, 4)


def score_universe(
    codes: list[str],
    *,
    universe_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame | None = None,
    klines: dict[str, pd.DataFrame] | None = None,
    benchmark_df: pd.DataFrame | None = None,
    flow_df: pd.DataFrame | None = None,
    lhb_df: pd.DataFrame | None = None,
    northbound_df: pd.DataFrame | None = None,
    listing_df: pd.DataFrame | None = None,
    config: ScanConfig | None = None,
    as_of: date | None = None,
) -> dict[str, ScoredStock]:
    """Score every eligible code on all three sub-scores (CONTRACT §6).

    Codes without a fundamentals row or without klines at ``min_kline_bars``
    are skipped (not eligible; the funnel already cut them). The returned dict
    covers the remaining codes, keyed by 6-digit string.

    Args:
        codes: Eligible codes, in universe order.
        universe_df: Spot snapshot (price/pe/pb/caps/turnover) per code.
        fundamentals_df: Latest published fundamentals per code.
        klines: ``code -> ohlcv frame`` (ascending, forward-adjusted).
        benchmark_df: CSI 300 (or configured) benchmark klines.
        flow_df: THS net flow per code (``flow_1d/3d/5d``, net CNY).
        lhb_df: Dragon-Tiger rows (``lhb_count/lhb_net_buy/lhb_last_date``).
        northbound_df: Northbound holdings — **info only, never scored**.
        listing_df: ``list_date`` / ``industry`` per code.
        config: Scan settings; defaults to :class:`ScanConfig`.
        as_of: Reference date for the late-filer rule (item C). ``None``
            disables the rule when the caller does not know it.

    Returns:
        ``code -> ScoredStock``.
    """
    config = config or ScanConfig()
    uni = ensure_code_index(universe_df)
    funds = ensure_code_index(fundamentals_df)
    flow = ensure_code_index(flow_df)
    lhb = ensure_code_index(lhb_df)
    north = ensure_code_index(northbound_df)
    listing = ensure_code_index(listing_df)
    klines = klines or {}
    bench = _benchmark_closes(benchmark_df)

    # Defence in depth: the funnel already dropped these, but scoring must
    # never invent data for an ineligible stock.
    eligible: list[str] = []
    for raw in codes:
        code = normalize_code(raw)
        if _row_at(funds, code) is None:
            continue
        frame = klines.get(code)
        if frame is None or frame.empty or len(frame) < config.min_kline_bars:
            continue
        eligible.append(code)
    if not eligible:
        return {}

    industry_of = {code: _industry_for(code, funds, listing) for code in eligible}
    fund_rows = {code: _fundamental_row(funds, code) for code in eligible}
    uni_rows = {code: _row_at(uni, code) for code in eligible}
    spot = {
        code: {
            "pe": _finite(_cell(uni_rows[code], "pe")),
            "pb": _finite(_cell(uni_rows[code], "pb")),
        }
        for code in eligible
    }
    trends = {code: _trend_components(klines[code], bench) for code in eligible}
    flows = {
        code: _flow_components(code, uni_rows[code], flow, lhb, north)
        for code in eligible
    }

    industry_groups: dict[str, list[str]] = {}
    for code in eligible:
        industry_groups.setdefault(industry_of[code], []).append(code)

    fund_basis = _fundamental_basis(
        eligible, fund_rows, spot, industry_groups, as_of
    )
    trend_raw = _trend_raw_batch(trends)
    trend_pct = _percentiles(trend_raw)
    flow_blend = {code: flows[code]["blend"] for code in eligible}
    flow_pct = _percentiles(flow_blend)

    results: dict[str, ScoredStock] = {}
    for code in eligible:
        components = fund_basis.components[code]
        sub_scores = {
            "fundamental": round(fund_basis.percentiles[code], 4),
            "trend": round(trend_pct[code], 4),
            "flow": round(flow_pct[code], 4),
        }
        score = composite_score(sub_scores, config.weights)
        flags = [
            *flows[code]["flags"],
            *_fund_flags(fund_rows[code], spot[code], industry_of[code], as_of),
            *trends[code]["flags"],
        ]
        metrics = _metrics_for(
            code,
            uni_rows[code],
            fund_rows[code],
            spot[code],
            industry_of[code],
            trends[code],
            flows[code],
            components=components,
        )
        reasons = _reasons_for(
            fund_rows[code],
            spot[code],
            industry_of[code],
            trends[code],
            flows[code],
            components,
            as_of,
        )
        results[code] = ScoredStock(
            code=code,
            fundamental=sub_scores["fundamental"],
            trend=sub_scores["trend"],
            flow=sub_scores["flow"],
            score=score,
            metrics=metrics,
            reasons=reasons,
            flags=_dedupe(flags),
        )
    return results


# --------------------------------------------------------------------------- #
# Fundamental components
# --------------------------------------------------------------------------- #


@dataclass
class _FundBasis:
    """Percentiles plus the raw component values used for reasons/metrics."""

    percentiles: dict[str, float]
    components: dict[str, dict[str, float | None]]


def _fundamental_basis(
    codes: list[str],
    fund_rows: dict[str, dict[str, Any]],
    spot: dict[str, dict[str, float | None]],
    industry_groups: dict[str, list[str]],
    as_of: date | None = None,
) -> _FundBasis:
    """Percentile every fundamental input across the scored set.

    Item C: ROE is ranked on its like-for-like (TTM, else annualised) value.
    Item D: the profit-growth input is recurring (扣非) profit YoY when the
    source provides it, else the damped reported figure.
    """
    roe_basis: dict[str, str] = {}
    growth_basis: dict[str, str] = {}
    roe_vals: dict[str, float | None] = {}
    profit_growth: dict[str, float | None] = {}
    for code in codes:
        roe_value, basis = _roe_like_for_like(fund_rows[code])
        roe_vals[code] = roe_value
        roe_basis[code] = basis
        growth_value, growth_kind = _profit_growth_input(fund_rows[code])
        profit_growth[code] = growth_value
        growth_basis[code] = growth_kind

    raw: dict[str, dict[str, float | None]] = {
        "roe": roe_vals,
        "gross_margin": {c: fund_rows[c].get("gross_margin") for c in codes},
        "cash_quality": {c: _cash_quality(fund_rows[c]) for c in codes},
        "revenue_yoy": _winsorise({c: fund_rows[c].get("revenue_yoy") for c in codes}),
        "profit_growth": _winsorise(profit_growth),
        "pe_invert": {c: _pe_invert(spot[c].get("pe")) for c in codes},
        "pb_invert": {c: _pb_invert(spot[c].get("pb")) for c in codes},
        "debt_low": {c: _neg(fund_rows[c].get("debt_ratio")) for c in codes},
    }
    pct = {name: _percentiles(values) for name, values in raw.items()}

    # PE is industry-relative when the industry group is big enough.
    pe_pct: dict[str, float] = {}
    fallback_codes: set[str] = set()
    for members in industry_groups.values():
        if len(members) >= MIN_INDUSTRY_GROUP:
            pe_pct.update(_percentiles({c: raw["pe_invert"][c] for c in members}))
        else:
            fallback_codes.update(members)
    if fallback_codes:
        whole_pe = _percentiles({c: raw["pe_invert"][c] for c in codes})
        for code in fallback_codes:
            pe_pct[code] = whole_pe[code]

    components: dict[str, dict[str, float | None]] = {}
    for code in codes:
        industry = _industry_of_code(code, industry_groups)
        quality = _mean_neutral(
            [
                pct["roe"][code] if raw["roe"][code] is not None else None,
                pct["gross_margin"][code]
                if raw["gross_margin"][code] is not None
                else None,
                pct["cash_quality"][code] if raw["cash_quality"][code] is not None else None,
            ]
        )
        growth = _mean_neutral(
            [
                pct["revenue_yoy"][code] if raw["revenue_yoy"][code] is not None else None,
                pct["profit_growth"][code]
                if raw["profit_growth"][code] is not None
                else None,
            ]
        )
        pe_score = pe_pct.get(code) if raw["pe_invert"][code] is not None else None
        pb_score = pct["pb_invert"][code] if raw["pb_invert"][code] is not None else None
        valuation = _mean_neutral([pe_score, pb_score])
        leverage = (
            NEUTRAL if _is_financial(industry) else pct["debt_low"][code]
        )
        blend = (
            FUNDAMENTAL_COMPONENT_WEIGHTS["quality"] * quality
            + FUNDAMENTAL_COMPONENT_WEIGHTS["growth"] * growth
            + FUNDAMENTAL_COMPONENT_WEIGHTS["valuation"] * valuation
            + FUNDAMENTAL_COMPONENT_WEIGHTS["leverage"] * leverage
        )
        late = is_late_filer(fund_rows[code].get("report_period"), as_of)
        if late:
            blend -= LATE_FILER_PENALTY
        components[code] = {
            "quality": round(quality, 4),
            "growth": round(growth, 4),
            "valuation": round(valuation, 4),
            "leverage": round(leverage, 4),
            "fundamental_raw": round(blend, 4),
            "valuation_industry_fallback": 1.0 if code in fallback_codes else 0.0,
            "roe_basis": roe_basis[code],
            "profit_growth_basis": growth_basis[code],
            "late_filer": 1.0 if late else 0.0,
        }

    # Publish the sub-score itself as a true cross-set percentile.
    final_pct = _percentiles(
        {code: components[code]["fundamental_raw"] for code in codes}  # type: ignore[arg-type]
    )
    return _FundBasis(percentiles=final_pct, components=components)


def _cash_quality(row: dict[str, Any]) -> float | None:
    """OCF per share vs EPS; only meaningful for profitable quarters."""
    eps = row.get("eps")
    ocf = row.get("ocf_per_share")
    if eps is None or ocf is None or eps <= 0:
        return None
    return float(np.clip(ocf / eps, 0.0, 3.0))


def _pe_invert(pe: float | None) -> float | None:
    """Inverse PE; loss-makers get the strictly-lowest slot (-1)."""
    if pe is None:
        return None
    return 1.0 / pe if pe > 0 else -1.0


def _pb_invert(pb: float | None) -> float | None:
    """Inverse PB; non-positive PB gets the strictly-lowest slot (-1)."""
    if pb is None:
        return None
    return 1.0 / pb if pb > 0 else -1.0


# --------------------------------------------------------------------------- #
# Period comparability helpers (TODO_R2 item C)
# --------------------------------------------------------------------------- #


def latest_mandatory_period(as_of: date | None) -> str | None:
    """The most recent report period that MUST be published by ``as_of``.

    Mandatory periods (TODO_R2 item C): Q1 by 04-30, H1 by 08-31, Q3 by
    10-31, FY by 04-30 of the following year. A FY period therefore stays
    the mandatory one from its 12-31 period end until the next 04-30.

    Args:
        as_of: Scan reference date. ``None`` (no calendar supplied) disables
            the late-filer rule, because we cannot know what is mandatory.

    Returns:
        ``"YYYYQn"`` (e.g. ``"2026Q2"``), or ``None`` when unknown.
    """
    if as_of is None:
        return None
    # Newest deadline that has passed, walking the year's quarters. Q4's
    # deadline falls in the NEXT year, so its period year is as_of.year - 1.
    best: str | None = None
    for quarter in ("Q1", "Q2", "Q3", "Q4"):
        (month, day), next_year = MANDATORY_DEADLINES[quarter]
        deadline = _dt.date(as_of.year + 1 if next_year else as_of.year, month, day)
        if as_of > deadline:
            label = f"{as_of.year - 1 if next_year else as_of.year}{quarter}"
            if best is None or _quarter_rank(label) > _quarter_rank(best):
                best = label
    if best is None:
        # Before 04-30: the previous FY is still the mandatory period.
        return f"{as_of.year - 1}Q4"
    return best


def _quarter_rank(label: str) -> int:
    """Sortable rank of a ``YYYYQn`` label (older = smaller)."""
    try:
        return int(label[:4]) * 10 + int(label[-1])
    except ValueError:  # pragma: no cover — malformed labels only
        return 0


def _dt_iso_deadline(year: int, month_day: tuple[int, int]) -> date:
    """``(year, (m, d))`` -> ``date``."""
    return _dt.date(year, month_day[0], month_day[1])


def _period_label(report_period: Any) -> str | None:
    """``"2026-06-30"`` / ``"2026Q2"`` / ``"20260630"`` -> ``"2026Q2"``."""
    if report_period is None:
        return None
    text = str(report_period).strip()
    if not text:
        return None
    if text.endswith(("Q1", "Q2", "Q3", "Q4")):
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 8:
        # '20260630' -> MMDD = 0630 -> Q2 (the period END, not the quarter
        # the report covers: 0331=Q1, 0630=Q2, 0930=Q3, 1231=Q4).
        quarter = {"03": "Q1", "06": "Q2", "09": "Q3", "12": "Q4"}.get(digits[4:6])
        return f"{digits[:4]}{quarter}" if quarter else None
    return None


def _period_factor(report_period: Any) -> float | None:
    """Annualisation factor for a report period (Q1 -> 4, H1 -> 2, ...)."""
    label = _period_label(report_period)
    if label is None:
        return None
    return PERIOD_ANNUALISE.get(label[4:])


def is_late_filer(report_period: Any, as_of: date | None) -> bool:
    """True when a stock's latest report is older than the mandatory period.

    Args:
        report_period: The stock's latest published period.
        as_of: Scan reference date; ``None`` disables the rule.
    """
    mandatory = latest_mandatory_period(as_of)
    label = _period_label(report_period)
    if mandatory is None or label is None:
        return False
    try:
        got_year = int(label[:4])
        want_year = int(mandatory[:4])
    except ValueError:  # pragma: no cover — malformed periods only
        return False
    if got_year != want_year:
        return got_year < want_year
    return _quarter_index(label) < _quarter_index(mandatory)


def _quarter_index(label: str) -> int:
    """``"2026Q2"`` -> 2."""
    try:
        return int(label[-1])
    except ValueError:  # pragma: no cover
        return 0


def _roe_like_for_like(fund: dict[str, Any]) -> tuple[float | None, str]:
    """ROE made comparable across report periods (TODO_R2 item C).

    Preference order:
      1. ``roe_ttm`` — already a trailing-twelve-month figure, directly
         comparable across stocks (preferred).
      2. Annualise the reported period-level ROE by its period factor
         (Q1×4, H1×2, Q3×4/3, FY×1).
    A single-quarter or single-half-year figure is never ranked as-is.

    Returns:
        ``(value, basis)`` where basis is ``"ttm"``, ``"annualised"`` or
        ``"reported"``.
    """
    ttm = _finite(fund.get("roe_ttm"))
    if ttm is not None:
        return ttm, "ttm"
    roe = _finite(fund.get("roe"))
    if roe is None:
        return None, "reported"
    factor = _period_factor(fund.get("report_period"))
    if factor is None:
        return roe, "reported"
    return roe * factor, "annualised"


def _eps_like_for_like(fund: dict[str, Any]) -> float | None:
    """EPS annualised the same way as ROE (period-level metric)."""
    eps = _finite(fund.get("eps"))
    if eps is None:
        return None
    factor = _period_factor(fund.get("report_period"))
    return eps if factor is None else eps * factor


def _profit_growth_input(fund: dict[str, Any]) -> tuple[float | None, str]:
    """Recurring (扣非) net-profit YoY when available, else damped profit YoY.

    TODO_R2 item D: reported net profit is contaminated by non-recurring
    items, so the growth component prefers recurring profit growth when the
    source supplies it whole-market. **Choice made here: the source does.**
    ``RPT_F10_FINANCE_MAINFINADATA`` (EM datacenter, the same table family
    the fundamentals source already uses) returns ``DJD_DEDUCTDPNP_YOY``
    (扣非归母净利润同比) for every A share at one report period — verified
    live for 2026-06-30: 13,470 rows, 100% overlap with the yjbb board's
    11,448 codes. It is fetched by
    :mod:`src.market_scanner.sources.fundamentals` (one extra board per
    period, cached) and lands in the ``deduct_profit_yoy`` column.

    When it is missing (cache miss, source degraded, or a period the fetch
    could not cover) the fallback *damps* the reported figure: with revenue
    YoY below ``REVENUE_DECLINE_DAMP_AT`` the input becomes
    ``max(revenue_yoy, 0)`` — growth that the top line does not support must
    not score as growth.

    Returns:
        ``(value, basis)`` where basis is ``"recurring"`` or ``"damped"``.
    """
    recurring = _finite(fund.get("deduct_profit_yoy"))
    if recurring is not None:
        return recurring, "recurring"
    profit = _finite(fund.get("profit_yoy"))
    revenue = _finite(fund.get("revenue_yoy"))
    if profit is None:
        return None, "reported"
    if revenue is not None and revenue < REVENUE_DECLINE_DAMP_AT:
        return max(revenue, 0.0), "damped"
    return profit, "reported"


# --------------------------------------------------------------------------- #
# Trend components
# --------------------------------------------------------------------------- #


def _trend_components(kdf: pd.DataFrame, bench: pd.Series | None) -> dict[str, Any]:
    """Extract the raw trend inputs from one stock's kline frame."""
    close = pd.to_numeric(kdf["close"], errors="coerce").dropna()
    high = (
        pd.to_numeric(kdf["high"], errors="coerce").dropna()
        if "high" in kdf.columns
        else close
    )
    volume = (
        pd.to_numeric(kdf["volume"], errors="coerce").dropna()
        if "volume" in kdf.columns
        else None
    )
    last = float(close.iloc[-1])

    def _ma(window: int) -> float | None:
        return float(close.tail(window).mean()) if len(close) >= window else None

    ma20, ma60, ma120 = _ma(20), _ma(60), _ma(120)
    checks = [
        last > ma20 if ma20 is not None else None,
        last > ma60 if ma60 is not None else None,
        last > ma120 if ma120 is not None else None,
        ma20 > ma60 if (ma20 is not None and ma60 is not None) else None,
        ma60 > ma120 if (ma60 is not None and ma120 is not None) else None,
    ]
    answered = [bool(c) for c in checks if c is not None]
    ma_score = 100.0 * sum(answered) / len(answered) if answered else NEUTRAL

    high_250 = float(high.tail(250).max()) if len(high) else None
    dist_to_high = last / high_250 if (high_250 and high_250 > 0) else None

    def _ret(bars: int) -> float | None:
        if len(close) <= bars:
            return None
        base = float(close.iloc[-(bars + 1)])
        if base <= 0:
            return None
        return (last / base - 1.0) * 100.0

    def _bench_ret(bars: int) -> float | None:
        if bench is None or len(bench) <= bars:
            return None
        base = float(bench.iloc[-(bars + 1)])
        end = float(bench.iloc[-1])
        if base <= 0:
            return None
        return (end / base - 1.0) * 100.0

    vol_expansion: float | None = None
    if volume is not None and len(volume) >= 60:
        v60 = float(volume.tail(60).mean())
        if v60 > 0:
            vol_expansion = float(volume.tail(20).mean()) / v60

    vol60_std: float | None = None
    if len(close) >= 21:
        daily = close.pct_change().dropna().tail(60) * 100.0
        if len(daily) >= 2:
            vol60_std = float(daily.std(ddof=1))

    ret_20d, ret_60d, ret_120d = _ret(20), _ret(60), _ret(120)
    bench_ret_60d, bench_ret_120d = _bench_ret(60), _bench_ret(120)

    flags: list[str] = []
    if ret_20d is not None and ret_20d > OVERHEAT_RET_20D_PCT:
        flags.append("overheated_move")
    if vol60_std is not None and vol60_std > EXTREME_VOL_DAILY_PCT:
        flags.append("extreme_volatility")

    return {
        "bars": int(len(close)),
        "close": last,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
        "ma_score": ma_score,
        "dist_to_high_250": dist_to_high,
        "ret_20d": ret_20d,
        "ret_60d": ret_60d,
        "ret_120d": ret_120d,
        "bench_ret_60d": bench_ret_60d,
        "bench_ret_120d": bench_ret_120d,
        "rel_60d": (
            ret_60d - bench_ret_60d
            if (ret_60d is not None and bench_ret_60d is not None)
            else None
        ),
        "rel_120d": (
            ret_120d - bench_ret_120d
            if (ret_120d is not None and bench_ret_120d is not None)
            else None
        ),
        "vol_expansion": vol_expansion,
        "vol60_std": vol60_std,
        "flags": flags,
    }


def _trend_raw_batch(trends: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Blend the trend inputs into raw values; percentileed afterwards."""
    dist_pct = _percentiles({c: t["dist_to_high_250"] for c, t in trends.items()})
    rel60_pct = _percentiles({c: t["rel_60d"] for c, t in trends.items()})
    rel120_pct = _percentiles({c: t["rel_120d"] for c, t in trends.items()})
    vol_pct = _percentiles({c: t["vol_expansion"] for c, t in trends.items()})
    mom_pct = _percentiles({c: t["ret_20d"] for c, t in trends.items()})

    out: dict[str, float] = {}
    for code, t in trends.items():
        raw = (
            TREND_COMPONENT_WEIGHTS["ma_alignment"] * float(t["ma_score"])
            + TREND_COMPONENT_WEIGHTS["breakout"] * dist_pct[code]
            + TREND_COMPONENT_WEIGHTS["rel_60d"] * rel60_pct[code]
            + TREND_COMPONENT_WEIGHTS["rel_120d"] * rel120_pct[code]
            + TREND_COMPONENT_WEIGHTS["volume_expansion"] * vol_pct[code]
            + TREND_COMPONENT_WEIGHTS["momentum_20d"] * mom_pct[code]
        )
        if "overheated_move" in t["flags"]:
            raw -= OVERHEAT_PENALTY
        if "extreme_volatility" in t["flags"]:
            raw -= EXTREME_VOL_PENALTY
        out[code] = float(np.clip(raw, 0.0, 100.0))
    return out


# --------------------------------------------------------------------------- #
# Flow components
# --------------------------------------------------------------------------- #


def _flow_components(
    code: str,
    uni_row: pd.Series | None,
    flow: pd.DataFrame | None,
    lhb: pd.DataFrame | None,
    north: pd.DataFrame | None,
) -> dict[str, Any]:
    """THS net flow scaled by float cap + Dragon-Tiger bonus.

    Northbound holding data is read here but only ever lands in the returned
    ``northbound`` info block — never in ``blend``, hence never in the score.
    """
    float_mv = _finite(_cell(uni_row, "float_mv")) or _finite(_cell(uni_row, "total_mv"))
    row = _row_at(flow, code) if flow is not None else None

    f1 = _finite(_cell(row, "flow_1d"))
    f3 = _finite(_cell(row, "flow_3d"))
    f5 = _finite(_cell(row, "flow_5d"))
    has_flow = any(v is not None for v in (f1, f3, f5))

    lhb_row = _row_at(lhb, code) if lhb is not None else None
    lhb_count_raw = _finite(_cell(lhb_row, "lhb_count"))
    lhb_count = None if lhb_count_raw is None else int(lhb_count_raw)
    lhb_net_buy = _finite(_cell(lhb_row, "lhb_net_buy"))

    north_row = _row_at(north, code) if north is not None else None
    northbound = {
        "nb_hold_ratio": _finite(_cell(north_row, "nb_hold_ratio")),
        "nb_hold_mv": _finite(_cell(north_row, "nb_hold_mv")),
        "nb_report_date": _iso(_cell(north_row, "nb_report_date")),
    }

    flags: list[str] = []
    ratios: dict[str, float | None] = {
        "flow_ratio_1d": None,
        "flow_ratio_3d": None,
        "flow_ratio_5d": None,
    }
    blend: float | None = None

    if not has_flow or float_mv is None or float_mv <= 0:
        # HARD RULE (CONTRACT §6): missing flow -> 50 + flag + reason line.
        flags.append("flow_missing")
        blend = None
    else:
        present: dict[str, float] = {}
        for horizon, value in (("1d", f1), ("3d", f3), ("5d", f5)):
            if value is None:
                continue
            ratio = value / float_mv
            ratios[f"flow_ratio_{horizon}"] = ratio
            present[f"flow_{horizon}"] = ratio
        if present:
            # Horizon weights re-balanced over present horizons only; this is
            # inside the flow component, not the top-level sub-score weights.
            weight_sum = sum(FLOW_HORIZON_WEIGHTS[k] for k in present)
            blend = (
                sum(FLOW_HORIZON_WEIGHTS[k] * v for k, v in present.items()) / weight_sum
            )
            if lhb_count and lhb_count > 0 and lhb_net_buy and lhb_net_buy > 0:
                blend += min(LHB_BONUS_CAP, lhb_net_buy / float_mv)
                flags.append("dragon_tiger_buy")

    return {
        "flow_1d": f1,
        "flow_3d": f3,
        "flow_5d": f5,
        **ratios,
        "float_mv": float_mv,
        "blend": blend,
        "lhb_count": lhb_count,
        "lhb_net_buy": lhb_net_buy,
        "lhb_last_date": _iso(_cell(lhb_row, "lhb_last_date")),
        "lhb_reasons": _cell(lhb_row, "lhb_reasons"),
        "northbound": northbound,
        "flags": flags,
    }


# --------------------------------------------------------------------------- #
# Metrics / reasons
# --------------------------------------------------------------------------- #


def _metrics_for(
    code: str,
    uni_row: pd.Series | None,
    fund: dict[str, Any],
    spot: dict[str, float | None],
    industry: str,
    trend: dict[str, Any],
    flow: dict[str, Any],
    components: dict[str, float | None],
) -> dict[str, Any]:
    """Assemble the JSON-serialisable metrics block for one stock."""
    return {
        "code": code,
        "industry": industry,
        "price": _finite(_cell(uni_row, "price")),
        "pct_chg": _finite(_cell(uni_row, "pct_chg")),
        "turnover_rate": _finite(_cell(uni_row, "turnover_rate")),
        "total_mv": _finite(_cell(uni_row, "total_mv")),
        "float_mv": _finite(_cell(uni_row, "float_mv")),
        "pe": spot.get("pe"),
        "pb": spot.get("pb"),
        "report_period": _iso(fund.get("report_period")),
        "roe": fund.get("roe"),
        "roe_ttm": _finite(fund.get("roe_ttm")),
        "roe_annualised": (
            None
            if components.get("roe_basis") != "annualised"
            else _roe_like_for_like(fund)[0]
        ),
        "eps": fund.get("eps"),
        "eps_annualised": _eps_like_for_like(fund),
        "bps": fund.get("bps"),
        "revenue_yoy": fund.get("revenue_yoy"),
        "profit_yoy": fund.get("profit_yoy"),
        "deduct_profit_yoy": _finite(fund.get("deduct_profit_yoy")),
        "profit_growth_input": _profit_growth_input(fund)[0],
        "gross_margin": fund.get("gross_margin"),
        "ocf_per_share": fund.get("ocf_per_share"),
        "debt_ratio": fund.get("debt_ratio"),
        "bars": trend.get("bars"),
        "close": trend.get("close"),
        "ma20": trend.get("ma20"),
        "ma60": trend.get("ma60"),
        "ma120": trend.get("ma120"),
        "dist_to_high_250": trend.get("dist_to_high_250"),
        "ret_20d": trend.get("ret_20d"),
        "ret_60d": trend.get("ret_60d"),
        "ret_120d": trend.get("ret_120d"),
        "bench_ret_60d": trend.get("bench_ret_60d"),
        "bench_ret_120d": trend.get("bench_ret_120d"),
        "rel_60d": trend.get("rel_60d"),
        "rel_120d": trend.get("rel_120d"),
        "vol_expansion": trend.get("vol_expansion"),
        "vol60_std": trend.get("vol60_std"),
        "flow_1d": flow.get("flow_1d"),
        "flow_3d": flow.get("flow_3d"),
        "flow_5d": flow.get("flow_5d"),
        "flow_ratio_1d": flow.get("flow_ratio_1d"),
        "flow_ratio_3d": flow.get("flow_ratio_3d"),
        "flow_ratio_5d": flow.get("flow_ratio_5d"),
        "lhb_count": flow.get("lhb_count"),
        "lhb_net_buy": flow.get("lhb_net_buy"),
        "lhb_last_date": flow.get("lhb_last_date"),
        # Northbound: info only, never part of any score (CONTRACT §6).
        "northbound_hold_ratio": flow["northbound"].get("nb_hold_ratio"),
        "northbound_hold_mv": flow["northbound"].get("nb_hold_mv"),
        "northbound_report_date": flow["northbound"].get("nb_report_date"),
        "components": {k: _round(v) for k, v in components.items()},
    }


def _reasons_for(
    fund: dict[str, Any],
    spot: dict[str, float | None],
    industry: str,
    trend: dict[str, Any],
    flow: dict[str, Any],
    components: dict[str, float | None],
    as_of: date | None = None,
) -> list[str]:
    """Plain-language reasons carrying real numbers (CONTRACT §6)."""
    reasons: list[str] = []
    pe = spot.get("pe")
    flags = _fund_flags(fund, spot, industry, as_of)
    growth_basis = components.get("profit_growth_basis")

    if fund.get("roe") is not None:
        line = f"ROE {fmt_pct(fund['roe'])}" + _top_suffix(components.get("quality"))
        if components.get("roe_basis") == "annualised":
            factor = _period_factor(fund.get("report_period"))
            if factor is not None:
                line += (
                    f" (period ROE ×{factor:.2f} to annualise — period-level metric)"
                )
        reasons.append(line)
    if fund.get("gross_margin") is not None:
        reasons.append(f"gross margin {fmt_pct(fund['gross_margin'])}")
    if fund.get("revenue_yoy") is not None:
        reasons.append(f"revenue YoY {fmt_pct(fund['revenue_yoy'], signed=True)}")
    profit_reported = fund.get("profit_yoy")
    if profit_reported is not None:
        if growth_basis == "recurring" and fund.get("deduct_profit_yoy") is not None:
            reasons.append(
                f"recurring (扣非) profit YoY {fmt_pct(fund['deduct_profit_yoy'], signed=True)}"
                f" (reported {fmt_pct(profit_reported, signed=True)})"
            )
        elif growth_basis == "damped":
            revenue = fund.get("revenue_yoy")
            reasons.append(
                "profit growth not supported by revenue "
                f"({fmt_pct(revenue, signed=True)} revenue) — "
                f"reported profit YoY {fmt_pct(profit_reported, signed=True)} damped"
            )
        else:
            reasons.append(f"profit YoY {fmt_pct(profit_reported, signed=True)}")
    if "late_filer" in flags:
        reasons.append(
            f"late filer: latest report {_period_label(fund.get('report_period')) or 'unknown'}"
            f" is older than the mandatory period"
        )
    if pe is None:
        pass
    elif pe <= 0:
        reasons.append(f"loss-making (PE {pe:.1f}) — low valuation score")
    else:
        valuation = components.get("valuation")
        suffix = "" if valuation is None else f" (valuation {valuation:.0f}/100)"
        reasons.append(f"PE {pe:.1f}{suffix}")
    if fund.get("debt_ratio") is not None:
        if "leverage_neutral_financial" in flags:
            reasons.append(
                f"debt ratio {fmt_pct(fund['debt_ratio'])} — financial firm, "
                "leverage not penalised (info)"
            )
        else:
            reasons.append(f"debt ratio {fmt_pct(fund['debt_ratio'])} (low-leverage score)")

    if trend.get("ma60") is not None and trend.get("close") is not None:
        delta = (trend["close"] / trend["ma60"] - 1.0) * 100.0
        reasons.append(f"close {fmt_pct(delta, signed=True)} vs MA60")
    if trend.get("ret_120d") is not None:
        line = f"120d return {fmt_pct(trend['ret_120d'], signed=True)}"
        if trend.get("bench_ret_120d") is not None:
            line += f" vs CSI300 {fmt_pct(trend['bench_ret_120d'], signed=True)}"
        reasons.append(line)
    if trend.get("dist_to_high_250") is not None:
        reasons.append(
            "distance to 250d high "
            f"{fmt_pct((trend['dist_to_high_250'] - 1) * 100, signed=True)}"
        )
    if trend.get("vol_expansion") is not None:
        reasons.append(f"20d volume {fmt_ratio(trend['vol_expansion'])} the 60d average")
    if "overheated_move" in trend.get("flags", []):
        reasons.append(f"overheated: {fmt_pct(trend.get('ret_20d'), signed=True)} in 20d")
    if "extreme_volatility" in trend.get("flags", []):
        reasons.append(
            f"extreme volatility: daily σ {fmt_pct(trend.get('vol60_std'))} over 60d"
        )

    if "flow_missing" in flow.get("flags", []):
        reasons.append("flow data unavailable — neutral flow score 50 (flow_missing)")
    else:
        for horizon, label in (("5", "5-day"), ("3", "3-day"), ("1", "1-day")):
            amount = flow.get(f"flow_{horizon}d")
            ratio = flow.get(f"flow_ratio_{horizon}d")
            if amount is None:
                continue
            pct_of_cap = None if ratio is None else ratio * 100.0
            reasons.append(
                f"{label} THS net inflow {fmt_cny(amount)}"
                + (
                    f" ({pct_of_cap:.2f}% of float cap)"
                    if pct_of_cap is not None
                    else ""
                )
            )
            break  # lead with the longest available horizon
    if flow.get("lhb_net_buy") and flow["lhb_net_buy"] > 0:
        reasons.append(
            f"Dragon-Tiger net buy {fmt_cny(flow['lhb_net_buy'])} "
            f"in last 30 days ({flow.get('lhb_count')} hits)"
        )
    nb_ratio = flow["northbound"].get("nb_hold_ratio")
    if nb_ratio is not None:
        reasons.append(f"northbound holding {nb_ratio:.2f}% (info only, not scored)")
    return reasons[:12]


def _fund_flags(
    fund: dict[str, Any],
    spot: dict[str, float | None],
    industry: str,
    as_of: date | None = None,
) -> list[str]:
    flags: list[str] = []
    pe = spot.get("pe")
    if pe is not None and pe <= 0:
        flags.append("loss_maker")
    if _is_financial(industry):
        flags.append("leverage_neutral_financial")
    # TODO_R2 item D: damped growth needs to be visible in the flags list so
    # the UI/reviewer can see WHY the growth input was capped.
    if _profit_growth_is_damped(fund):
        flags.append("profit_growth_damped")
    if is_late_filer(fund.get("report_period"), as_of):
        flags.append("late_filer")
    return flags


def _profit_growth_is_damped(fund: dict[str, Any]) -> bool:
    """True when this row's profit-growth input would be damped."""
    value, kind = _profit_growth_input(fund)
    return kind == "damped" and value is not None


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _percentiles(values: dict[str, float | None]) -> dict[str, float]:
    """Rank-based 0-100 percentiles within the set; missing -> NEUTRAL.

    Ties get the average rank. Fewer than two present members means there is
    no cross-section to compare against, so everyone present gets 50.
    """
    series = pd.Series(
        {k: (np.nan if v is None else float(v)) for k, v in values.items()},
        dtype=float,
    )
    count = int(series.notna().sum())
    if count < 2:
        return {k: NEUTRAL for k in values}
    ranked = series.rank(method="average")
    pct = ((ranked - 1.0) / (count - 1) * 100.0).clip(0.0, 100.0)
    out: dict[str, float] = {}
    for key in values:
        value = pct[key] if key in pct.index else np.nan
        out[key] = float(value) if value == value else NEUTRAL
    return out


def _winsorise(
    values: dict[str, float | None], lower: float = 0.05, upper: float = 0.95
) -> dict[str, float | None]:
    """Clip extreme growth values to the set's p5/p95 quantiles."""
    series = pd.Series(
        {k: (np.nan if v is None else float(v)) for k, v in values.items()},
        dtype=float,
    )
    valid = series.dropna()
    if len(valid) < 3:
        return dict(values)
    lo, hi = valid.quantile(lower), valid.quantile(upper)
    clipped = series.clip(lo, hi)
    return {
        k: (float(clipped[k]) if clipped[k] == clipped[k] else None) for k in values
    }


def _mean_neutral(values: list[float | None]) -> float:
    """Mean of the available values; NEUTRAL when none are available.

    Internal component weights re-balance over the *available* components
    only. This is NOT the top-level sub-score renormalisation the contract
    forbids — the three sub-score weights never pass through here.
    """
    present = [v for v in values if v is not None]
    if not present:
        return NEUTRAL
    return float(sum(present) / len(present))


def _neg(value: float | None) -> float | None:
    return None if value is None else -float(value)


def _top_suffix(percentile: float | None) -> str:
    """`` (top 12%)``-style suffix from a higher-is-better percentile."""
    if percentile is None:
        return ""
    top = max(1, min(99, int(round(100.0 - float(percentile)))))
    return f" (top {top}%)"


def _benchmark_closes(benchmark_df: pd.DataFrame | None) -> pd.Series | None:
    if benchmark_df is None or benchmark_df.empty or "close" not in benchmark_df.columns:
        return None
    closes = pd.to_numeric(benchmark_df["close"], errors="coerce").dropna()
    return closes if len(closes) >= 2 else None


def _fundamental_row(funds: pd.DataFrame | None, code: str) -> dict[str, Any]:
    row = _row_at(funds, code)
    if row is None:
        return {}
    out: dict[str, Any] = {}
    for key in (
        "report_period",
        "roe",
        "roe_ttm",
        "eps",
        "bps",
        "revenue_yoy",
        "profit_yoy",
        "deduct_profit_yoy",
        "gross_margin",
        "ocf_per_share",
        "debt_ratio",
        "industry",
    ):
        out[key] = _cell(row, key)
    for key in (
        "roe",
        "roe_ttm",
        "eps",
        "bps",
        "revenue_yoy",
        "profit_yoy",
        "deduct_profit_yoy",
        "gross_margin",
        "ocf_per_share",
        "debt_ratio",
    ):
        out[key] = _finite(out.get(key))
    return out


def _industry_for(
    code: str, funds: pd.DataFrame | None, listing: pd.DataFrame | None
) -> str:
    fund_row = _row_at(funds, code)
    if fund_row is not None:
        industry = _cell(fund_row, "industry")
        if industry:
            return str(industry)
    listing_row = _row_at(listing, code)
    if listing_row is not None:
        industry = _cell(listing_row, "industry")
        if industry:
            return str(industry)
    return "未分类"


def _industry_of_code(code: str, industry_groups: dict[str, list[str]]) -> str:
    for industry, members in industry_groups.items():
        if code in members:
            return industry
    return "未分类"


def _is_financial(industry: str | None) -> bool:
    if not industry:
        return False
    lowered = str(industry).lower()
    return any(keyword in lowered for keyword in FINANCIAL_INDUSTRY_KEYWORDS)


def _row_at(df: pd.DataFrame | None, code: str) -> pd.Series | None:
    if df is None or df.empty or code not in df.index:
        return None
    row = df.loc[code]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _cell(row: pd.Series | None, key: str) -> Any:
    if row is None:
        return None
    try:
        value = row[key]
    except Exception:  # noqa: BLE001 — partial rows are normal here
        return None
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:  # pragma: no cover — non-float scalars
        pass
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


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


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:  # pragma: no cover
            return str(value)
    return str(value)


def _round(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)
