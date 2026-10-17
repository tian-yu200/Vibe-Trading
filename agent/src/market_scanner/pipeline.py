"""Scan pipeline: sources -> funnel -> scoring -> picks -> AI review (section 7).

``run_scan`` orchestrates the injectable source bundle, the ordered exclusion
funnel, cross-sectional scoring, the industry cap, the AI review step
(review.py, section 8) and suppressed/pinned handling. It never raises on a
source failure — a failed or degraded source degrades the run ``status`` to
``partial`` and is listed in ``sources``.
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import ScanConfig
from .exclusions import (
    LIQUIDITY_SNAPSHOT_FLAG,
    ensure_code_index,
    run_exclusions,
)
from .models import (
    AI_STATUS_PENDING,
    STAGE_AI_DEMOTED,
    STAGE_FINALISTS,
    STAGE_SCORED,
    STAGE_UNIVERSE,
    STAGE_WATCHLIST,
    ai_pending,
    board_from_code,
    make_funnel_entry,
    normalize_code,
    symbol_from_code,
)
from .review import (
    AIReviewSummary,
    NEUTRAL_RANK_PENALTY,
    aggregate_source_status,
    ai_stats_from_results,
    build_data_pack,
    build_default_llm_callable,
    resolve_llm_callable,
    run_ai_review,
    summarize_ai_results,
)
from .scoring import ScoredStock, score_universe

logger = logging.getLogger(__name__)

#: Asia/Shanghai has no DST; a fixed offset keeps this dependency-free.
SHANGHAI = timezone(timedelta(hours=8), "Asia/Shanghai")

#: A-share regular session window in Asia/Shanghai.
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 0)

_SOURCE_NAMES: tuple[str, ...] = (
    "fetch_universe",
    "fetch_listing_info",
    "fetch_fundamentals",
    "fetch_klines",
    "fetch_benchmark",
    "fetch_fund_flow",
    "fetch_dragon_tiger",
    "fetch_northbound_info",
    "fetch_news",
    "fetch_announcements",
    "fetch_red_flags",
    "load_calendar",
)

try:  # pragma: no cover — exercised implicitly by every pipeline test
    from .sources.status import SourceStatus
except Exception as _status_import_error:  # pragma: no cover — defensive mirror
    logger.warning("market_scanner.sources.status unavailable (%s); using mirror", _status_import_error)
    from dataclasses import dataclass as _dataclass
    from typing import Literal as _Literal

    @_dataclass
    class SourceStatus:  # type: ignore[no-redef]
        """Mirror of sources.status.SourceStatus (kept shape-compatible)."""

        name: str
        status: _Literal["ok", "degraded", "failed", "skipped"] = "ok"
        rows: int = 0
        latency_s: float = 0.0
        error: str | None = None
        detail: str | None = None
        as_of: str | None = None

        def to_dict(self) -> dict[str, Any]:
            """JSON-serialisable dict for the ScanRun envelope."""
            return {
                "name": self.name,
                "status": self.status,
                "rows": self.rows,
                "latency_s": round(self.latency_s, 3),
                "error": self.error,
                "detail": self.detail,
                "as_of": self.as_of,
            }


# --------------------------------------------------------------------------- #
# Injectable source bundle
# --------------------------------------------------------------------------- #


@dataclass
class SourceBundle:
    """Bundle of the CONTRACT section 3 fetch callables.

    Any member may be ``None`` (source not provided) — the pipeline records a
    ``skipped`` status for it and continues.
    """

    fetch_universe: Callable | None = None
    fetch_listing_info: Callable | None = None
    fetch_fundamentals: Callable | None = None
    fetch_klines: Callable | None = None
    fetch_benchmark: Callable | None = None
    fetch_fund_flow: Callable | None = None
    fetch_dragon_tiger: Callable | None = None
    fetch_northbound_info: Callable | None = None
    fetch_news: Callable | None = None
    fetch_announcements: Callable | None = None
    fetch_red_flags: Callable | None = None
    load_calendar: Callable | None = None

    @classmethod
    def resolve(cls, sources: object | None) -> SourceBundle:
        """Coerce ``sources`` (None / bundle / duck-typed object) to a bundle."""
        if sources is None:
            return cls.default()
        if isinstance(sources, cls):
            return sources
        return cls(
            **{name: getattr(sources, name, None) for name in _SOURCE_NAMES}
        )

    @classmethod
    def default(cls) -> SourceBundle:
        """Real source adapters from ``market_scanner.sources`` (section 3).

        Import is lazy and per-module: while hermes is still writing the
        package, a missing module yields a callable that raises a clear error,
        which the pipeline converts into a ``failed`` SourceStatus instead of
        crashing the scan.
        """
        modules = _import_default_source_modules()
        return cls(
            fetch_universe=_module_callable(modules, "universe", "fetch_universe"),
            fetch_listing_info=_module_callable(modules, "listing", "fetch_listing_info"),
            fetch_fundamentals=_module_callable(modules, "fundamentals", "fetch_fundamentals"),
            fetch_klines=_module_callable(modules, "kline", "fetch_klines"),
            fetch_benchmark=_module_callable(modules, "kline", "fetch_benchmark"),
            fetch_fund_flow=_module_callable(modules, "flows", "fetch_fund_flow"),
            fetch_dragon_tiger=_module_callable(modules, "flows", "fetch_dragon_tiger"),
            fetch_northbound_info=_module_callable(modules, "flows", "fetch_northbound_info"),
            fetch_news=_module_callable(modules, "news", "fetch_news"),
            fetch_announcements=_module_callable(modules, "news", "fetch_announcements"),
            fetch_red_flags=_module_callable(modules, "news", "fetch_red_flags"),
            load_calendar=_module_callable(modules, "calendar", "load_calendar"),
        )


def _import_default_source_modules() -> dict[str, Any]:
    """Import the ``sources.*`` modules; failures are recorded, not raised."""
    # The adapters under sources/ import each other as top-level ``sources.*``
    # modules, so the package directory itself needs to be importable.
    package_dir = str(Path(__file__).resolve().parent)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    out: dict[str, Any] = {}
    for name in ("universe", "listing", "fundamentals", "kline", "flows", "calendar", "news"):
        try:
            out[name] = importlib.import_module(f"sources.{name}")
        except Exception as exc:  # noqa: BLE001 — hermes may still be mid-write
            logger.warning("market_scanner.sources.%s not importable yet: %s", name, exc)
            out[name] = exc
    return out


def _module_callable(modules: dict[str, Any], module_name: str, func_name: str) -> Callable | None:
    entry = modules.get(module_name)
    if entry is None:
        return None
    if isinstance(entry, Exception):
        error = entry

        def _unavailable(*_args: Any, _error=error, **_kwargs: Any) -> Any:
            raise RuntimeError(
                f"market_scanner.sources.{module_name} not importable yet: {_error}"
            )

        return _unavailable
    return getattr(entry, func_name, None)


def _invoke(
    name: str, fn: Callable | None, *args: Any, **kwargs: Any
) -> tuple[Any, SourceStatus]:
    """Call one source; never raises. Returns ``(data, SourceStatus)``."""
    t0 = time.monotonic()
    if fn is None:
        return None, SourceStatus(
            name=name,
            status="skipped",
            detail="not provided by the source bundle",
        )
    try:
        out = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — sources must not crash a scan
        logger.warning("source %s failed: %s", name, exc)
        return None, SourceStatus(
            name=name,
            status="failed",
            latency_s=time.monotonic() - t0,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )
    if not isinstance(out, tuple) or len(out) != 2:
        return None, SourceStatus(
            name=name,
            status="failed",
            latency_s=time.monotonic() - t0,
            error=f"bad return shape: expected (data, SourceStatus), got {type(out).__name__}",
        )
    data, status = out
    if status is None:
        status = SourceStatus(name=name, status="ok")
    return data, status


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run_scan(
    config: ScanConfig,
    *,
    suppressed: set[str] = frozenset(),  # type: ignore[assignment]
    pinned: set[str] = frozenset(),  # type: ignore[assignment]
    trigger: str = "cli",
    progress: Callable[[str, float], None] | None = None,
    now: datetime | None = None,
    llm: Callable | None = None,
    sources: object | None = None,
    lock_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run one full scan and return the ScanRun envelope (CONTRACT §7).

    Args:
        config: Validated scan settings.
        suppressed: Codes the user removed; never enter ``picks``, land in
            ``requalified`` when they would have made the top_n.
        pinned: Manual adds; scored when data exists and reported in
            ``pinned`` (bypassing the top_n / industry cap when scored).
        trigger: ``cli`` / ``api`` / ``schedule`` / ``agent``.
        progress: Optional ``(stage, fraction)`` callback; never breaks a scan.
        now: Injected clock (Asia/Shanghai semantics); defaults to now.
        llm: Injected completion callable for the AI review step.
        sources: Injectable :class:`SourceBundle` (or duck-typed object with
            the same attribute names); ``None`` uses the real adapters.
        lock_check: Optional no-arg callable (hermes's ``JobHandle.check``).
            Called at stage boundaries and inside kline fetching; a lost job
            lock aborts the run as failed with the F2 reason instead of
            crashing the scan.

    Returns:
        The ScanRun dict from CONTRACT section 7.
    """
    started = _coerce_now(now)
    t0 = time.monotonic()
    timings: dict[str, Any] = {}
    run_id = uuid.uuid4().hex
    # Run-scoped per-host request meter (section 9b) — fresh per run so two
    # runs never mix.
    from src.market_scanner.sources._common import HostMeter, set_active_meter

    meter = HostMeter()
    set_active_meter(meter)
    suppressed_codes = {normalize_code(c) for c in (suppressed or frozenset())}
    pinned_codes = {normalize_code(c) for c in (pinned or frozenset())}
    # A manual add beats suppression: the user pinned it after removing it.
    suppressed_effective = suppressed_codes - pinned_codes
    try:
        return _run_scan_body(
            config, run_id=run_id, meter=meter, t0=t0, started=started,
            timings=timings, suppressed=suppressed, pinned=pinned,
            suppressed_effective=suppressed_effective,
            suppressed_codes=suppressed_codes, pinned_codes=pinned_codes,
            trigger=trigger, progress=progress, llm=llm, sources=sources,
            now=now, lock_check=lock_check,
        )
    finally:
        set_active_meter(None)


def _run_scan_body(
    config, *, run_id, meter, t0, started, timings, suppressed, pinned,
    suppressed_effective, suppressed_codes, pinned_codes, trigger, progress,
    llm, sources, now, lock_check=None,
) -> dict[str, Any]:
    bundle = SourceBundle.resolve(sources)

    _notify(progress, "start", 0.0)

    # ------------------------------------------------------------------ #
    # Calendar: trade_date + intraday (CONTRACT §3/§7)
    # ------------------------------------------------------------------ #
    calendar_obj, calendar_status = _invoke("calendar", bundle.load_calendar)
    statuses: list[SourceStatus] = [calendar_status]
    trade_date, intraday = _resolve_trade_date(calendar_obj, started)

    # ------------------------------------------------------------------ #
    # Universe (fatal when it fails)
    # ------------------------------------------------------------------ #
    _notify(progress, "universe", 0.10)
    _t = time.monotonic()
    universe_df, universe_status = _invoke("universe", bundle.fetch_universe)
    timings["universe"] = round(time.monotonic() - _t, 4)
    statuses.append(universe_status)
    uni = ensure_code_index(universe_df if isinstance(universe_df, pd.DataFrame) else None)
    if uni is None or uni.empty:
        return _failed_run(
            run_id=run_id,
            started=started,
            t0=t0,
            trigger=trigger,
            config=config,
            statuses=statuses,
            trade_date=trade_date,
            intraday=intraday,
            progress=progress,
            note="universe fetch returned no rows",
        )

    # ------------------------------------------------------------------ #
    # Listing info
    # ------------------------------------------------------------------ #
    _notify(progress, "listing", 0.20)
    _t = time.monotonic()
    listing_raw, listing_status = _invoke("listing", bundle.fetch_listing_info)
    timings["listing"] = round(time.monotonic() - _t, 4)
    statuses.append(listing_status)
    listing = ensure_code_index(
        listing_raw if isinstance(listing_raw, pd.DataFrame) else None
    )

    survivors = _early_survivors(uni, config, listing, trade_date, klines=None)

    # ------------------------------------------------------------------ #
    # Fundamentals + klines for the survivors only
    # ------------------------------------------------------------------ #
    _notify(progress, "fundamentals", 0.35)
    _t = time.monotonic()
    fund_raw, fund_status = _invoke(
        "fundamentals", bundle.fetch_fundamentals, trade_date
    )
    timings["fundamentals"] = round(time.monotonic() - _t, 4)
    statuses.append(fund_status)
    funds = ensure_code_index(fund_raw if isinstance(fund_raw, pd.DataFrame) else None)

    _notify(progress, "klines", 0.45)
    _t = time.monotonic()
    kline_symbols = _symbols_for(uni, survivors)
    kline_budget_s = config.kline_time_budget_s
    kline_limiter = None
    try:
        from src.market_scanner.sources import kline as _kline_engine
    except ImportError:
        _kline_engine = None  # type: ignore[assignment]
    if _kline_engine is not None:
        try:
            kline_limiter = _kline_engine.KlineRateLimiter(
                stop_consecutive=config.kline_stop_consecutive,
                stop_window=config.kline_stop_window,
                stop_min_error_rate=config.kline_stop_min_error_rate,
                stop_min_elapsed_s=config.kline_stop_min_elapsed_s,
            )
        except Exception:  # noqa: BLE001 — config errors surface in validate
            kline_limiter = None
    if _kline_engine is not None and calendar_obj is not None \
            and kline_symbols:
        # F5: scale the kline budget with the remaining shortfall (one
        # meta query, no bar reads). Skipped without a calendar; falls back
        # to the base budget on any assessment failure.
        try:
            shortfall = _kline_engine.assess_shortfall(
                kline_symbols, trade_date, calendar=calendar_obj)
            kline_budget_s = _kline_engine.resolve_kline_budget(
                shortfall.est_seconds, config.kline_time_budget_s,
                config.kline_time_budget_cap_s)
        except Exception as exc:  # noqa: BLE001
            if _is_home_mismatch(exc):
                return _abort_run(
                    run_id, started, t0, trigger, config,
                    statuses, trade_date, intraday, progress, str(exc))
            logger.warning("kline shortfall assessment failed: %s", exc)
    abort = _lock_abort_reason(lock_check, "klines")
    if abort is not None:
        return _abort_run(
            run_id, started, t0, trigger, config,
            statuses, trade_date, intraday, progress, abort)
    klines_raw, kline_status = _invoke(
        "klines",
        bundle.fetch_klines,
        kline_symbols,
        bars=config.kline_bars,
        workers=config.kline_workers,
        progress=_kline_progress(progress),
        snapshot=uni,
        trade_date=trade_date,
        calendar=calendar_obj,
        limiter=kline_limiter,
        time_budget_s=kline_budget_s,
        lock_check=lock_check,
        local_budget_s=config.kline_local_budget_s,
    )
    if getattr(kline_status, "lock_aborted", False):
        return _abort_run(
            run_id, started, t0, trigger, config,
            statuses + [kline_status], trade_date, intraday, progress,
            LOCK_LOST_REASON)
    abort = _lock_abort_reason(lock_check, "klines-done")
    if abort is not None:
        return _abort_run(
            run_id, started, t0, trigger, config,
            statuses + [kline_status], trade_date, intraday, progress,
            abort)
    timings["klines"] = round(time.monotonic() - _t, 4)
    statuses.append(kline_status)
    klines = _normalize_klines(klines_raw, uni)

    _notify(progress, "benchmark", 0.55)
    bench_raw, bench_status = _invoke(
        "benchmark", bundle.fetch_benchmark, trade_date=trade_date
    )
    statuses.append(bench_status)
    benchmark = bench_raw if isinstance(bench_raw, pd.DataFrame) else None

    # ------------------------------------------------------------------ #
    # Flows (THS net flow, Dragon-Tiger, northbound info-only)
    # ------------------------------------------------------------------ #
    _notify(progress, "flows", 0.65)
    _t = time.monotonic()
    flow_raw, flow_status = _invoke("flows", bundle.fetch_fund_flow)
    statuses.append(flow_status)
    flows = ensure_code_index(flow_raw if isinstance(flow_raw, pd.DataFrame) else None)

    lhb_raw, lhb_status = _invoke(
        "dragon_tiger", bundle.fetch_dragon_tiger, trade_date, days=30
    )
    statuses.append(lhb_status)
    lhb = ensure_code_index(lhb_raw if isinstance(lhb_raw, pd.DataFrame) else None)

    north_raw, north_status = _invoke("northbound", bundle.fetch_northbound_info)
    statuses.append(north_status)
    north = ensure_code_index(
        north_raw if isinstance(north_raw, pd.DataFrame) else None
    )
    timings["flows"] = round(time.monotonic() - _t, 4)

    # ------------------------------------------------------------------ #
    # Funnel + scoring
    # ------------------------------------------------------------------ #
    _notify(progress, "exclusions", 0.75)
    abort = _lock_abort_reason(lock_check, "exclusions")
    if abort is not None:
        return _abort_run(
            run_id, started, t0, trigger, config,
            statuses, trade_date, intraday, progress, abort)
    _t = time.monotonic()
    exclusion = run_exclusions(
        uni,
        config=config,
        listing_df=listing,
        klines=klines,
        fundamentals_df=funds,
        trade_date=trade_date,
    )
    timings["exclusions"] = round(time.monotonic() - _t, 4)

    _notify(progress, "scoring", 0.85)
    _t = time.monotonic()
    scored = score_universe(
        exclusion.passed,
        universe_df=uni,
        fundamentals_df=funds,
        klines=klines,
        benchmark_df=benchmark,
        flow_df=flows,
        lhb_df=lhb,
        northbound_df=north,
        listing_df=listing,
        config=config,
    )
    timings["scoring"] = round(time.monotonic() - _t, 4)

    # Liquidity snapshot fallback (opencode2 rule (b)): codes that passed the
    # liquidity bar via the Sina snapshot day-amount proxy carry the flag on
    # the scored item, so picks / requalified / ai_demoted inherit it.
    for code in exclusion.liquidity_snapshot_fallback:
        item = scored.get(code)
        if item is not None and LIQUIDITY_SNAPSHOT_FLAG not in item.flags:
            item.flags.append(LIQUIDITY_SNAPSHOT_FLAG)

    # ------------------------------------------------------------------ #
    # Review pool: top ai_review_pool candidates (industry cap applied)
    # ------------------------------------------------------------------ #
    _notify(progress, "select", 0.90)
    ranked = sorted(scored, key=lambda c: (-scored[c].score, c))
    rule_rank = {code: i + 1 for i, code in enumerate(ranked)}
    pool_codes: list[str] = []
    industry_count: dict[str, int] = {}
    for code in ranked:
        industry = _industry_of(scored[code])
        if industry_count.get(industry, 0) >= config.max_per_industry:
            continue
        pool_codes.append(code)
        industry_count[industry] = industry_count.get(industry, 0) + 1
        if len(pool_codes) >= config.ai_review_pool:
            break
    finalists_count = min(config.top_n, len(pool_codes))

    # ------------------------------------------------------------------ #
    # AI review (review.py, CONTRACT §8)
    # ------------------------------------------------------------------ #
    _notify(progress, "ai_review", 0.95)
    abort = _lock_abort_reason(lock_check, "ai_review")
    if abort is not None:
        return _abort_run(
            run_id, started, t0, trigger, config,
            statuses, trade_date, intraday, progress, abort)
    llm_fn, llm_skip_note = _resolve_llm(llm, config)
    _t = time.monotonic()
    if llm_fn is None:
        news_map: dict[str, list] = {}
        ann_map: dict[str, list] = {}
        red_map: dict[str, list] = {}
        news_statuses = [
            SourceStatus(name=name, status="skipped", detail="AI review skipped")
            for name in ("news", "announcements", "red_flags")
        ]
    else:
        _notify(progress, "news", 0.93)
        news_map, ann_map, red_map, news_statuses = _fetch_review_inputs(
            bundle, pool_codes, config, trade_date
        )
    timings["news_red_flags"] = round(time.monotonic() - _t, 4)
    statuses.extend(news_statuses)
    _t = time.monotonic()
    ai_summary, ai_results, ai_pool_rows = _run_review(
        pool_codes, scored, uni, news_map, ann_map, red_map, config, llm_fn,
        llm_skip_note, rule_rank=rule_rank
    )
    timings["ai_review"] = round(time.monotonic() - _t, 4)

    picks, ai_demoted, requalified, pinned_entries, funnel_tail = _select_picks(
        uni=uni,
        scored=scored,
        exclusion_reasons=exclusion.reasons,
        config=config,
        suppressed=suppressed_effective,
        pinned=pinned_codes,
        pool_codes=pool_codes,
        rule_rank=rule_rank,
        ai_results=ai_results,
        finalists_count=finalists_count,
    )

    # ------------------------------------------------------------------ #
    # Envelope
    # ------------------------------------------------------------------ #
    funnel = list(exclusion.funnel) + funnel_tail
    funnel = _make_funnel_consistent(funnel)
    status = _overall_status(statuses)
    finished = _coerce_now(now)
    scores = _score_rows(uni, scored, exclusion.reasons)

    coverage = _build_coverage(exclusion.funnel, scored)
    # kline_full_history: stocks that passed exclusions AND have a kline frame
    # with at least config.min_kline_bars bars (CONTRACT section 9b).
    passed_set = set(exclusion.passed)
    coverage["kline_full_history"] = sum(
        1
        for code, frame in klines.items()
        if code in passed_set and len(frame) >= config.min_kline_bars
    )
    invariant_ok = _coverage_invariant_holds(coverage)
    coverage_note: str | None = None
    if not invariant_ok:
        coverage_note = (
            "coverage invariant failed: "
            f"passed_exclusions={coverage['passed_exclusions']} "
            f"insufficient={coverage['insufficient']} "
            f"scored={coverage['scored']} "
            f"trend_scored={coverage['trend_scored']} "
            f"flow_real={coverage['flow_real']} "
            f"flow_neutral={coverage['flow_neutral']}"
        )
        logger.warning(coverage_note)

    # Coverage guard: after scoring, a run whose trend or fundamentals
    # coverage falls below config.min_coverage is failed outright (kept for
    # history; it must not move the watchlist or become a diff baseline).
    trend_cov, fund_cov = _coverage_ratios(coverage)
    coverage["trend_coverage"] = round(trend_cov, 4)
    coverage["fundamentals_coverage"] = round(fund_cov, 4)
    coverage["min_coverage"] = config.min_coverage
    failure_reason = _coverage_guard_failure(coverage, config.min_coverage)
    kline_unavailable = bool(getattr(kline_status, "unavailable", False))
    if kline_unavailable:
        # F4: the price source died mid-run — surface it alongside (or
        # instead of) the coverage reason so the failure is diagnosable.
        kline_note = f"price source unavailable ({kline_status.error or 'outage'})"
        failure_reason = (f"{kline_note}; {failure_reason}"
                          if failure_reason else kline_note)
    if failure_reason is not None:
        status = "failed"
        logger.warning("watchlist pipeline: %s", failure_reason)

    timings["total"] = round(time.monotonic() - t0, 3)

    requests_metrics = meter.to_dict()

    run: dict[str, Any] = {
        "run_id": run_id,
        "trade_date": trade_date.isoformat(),
        "intraday": intraday,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": timings["total"],
        "trigger": trigger,
        "status": status,
        "failure_reason": failure_reason,
        "ai_status": ai_summary.status,
        "ai_note": ai_summary.note,
        "ai_stats": ai_summary.ai_stats,
        "coverage": coverage,
        "coverage_note": coverage_note,
        "kline_reasons": getattr(kline_status, "kline_reasons", None),
        "home": _home_fields_safe(),
        "config": config.to_dict(),
        "funnel": funnel,
        "sources": [s.to_dict() for s in statuses],
        "picks": picks,
        "ai_demoted": ai_demoted,
        "requalified": requalified,
        "pinned": pinned_entries,
        "scores": scores,
        "timings": timings,
        "requests": requests_metrics,
        # AI review pool (top ai_review_pool candidates, pick-shaped, compact)
        # so a re-review can backfill after 'avoid' without a market rescan.
        "ai_pool": ai_pool_rows,
        "review_revision": 0,
        "reviewed_at": None,
    }
    _notify(progress, "done", 1.0)
    return run


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def _select_picks(
    *,
    uni: pd.DataFrame,
    scored: dict[str, ScoredStock],
    exclusion_reasons: dict[str, str | None],
    config: ScanConfig,
    suppressed: set[str],
    pinned: set[str],
    pool_codes: list[str],
    rule_rank: dict[str, int],
    ai_results: dict[str, dict],
    finalists_count: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Apply AI verdicts, then build picks / ai_demoted / requalified / pinned.

    ``avoid`` pool members are demoted (kept visible in ``ai_demoted``) and
    the next pool candidates backfill up to ``top_n``. ``neutral`` members
    stay in picks but are ordered :data:`NEUTRAL_RANK_PENALTY` points lower
    (score / sub_scores / rule_rank untouched).
    """

    def adjusted(code: str) -> float:
        score = scored[code].score
        ai = ai_results.get(code) or {}
        if ai.get("verdict") == "neutral":
            return score - NEUTRAL_RANK_PENALTY
        return score

    ordered = sorted(pool_codes, key=lambda c: (-adjusted(c), c))
    avoided = {
        c for c in pool_codes if (ai_results.get(c) or {}).get("verdict") == "avoid"
    }

    selected: list[str] = []
    for code in ordered:
        if code in avoided:
            continue
        selected.append(code)
        if len(selected) >= config.top_n:
            break

    picks: list[dict] = []
    requalified: list[dict] = []
    selected_set = set(selected)
    for code in selected:
        pick = _make_pick(
            uni, scored[code], rule_rank=rule_rank[code], flags=(),
            ai=ai_results.get(code),
        )
        if code in suppressed:
            # Would have been a pick; the user removed it -> requalified.
            requalified.append(pick)
        else:
            pick["rank"] = len(picks) + 1
            picks.append(pick)

    ai_demoted: list[dict] = []
    for code in pool_codes:
        if code in avoided:
            ai_demoted.append(
                _make_pick(
                    uni, scored[code], rule_rank=rule_rank[code], flags=(),
                    ai=ai_results.get(code),
                )
            )

    pinned_entries: list[dict] = []
    for code in sorted(pinned):
        if code in scored and code in selected_set:
            for pick in picks:
                if pick["code"] == code:
                    _add_flag(pick, "pinned")
            pinned_entries.append(
                next(p for p in picks if p["code"] == code)
            )
            continue
        if code in scored:
            pick = _make_pick(
                uni, scored[code], rule_rank=rule_rank.get(code), flags=("pinned",),
                ai=ai_results.get(code),
            )
            pick["rank"] = len(picks) + 1
            picks.append(pick)
            pinned_entries.append(pick)
            continue
        # Not scored: report the exclusion reason (or absence from the universe).
        reason = exclusion_reasons.get(code)
        if reason is None and code not in uni.index:
            reason = "not in the current universe snapshot"
        elif reason is None:
            reason = "not scored"
        name = _cell(uni, code, "name")
        pinned_entries.append(
            {
                "code": code,
                "name": name,
                "board": _cell(uni, code, "board") or board_from_code(code),
                "exclusion_reason": reason,
            }
        )

    funnel_tail = [
        make_funnel_entry(
            STAGE_SCORED,
            len(scored),
            # scored removed nothing: the cuts live on the exclusion stages
            # above. removed = 0 keeps the invariant count[i-1] - count[i].
            0,
        ),
        make_funnel_entry(
            STAGE_FINALISTS,
            finalists_count,
            max(0, len(scored) - finalists_count),
            f"top_n={config.top_n}, max_per_industry={config.max_per_industry}",
        ),
        make_funnel_entry(
            STAGE_AI_DEMOTED,
            len(ai_demoted),
            # ai_demoted is a side report (picks backfill from the next pool
            # candidates), not a funnel cut — removed stays 0 and the
            # next stage's count is unchanged, so the invariant holds.
            0,
            f"{len(ai_demoted)} demoted by AI review" if ai_demoted else None,
        ),
        make_funnel_entry(
            STAGE_WATCHLIST,
            len(picks),
            max(0, finalists_count - len(picks)),
        ),
    ]
    return picks, ai_demoted, requalified, pinned_entries, funnel_tail


def _make_funnel_consistent(funnel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute each stage's ``removed`` as ``count[i-1] - count[i]``.

    The first entry (universe) keeps ``removed=0``. This guards against a
    stage accidentally reporting a universe-scale removal that belongs to an
    earlier exclusion stage (e.g. the old ``scored`` removed=len(uni)-len(scored)).
    """
    if not funnel:
        return funnel
    fixed: list[dict[str, Any]] = []
    prev_count: int | None = None
    for entry in funnel:
        count = int(entry.get("count") or 0)
        if prev_count is None or entry.get("stage") == STAGE_AI_DEMOTED:
            # ai_demoted is a side report (picks backfill from the next pool
            # candidates), not a funnel cut — removed stays 0.
            removed = int(entry.get("removed") or 0)
        else:
            removed = max(0, prev_count - count)
        new_entry = dict(entry)
        new_entry["removed"] = removed
        fixed.append(new_entry)
        prev_count = count
    return fixed


def _make_pick(
    uni: pd.DataFrame,
    item: ScoredStock,
    *,
    rule_rank: int | None,
    flags: tuple[str, ...] = (),
    ai: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one Pick dict (CONTRACT §7) from a ScoredStock."""
    board = _cell(uni, item.code, "board") or board_from_code(item.code)
    name = _cell(uni, item.code, "name")
    return {
        "code": item.code,
        "name": name,
        "board": board,
        "industry": _industry_of(item),
        "rank": None,
        "rule_rank": rule_rank,
        "score": item.score,
        "sub_scores": {
            "fundamental": item.fundamental,
            "trend": item.trend,
            "flow": item.flow,
        },
        "metrics": dict(item.metrics),
        "reasons": list(item.reasons),
        "flags": _dedupe([*item.flags, *flags]),
        # AI review block (CONTRACT §8); not_reviewed until review.py runs.
        "ai": dict(ai) if ai is not None else ai_pending(),
    }


def _add_flag(pick: dict[str, Any], flag: str) -> None:
    if flag not in pick["flags"]:
        pick["flags"].append(flag)


def _industry_of(item: ScoredStock) -> str:
    return str(item.metrics.get("industry") or "未分类")


def _score_rows(
    uni: pd.DataFrame,
    scored: dict[str, ScoredStock],
    exclusion_reasons: dict[str, str | None],
) -> list[dict[str, Any]]:
    """The ``scores`` list: the whole universe, scored numbers or a reason."""
    rows: list[dict[str, Any]] = []
    ranked = sorted(scored, key=lambda c: (-scored[c].score, c))
    for code in ranked:
        item = scored[code]
        rows.append(
            {
                "code": code,
                "name": _cell(uni, code, "name"),
                "board": _cell(uni, code, "board") or board_from_code(code),
                "industry": _industry_of(item),
                "score": item.score,
                "fundamental": item.fundamental,
                "trend": item.trend,
                "flow": item.flow,
                "exclusion_reason": None,
            }
        )
    for code in uni.index:
        if code in scored:
            continue
        rows.append(
            {
                "code": code,
                "name": _cell(uni, code, "name"),
                "board": _cell(uni, code, "board") or board_from_code(code),
                "industry": None,
                "score": None,
                "fundamental": None,
                "trend": None,
                "flow": None,
                "exclusion_reason": exclusion_reasons.get(code),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Calendar / status / progress helpers
# --------------------------------------------------------------------------- #


def _resolve_trade_date(calendar_obj: Any, now: datetime) -> tuple[date, bool]:
    """Resolve the snapshot date and whether the current session is live."""
    try:
        today_is_trading = (
            bool(calendar_obj.is_trading_day(now.date()))
            if calendar_obj is not None
            else now.weekday() < 5
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("calendar is_trading_day failed: %s", exc)
        today_is_trading = now.weekday() < 5

    intraday = (
        today_is_trading
        and SESSION_OPEN <= now.time() < SESSION_CLOSE
    )
    if intraday:
        return now.date(), True

    if calendar_obj is None:
        return _fallback_trade_date(now, intraday), False
    try:
        raw = calendar_obj.latest_completed_trading_day(now)
        trade_date = _as_date(raw) or _fallback_trade_date(now, intraday)
    except Exception as exc:  # noqa: BLE001 — calendar must not kill a scan
        logger.warning("calendar latest_completed_trading_day failed: %s", exc)
        trade_date = _fallback_trade_date(now, intraday)
    return trade_date, False


def _fallback_trade_date(now: datetime, intraday: bool) -> date:
    """Weekday fallback used when the calendar source is unavailable."""
    day = now.date()
    if intraday or (day.weekday() < 5 and now.time() >= SESSION_OPEN):
        return day
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    if now.weekday() < 5 and now.time() < SESSION_OPEN:
        day -= timedelta(days=1)
    return day


def _overall_status(statuses: list[SourceStatus]) -> str:
    if any(s.status in {"failed", "degraded"} for s in statuses):
        return "partial"
    return "ok"


def _notify(progress: Callable[[str, float], None] | None, stage: str, fraction: float) -> None:
    if progress is None:
        return
    try:
        progress(stage, float(fraction))
    except Exception as exc:  # noqa: BLE001 — progress must never break a scan
        logger.debug("progress callback raised: %s", exc)


#: Exact F2 abort reason when the job lock is lost mid-run.
LOCK_LOST_REASON = "job lock lost (deleted or taken over) - aborted"


class _LockAbort(Exception):
    """Internal: a lock check failed — abort the run cleanly as failed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _is_home_mismatch(exc: BaseException) -> bool:
    """True for pi's HomeMismatch (matched by name until home.py lands)."""
    if type(exc).__name__ == "HomeMismatch":
        return True
    return "pinned home" in str(exc).lower() and "home" in str(exc).lower()


def _require_home(context: str) -> Any:
    """Home pin guard (F1); fallback until pi's home module lands."""
    try:
        from src.watchlist.home import require_home as _pi_require_home
    except ImportError:
        from src.config.paths import get_runtime_root

        return get_runtime_root()
    return _pi_require_home(context)


def _home_fields_safe() -> dict[str, Any] | None:
    """``{"home", "pid"}`` for run rows; None when unresolvable."""
    try:
        from src.watchlist.home import home_fields as _pi_home_fields
    except ImportError:
        try:
            import os

            from src.config.paths import get_runtime_root

            return {"home": str(get_runtime_root()), "pid": os.getpid()}
        except Exception:  # noqa: BLE001
            return None
    try:
        return dict(_pi_home_fields())
    except Exception:  # noqa: BLE001
        return None


def _check_lock(lock_check: Callable[[], None] | None, stage: str) -> None:
    """Run one lock checkpoint; raise :class:`_LockAbort` when lost.

    Any exception from the check aborts the run as failed (never a crash):
    a ``JobLockLost``-shaped error maps to the exact F2 reason, anything else
    to a ``lock check failed at <stage>`` reason.
    """
    if lock_check is None:
        return
    try:
        lock_check()
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if name == "JobLockLost" or "lock lost" in str(exc).lower():
            raise _LockAbort(LOCK_LOST_REASON) from exc
        raise _LockAbort(
            f"lock check failed at {stage}: {name}: {exc}"[:300]) from exc


def _lock_abort_reason(lock_check: Callable[[], None] | None,
                       stage: str) -> str | None:
    """One lock checkpoint; the abort reason, or None when the lock holds."""
    try:
        _check_lock(lock_check, stage)
    except _LockAbort as abort:
        return abort.reason
    return None


def _kline_progress(progress: Callable[[str, float], None] | None) -> Callable | None:
    """Adapt the kline source's progress shape to ``(stage, fraction)``."""

    def _adapter(*args: Any, **kwargs: Any) -> None:
        fraction: float | None = None
        if len(args) >= 2 and all(isinstance(a, (int, float)) for a in args[:2]):
            done, total = float(args[0]), float(args[1])
            fraction = done / total if total else None
        elif len(args) == 1 and isinstance(args[0], (int, float)):
            value = float(args[0])
            fraction = value if 0.0 <= value <= 1.0 else None
        if fraction is not None:
            _notify(progress, "klines", 0.45 + 0.10 * min(1.0, max(0.0, fraction)))

    return _adapter if progress is not None else None


# --------------------------------------------------------------------------- #
# AI review helpers (review.py, CONTRACT §8)
# --------------------------------------------------------------------------- #


def _resolve_llm(
    llm: Callable | None, config: ScanConfig
) -> tuple[Callable | None, str | None]:
    """Resolve the completion callable for the review step.

    Returns ``(callable, None)`` when a review will run, else
    ``(None, skip_note)`` explaining why it will not (disabled config, a
    non-callable/None input, or no constructible provider). Never raises.

    Model selection: ``config.ai_model`` (None = app default from agent\\.env)
    is passed into ``ChatLLM(model_name=...)``; ``config.ai_fallback_model``
    (None = OFF) retries a quota/model error once on the fallback model.
    """
    if not config.ai_enabled:
        return None, "AI review disabled by config"
    if llm is not None:
        # Adapt ChatLLM clients to callable(messages) -> str; plain callables
        # (the test-injection path) pass through untouched. An object that is
        # neither is a hard skip rather than a TypeError on every pick.
        adapted = resolve_llm_callable(llm, timeout=config.ai_call_timeout_s)
        if adapted is None:
            return None, (
                f"llm input not usable: expected a callable or ChatLLM, "
                f"got {type(llm).__name__}"
            )
        return adapted, None
    try:
        return build_default_llm_callable(
            timeout=config.ai_call_timeout_s,
            model=config.ai_model, fallback_model=config.ai_fallback_model
        ), None
    except Exception as exc:  # noqa: BLE001 — provider misconfig must not kill a scan
        logger.info("no LLM provider available: %s", exc)
        return None, f"no LLM provider configured ({type(exc).__name__}: {exc})"


def _fetch_review_inputs(
    bundle: SourceBundle,
    pool_codes: list[str],
    config: ScanConfig,
    trade_date: date,
) -> tuple[dict[str, list], dict[str, list], dict[str, list], list[SourceStatus]]:
    """Fetch news / announcements / red flags for the review pool.

    Red flags come from one batch call; news and announcements are fetched
    per stock. Per-stock failures are isolated (section 8); the per-stock
    statuses collapse into one aggregated status per source via
    :func:`aggregate_source_status`.
    """
    news_map: dict[str, list] = {}
    ann_map: dict[str, list] = {}
    red_map: dict[str, list] = {}

    red_raw, red_status = _invoke(
        "red_flags", bundle.fetch_red_flags, pool_codes, today=trade_date
    )
    if isinstance(red_raw, dict):
        red_map = {normalize_code(k): list(v or []) for k, v in red_raw.items()}

    news_statuses: list[SourceStatus] = []
    ann_statuses: list[SourceStatus] = []
    for code in pool_codes:
        n_raw, n_status = _invoke(
            "news", bundle.fetch_news, code, days=config.news_days, today=trade_date
        )
        news_statuses.append(n_status)
        news_map[code] = list(n_raw) if isinstance(n_raw, list) else []

        a_raw, a_status = _invoke(
            "announcements",
            bundle.fetch_announcements,
            code,
            days=config.news_days,
            today=trade_date,
        )
        ann_statuses.append(a_status)
        ann_map[code] = list(a_raw) if isinstance(a_raw, list) else []

    statuses = [
        aggregate_source_status("news", news_statuses),
        aggregate_source_status("announcements", ann_statuses),
        red_status,
    ]
    return news_map, ann_map, red_map, statuses


def _pool_rows(
    pool_codes: list[str],
    scored: dict[str, ScoredStock],
    uni: pd.DataFrame,
    rule_rank: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Pick-shaped, compact rows for the AI review pool (persisted on the run).

    Reused by ``rerun_ai_review`` so a re-review can backfill after ``avoid``
    without a market rescan.
    """
    rows: list[dict[str, Any]] = []
    for code in pool_codes:
        item = scored[code]
        row = _make_pick(
            uni,
            item,
            rule_rank=(rule_rank or {}).get(code),
            flags=(),
            ai=ai_pending(),
        )
        rows.append(row)
    return rows


def _run_review(
    pool_codes: list[str],
    scored: dict[str, ScoredStock],
    uni: pd.DataFrame,
    news_map: dict[str, list],
    ann_map: dict[str, list],
    red_map: dict[str, list],
    config: ScanConfig,
    llm_fn: Callable | None,
    llm_skip_note: str | None,
    *,
    rule_rank: dict[str, int] | None = None,
) -> tuple[AIReviewSummary, dict[str, dict], list[dict[str, Any]]]:
    """Build one data pack per pool stock and run the AI review pass.

    Returns ``(summary, results, pool_rows)``.
    """
    pool_rows = _pool_rows(pool_codes, scored, uni, rule_rank=rule_rank)
    if not pool_codes:
        return (
            AIReviewSummary(
                status="not_reviewed",
                note="no candidates to review",
                ai_stats=ai_stats_from_results({}),
            ),
            {},
            pool_rows,
        )
    if llm_fn is None:
        note = llm_skip_note or "AI review skipped"
        pending = {code: ai_pending(note) for code in pool_codes}
        for row in pool_rows:
            row["ai"] = dict(pending[normalize_code(row.get("code"))])
        return (
            summarize_ai_results(pending, note=note),
            pending,
            pool_rows,
        )

    packs: dict[str, dict] = {}
    for code in pool_codes:
        item = scored[code]
        packs[code] = build_data_pack(
            code,
            name=_cell(uni, code, "name"),
            board=_cell(uni, code, "board") or board_from_code(code),
            industry=_industry_of(item),
            score=item.score,
            sub_scores={
                "fundamental": item.fundamental,
                "trend": item.trend,
                "flow": item.flow,
            },
            metrics=item.metrics,
            rule_reasons=item.reasons,
            news=news_map.get(code) or [],
            announcements=ann_map.get(code) or [],
            red_flags=red_map.get(code) or [],
        )
    # Meter the LLM calls (section 9b): count completions against the LLM
    # base-URL host, without ever recording the key.
    llm_host = "llm"
    try:
        llm_host = _llm_host(llm_fn)
    except Exception:  # noqa: BLE001 — metering must never break the review
        pass

    def metered_llm(messages):
        from src.market_scanner.sources._common import get_active_meter

        meter = get_active_meter()
        t0 = time.monotonic()
        try:
            result = llm_fn(messages)
        except Exception:
            if meter is not None:
                meter.record(llm_host, ok=False, elapsed_s=time.monotonic() - t0)
            raise
        if meter is not None:
            meter.record(llm_host, ok=True, elapsed_s=time.monotonic() - t0)
        return result

    summary = run_ai_review(
        pool_codes, packs_by_code=packs, llm=metered_llm, config=config
    )
    for row in pool_rows:
        code = normalize_code(row.get("code"))
        row["ai"] = dict(summary.results.get(code) or ai_pending())
    return summary, summary.results, pool_rows


def _llm_host(llm_fn: Callable) -> str:
    """Best-effort base-URL host from the LLM client object (never the key)."""
    import urllib.parse

    from src.providers.capabilities import get_llm_credentials  # config layer

    creds = get_llm_credentials(None, None)  # config layer; no raw env reads
    for attr in ("openai_api_base", "openai_base_url", "base_url", "_base_url"):
        client = getattr(llm_fn, "__self__", None) or llm_fn
        value = getattr(client, attr, None) or creds.get("base_url")
        if value:
            return urllib.parse.urlparse(str(value)).hostname or "llm"
    return "llm"


# --------------------------------------------------------------------------- #
# AI re-review (no market rescan)
# --------------------------------------------------------------------------- #


def _cell_or(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key)
    return default if value is None else value


def _packs_from_rows(
    pool_rows: list[dict[str, Any]],
    news_map: dict[str, list],
    ann_map: dict[str, list],
    red_map: dict[str, list],
) -> dict[str, dict]:
    """Rebuild one citable data pack per pool row (re-review path)."""
    packs: dict[str, dict] = {}
    for row in pool_rows:
        code = normalize_code(row.get("code"))
        sub = row.get("sub_scores") or {}
        packs[code] = build_data_pack(
            code,
            name=_cell_or(row, "name"),
            board=_cell_or(row, "board"),
            industry=_cell_or(row, "industry"),
            score=_cell_or(row, "score"),
            sub_scores={
                "fundamental": sub.get("fundamental"),
                "trend": sub.get("trend"),
                "flow": sub.get("flow"),
            },
            metrics=row.get("metrics"),
            rule_reasons=row.get("reasons") or [],
            news=news_map.get(code) or [],
            announcements=ann_map.get(code) or [],
            red_flags=red_map.get(code) or [],
        )
    return packs


def _rebuild_ai_run(
    run: dict[str, Any],
    pool_rows: list[dict[str, Any]],
    ai_results: dict[str, dict],
    ai_summary: AIReviewSummary,
    *,
    revision: int,
    reviewed_at: str,
    note: str,
    picks_only: bool,
    pinned: Any = frozenset(),
    suppressed: Any = frozenset(),
) -> dict[str, Any]:
    """Recompute picks / ai_demoted / requalified / funnel for a re-review.

    ``pinned`` / ``suppressed`` carry the watchlist's manual pins and
    removals into the re-selection so it honours them exactly like a scan
    (pinned beats suppression; a pinned code is never AI-demoted out of
    picks).
    """
    pool_codes = [normalize_code(r.get("code")) for r in pool_rows]
    rule_rank = {c: int(_cell_or(r, "rule_rank") or i + 1) for i, (c, r) in enumerate(zip(pool_codes, pool_rows))}
    scored: dict[str, ScoredStock] = {}
    uni_rows: dict[str, dict] = {}
    for row in pool_rows:
        code = normalize_code(row.get("code"))
        sub = row.get("sub_scores") or {}
        metrics = dict(row.get("metrics") or {})
        industry = _cell_or(row, "industry") or metrics.get("industry")
        if industry and not metrics.get("industry"):
            metrics["industry"] = industry
        scored[code] = ScoredStock(
            code=code,
            score=float(_cell_or(row, "score", 0.0) or 0.0),
            fundamental=float(sub.get("fundamental") or 0.0),
            trend=float(sub.get("trend") or 0.0),
            flow=float(sub.get("flow") or 0.0),
            metrics=metrics,
            reasons=list(row.get("reasons") or []),
            flags=list(row.get("flags") or []),
        )
        # Minimal universe frame so _make_pick recovers name/board/industry.
        uni_rows[code] = {
            "name": _cell_or(row, "name"),
            "board": _cell_or(row, "board") or board_from_code(code),
            "industry": industry,
            "symbol": symbol_from_code(code),
        }
    config = ScanConfig.from_dict(run.get("config") or {})
    suppressed = {normalize_code(c) for c in (suppressed or set())}
    pinned = {normalize_code(c) for c in (pinned or set())}
    if picks_only:
        # No backfill: only the previously-finalised picks are re-ordered.
        pool_codes = [p.get("code") for p in run.get("picks") or []]
        rule_rank = {c: i + 1 for i, c in enumerate(pool_codes)}
    uni = pd.DataFrame.from_dict(uni_rows, orient="index")
    if not uni.empty:
        uni.index.name = "code"
    picks, ai_demoted, requalified, pinned_entries, funnel_tail = _select_picks(
        uni=uni,
        scored=scored,
        exclusion_reasons={},
        config=config,
        suppressed=suppressed,
        pinned=pinned,
        pool_codes=pool_codes,
        rule_rank=rule_rank,
        ai_results=ai_results,
        finalists_count=min(config.top_n, len(pool_codes)),
    )
    funnel = list(run.get("funnel") or [])[: -len(funnel_tail)] if funnel_tail else list(run.get("funnel") or [])
    funnel = funnel + funnel_tail
    funnel = _make_funnel_consistent(funnel)
    updated = dict(run)
    updated.update(
        {
            "picks": picks,
            "ai_demoted": ai_demoted,
            "requalified": requalified,
            "pinned": pinned_entries,
            "ai_pool": pool_rows,
            "ai_status": ai_summary.status,
            "ai_note": note,
            "ai_stats": ai_summary.ai_stats,
            "funnel": funnel,
            "review_revision": revision,
            "reviewed_at": reviewed_at,
        }
    )
    return updated


def rerun_ai_review(
    run: dict[str, Any],
    config: ScanConfig,
    *,
    llm: Any = None,
    sources: object | None = None,
    progress: Callable[[str, float], None] | None = None,
    now: datetime | None = None,
    pinned: Any = frozenset(),
    suppressed: Any = frozenset(),
) -> dict[str, Any]:
    """Re-run only the AI review on a stored ScanRun; no market rescan.

    Re-fetches news / announcements / red flags for the review pool (cheap),
    reviews the pool again, recomputes picks / ai_demoted / requalified
    ordering exactly like a scan, and returns an updated ScanRun with
    ``review_revision += 1`` and ``reviewed_at`` set.

    For old runs without ``ai_pool``, only the stored picks are reviewed (no
    backfill) and the note says so.

    Args:
        run: A stored ScanRun dict.
        config: Scan settings for this re-review.
        llm: Injected completion callable (or ChatLLM); ``None`` = provider.
        sources: Source bundle for the cheap news/ann/red-flag fetches.
        progress: Optional ``(stage, fraction)`` callback.
        now: Injected clock; defaults to now (Asia/Shanghai).
        pinned: Manual watchlist pins to carry into re-selection (pinned
            beats suppression; a pinned code is never AI-demoted out of
            picks — exactly like a scan).
        suppressed: Suppressed watchlist codes (``removed`` state); selected
            ones surface as ``requalified``, like in a scan.

    Returns:
        An updated ScanRun dict (the input ``run`` is not mutated).
    """
    if not isinstance(run, dict):
        raise ValueError(f"run must be a ScanRun dict, got {type(run).__name__}")
    started = _coerce_now(now)
    bundle = SourceBundle.resolve(sources)
    _notify(progress, "ai_review", 0.1)

    pool_rows_raw = run.get("ai_pool") or []
    picks = [dict(p) for p in run.get("picks") or []]
    picks_by_code = {
        normalize_code(p.get("code")): p for p in picks if p.get("code") is not None
    }
    picks_only = not pool_rows_raw
    if picks_only:
        pool_rows = picks
    else:
        pool_rows = [dict(r) for r in pool_rows_raw]
        for row in pool_rows:
            code = normalize_code(row.get("code"))
            pick_ai = picks_by_code.get(code, {}).get("ai")
            row_ai = row.get("ai") or {}
            if (
                isinstance(pick_ai, dict)
                and pick_ai.get("status") in {"reviewed", "failed"}
                and row_ai.get("status") == "not_reviewed"
            ):
                row["ai"] = dict(pick_ai)
    pool_codes = [normalize_code(r.get("code")) for r in pool_rows]

    trade_date = _as_date(run.get("trade_date")) or started.date()

    llm_fn, llm_skip_note = _resolve_llm(llm, config)
    existing_results = {
        normalize_code(row.get("code")): dict(row.get("ai") or ai_pending())
        for row in pool_rows
    }
    review_codes = [
        code
        for code in pool_codes
        if existing_results[code].get("status") in {"failed", "not_reviewed"}
    ]
    skipped_reviewed = len(pool_codes) - len(review_codes)
    if llm_fn is None:
        ai_results = existing_results
        note = (
            f"{llm_skip_note or 'AI review skipped'}; "
            f"skipped already reviewed: {skipped_reviewed}"
        )
        summary = summarize_ai_results(ai_results, note=note)
    elif review_codes:
        _notify(progress, "news", 0.3)
        news_map, ann_map, red_map, _statuses = _fetch_review_inputs(
            bundle, review_codes, config, trade_date
        )
        review_rows = [
            row
            for row in pool_rows
            if normalize_code(row.get("code")) in review_codes
        ]
        packs = _packs_from_rows(review_rows, news_map, ann_map, red_map)
        _notify(progress, "ai_review", 0.6)
        review_summary = run_ai_review(
            review_codes, packs_by_code=packs, llm=llm_fn, config=config
        )
        ai_results = dict(existing_results)
        ai_results.update(review_summary.results)
        note = (
            f"{review_summary.note}; skipped already reviewed: "
            f"{skipped_reviewed}"
        )
        summary = summarize_ai_results(
            ai_results, note=note, forced=review_summary.forced
        )
    else:
        ai_results = existing_results
        note = f"AI review: no calls; skipped already reviewed: {skipped_reviewed}"
        summary = summarize_ai_results(ai_results, note=note)

    for row in pool_rows:
        row["ai"] = dict(ai_results[normalize_code(row.get("code"))])

    if picks_only:
        note = f"{summary.note}; no ai_pool on this run, picks only (no backfill)"
        summary = AIReviewSummary(
            status=summary.status,
            note=note,
            results=summary.results,
            reviewed=summary.reviewed,
            failed=summary.failed,
            demoted=summary.demoted,
            forced=summary.forced,
            quota_failed=summary.quota_failed,
            ai_stats=summary.ai_stats,
        )
    revision = int(run.get("review_revision") or 0) + 1
    reviewed_at = started.isoformat()
    updated = _rebuild_ai_run(
        run,
        pool_rows,
        ai_results,
        summary,
        revision=revision,
        reviewed_at=reviewed_at,
        note=summary.note,
        picks_only=picks_only,
        pinned=pinned,
        suppressed=suppressed,
    )
    _notify(progress, "done", 1.0)
    return updated


# --------------------------------------------------------------------------- #
# Envelope helpers
# --------------------------------------------------------------------------- #


def _failed_run(
    *,
    run_id: str,
    started: datetime,
    t0: float,
    trigger: str,
    config: ScanConfig,
    statuses: list[SourceStatus],
    trade_date: date,
    intraday: bool,
    progress: Callable[[str, float], None] | None,
    note: str,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Minimal ScanRun for the fatal case where the universe is empty."""
    finished = _coerce_now(None)
    _notify(progress, "done", 1.0)
    _total = round(time.monotonic() - t0, 3)
    return {
        "run_id": run_id,
        "trade_date": trade_date.isoformat(),
        "intraday": intraday,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": _total,
        "trigger": trigger,
        "status": "failed",
        "failure_reason": failure_reason,
        "ai_status": AI_STATUS_PENDING,
        "ai_note": note,
        "ai_stats": ai_stats_from_results({}),
        "kline_reasons": None,
        "home": _home_fields_safe(),
        "coverage": {
            "passed_exclusions": 0,
            "insufficient": 0,
            "insufficient_by_key": {},
            "scored": 0,
            "trend_scored": 0,
            "flow_real": 0,
            "flow_neutral": 0,
            "trend_coverage": 1.0,
            "fundamentals_coverage": 1.0,
            "min_coverage": config.min_coverage,
        },
        "coverage_note": None,
        "config": config.to_dict(),
        "funnel": [make_funnel_entry(STAGE_UNIVERSE, 0, 0, note)],
        "sources": [s.to_dict() for s in statuses],
        "picks": [],
        "ai_demoted": [],
        "requalified": [],
        "pinned": [],
        "scores": [],
        "timings": {"total": _total},
    }


def _abort_run(
    run_id: str,
    started: datetime,
    t0: float,
    trigger: str,
    config: ScanConfig,
    statuses: list[SourceStatus],
    trade_date: date,
    intraday: bool,
    progress: Callable[[str, float], None] | None,
    reason: str,
) -> dict[str, Any]:
    """Failed envelope for a lock abort (F2) or home refusal (F1).

    Same shape as :func:`_failed_run` with the abort reason surfaced as the
    failure reason, so history/CLI render it instead of a crash.
    """
    return _failed_run(
        run_id=run_id, started=started, t0=t0, trigger=trigger,
        config=config, statuses=statuses, trade_date=trade_date,
        intraday=intraday, progress=progress, note=reason,
        failure_reason=reason,
    )


def _early_survivors(
    uni: pd.DataFrame,
    config: ScanConfig,
    listing: pd.DataFrame | None,
    trade_date: date,
    klines: dict[str, pd.DataFrame] | None = None,
) -> list[str]:
    """Codes that pass the data-free rules — the kline/fundamentals fetch set.

    ``klines`` is optional (and usually ``None`` at call time): the K-line
    history fallback for the ``new_listing`` age rule cannot run before the
    fetch. Any stock whose age is still unknown at this point is conservatively
    kept in the fetch set (so it can later be judged by its history) rather
    than being mis-labelled ``new_listing``.
    """
    from .exclusions import apply_early_rules

    early = apply_early_rules(
        uni, config, listing_df=listing, trade_date=trade_date, klines=klines or {}
    )
    return [code for code in uni.index if early.get(code) is None]


def _symbols_for(uni: pd.DataFrame, codes: list[str]) -> list[str]:
    symbols: list[str] = []
    for code in codes:
        symbol = _cell(uni, code, "symbol")
        symbols.append(str(symbol) if symbol else symbol_from_code(code))
    return symbols


def _normalize_klines(raw: Any, uni: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Accept ``dict[code, df]`` (contract) and tolerate symbol keys."""
    out: dict[str, pd.DataFrame] = {}
    if not isinstance(raw, dict):
        return out
    symbol_to_code = {
        str(_cell(uni, code, "symbol") or symbol_from_code(code)): code
        for code in uni.index
    }
    for key, frame in raw.items():
        key_str = str(key)
        if key_str in uni.index:
            code = key_str
        else:
            code = symbol_to_code.get(key_str)
        if code is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        out[code] = frame
    return out


def _cell(uni: pd.DataFrame, code: str, key: str) -> Any:
    if uni is None or uni.empty or code not in uni.index:
        return None
    row = uni.loc[code]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    try:
        value = row[key]
    except Exception:  # noqa: BLE001
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


def _coerce_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(tz=SHANGHAI)
    if now.tzinfo is None:
        return now.replace(tzinfo=SHANGHAI)
    return now.astimezone(SHANGHAI)


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


# --------------------------------------------------------------------------- #
# Coverage (CONTRACT §7 diagnostic block)
# --------------------------------------------------------------------------- #


def _build_coverage(
    funnel: list[dict[str, Any]], scored: dict[str, ScoredStock]
) -> dict[str, Any]:
    """Derive the coverage block from the funnel and scored set.

    ``passed_exclusions`` counts stocks surviving every ``excluded:*`` stage;
    ``insufficient`` counts ``insufficient_data:*`` removals; ``scored`` counts
    composite scores; ``trend_scored`` counts trend sub-scores; ``flow_real`` /
    ``flow_neutral`` split real flow data from the neutral-50 ``flow_missing``
    fill. The caller asserts ``passed_exclusions - insufficient == scored ==
    trend_scored == flow_real + flow_neutral``.
    """
    universe_count = 0
    excluded_removed = 0
    insufficient = 0
    insufficient_by_key: dict[str, int] = {}
    for entry in funnel:
        stage = str(entry.get("stage") or "")
        if stage == STAGE_UNIVERSE:
            universe_count = int(entry.get("count") or 0)
        elif stage.startswith("excluded:"):
            excluded_removed += int(entry.get("removed") or 0)
        elif stage.startswith("insufficient_data:"):
            removed = int(entry.get("removed") or 0)
            insufficient += removed
            if removed:
                insufficient_by_key[stage] = insufficient_by_key.get(stage, 0) + removed
    passed_exclusions = universe_count - excluded_removed
    scored_count = len(scored)
    trend_scored = sum(1 for item in scored.values() if item.trend is not None)
    flow_neutral = sum(
        1 for item in scored.values() if "flow_missing" in (item.flags or [])
    )
    flow_real = scored_count - flow_neutral
    return {
        "passed_exclusions": passed_exclusions,
        "insufficient": insufficient,
        "insufficient_by_key": insufficient_by_key,
        "scored": scored_count,
        "trend_scored": trend_scored,
        "flow_real": flow_real,
        "flow_neutral": flow_neutral,
    }


def _coverage_invariant_holds(coverage: dict[str, int]) -> bool:
    """True when the coverage arithmetic is internally consistent."""
    base = coverage["passed_exclusions"] - coverage["insufficient"]
    return (
        base == coverage["scored"]
        == coverage["trend_scored"]
        == coverage["flow_real"] + coverage["flow_neutral"]
    )


def _coverage_ratios(coverage: dict[str, Any]) -> tuple[float, float]:
    """``(trend_coverage, fundamentals_coverage)`` in [0, 1].

    trend = trend_scored / passed_exclusions; fundamentals =
    (passed_exclusions - insufficient_data:fundamentals) / passed_exclusions.
    A run that passed no exclusions has vacuous coverage (1.0).
    """
    passed = int(coverage.get("passed_exclusions") or 0)
    if passed <= 0:
        return 1.0, 1.0
    trend_cov = float(coverage.get("trend_scored") or 0) / passed
    insuff_fund = int(
        (coverage.get("insufficient_by_key") or {}).get(
            "insufficient_data:fundamentals", 0
        )
    )
    fund_cov = (passed - insuff_fund) / passed
    return trend_cov, fund_cov


def _coverage_guard_failure(
    coverage: dict[str, Any], min_coverage: float
) -> str | None:
    """Failure reason when either coverage ratio is below the floor."""
    trend_cov, fund_cov = _coverage_ratios(coverage)
    if trend_cov < min_coverage:
        return (
            f"insufficient coverage (trend {trend_cov * 100:.1f}% "
            f"< {min_coverage * 100:g}%)"
        )
    if fund_cov < min_coverage:
        return (
            f"insufficient coverage (fundamentals {fund_cov * 100:.1f}% "
            f"< {min_coverage * 100:g}%)"
        )
    return None
