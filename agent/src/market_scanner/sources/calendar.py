"""Trading calendar: sina hist via akshare, cached JSON, weekday fallback."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.market_scanner.sources._common import meter_akshare_call
from src.market_scanner.sources.status import SourceStatus

logger = logging.getLogger(__name__)

_NAME = "calendar"

_CACHE_SUBDIR = Path("market_scanner") / "cache"
_CACHE_FILE = "trade_calendar.json"


@dataclass
class TradingCalendar:
    """Trading-day set with helpers; weekday fallback when fetch fails."""

    trading_days: frozenset[str] = field(default_factory=frozenset)
    source: str = "weekday-fallback"

    def is_trading_day(self, d: _dt.date) -> bool:
        """True when ``d`` is a known trading day (weekday fallback applies)."""
        key = d.isoformat()
        if self.trading_days:
            return key in self.trading_days
        return d.weekday() < 5

    def previous_trading_day(self, d: _dt.date) -> _dt.date:
        """The latest trading day strictly before ``d``."""
        cur = d - _dt.timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(cur):
                return cur
            cur -= _dt.timedelta(days=1)
        return cur

    def latest_completed_trading_day(self, now: _dt.datetime) -> _dt.date:
        """Today when today trades and now >= 15:00 Asia/Shanghai, else previous.

        Args:
            now: Naive or tz-aware datetime interpreted as Asia/Shanghai.
        """
        today = now.date() if not now.tzinfo else now.astimezone(
            _dt.timezone(_dt.timedelta(hours=8))
        ).date()
        clock = now.time() if not now.tzinfo else now.astimezone(
            _dt.timezone(_dt.timedelta(hours=8))
        ).time()
        if self.is_trading_day(today) and clock >= _dt.time(15, 0):
            return today
        return self.previous_trading_day(today)


def _cache_path() -> Path:
    """Cache file path under get_runtime_root()/market_scanner/cache."""
    from src.config.paths import get_runtime_root

    path = get_runtime_root() / _CACHE_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path / _CACHE_FILE


def _read_cache() -> list[str] | None:
    """Cached day list, or None when absent/invalid."""
    path = _cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        days = payload.get("trading_days")
        if isinstance(days, list) and days:
            return days
    except (OSError, ValueError):
        return None
    return None


def _write_cache(days: list[str]) -> None:
    """Best-effort cache write; failure is logged, never raised."""
    try:
        _cache_path().write_text(
            json.dumps({"saved_at": _dt.date.today().isoformat(), "trading_days": days}),
            encoding="utf-8",
        )
    except OSError:
        logger.exception("calendar cache write failed")


def load_calendar(
    hist_fn: Callable | None = None,
    force_refresh: bool = False,
) -> tuple[TradingCalendar, SourceStatus]:
    """Trading calendar with JSON cache and weekday fallback.

    Args:
        hist_fn: Injectable for ``ak.tool_trade_date_hist_sina`` (tests pass a
            fixture returning a DataFrame with a ``trade_date`` column).
        force_refresh: Ignore any cached copy.

    Returns:
        ``(calendar, st)``. ``degraded`` when falling back to weekdays or the
        cache; ``ok`` when freshly fetched.
    """
    t0 = time.monotonic()

    days: list[str] | None = None
    source = "weekday-fallback"
    error: str | None = None

    if not force_refresh:
        days = _read_cache()
        if days:
            source = "cache"

    if days is None:
        if hist_fn is None:
            import akshare as ak
            hist_fn = ak.tool_trade_date_hist_sina
        try:
            df = meter_akshare_call(hist_fn, host="sina")
            days = [str(v)[:10] for v in df["trade_date"].tolist()]
            source = "sina"
            _write_cache(days)
        except Exception as exc:  # noqa: BLE001 — fallback, never raise
            error = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("calendar fetch failed (%s); weekday fallback", error)
            days = None

    if days is None:
        # Pure weekday fallback: still store the (empty) fact so subsequent
        # calls do not re-hit the network every time.
        calendar = TradingCalendar(frozenset(), source="weekday-fallback")
        st = SourceStatus(
            name=_NAME, status="degraded", rows=0,
            latency_s=time.monotonic() - t0,
            error=error,
            detail="weekday fallback (Mon–Fri treated as trading days); CN holidays not excluded",
        )
        return calendar, st

    calendar = TradingCalendar(frozenset(days), source=source)
    # Fresh cache read is fine: it was written from a successful Sina fetch.
    # 'degraded' only for the weekday fallback (no holiday knowledge) or an
    # expired cache that could not refresh (source=cache AND error set).
    status = "degraded" if source == "weekday-fallback" else "ok"
    st = SourceStatus(
        name=_NAME, status=status, rows=len(days),
        latency_s=time.monotonic() - t0,
        error=error if (source == "cache" and error) else None,
        detail=f"source={source} range={days[0]}..{days[-1]}" if days else None,
        as_of=days[-1] if days else None,  # calendar range end
    )
    return calendar, st
