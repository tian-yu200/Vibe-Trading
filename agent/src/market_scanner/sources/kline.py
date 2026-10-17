"""Daily K-lines (qfq) from Tencent fqkline: SQLite cache + snapshot append.

Fetch strategy (``fetch_klines`` with ``snapshot``/``trade_date``/``calendar``):
* (a) cached history whose last bar is the previous trading day AND whose
  last close matches the snapshot settlement (prev close) within 0.3% ->
  append trade_date's bar from the snapshot. No network call.
* (b) mismatch (dividend/split), gap, no cache, or cache older than 30 days
  -> full refetch from Tencent (rate-limited, budget-capped). Small gaps
  (missing trading days <= _MAX_GAPFILL_DAYS) are tail-filled instead:
  fetch gap+overlap bars, verify the overlap closes (and STAR volume units),
  merge; full refetch only on a mismatch.
* suspended / no snapshot bar -> cached frame kept as is.
* Throttle stop: 10 consecutive stop-class errors (HTTP 501/403/429,
  connection reset, OR transport errors: ReadTimeout/ConnectTimeout/
  ConnectionError/RemoteDisconnected) stop ALL further kline fetches.
  A second rule stops on >50% errors over the last 60 requests once the
  kline stage is >= 120 s old. Both mark the source unavailable
  ("price source unavailable"). Transport errors never feed known-missing:
  the next run retries them (F5d).
* Time budget: once exceeded, remaining symbols stay missing (they become
  ``insufficient_data:trend`` in the funnel). The scan budget scales with
  the remaining shortfall (F5) up to a cap.
"""

from __future__ import annotations

import datetime as _dt
import dataclasses
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd

from src.market_scanner.sources._common import get_active_meter
from src.market_scanner.sources.kline_cache import KlineCache, MAX_AGE_DAYS
from src.market_scanner.sources.status import SourceStatus

logger = logging.getLogger(__name__)

_NAME = "kline"

# web.ifzq.gtimg.cn started returning 501 (2026-10-13); the bare ifzq.gtimg.cn
# host serves the same fqkline endpoint and works when the block lifts.
_FQKLINE_URL = "https://ifzq.gtimg.cn/appstock/app/fqkline/get"
_KLINE_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]
_MAX_WORKERS = 6
_LOT_SIZE = 100  # A-shares trade in lots of 100 shares; qfq volume is lots.
_CLOSE_MATCH_TOLERANCE = 0.003  # 0.3% settlement vs cached close

_THROTTLE_STATUS_CODES = {501, 403, 429}
_THROTTLE_STOP_AFTER = 10  # consecutive stop-class errors -> stop the run

# Gap-fill policy (F5a): a gap of this many trading days or fewer is repaired
# by fetching only the tail (gap + overlap); larger gaps refetch fully.
_MAX_GAPFILL_DAYS = 10
_GAPFILL_OVERLAP = 3  # overlapping bars verified before a merge

# Volume-unit sanity band for a gap-fill merge: STAR lots vs shares differ by
# 100x, so a median overlap-volume ratio outside [0.2, 5] means a unit
# mismatch -> full refetch instead of a merge.
_VOLUME_RATIO_LO = 0.2
_VOLUME_RATIO_HI = 5.0

import os  # noqa: E402

_JITTER_MAX_S = 0.05


class KlineRateLimiter:
    """K-line spacing gate (rate configurable) + throttle circuit breaker."""

    _WINDOW = 50
    _ERROR_THRESHOLD = 0.20

    def __init__(self, rate_per_sec: float | None = None, *,
                 stop_consecutive: int = _THROTTLE_STOP_AFTER,
                 stop_window: int = 60,
                 stop_min_error_rate: float = 0.5,
                 stop_min_elapsed_s: float = 120.0,
                 clock: Callable[[], float] | None = None) -> None:
        rate = rate_per_sec
        if rate is None:
            # env-schema gated (CI env-var gate): kline default 1 req/s
            from src.config.accessor import get_env_config

            rate = get_env_config().agent_tuning.vibe_trading_kline_rate_per_sec
        self._base_rate = max(0.1, min(20.0, float(rate)))
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._slowdowns = 0
        self._results: list[bool] = []
        self.throttle_stop = False
        self.last_throttle_at: str | None = None
        self.last_throttle_desc: str | None = None
        # F4 outage stop (transport errors count alongside throttle errors).
        self._stop_consecutive = max(1, int(stop_consecutive))
        self._stop_window = max(10, int(stop_window))
        self._stop_min_error_rate = float(stop_min_error_rate)
        self._stop_min_elapsed_s = float(stop_min_elapsed_s)
        self._clock = clock or time.monotonic
        self._consecutive_stop_errors = 0
        self._stop_window_results: list[bool] = []
        self._run_start: float | None = None
        self._last_stop_kind: str | None = None  # "throttle" | "transport"
        self.unavailable = False
        self.unavailable_reason: str | None = None
        # F10: accumulated per-request network seconds (from elapsed_s).
        # The scan budget gates on this, never on wall time, so local cache
        # I/O cannot eat the network budget.
        self._network_s = 0.0

    def over_network_budget(self, budget_s: float | None) -> bool:
        """True when the accumulated network time exceeds ``budget_s``."""
        if budget_s is None:
            return False
        with self._lock:
            return self._network_s >= float(budget_s)

    @property
    def network_s(self) -> float:
        """Total per-request network seconds recorded so far."""
        with self._lock:
            return self._network_s

    @property
    def min_interval(self) -> float:
        return 1.0 / (self._base_rate / (2**self._slowdowns))

    def wait(self, clock: Callable[[], float] = time.monotonic,
             sleeper: Callable[[float], None] = time.sleep) -> None:
        """Block until the next slot (fake clock injectable)."""
        with self._lock:
            now = clock()
            start = max(now, self._next_slot)
            jitter = _JITTER_MAX_S * (0 if self._slowdowns else 1)
            import random
            fire_at = start + random.uniform(0.0, jitter)
            self._next_slot = fire_at + self.min_interval
        delay = fire_at - clock()
        if delay > 0:
            sleeper(delay)

    def note_result(self, ok: bool, *, throttle: bool = False,
                    transport: bool = False,
                    desc: str | None = None, elapsed_s: float = 0.0) -> dict[str, bool]:
        """Record one request outcome.

        Also feeds the run-scoped host meter (section 9b) for the tencent host.

        ``throttle`` marks throttle-type failures (HTTP 501/403/429,
        connection reset); ``transport`` marks transport failures
        (ReadTimeout/ConnectTimeout/ConnectionError/RemoteDisconnected, F4).
        Either counts toward the stop rules; only ``ok`` resets the
        consecutive counter.

        Returns:
            ``{"breaker_tripped": bool, "throttle_stop": bool,
            "unavailable": bool}``.
        """
        meter = get_active_meter()
        if meter is not None:
            meter.record("tencent", ok=ok, elapsed_s=elapsed_s)
        with self._lock:
            flags: dict[str, bool] = {"breaker_tripped": False,
                                       "throttle_stop": False,
                                       "unavailable": False}
            try:
                self._network_s += max(0.0, float(elapsed_s))
            except (TypeError, ValueError):
                pass
            now = self._clock()
            if self._run_start is None:
                self._run_start = now
            stop_error = bool(throttle or transport) and not ok
            if ok:
                self._consecutive_stop_errors = 0
            elif stop_error:
                self._consecutive_stop_errors += 1
                self._last_stop_kind = "transport" if transport else "throttle"
                self.last_throttle_at = _dt.datetime.now().strftime("%H:%M")
                self.last_throttle_desc = desc
                if self._consecutive_stop_errors >= self._stop_consecutive:
                    self._mark_unavailable(
                        f"{self._consecutive_stop_errors} consecutive "
                        f"{self._last_stop_kind} failures")
            if not ok and not stop_error:
                # Permanent per-symbol failure (bad symbol, empty payload):
                # not stop-class, so it neither increments nor resets the
                # outage counter. F5d: these are the only failures that feed
                # the 24 h known-missing registry.
                pass

            self._results.append(ok)
            if len(self._results) > self._WINDOW:
                self._results.pop(0)
            if len(self._results) >= self._WINDOW:
                error_rate = 1.0 - sum(self._results) / len(self._results)
                if error_rate > self._ERROR_THRESHOLD:
                    self._slowdowns += 1
                    self._results.clear()
                    flags["breaker_tripped"] = True

            self._stop_window_results.append(bool(ok))
            if len(self._stop_window_results) > self._stop_window:
                self._stop_window_results.pop(0)
            if (not self.unavailable
                    and len(self._stop_window_results) >= self._stop_window
                    and (now - (self._run_start or now)) >= self._stop_min_elapsed_s):
                window = self._stop_window_results
                error_rate = 1.0 - sum(window) / len(window)
                if error_rate > self._stop_min_error_rate:
                    self._mark_unavailable(
                        f"{error_rate:.0%} errors over the last "
                        f"{len(window)} requests")

            flags["throttle_stop"] = self.throttle_stop
            flags["unavailable"] = self.unavailable
            return flags

    def _mark_unavailable(self, reason: str) -> None:
        """Latch the F4 outage stop (caller holds the lock)."""
        self.throttle_stop = True
        self.unavailable = True
        self.unavailable_reason = reason

    @property
    def breaker_trips(self) -> int:
        """Number of circuit-breaker trips (for status.detail)."""
        return self._slowdowns


_KLINE_LIMITER = KlineRateLimiter()


def is_throttle_error(exc: BaseException, status_code: int | None = None) -> bool:
    """True for throttle-type failures (HTTP 501/403/429, connection reset)."""
    if status_code is not None and status_code in _THROTTLE_STATUS_CODES:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    for code in _THROTTLE_STATUS_CODES:
        if str(code) in text and ("http" in text or "server error" in text
                                  or "too many" in text):
            return True
    return "connection reset" in text or "remote end closed" in text


# Transport failures that count toward the F4 outage stop alongside throttle
# errors. Matched by exception name/text (no hard dependency on requests /
# urllib3, so fake fetchers in tests classify the same way). Deliberately NOT
# matching is_throttle_error's "connection reset"/"remote end closed": those
# stay throttle-class and are checked first.
_TRANSPORT_MARKERS = (
    "readtimeout",
    "read timed out",
    "connecttimeout",
    "connect timed out",
    "connectionerror",
    "connection aborted",
    "remotedisconnected",
    "remote disconnected",
    "timeouterror",  # LLM-style "timed out after Ns" surfaced per-symbol
)


def is_transport_error(exc: BaseException) -> bool:
    """True for transport failures (F4 stop-class, retried next run)."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _TRANSPORT_MARKERS)


def _parse_kline_payload(text: str, symbol: str) -> pd.DataFrame | None:
    """Parse one fqkline response body into a contract frame (or None)."""
    data = json.loads(text)
    stock = (data.get("data") or {}).get(symbol) or {}
    rows_raw = stock.get("qfqday") or stock.get("day") or []
    if not rows_raw:
        return None
    records = []
    for k in rows_raw:
        if len(k) < 6:
            continue
        try:
            records.append({
                "date": k[0],
                "open": float(k[1]),
                "close": float(k[2]),
                "high": float(k[3]),
                "low": float(k[4]),
                "volume": float(k[5]),
                "amount": None,
            })
        except (TypeError, ValueError):
            continue
    if not records:
        return None
    df = pd.DataFrame.from_records(records)[_KLINE_COLUMNS]
    # Volume units (live-verified 2026-10-13 vs Sina): Tencent fqkline qfq
    # volume is LOTS (x100 shares) for SH-main/SZ-main/ChiNext, but SHARES
    # for STAR (688/689 prefixes) — and (unverified for BSE 4/8/9xx) we apply
    # the same shares->lots normalisation to BSE codes as a precaution.
    # Normalise STAR/BSE to LOTS and store amount = close*volume*100.
    code = symbol[2:] if len(symbol) > 6 else symbol
    if code.startswith(("688", "689")):
        df["volume"] = df["volume"] / _LOT_SIZE
    df["amount"] = df["close"] * df["volume"] * _LOT_SIZE
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _fetch_one(
    symbol: str,
    bars: int,
    limiter: KlineRateLimiter | None = None,
    fetch: Callable | None = None,
) -> pd.DataFrame:
    """Fetch one symbol's last ``bars`` qfq daily bars. Raises on failure."""
    param = f"{symbol},day,,,{bars},qfq"
    limiter = limiter or _KLINE_LIMITER
    if fetch is None:
        from src.market_scanner.sources._common import make_session
        from backtest.loaders._http import DEFAULT_USER_AGENT

        session = make_session()

        def _do_get() -> tuple[str, int | None]:
            limiter.wait()
            resp = session.get(
                _FQKLINE_URL, params={"param": param},
                headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=15,
            )
            return resp.text, resp.status_code

        text, status = _do_get()
    else:
        text = fetch(_FQKLINE_URL, {"param": param})
        status = None
    if status is not None and status in _THROTTLE_STATUS_CODES:
        raise ThrottleError(f"HTTP {status} from fqkline", status_code=status)
    df = _parse_kline_payload(text, symbol)
    if df is None or df.empty:
        raise ValueError(f"no kline rows for {symbol}")
    return df


class ThrottleError(RuntimeError):
    """A throttle-type provider failure (HTTP 501/403/429 or connection reset)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# --------------------------------------------------------------------------- #
# One-time cache migration: STAR volume shares->lots (2026-10-13 finding)
# --------------------------------------------------------------------------- #
STAR_MIGRATION_KEY = "star_volume_migration_v1"


def migrate_star_volume(cache: KlineCache) -> int:
    """One-time in-place cache migration (NO network): STAR (688/689) bars
    were ingested with Tencent's SHARES volume under the lots convention, so
    their volume is 100x too large and amount (=close*volume*100) 100x too
    large. Divide volume and amount by 100. Guarded by a marker in kline_meta
    so it can never run twice. Returns rows changed."""
    import sqlite3 as _sq

    with cache._connect() as db:  # noqa: SLF001 — migration is cache-internal
        # kline_meta is keyed by code (no generic key/value table): the marker
        # is a dedicated row with code='_migration', source='migration',
        # last_date=STAR_MIGRATION_KEY.
        marker = db.execute(
            "SELECT 1 FROM kline_meta WHERE code = '_migration' "
            "AND last_date = ? AND source = 'migration'",
            (STAR_MIGRATION_KEY,),
        ).fetchone()
        if marker:
            return 0
        cur = db.execute(
            "UPDATE kline_bars SET volume = volume / 100.0, amount = amount / 100.0 "
            "WHERE (code LIKE '688%' OR code LIKE '689%')"
        )
        db.execute(
            "INSERT INTO kline_meta (code, last_date, fetched_at, source) "
            "VALUES ('_migration', ?, ?, 'migration')",
            (STAR_MIGRATION_KEY, _dt.datetime.now().isoformat(timespec="seconds")),
        )
        return cur.rowcount


# ---------------------------------------------------------------------------
# snapshot append helpers
# ---------------------------------------------------------------------------
def _prev_trading_day(calendar, trade_date: _dt.date) -> _dt.date | None:
    """The trading day strictly before trade_date (calendar-aware)."""
    try:
        return calendar.previous_trading_day(trade_date)
    except Exception:  # noqa: BLE001 — weekday fallback
        d = trade_date - _dt.timedelta(days=1)
        while d.weekday() >= 5:
            d -= _dt.timedelta(days=1)
        return d


def _default_calendar():
    """The trading calendar, loaded on demand (cached on disk).

    Returns ``None`` when the calendar cannot be loaded (offline test, degraded
    source); callers must then fall back to the weekday heuristic in
    :func:`_prev_trading_day` — which is what ``calendar=None`` used to mean,
    except that it was resolved to ``prev_day=None`` upstream and turned every
    append into a refetch.
    """
    try:
        from src.market_scanner.sources.calendar import load_calendar

        loaded = load_calendar()
        # load_calendar returns (TradingCalendar, SourceStatus).
        if isinstance(loaded, tuple):
            return loaded[0]
        return loaded
    except Exception:  # noqa: BLE001 — calendar is best-effort
        return None


def _snapshot_row(snapshot: pd.DataFrame, code: str) -> pd.Series | None:
    """One universe row by code, or None."""
    if snapshot is None or snapshot.empty or code not in snapshot.index:
        return None
    return snapshot.loc[code]


def _normalise_volume(volume: float | None, source: str) -> float | None:
    """Normalise a snapshot volume to LOTS (cache convention).

    ``source='sina'`` volume is shares; ``source='tencent'`` is lots.
    """
    if volume is None:
        return None
    if str(source).lower() == "sina":
        return volume / _LOT_SIZE
    return volume


def _snapshot_bar(snapshot_row: pd.Series, trade_date: _dt.date,
                  snapshot_source: str) -> dict | None:
    """One daily bar dict from a snapshot row, or None when unusable.

    Same validation and amount rule as :func:`_append_from_snapshot`
    (homogeneous ``close * volume * 100`` estimate, Sina turnover kept as
    ``amount_reported`` for audit), without needing the cached frame.
    """
    price = snapshot_row.get("price")
    if price is None or pd.isna(price) or float(price) <= 0:
        return None  # suspended or no data
    prev_close = snapshot_row.get("settlement")
    if prev_close is None or pd.isna(prev_close) or float(prev_close) <= 0:
        return None
    volume = _normalise_volume(snapshot_row.get("volume"), snapshot_source)
    amount_reported = snapshot_row.get("amount")
    return {
        "date": pd.Timestamp(trade_date),
        "open": snapshot_row.get("open") or price,
        "high": snapshot_row.get("high") or price,
        "low": snapshot_row.get("low") or price,
        "close": float(price),
        "volume": volume,
        "amount": float(price) * (volume or 0) * _LOT_SIZE,
        "amount_reported": (float(amount_reported)
                            if amount_reported is not None
                            and not pd.isna(amount_reported)
                            else None),
    }


def _append_from_snapshot(
    cached_df: pd.DataFrame,
    snapshot_row: pd.Series,
    trade_date: _dt.date,
    snapshot_source: str,
) -> pd.DataFrame | None:
    """Build the extended frame with trade_date's bar from the snapshot.

    Amount rule (CONTRACT 9b, verified live 2026-10-13): the cached Tencent
    history estimates amount as ``close * volume * 100`` (Sina's reported
    turnover differs by up to ±1.3% because it uses actual trade prices, not
    the close). To keep every appended bar homogeneous with the history, the
    appended bar uses the SAME estimate; Sina's reported turnover is preserved
    in an extra ``amount_reported`` column for audit.

    Returns None when the snapshot has no usable bar (suspended / price 0).
    """
    bar = _snapshot_bar(snapshot_row, trade_date, snapshot_source)
    if bar is None:
        return None
    out = pd.concat([cached_df, pd.DataFrame([bar])], ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _settlement_matches(cached_last_close: float | None, settlement) -> bool:
    """True when the snapshot's settlement matches the cached close within 0.3%."""
    if cached_last_close is None or settlement is None or pd.isna(settlement):
        return False
    try:
        settlement = float(settlement)
    except (TypeError, ValueError):
        return False
    if settlement <= 0 or cached_last_close <= 0:
        return False
    return abs(cached_last_close - settlement) / settlement <= _CLOSE_MATCH_TOLERANCE


def _snapshot_session_completed(trade_date: _dt.date,
                                now: _dt.datetime) -> bool:
    """True when the snapshot represents a COMPLETED session for trade_date.

    The A-share session runs 09:15–15:00 Asia/Shanghai (09:15 pre-open).
    - On trade_date itself: completed only at/after 15:00.
    - Before 09:15 on trade_date: the snapshot still shows the PREVIOUS
      session, not trade_date -> not completed for trade_date.
    - On any day after trade_date (or intraday on later days): the last
      completed session ending trade_date has long closed -> completed.
    """
    if now.date() > trade_date:
        return True
    if now.date() < trade_date:
        return False
    # same day: completed only after the 15:00 close
    return now.time() >= _dt.time(15, 0)


# ---------------------------------------------------------------------------
# F7 — per-symbol kline reason vocabulary (opencode owns the dict shape)
# ---------------------------------------------------------------------------
KLINE_REASONS: tuple[str, ...] = (
    "cache_hit",
    "appended",
    "gap_filled",
    "refetch_no_cache",
    "refetch_gap_mismatch",
    "refetch_prev_close_mismatch",
    "refetch_last_bar_mismatch",
    "known_missing",
    "transport_error",
    "throttle_error",
    "permanent_error",
    "refetched",
    "budget_not_fetched",
    "stopped_unavailable",
    "intraday_dropped",
)
_MAX_REASON_EXAMPLES = 20


def _new_kline_reasons() -> dict[str, dict[str, Any]]:
    """Empty ``{"counts": {reason: 0}, "examples": {reason: [codes]}}``."""
    return {
        "counts": {reason: 0 for reason in KLINE_REASONS},
        "examples": {reason: [] for reason in KLINE_REASONS},
    }


def _note_reason(reasons: dict[str, dict[str, Any]], key: str, code: str) -> None:
    """Count one per-symbol reason; keep the first 20 example codes."""
    counts = reasons["counts"]
    examples = reasons["examples"]
    counts[key] = counts.get(key, 0) + 1
    if len(examples[key]) < _MAX_REASON_EXAMPLES:
        examples[key].append(code)


# ---------------------------------------------------------------------------
# F5a — tail gap-fill
# ---------------------------------------------------------------------------
def _gap_trading_days(calendar, last_date: _dt.date,
                      trade_date: _dt.date) -> int:
    """Trading days strictly between ``last_date`` and ``trade_date``.

    Those are the missed sessions a tail fill must cover (the tail itself
    always ends at ``trade_date``). Falls back to weekdays without any I/O.
    """
    missing = 0
    day = last_date + _dt.timedelta(days=1)
    guard = 0
    while day < trade_date and guard < 400:
        guard += 1
        try:
            trading = calendar.is_trading_day(day) if calendar is not None \
                else day.weekday() < 5
        except Exception:  # noqa: BLE001 — fall back to weekdays
            trading = day.weekday() < 5
        if trading:
            missing += 1
        day += _dt.timedelta(days=1)
    return missing


def _verify_tail_overlap(cached_df: pd.DataFrame, tail_df: pd.DataFrame,
                         code: str) -> bool:
    """True when the fetched tail verifies against the cached history.

    Requires at least one intersecting date among the last
    ``_GAPFILL_OVERLAP`` cached bars, every intersecting close within 0.3%,
    and (when both sides have positive volume) a median volume ratio inside
    the sanity band — the STAR lots-vs-shares guard (100x off on a unit bug).
    """
    try:
        cached_dates = [d.date() if hasattr(d, "date") else d
                        for d in cached_df["date"].tolist()[-_GAPFILL_OVERLAP:]]
        tail_by_date = {}
        for _, row in tail_df.iterrows():
            d = row["date"]
            tail_by_date[d.date() if hasattr(d, "date") else d] = row
        overlap = [d for d in cached_dates if d in tail_by_date]
        if not overlap:
            return False
        for d in overlap:
            cached_close = float(
                cached_df[cached_df["date"].dt.date == d].iloc[0]["close"])
            tail_close = float(tail_by_date[d]["close"])
            if not _settlement_matches(cached_close, tail_close):
                return False
        cached_vols = [float(cached_df[cached_df["date"].dt.date == d].iloc[0]["volume"])
                       for d in overlap]
        tail_vols = [float(tail_by_date[d]["volume"]) for d in overlap]
        import statistics as _st

        cached_med = _st.median(cached_vols)
        tail_med = _st.median(tail_vols)
        if cached_med > 0 and tail_med > 0:
            ratio = tail_med / cached_med
            if not (_VOLUME_RATIO_LO <= ratio <= _VOLUME_RATIO_HI):
                return False
        return True
    except Exception:  # noqa: BLE001 — any doubt means full refetch
        return False


def _merge_tail(cached_df: pd.DataFrame, tail_df: pd.DataFrame) -> pd.DataFrame:
    """Cached bars plus the tail's strictly-newer rows, sorted by date."""
    last_cached = cached_df["date"].max()
    new_rows = tail_df[tail_df["date"] > last_cached]
    out = pd.concat([cached_df, new_rows], ignore_index=True)
    return out.sort_values("date").reset_index(drop=True)


def _latest_completed_date(calendar, now: _dt.datetime) -> _dt.date | None:
    """Latest trading day whose daily bar is final (intraday guard).

    A bar dated today is complete only once the A-share session has closed
    (09:15-15:00 Asia/Shanghai: ``pipeline.SESSION_CLOSE`` / the scheduler's
    trading block). Before 15:00 the fetched "today" bar is partial and must
    never be stored as completed; on a non-trading day the latest session's
    bar is final. Naive datetimes are interpreted as Asia/Shanghai (the same
    convention as ``_snapshot_session_completed`` and the scheduler clock).
    Falls back to the weekday heuristic without any I/O.
    """
    try:
        if calendar is not None and hasattr(
                calendar, "latest_completed_trading_day"):
            day = calendar.latest_completed_trading_day(now)
            if isinstance(day, _dt.datetime):
                return day.date()
            if isinstance(day, _dt.date):
                return day
    except Exception:  # noqa: BLE001 — fall back to weekdays
        pass
    day = now.date()
    if day.weekday() < 5 and now.time() >= _dt.time(15, 0):
        return day
    day -= _dt.timedelta(days=1)
    while day.weekday() >= 5:
        day -= _dt.timedelta(days=1)
    return day


def _resolve_cutoff(calendar, trade_date: _dt.date | None,
                    now: _dt.datetime) -> _dt.date | None:
    """Newest bar date allowed into the cache (intraday guard).

    The minimum of the scan's trade date and the latest completed session:
    anything newer is still forming and must be dropped before storing, and
    must never set ``last_date``.
    """
    latest = _latest_completed_date(calendar, now)
    if latest is None:
        return trade_date
    if trade_date is None:
        return latest
    return min(trade_date, latest)


def _drop_incomplete_bars(df: pd.DataFrame,
                          cutoff_date: _dt.date | None) -> pd.DataFrame | None:
    """Drop bars dated after ``cutoff_date`` (intraday guard).

    Returns the filtered frame, or None when nothing storable remains.
    Never raises: on any doubt the frame passes through unfiltered.
    """
    if df is None or len(df) == 0 or cutoff_date is None:
        return df
    try:
        dates = pd.to_datetime(df["date"])
        keep = dates.dt.date <= cutoff_date
        out = df.loc[keep.values].reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return df
    if len(out) == 0:
        return None
    return out


def fetch_klines(
    symbols: list[str],
    bars: int = 260,
    workers: int = 6,
    progress: Callable[[int, int], None] | None = None,
    fetch: Callable | None = None,
    limiter: KlineRateLimiter | None = None,
    *,
    snapshot: pd.DataFrame | None = None,
    trade_date: _dt.date | None = None,
    calendar=None,
    cache: KlineCache | None = None,
    time_budget_s: float | None = None,
    snapshot_source: str = "sina",
    now_fn: Callable[[], _dt.datetime] | None = None,
    lock_check: Callable[[], None] | None = None,
    local_budget_s: float | None = None,
) -> tuple[dict[str, pd.DataFrame], SourceStatus]:
    """Bulk daily qfq K-lines with cache + snapshot append (see module doc).

    Args:
        symbols: Prefixed symbols (``sh600519`` ...).
        bars: Bars per full refetch.
        workers: Thread pool size (capped at 6).
        progress: Optional ``(done, total)`` callback.
        fetch: Injectable ``(url, params) -> str`` for offline tests.
        limiter: Injectable :class:`KlineRateLimiter`.
        snapshot: Universe frame (indexed by code) with price/settlement/
            volume/open/high/low/amount columns for the append path.
        trade_date: The snapshot's trading day.
        calendar: Trading calendar for previous-trading-day resolution.
        cache: Injectable :class:`KlineCache` (tests pass a tmp_path one).
        time_budget_s: Wall-clock NETWORK budget for refetches (F10: phase-2
            requests only; local cache I/O never consumes it).
        snapshot_source: ``'sina'`` (volume in shares) or ``'tencent'`` (lots).
        now_fn: Injectable clock for budget accounting.
        lock_check: Optional no-arg callable (hermes's ``JobHandle.check``);
            every 25 completions (and once up front) it is called and any
            exception aborts the fetch with ``st.lock_aborted`` set.
        local_budget_s: Generous cap (s) for the LOCAL phase (cache reads +
            batched appends). Leftovers become ordinary refetch needs instead
            of missing data; ``None`` disables the cap.

    Returns:
        ``(frames, st)`` — frames keyed by 6-digit code; ``st.detail`` carries
        cache_hit / appended_from_snapshot / gap_filled / refetched / missing /
        throttled / budget_exhausted counters plus ``local_s``,
        ``network_s`` and ``local_capped``; ``st.kline_reasons`` carries the
        F7 per-symbol reason counts + examples; ``st.unavailable`` flags the
        F4 outage stop; ``st.lock_aborted`` flags a lock-check abort.
    """
    t0 = time.monotonic()
    now_fn = now_fn or (lambda: _dt.datetime.now())
    workers = max(1, min(_MAX_WORKERS, workers))
    limiter = limiter or _KLINE_LIMITER
    cache = cache or KlineCache()
    trade_date = trade_date or now_fn().date()

    frames: dict[str, pd.DataFrame] = {}
    cache_hits = 0
    appended = 0
    refetched = 0
    missing = 0
    failures: list[tuple[str, str]] = []
    budget_exhausted = False
    reasons = _new_kline_reasons()
    lock_aborted = False
    lock_abort_error: str | None = None
    stop_cause: str | None = None  # "budget" | "unavailable" | "lock"
    local_capped = False

    # Snapshot-append session guard (CONTRACT 9b): a Sina snapshot is only a
    # COMPLETED session for trade_date when "now" is after 15:00 Asia/Shanghai
    # on trade_date, or before 09:15 on a later day (previous session), or on
    # any day after trade_date. During 09:15–15:00 on trade_date it is
    # intraday for TODAY -> never append it as a daily bar.
    now = now_fn()
    snap_completed = _snapshot_session_completed(trade_date, now)

    # ---- phase 1: cache + snapshot append (no network) --------------------
    # F10: light decisions only — one meta query + one newest-bar query for
    # the whole universe, NO per-symbol frame loads. Appends are collected
    # and flushed in ONE transaction; full frames are bulk-loaded once.
    # needs: (symbol, kind, bars_to_fetch, rank); kind "full" = wholesale
    # refetch, kind "gap" = tail fill (F5a). Rank orders phase 2 so
    # corporate-action refetches run first (0 = prev-close mismatch,
    # 1 = gap, 2 = missing/stale).
    needs: list[tuple[str, str, int, int]] = []
    # A caller that passes a snapshot/trade_date but no calendar (the scan
    # pipeline did exactly this) must not silently degrade every appended bar
    # into a full refetch: ``prev_day = None`` makes ``expected_last`` fall back
    # to trade_date itself, so a cache ending on the previous trading day looks
    # like a gap and ~4200 symbols get refetched for no reason. Resolve the
    # real previous trading day lazily here; an explicit calendar always wins.
    if calendar is None:
        calendar = _default_calendar()
    prev_day = _prev_trading_day(calendar, trade_date) if calendar is not None else None
    known_missing = (cache.known_missing(now) if cache is not None else {})
    try:
        meta = cache.meta_map()
    except Exception:  # noqa: BLE001 — fall back to per-symbol reads
        meta = {}
    all_codes = [s[2:] if len(s) > 6 else s for s in symbols]
    try:
        lasts = cache.last_bars(all_codes)
    except Exception:  # noqa: BLE001
        lasts = {}
    cutoff_date = _resolve_cutoff(calendar, trade_date, now)
    cutoff_iso = cutoff_date.isoformat() if cutoff_date is not None else None
    expected_last = prev_day or trade_date
    append_rows: list[tuple[str, dict]] = []
    keep_codes: list[str] = []
    phase1_t0 = time.monotonic()

    def _local_expired() -> bool:
        return (local_budget_s is not None
                and time.monotonic() - phase1_t0 > local_budget_s)

    for sym in symbols:
        if _local_expired():
            break
        code = sym[2:] if len(sym) > 6 else sym
        if known_missing.get(code):
            # known-missing (recently failed) with live retry-after: report
            # as missing WITHOUT refetching; expired markers are dropped by
            # the registry and retried normally.
            missing += 1
            failures.append((sym, "known-missing (retry after "
                             + known_missing[code] + ")"))
            _note_reason(reasons, "known_missing", code)
            continue
        row = meta.get(code)
        if row is None:
            needs.append((sym, "full", bars, 2))
            _note_reason(reasons, "refetch_no_cache", code)
            continue
        # stale cache -> full refetch
        fetched_at = row.get("fetched_at")
        stale = True
        if fetched_at:
            try:
                age = (now_fn() - _dt.datetime.fromisoformat(fetched_at)).days
                stale = age > MAX_AGE_DAYS
            except ValueError:
                stale = True
        if stale:
            needs.append((sym, "full", bars, 2))
            _note_reason(reasons, "refetch_no_cache", code)
            continue
        last = lasts.get(code)
        if last is None:
            # meta row but no bars: treat like a missing cache.
            needs.append((sym, "full", bars, 2))
            _note_reason(reasons, "refetch_no_cache", code)
            continue
        last_cached_date = _dt.date.fromisoformat(last["date"])
        snap_row = _snapshot_row(snapshot, code)
        if snap_row is None:
            # suspended / no snapshot bar: keep cached frame as is
            keep_codes.append(code)
            cache_hits += 1
            _note_reason(reasons, "cache_hit", code)
            continue
        if last_cached_date == trade_date:
            # Cache already contains trade_date's bar (e.g. backfilled after
            # the close): HIT — but only if the snapshot close matches the
            # cached close (corporate action / bad data guard). No refetch.
            try:
                cached_close = float(last["close"])
            except (TypeError, ValueError):
                cached_close = None
            snap_close = snap_row.get("price")
            if snap_close is None or pd.isna(snap_close):
                keep_codes.append(code)
                cache_hits += 1
                _note_reason(reasons, "cache_hit", code)
                continue
            if cached_close is None or not _settlement_matches(
                    cached_close, snap_close):
                # snapshot disagrees with the stored bar -> full refetch
                needs.append((sym, "full", bars, 0))
                _note_reason(reasons, "refetch_last_bar_mismatch", code)
                continue
            keep_codes.append(code)
            cache_hits += 1
            _note_reason(reasons, "cache_hit", code)
            continue
        if last_cached_date != expected_last:
            # gap: route small gaps to the tail fill (F5a); anything larger
            # (or a gap with no usable calendar) refetches fully.
            gap_days = 0
            if last_cached_date < expected_last:
                gap_days = _gap_trading_days(calendar, last_cached_date,
                                             trade_date)
            if 1 <= gap_days <= _MAX_GAPFILL_DAYS:
                needs.append((sym, "gap", gap_days + _GAPFILL_OVERLAP, 1))
                continue
            needs.append((sym, "full", bars, 2))
            _note_reason(reasons, "refetch_no_cache", code)
            continue
        try:
            cached_close = float(last["close"])
        except (TypeError, ValueError):
            cached_close = None
        if cached_close is None or not _settlement_matches(
                cached_close, snap_row.get("settlement")):
            # corporate action (dividend/split) -> full refetch
            needs.append((sym, "full", bars, 0))
            _note_reason(reasons, "refetch_prev_close_mismatch", code)
            continue
        if not snap_completed:
            # Intraday snapshot: appending it would write a partial day as a
            # completed daily bar. Keep the cached frame (no network either).
            keep_codes.append(code)
            cache_hits += 1
            _note_reason(reasons, "cache_hit", code)
            continue
        # (a) append trade_date's bar from the snapshot — no network. The bar
        # is validated here; the write is batched below and the frames come
        # from the bulk load afterwards.
        bar = _snapshot_bar(snap_row, trade_date, snapshot_source)
        if bar is None:
            keep_codes.append(code)  # suspended today: keep as is
            cache_hits += 1
            _note_reason(reasons, "cache_hit", code)
            continue
        append_rows.append((code, bar))
        keep_codes.append(code)
        cache_hits += 1
        appended += 1
        _note_reason(reasons, "appended", code)

    local_capped = _local_expired()
    if local_capped:
        # Local budget blown: everything not yet decided becomes an ordinary
        # full refetch (the untouched network budget still protects
        # coverage) instead of missing data.
        decided = set()
        for _sym, _kind, _bars_needed, _rank in needs:
            decided.add(_sym[2:] if len(_sym) > 6 else _sym)
        decided.update(keep_codes)
        for _code, _sym in ((c, s) for s, c in zip(symbols, all_codes)
                            if c not in decided and c not in known_missing):
            needs.append((_sym, "full", bars, 2))
            _note_reason(reasons, "refetch_no_cache", _code)
        for _code in [c for c in all_codes
                      if c not in decided and c in known_missing]:
            missing += 1
            failures.append((_code, "known-missing (retry after "
                             + str(known_missing[_code]) + ")"))
            _note_reason(reasons, "known_missing", _code)

    # ---- phase 2: fetches (network, rate-limited, budget-capped) --------
    def _run_lock_check() -> str | None:
        """Run the injected lock check; return an error text or None."""
        if lock_check is None:
            return None
        try:
            lock_check()
        except Exception as exc:  # noqa: BLE001 — abort, never crash
            return f"{type(exc).__name__}: {exc}"[:200]
        return None

    # Intraday guard: bars newer than the latest completed session must
    # never reach the cache (a fetch during the session returns today's
    # partial bar). Resolved once here; every store path below filters.
    cutoff_date = _resolve_cutoff(calendar, trade_date, now)
    cutoff_iso = cutoff_date.isoformat() if cutoff_date is not None else None

    # F10: flush the collected appends in ONE transaction per chunk, then
    # bulk-load every still-needed cached frame once (no per-symbol
    # get_frame at all). Gap merges below read their cached history from
    # this dict; refetched codes are overwritten from the network.
    if append_rows:
        _CHUNK = 500
        for _i in range(0, len(append_rows), _CHUNK):
            cache.append_bars(append_rows[_i:_i + _CHUNK], "snapshot")
    gap_codes = [s[2:] if len(s) > 6 else s
                 for s, kind, _, _ in needs if kind == "gap"]
    load_codes = sorted(set(keep_codes) | set(gap_codes))
    cached_frames: dict[str, Any] = {}
    if load_codes:
        try:
            cached_frames = cache.load_frames(load_codes,
                                              cutoff=cutoff_iso)
        except Exception:  # noqa: BLE001 — fall back to per-symbol reads
            cached_frames = {}
            for _code in load_codes:
                try:
                    _got = cache.get_frame(_code)
                except Exception:  # noqa: BLE001
                    _got = None
                if _got is not None:
                    cached_frames[_code] = _got
    for _code in keep_codes:
        _hit = cached_frames.get(_code)
        if _hit is not None and _hit.get("bars") is not None:
            frames[_code] = _hit["bars"]
    local_s = time.monotonic() - phase1_t0

    def _refetch(item: tuple[str, str, int, int]):
        sym, kind, need_bars, _rank = item
        code = sym[2:] if len(sym) > 6 else sym
        t_req = time.monotonic()

        def _keep_completed(df: pd.DataFrame | None) -> pd.DataFrame | None:
            kept = _drop_incomplete_bars(df, cutoff_date)
            try:
                dropped = (df is not None and len(df) > 0
                           and (kept is None or len(kept) < len(df)))
            except Exception:  # noqa: BLE001
                dropped = False
            if dropped:
                _note_reason(reasons, "intraday_dropped", code)
            return kept

        try:
            df = _fetch_one(sym, need_bars, limiter=limiter, fetch=fetch)
        except ThrottleError as exc:
            return (sym, None, str(exc), True, False,
                    time.monotonic() - t_req, None, False)
        except Exception as exc:  # noqa: BLE001 — per-symbol isolation
            transport = is_transport_error(exc)
            return (sym, None, f"{type(exc).__name__}: {exc}"[:200], False,
                    transport, time.monotonic() - t_req, None, False)
        req_elapsed = time.monotonic() - t_req
        df = _keep_completed(df)
        if df is None or len(df) == 0:
            # Only today's partial bar came back: nothing storable, and no
            # 24 h block (the next run after the close retries normally).
            return (sym, None, "no completed kline bars (intraday fetch)",
                    False, False, req_elapsed, None, True)
        if kind == "gap":
            cached_df = (cached_frames.get(code) or {}).get("bars")
            if cached_df is not None and _verify_tail_overlap(
                    cached_df, df, code):
                merged = _drop_incomplete_bars(
                    _merge_tail(cached_df, df), cutoff_date)
                return (sym, merged, None, False, False, req_elapsed,
                        "gap_filled", False)
            # overlap mismatch (corporate action / unit change): wholesale
            # refetch inline so the symbol is not left gapped.
            try:
                full_df = _fetch_one(sym, bars, limiter=limiter, fetch=fetch)
            except ThrottleError as exc:
                return (sym, None, str(exc), True, False,
                        time.monotonic() - t_req, None, False)
            except Exception as exc:  # noqa: BLE001
                transport = is_transport_error(exc)
                return (sym, None,
                        f"{type(exc).__name__}: {exc}"[:200], False,
                        transport, time.monotonic() - t_req, None, False)
            full_df = _keep_completed(full_df)
            if full_df is None or len(full_df) == 0:
                return (sym, None, "no completed kline bars (intraday fetch)",
                        False, False, time.monotonic() - t_req, None, True)
            return (sym, full_df, None, False, False,
                    time.monotonic() - t_req, "refetch_gap_mismatch", False)
        return (sym, df, None, False, False, req_elapsed, None, False)

    lock_err = _run_lock_check()
    if lock_err is not None:
        lock_aborted = True
        lock_abort_error = lock_err
        stop_cause = "lock"
    if needs and not lock_aborted:
        # F10: corporate-action refetches first, then gaps, then the bulk
        # missing/stale set — so a budget cut hits the least informative
        # symbols last.
        ordered = sorted(needs, key=lambda item: item[3])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Submit lazily: after a stop we cancel futures that have not
            # started (and count them missing).
            futures = {pool.submit(_refetch, item): item[0]
                       for item in ordered}
            done = 0
            stop = False
            tallied: set[int] = set()

            def _tally(sym, df, err, throttled, transport, req_elapsed,
                       outcome, no_mark):
                """Exactly-one-bucket accounting, shared by the main loop
                and the drain pass below: every completed future lands its
                symbol in frames (stored) or missing + failures, with its
                terminal kline_reasons entry. Never raises."""
                nonlocal refetched, missing
                code = sym[2:] if len(sym) > 6 else sym
                if err is None and df is not None:
                    frames[code] = df
                    cache.put_frame(code, df, "tencent")
                    cache.clear_known_missing(sym)
                    limiter.note_result(True, elapsed_s=req_elapsed)
                    if outcome == "gap_filled":
                        _note_reason(reasons, "gap_filled", code)
                    elif outcome == "refetch_gap_mismatch":
                        _note_reason(reasons, "refetch_gap_mismatch", code)
                        refetched += 1
                    else:
                        _note_reason(reasons, "refetched", code)
                        refetched += 1
                else:
                    missing += 1
                    failures.append((sym, err or "unknown"))
                    if transport:
                        _note_reason(reasons, "transport_error", code)
                        limiter.note_result(
                            False, throttle=False, transport=True,
                            desc=err, elapsed_s=req_elapsed)
                    elif throttled:
                        _note_reason(reasons, "throttle_error", code)
                        limiter.note_result(
                            False, throttle=True, transport=False,
                            desc=err, elapsed_s=req_elapsed)
                    else:
                        limiter.note_result(
                            False, throttle=False, transport=False,
                            desc=err, elapsed_s=req_elapsed)
                        if not no_mark:
                            # F5d: only permanent individual failures (bad
                            # symbol, delisted…) are skipped for 24 h.
                            # Transport errors and intraday-only fetches are
                            # retried on the next run.
                            cache.mark_known_missing(sym)
                            _note_reason(reasons, "permanent_error", code)

            for fut in as_completed(futures):
                sym, df, err, throttled, transport, req_elapsed, outcome, \
                    no_mark = fut.result()
                done += 1
                _tally(sym, df, err, throttled, transport, req_elapsed,
                       outcome, no_mark)
                tallied.add(id(fut))
                if limiter.unavailable:
                    stop = True
                    stop_cause = "unavailable"
                if progress is not None:
                    try:
                        progress(done + cache_hits, len(needs) + cache_hits)
                    except Exception:  # noqa: BLE001
                        logger.exception("progress callback raised")
                if done % 25 == 0 or stop:
                    lock_err = _run_lock_check()
                    if lock_err is not None:
                        lock_aborted = True
                        lock_abort_error = lock_err
                        stop = True
                        stop_cause = "lock"
                        break
                if not stop and time_budget_s is not None \
                        and limiter.over_network_budget(time_budget_s):
                    budget_exhausted = True
                    stop = True
                    stop_cause = "budget"
                if stop:
                    break  # leave as_completed; pending futures cancelled below
            if stop:
                # Cancel futures that have not started yet; the ones already
                # running finish (per-symbol isolation). Cancelled or still
                # pending futures count as missing — but results already
                # consumed in the as_completed loop are never double-counted.
                not_fetched_reason = (
                    "stopped_unavailable" if stop_cause == "unavailable"
                    else "budget_not_fetched")
                lock_prefix = " (lock check aborted)" if stop_cause == "lock" else ""
                consumed = set()
                for fut, sym in futures.items():
                    cancelled = fut.cancel()
                    if cancelled:
                        consumed.add(id(fut))
                        missing += 1
                        failures.append(
                            (sym, "not fetched (throttle stop / budget)"
                             + lock_prefix))
                        if stop_cause in ("budget", "unavailable"):
                            _note_reason(reasons, not_fetched_reason,
                                         sym[2:] if len(sym) > 6 else sym)
                deadline = time.monotonic() + 2.0
                for fut, sym in futures.items():
                    if id(fut) in consumed or id(fut) in tallied:
                        continue
                    remaining = max(0.0, deadline - time.monotonic())
                    try:
                        result = fut.result(timeout=remaining)
                    except Exception:  # noqa: BLE001 — timeout / cancelled
                        if not fut.cancelled():
                            missing += 1
                            failures.append(
                                (sym, "not fetched (throttle stop / budget)"
                                 + lock_prefix))
                            if stop_cause in ("budget", "unavailable"):
                                _note_reason(reasons, not_fetched_reason,
                                             sym[2:] if len(sym) > 6 else sym)
                        continue
                    # F10: a late completion already paid for is stored and
                    # counted like any other result instead of being dropped.
                    _tally(*result)

    latency = time.monotonic() - t0
    ok = len(frames)
    if lock_aborted:
        status = "failed" if ok == 0 else "degraded"
    elif limiter.unavailable:
        # F4 outage stop: the price source is down, not merely slow.
        status = "failed"
    elif not symbols:
        status = "failed"
    elif failures or budget_exhausted:
        status = "degraded" if ok else "failed"
    else:
        status = "ok"

    if lock_aborted:
        error_text = lock_abort_error or "job lock lost (deleted or taken over)"
    elif limiter.unavailable:
        error_text = ("price source unavailable "
                      f"({limiter.unavailable_reason or 'outage'})")
    else:
        error_text = None if not failures else f"{len(failures)} symbols failed: {failures[0][1]}"[:300]

    throttle_note = "no"
    if limiter.throttle_stop:
        if limiter.unavailable and limiter._last_stop_kind == "transport":
            throttle_note = (f"yes (transport outage at "
                             f"{limiter.last_throttle_at or '?'})")
        else:
            throttle_note = (f"yes (HTTP 501 at "
                             f"{limiter.last_throttle_at or '?'})")
    # as_of: the latest bar date across the returned frames (data date).
    last_bar = None
    for _f in frames.values():
        try:
            d = _f["date"].max()
            d = d.date() if hasattr(d, "date") else d
            if last_bar is None or d > last_bar:
                last_bar = d
        except Exception:  # noqa: BLE001
            continue
    st = SourceStatus(
        name=_NAME,
        status=status,
        rows=ok,
        latency_s=latency,
        error=error_text,
        as_of=last_bar.isoformat() if last_bar is not None else None,
        detail=(
            f"cache_hit={cache_hits} appended_from_snapshot={appended} "
            f"gap_filled={reasons['counts']['gap_filled']} "
            f"refetched={refetched} missing={missing} "
            f"failed={len(failures)} "
            f"known_missing={len(known_missing)} "
            f"transport_errors={reasons['counts']['transport_error']} "
            f"intraday_dropped={reasons['counts']['intraday_dropped']} "
            f"breaker_trips={limiter.breaker_trips} "
            f"throttled={throttle_note} "
            f"budget_exhausted={'yes' if budget_exhausted else 'no'} "
            f"local_s={round(local_s, 1)} "
            f"network_s={round(limiter.network_s, 1)} "
            f"local_capped={'yes' if local_capped else 'no'} "
            f"amount estimated from close*volume*100; "
            f"source=tencent fqkline qfq"
        ),
    )
    # F7 + F4/F2 handoff attributes (not part of to_dict(); the pipeline
    # reads them off the live object).
    st.kline_reasons = reasons  # type: ignore[attr-defined]
    st.unavailable = limiter.unavailable  # type: ignore[attr-defined]
    st.lock_aborted = lock_aborted  # type: ignore[attr-defined]
    st.lock_abort_error = lock_abort_error  # type: ignore[attr-defined]
    return frames, st


# ---------------------------------------------------------------------------
# F5b — shortfall assessment + warm-up engine
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Shortfall:
    """What a scan for ``target_trade_date`` still needs from the network.

    ``appendable`` codes need no request (the scan appends trade_date's bar
    from the snapshot); they are NOT part of the shortfall. ``est_requests``
    counts one request per missing/gap/stale code (a gap-fill tail is one
    request); ``est_seconds`` divides by the assumed ≤ 1 req/s rate.
    """

    missing: int = 0
    gap: int = 0
    stale: int = 0
    appendable: int = 0
    est_requests: int = 0
    est_seconds: float = 0.0


@dataclasses.dataclass
class WarmResult:
    """Outcome of one :func:`warm_cache` run."""

    attempted: int = 0
    filled: int = 0
    failed: int = 0
    skipped_known_missing: int = 0
    requests: int = 0
    intraday_dropped: int = 0
    duration_s: float = 0.0
    stopped_unavailable: bool = False
    aborted: bool = False
    deadline_hit: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict for scheduler warmup_finished logs."""
        return {
            "attempted": self.attempted,
            "filled": self.filled,
            "failed": self.failed,
            "skipped_known_missing": self.skipped_known_missing,
            "requests": self.requests,
            "intraday_dropped": self.intraday_dropped,
            "duration_s": round(self.duration_s, 1),
            "stopped_unavailable": self.stopped_unavailable,
            "aborted": self.aborted,
            "deadline_hit": self.deadline_hit,
            "error": self.error,
        }


def _prev_day_iso(calendar, trade_date: _dt.date) -> str | None:
    """Previous trading day ISO date, or None when unresolvable."""
    try:
        prev = _prev_trading_day(calendar, trade_date)
    except Exception:  # noqa: BLE001
        prev = None
    if prev is None:
        return None
    try:
        return prev.isoformat()
    except Exception:  # noqa: BLE001
        return None


def assess_shortfall(symbols: list[str], target_trade_date: _dt.date,
                     calendar=None, cache: KlineCache | None = None, *,
                     rate_per_sec: float = 1.0,
                     now_fn: Callable[[], _dt.datetime] | None = None,
                     ) -> Shortfall:
    """Count what a scan for ``target_trade_date`` still needs (no network).

    Reads only ``kline_meta`` (one query) plus the live known-missing
    registry. ``calendar=None`` falls back to the weekday heuristic (never
    any I/O): with no holiday knowledge the estimate stays conservative.
    Codes with a live known-missing marker are skipped, not shortfall.
    """
    now_fn = now_fn or (lambda: _dt.datetime.now())
    cache = cache or KlineCache()
    if calendar is None:
        prev_iso: str | None = None
        day = target_trade_date - _dt.timedelta(days=1)
        while day.weekday() >= 5:
            day -= _dt.timedelta(days=1)
        prev_iso = day.isoformat()
    else:
        prev_iso = _prev_day_iso(calendar, target_trade_date)
    now = now_fn()
    live_missing = set()
    try:
        live_missing = set(cache.known_missing(now).keys())
    except Exception:  # noqa: BLE001 — registry must not break assessment
        live_missing = set()
    try:
        meta = cache.meta_map()
    except Exception:  # noqa: BLE001
        meta = {}
    out = Shortfall()
    for sym in symbols:
        code = sym[2:] if len(sym) > 6 else sym
        if code in live_missing:
            continue
        row = meta.get(code)
        if row is None:
            out.missing += 1
            continue
        if row.get("source") == "known-missing":
            continue
        fetched_at = row.get("fetched_at")
        try:
            age_days = (now - _dt.datetime.fromisoformat(fetched_at)).days
            if age_days > MAX_AGE_DAYS:
                out.stale += 1
                continue
        except (TypeError, ValueError):
            out.stale += 1
            continue
        last = row.get("last_date")
        if prev_iso is not None and last == prev_iso:
            out.appendable += 1
        elif last is None or (prev_iso is not None and last < prev_iso):
            out.gap += 1
        elif prev_iso is None:
            # No calendar: cannot tell a gap from current; assume a need.
            out.gap += 1
        # else: last is trade_date or newer -> already current, no need.
    out.est_requests = out.missing + out.gap + out.stale
    rate = max(rate_per_sec, 1e-9)
    out.est_seconds = out.est_requests / rate
    return out


def resolve_kline_budget(est_seconds: float, base_s: float,
                         cap_s: float) -> float:
    """Scan kline budget scaled by the remaining shortfall (F5).

    Covers the shortfall at ~1 req/s when it exceeds the base budget, never
    above the cap.
    """
    return min(float(cap_s), max(float(base_s), float(est_seconds)))


def _stop_requested(stop: Any) -> bool:
    """A threading.Event, a zero-arg callable, or None -> stop or not."""
    if stop is None:
        return False
    try:
        if hasattr(stop, "is_set"):
            return bool(stop.is_set())
        if callable(stop):
            return bool(stop())
    except Exception:  # noqa: BLE001
        return False
    return False


def warm_cache(symbols: list[str], target_trade_date: _dt.date, *,
               rate_per_sec: float = 1.0,
               deadline: float | None = None,
               progress: Callable[[int, int], None] | None = None,
               lock_check: Callable[[], None] | None = None,
               stop: Any = None,
               fetch: Callable | None = None,
               calendar=None,
               cache: KlineCache | None = None,
               limiter: KlineRateLimiter | None = None,
               max_symbols: int | None = None,
               now_fn: Callable[[], _dt.datetime] | None = None,
               stop_consecutive: int = _THROTTLE_STOP_AFTER,
               stop_window: int = 60,
               stop_min_error_rate: float = 0.5,
               stop_min_elapsed_s: float = 120.0,
               ) -> WarmResult:
    """Fill missing/stale/gapped cache tails, politely (F5b, offline-safe).

    Sequential at ≤ 1 req/s (shared limiter spacing), resumable (every code
    is written on success, so a rerun only sees the remainder), and never
    deletes or replaces data: only per-code verified frames are written and
    the ``_migration`` marker row is never touched. F4 stop rules apply.
    Intraday-safe: bars newer than the latest completed session are dropped
    before storing (a fetch during the session returns today's partial bar).

    Args:
        symbols: Prefixed symbols (``sh600519`` ...).
        target_trade_date: Trading day the cache is warmed for.
        rate_per_sec: Request rate, capped at 1.0.
        deadline: ``time.monotonic()`` timestamp to stop at (None = no cap).
        progress: Optional ``(done, total)`` callback (never breaks a run).
        lock_check: Called at start, every 25 codes and at end; any
            exception aborts the warm-up (``aborted=True``).
        stop: ``threading.Event`` or zero-arg callable requesting a stop.
        fetch: Injectable ``(url, params) -> str``; ``None`` uses live HTTP.
        calendar: Trading calendar (weekday fallback when None — no I/O).
        cache: Injectable cache (default path honours the home pin).
        limiter: Injectable limiter; otherwise built capped at 1 req/s.
        max_symbols: Attempt at most this many symbols (CLI ``--max``).
        now_fn: Injectable clock for the intraday guard (naive = Shanghai).
        stop_*: F4 thresholds for the built limiter.

    Returns:
        :class:`WarmResult`.
    """
    t0 = time.monotonic()
    result = WarmResult()
    cache = cache or KlineCache()
    now_fn = now_fn or (lambda: _dt.datetime.now())
    rate = min(float(rate_per_sec or 1.0), 1.0)
    limiter = limiter or KlineRateLimiter(
        rate_per_sec=rate, stop_consecutive=stop_consecutive,
        stop_window=stop_window, stop_min_error_rate=stop_min_error_rate,
        stop_min_elapsed_s=stop_min_elapsed_s)

    def _checked() -> str | None:
        if lock_check is None:
            return None
        try:
            lock_check()
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"[:200]
        return None

    lock_err = _checked()
    if lock_err is not None:
        result.aborted = True
        result.error = lock_err
        result.duration_s = time.monotonic() - t0
        return result

    needs: list[str] = []
    skipped_known = 0
    try:
        need_meta = cache.meta_map()
    except Exception:  # noqa: BLE001
        need_meta = {}
    try:
        need_live = set(cache.known_missing().keys())
    except Exception:  # noqa: BLE001
        need_live = set()
    need_prev_iso = _prev_day_iso(calendar, target_trade_date)
    for _sym in symbols:
        _code = _sym[2:] if len(_sym) > 6 else _sym
        if _code in need_live:
            skipped_known += 1
            continue
        _row = need_meta.get(_code)
        if _row is None or _row.get("source") == "known-missing":
            if _row is None:
                needs.append(_sym)
            continue
        try:
            _age = (_dt.datetime.now() - _dt.datetime.fromisoformat(
                _row.get("fetched_at"))).days
        except (TypeError, ValueError):
            _age = MAX_AGE_DAYS + 1
        if _age > MAX_AGE_DAYS:
            needs.append(_sym)
            continue
        _last = _row.get("last_date")
        if need_prev_iso is None:
            if _last != target_trade_date.isoformat():
                needs.append(_sym)
        elif _last != need_prev_iso and _last != target_trade_date.isoformat():
            needs.append(_sym)
    total = len(needs)
    done = 0
    # Intraday guard: same cutoff rule as the scan path — bars newer than
    # the latest completed session never reach the cache.
    cutoff = _resolve_cutoff(calendar, target_trade_date, now_fn())

    def _keep(df: pd.DataFrame | None) -> pd.DataFrame | None:
        kept = _drop_incomplete_bars(df, cutoff)
        try:
            if (df is not None and len(df) > 0
                    and (kept is None or len(kept) < len(df))):
                result.intraday_dropped += 1
        except Exception:  # noqa: BLE001
            pass
        return kept

    for sym in needs:
        if max_symbols is not None and done >= max_symbols:
            break
        if deadline is not None and time.monotonic() > deadline:
            result.deadline_hit = True
            break
        if _stop_requested(stop):
            result.aborted = True
            result.error = "stopped"
            break
        done += 1
        if done % 25 == 0:
            lock_err = _checked()
            if lock_err is not None:
                result.aborted = True
                result.error = lock_err
                break
        code = sym[2:] if len(sym) > 6 else sym
        limiter.wait()
        t_req = time.monotonic()
        try:
            cached = cache.get_frame(code)
            last = (cached["bars"].iloc[-1]["date"].date()
                    if cached is not None and len(cached["bars"]) else None)
            prev_iso = _prev_day_iso(calendar, target_trade_date)
            gap_days = 0
            if (cached is not None and last is not None
                    and prev_iso is not None
                    and last.isoformat() < prev_iso):
                gap_days = _gap_trading_days(calendar, last,
                                             target_trade_date)
            if 1 <= gap_days <= _MAX_GAPFILL_DAYS:
                tail = _keep(_fetch_one(sym, gap_days + _GAPFILL_OVERLAP,
                                        limiter=limiter, fetch=fetch))
                limiter.note_result(True, elapsed_s=time.monotonic() - t_req)
                result.requests += 1
                if tail is not None and cached is not None and _verify_tail_overlap(
                        cached["bars"], tail, code):
                    merged = _drop_incomplete_bars(
                        _merge_tail(cached["bars"], tail), cutoff)
                    cache.put_frame(code, merged, "tencent")
                    result.filled += 1
                else:
                    full = _keep(_fetch_one(sym, 260, limiter=limiter,
                                            fetch=fetch))
                    limiter.note_result(
                        True, elapsed_s=time.monotonic() - t_req)
                    result.requests += 1
                    if full is None or len(full) == 0:
                        # Only today's partial bar came back: nothing
                        # storable, retried next run (no 24 h block).
                        result.failed += 1
                        limiter.note_result(
                            False, elapsed_s=time.monotonic() - t_req)
                    else:
                        cache.put_frame(code, full, "tencent")
                        cache.clear_known_missing(sym)
                        result.filled += 1
            else:
                full = _keep(_fetch_one(sym, 260, limiter=limiter,
                                        fetch=fetch))
                limiter.note_result(True, elapsed_s=time.monotonic() - t_req)
                result.requests += 1
                if full is None or len(full) == 0:
                    result.failed += 1
                    limiter.note_result(
                        False, elapsed_s=time.monotonic() - t_req)
                else:
                    cache.put_frame(code, full, "tencent")
                    cache.clear_known_missing(sym)
                    result.filled += 1
        except ThrottleError as exc:
            result.requests += 1
            result.failed += 1
            limiter.note_result(False, throttle=True, desc=str(exc),
                                elapsed_s=time.monotonic() - t_req)
        except Exception as exc:  # noqa: BLE001 — per-symbol isolation
            result.requests += 1
            result.failed += 1
            if is_transport_error(exc):
                # F5d: retried next run, never known-missing.
                limiter.note_result(False, transport=True,
                                    desc=f"{type(exc).__name__}: {exc}"[:200],
                                    elapsed_s=time.monotonic() - t_req)
            else:
                limiter.note_result(False,
                                    desc=f"{type(exc).__name__}: {exc}"[:200],
                                    elapsed_s=time.monotonic() - t_req)
                cache.mark_known_missing(sym)
        if progress is not None:
            try:
                progress(done, total)
            except Exception:  # noqa: BLE001
                logger.exception("warmup progress callback raised")
        if limiter.unavailable:
            result.stopped_unavailable = True
            result.error = ("price source unavailable "
                            f"({limiter.unavailable_reason or 'outage'})")
            break
    lock_err = _checked()
    if lock_err is not None and not result.aborted:
        result.aborted = True
        result.error = lock_err
    result.attempted = done
    result.skipped_known_missing = skipped_known
    result.duration_s = time.monotonic() - t0
    return result


def _parse_sina_quote(text: str, symbol: str) -> dict | None:
    """Parse one Sina ``hq.sinajs.cn`` quote line into a bar dict, or None.

    Line shape (long-stable): ``var hq_str_<sym>="name,open,prev_close,
    price,high,low,...,volume,amount,..."`` with volume in shares and amount
    in CNY. The line carries NO date — callers must only use it for the
    just-closed session. ANY shape mismatch returns None (caller keeps the
    fetched frame and notes why).
    """
    try:
        payload = text.split("=", 1)[1].strip().strip('";\n')
        fields = payload.split(",")
        if len(fields) < 10:
            return None
        price = float(fields[3])
        if price <= 0:
            return None
        volume_lots = float(fields[8]) / _LOT_SIZE
        amount_reported = float(fields[9]) if fields[9] else None
        return {
            "open": float(fields[1]),
            "prev_close": float(fields[2]),
            "price": price,
            "high": float(fields[4]),
            "low": float(fields[5]),
            "volume_lots": volume_lots,
            "amount_reported": amount_reported,
        }
    except (IndexError, ValueError, TypeError):
        return None


def _fetch_index_quote(symbol: str,
                       quote_fetch: Callable[[str], dict | None] | None,
                       timeout: float = 15.0) -> dict | None:
    """One index bar dict for the just-closed session, or None.

    ``quote_fetch`` (tests) returns the bar dict or None directly. The live
    path goes through the shared section-9b HTTP layer
    (:func:`_common.http_get_text` with the ``sina`` host bucket): per-host
    throttle, run-meter accounting, retries, no dead proxy, same
    Referer/User-Agent as the other Sina callers. Only ASCII numeric fields
    are parsed, so the GBK body needs no charset handling. Never raises:
    any failure degrades to ``None`` (the caller keeps the fetched frame).
    """
    if quote_fetch is not None:
        try:
            return quote_fetch(symbol)
        except Exception:  # noqa: BLE001
            return None
    try:
        from src.market_scanner.sources import _common as _http

        text = _http.http_get_text(
            "https://hq.sinajs.cn/list=" + symbol,
            host_key="sina", timeout=timeout,
        )
        return _parse_sina_quote(text, symbol)
    except Exception:  # noqa: BLE001 — quote is best-effort only
        return None


def fetch_benchmark(
    symbol: str = "sh000300",
    bars: int = 260,
    fetch: Callable | None = None,
    cache: KlineCache | None = None,
    limiter: KlineRateLimiter | None = None,
    now_fn: Callable[[], _dt.datetime] | None = None,
    trade_date: _dt.date | None = None,
    quote_fetch: Callable[[str], dict | None] | None = None,
) -> tuple[pd.DataFrame | None, SourceStatus]:
    """Benchmark index daily qfq bars (CSI 300), cached; stale fallback.

    Intraday-safe like the stock path: bars newer than the latest completed
    session are dropped before storing (a fetch during the session returns
    today's partial bar). When ``trade_date`` names a completed session the
    fetched history does not reach yet, one Sina quote appends trade_date's
    bar (F10; fully injectable for offline tests, graceful fallback).
    """
    t0 = time.monotonic()
    cache = cache or KlineCache()
    limiter = limiter or _KLINE_LIMITER
    now_fn = now_fn or (lambda: _dt.datetime.now())
    now = now_fn()
    try:
        df = _fetch_one(symbol, bars, limiter=limiter, fetch=fetch)
        cutoff = _resolve_cutoff(None, trade_date, now)
        df = _drop_incomplete_bars(df, cutoff)
        if df is None or len(df) == 0:
            raise ValueError("no completed benchmark bars (intraday fetch)")
        note = ""
        last_bar = df["date"].max()
        last_day = last_bar.date() if hasattr(last_bar, "date") else last_bar
        if (trade_date is not None and last_day < trade_date
                and trade_date == now.date()
                and now.time() >= _dt.time(15, 0)):
            # Today's session just closed but fqkline does not carry it yet:
            # append it from one quote (never fabricate, never intraday).
            quote = _fetch_index_quote(symbol, quote_fetch)
            if quote is not None and quote.get("price"):
                try:
                    ref_close = float(df.iloc[-1]["close"])
                    q_close = float(quote["price"])
                    tail_vol = float(
                        df["volume"].tail(5).median() or 0)
                    q_vol = float(quote.get("volume_lots") or 0)
                    vol_ok = (tail_vol <= 0 or q_vol <= 0 or
                              (_VOLUME_RATIO_LO
                               <= q_vol / tail_vol <= _VOLUME_RATIO_HI))
                    close_ok = (ref_close > 0
                                and abs(q_close - ref_close) / ref_close
                                <= 0.10)
                except (TypeError, ValueError, ZeroDivisionError):
                    close_ok, vol_ok = False, False
                if close_ok and vol_ok:
                    new_bar = pd.DataFrame([{
                        "date": pd.Timestamp(trade_date),
                        "open": quote.get("open", q_close),
                        "high": quote.get("high", q_close),
                        "low": quote.get("low", q_close),
                        "close": q_close,
                        "volume": q_vol,
                        "amount": q_close * q_vol * _LOT_SIZE,
                    }])
                    df = pd.concat([df, new_bar], ignore_index=True)
                    df = df.sort_values("date").reset_index(drop=True)
                    note = " +quote-appended"
                else:
                    note = " (quote failed verification)"
            else:
                note = " (no quote)"
        cache.put_frame(symbol, df, "tencent")
        _last = df["date"].max()
        _last = _last.date() if hasattr(_last, "date") else _last
        return df, SourceStatus(
            name="benchmark", status="ok", rows=len(df),
            latency_s=time.monotonic() - t0,
            as_of=_last.isoformat(),
            detail=f"symbol={symbol} bars={bars} source=tencent fqkline{note}",
        )
        _last = df["date"].max()
        _last = _last.date() if hasattr(_last, "date") else _last
        return df, SourceStatus(
            name="benchmark", status="ok", rows=len(df),
            latency_s=time.monotonic() - t0,
            as_of=_last.isoformat(),
            detail=f"symbol={symbol} bars={bars} source=tencent fqkline",
        )
    except Exception as exc:  # noqa: BLE001 — stale cache fallback
        cached = cache.get_frame(symbol)
        if cached is not None and not cached["bars"].empty:
            last = cached["bars"]["date"].max().date()
            return cached["bars"], SourceStatus(
                name="benchmark", status="degraded", rows=len(cached["bars"]),
                latency_s=time.monotonic() - t0,
                error=f"{type(exc).__name__}: {exc}"[:200],
                detail=f"benchmark stale (last bar {last}); served from cache",
                as_of=last.isoformat(),
            )
        return None, SourceStatus(
            name="benchmark", status="failed", rows=0,
            latency_s=time.monotonic() - t0,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )
