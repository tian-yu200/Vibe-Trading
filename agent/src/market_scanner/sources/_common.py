"""Shared plumbing for market_scanner sources: board mapping, throttled HTTP,
retries with backoff, unit parsers."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, TypeVar

import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Proxy guard — the system proxy 127.0.0.1:10808 is dead on this machine.
# requests honours HTTP(S)_PROXY env vars (and on Windows some stacks read the
# registry); we pin trust_env off on the sessions we create so nothing routes
# through any inherited proxy. System settings are never modified.
# ---------------------------------------------------------------------------
DEAD_PROXY_HOST = "127.0.0.1:10808"


def make_session() -> requests.Session:
    """Session that bypasses every inherited proxy setting."""
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": "", "https": ""}
    return session


def assert_no_dead_proxy(url: str) -> None:
    """Defensive check: fail loudly in tests if a dead proxy ever leaks in."""
    proxies = requests.utils.get_environ_proxies(url, no_proxy=None)
    for scheme_proxy in proxies.values():
        if DEAD_PROXY_HOST in str(scheme_proxy):
            raise RuntimeError(f"dead proxy {DEAD_PROXY_HOST} leaked into request env for {url}")


# ---------------------------------------------------------------------------
# Board mapping (CONTRACT section 2)
# ---------------------------------------------------------------------------
def board_from_code(code: str) -> str:
    """Map a 6-digit A-share code prefix to its board.

    Args:
        code: 6-digit code string (no exchange prefix).

    Returns:
        One of ``main_sh``, ``main_sz``, ``star``, ``chinext``, ``bse``,
        or ``unknown`` when the prefix matches nothing.
    """
    c = str(code).zfill(6)
    if c[:3] in {"600", "601", "603", "605"}:
        return "main_sh"
    if c[:3] in {"000", "001", "002", "003"}:
        return "main_sz"
    if c[:3] in {"688", "689"}:
        return "star"
    if c[:3] in {"300", "301", "302"}:
        return "chinext"
    if c[0] in {"4", "8"} or c[:2] == "92":
        return "bse"
    return "unknown"


#: Boards whose membership is fully determined by the code prefix. The
#: listing/name sources only carry an exchange, never a board; a STAR or
#: ChiNext code is unambiguously identifiable from its prefix, so a vendor
#: row that disagrees (Sina's hs_a list reporting 300xxx as main_sz, an
#: akshare name list missing the STAR split) must never win.
_PREFIX_DETERMINED = frozenset({"star", "chinext", "bse"})


def resolve_board(code: Any, row_board: Any = None) -> str:
    """Code prefix wins over any vendor-supplied ``board`` string.

    ``board_from_code`` is the single source of truth for board identity:
    every prefix in the CONTRACT table maps to exactly one board, so a
    disagreement means the vendor row is wrong, not the prefix.

    Args:
        code: 6-digit code (str/int) — the authoritative signal.
        row_board: Optional board value from a universe/listing row.

    Returns:
        The prefix-derived board when the prefix is in the contract table,
        else ``row_board`` (str) or ``"unknown"``.
    """
    from_code = board_from_code(code)
    if from_code in _PREFIX_DETERMINED:
        return from_code
    text = str(row_board).strip() if row_board is not None else ""
    return text or (from_code if from_code != "unknown" else "unknown")


def exchange_from_code(code: str) -> str:
    """Exchange for a 6-digit code: SH / SZ / BJ (or '' when unknown)."""
    board = board_from_code(code)
    return {"main_sh": "SH", "star": "SH", "main_sz": "SZ", "chinext": "SZ", "bse": "BJ"}.get(
        board, ""
    )


def symbol_from_code(code: str) -> str:
    """Prefixed symbol (``sh600519`` / ``sz000001`` / ``bj920002``)."""
    ex = exchange_from_code(code)
    return f"{ex.lower()}{str(code).zfill(6)}" if ex else ""


# ---------------------------------------------------------------------------
# HTTP fetch with per-host spacing + retries (mirrors backtest.loaders._http
# semantics; a local copy so sources do not depend on loader registry side
# effects, but using the identical HostThrottle from _http for spacing).
# ---------------------------------------------------------------------------
from backtest.loaders._http import DEFAULT_USER_AGENT, HostThrottle  # noqa: E402

_THROTTLE = HostThrottle()
_SESSIONS: dict[str, requests.Session] = {}

_SINA_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Referer": "https://finance.sina.com.cn/",
}
_TENCENT_HEADERS = {"User-Agent": DEFAULT_USER_AGENT}


def _session_for(bucket: str) -> requests.Session:
    """Per-bucket session with proxies disabled."""
    session = _SESSIONS.get(bucket)
    if session is None:
        session = make_session()
        _SESSIONS[bucket] = session
    return session


# ---------------------------------------------------------------------------
# Per-host politeness defaults + request meter (CONTRACT section 9b)
# ---------------------------------------------------------------------------
# Rates come from src/config/env_schema.py (agent_tuning.*_rate_per_sec,
# defaults Sina 5 / Eastmoney 3 / THS 5 / Tencent kline 1; every other host
# 5) — see host_rate().

_THROTTLE_STOP_AFTER = 10  # consecutive throttle-type errors -> source stops


def _politeness_bucket(host_key: str) -> str:
    """Canonical politeness bucket for a host key (substring-classified).

    Callers pass either bucket names (``sina``, ``eastmoney``, ``ths``) or
    full hostnames (``search-api-web.eastmoney.com``,
    ``vip.stock.finance.sina.com.cn``). Classification is by substring so any
    ``*.eastmoney.com`` host shares one Eastmoney throttle bucket and cap.
    """
    key = (host_key or "").lower()
    if "eastmoney" in key:
        return "eastmoney"
    if "sina" in key:
        return "sina"
    if "ths" in key or "hexin" in key:
        return "ths"
    if "tencent" in key or "gtimg" in key:
        return "tencent"
    return key or "other"


def host_rate(host_key: str) -> float:
    """Conservative sustained rate for a host bucket (env-schema gated).

    Reads ``get_env_config().agent_tuning.{sina,eastmoney,ths,kline}_rate_per_sec``
    (defaults 5/3/5/1) — never ``os.environ`` directly (CI env-var gate).
    Any host outside those families caps at 5 req/s (section 9b).
    """
    from src.config.accessor import get_env_config

    tuning = get_env_config().agent_tuning
    rates = {
        "sina": tuning.vibe_trading_sina_rate_per_sec,
        "eastmoney": tuning.vibe_trading_eastmoney_rate_per_sec,
        "ths": tuning.vibe_trading_ths_rate_per_sec,
        "tencent": tuning.vibe_trading_kline_rate_per_sec,
    }
    return max(0.1, float(rates.get(_politeness_bucket(host_key), 5.0)))


def host_min_interval(host_key: str) -> float:
    """Minimum spacing (s) between requests to ``host_key``."""
    return 1.0 / host_rate(host_key)


class HostMeter:
    """Thread-safe per-host request meter, scoped to one run.

    Fed by :func:`http_get_text`/:func:`http_get_json`, the kline limiter and
    :func:`meter_akshare_call`. Reset per run: the pipeline creates a fresh
    meter and installs it via :func:`set_active_meter`, so two runs never mix.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, dict[str, float]] = {}
        self._t0 = time.monotonic()

    def record(self, host: str, *, ok: bool, elapsed_s: float) -> None:
        """Record one request outcome against ``host``.

        Tracks the wall-clock span (first request start -> last request end)
        in ``elapsed_s`` and the summed in-flight time in ``busy_s``, so
        ``effective_rps = count / wall span`` (not count / busy time — with
        concurrency the old summing understated the span and overstated rps).
        """
        with self._lock:
            stat = self._stats.setdefault(
                host, {"count": 0, "errors": 0, "elapsed_s": 0.0, "busy_s": 0.0,
                       "_first_start": time.monotonic(), "_last_end": 0.0}
            )
            stat["count"] += 1
            if not ok:
                stat["errors"] += 1
            stat["busy_s"] += elapsed_s
            now = time.monotonic()
            # first request start: we only see the end, so take the min
            stat["_first_start"] = min(stat["_first_start"], now - elapsed_s)
            stat["_last_end"] = max(stat["_last_end"], now)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """``{host: {count, errors, elapsed_s, busy_s, effective_rps}}`` + totals."""
        with self._lock:
            stats = {h: dict(v) for h, v in self._stats.items()}
        for v in stats.values():
            first = v.pop("_first_start")
            last = v.pop("_last_end")
            span = max(last - first, 1e-9)
            v["elapsed_s"] = round(span, 2)
            v["busy_s"] = round(v["busy_s"], 2)
            v["effective_rps"] = round(v["count"] / span, 2)
        total_count = sum(v["count"] for v in stats.values())
        total_errors = sum(v["errors"] for v in stats.values())
        total_busy = sum(v.get("busy_s", 0.0) for v in stats.values())
        wall = max(time.monotonic() - self._t0, 1e-9)
        stats["_totals"] = {
            "count": total_count,
            "errors": total_errors,
            "elapsed_s": round(wall, 2),
            "busy_s": round(total_busy, 2),
            "effective_rps": round(total_count / wall, 2),
        }
        return stats


_ACTIVE_METER: HostMeter | None = None
_ACTIVE_METER_LOCK = threading.Lock()


def set_active_meter(meter: HostMeter | None) -> None:
    """Install (or clear, with None) the run-scoped meter."""
    global _ACTIVE_METER
    with _ACTIVE_METER_LOCK:
        _ACTIVE_METER = meter


def get_active_meter() -> HostMeter | None:
    """The run-scoped meter, or None outside a scan."""
    with _ACTIVE_METER_LOCK:
        return _ACTIVE_METER


def meter_akshare_call(fn: Callable, *args: Any, host: str = "eastmoney",
                       pages: int = 1, **kwargs: Any) -> Any:
    """Call one akshare function, metering it against ``host``.

    Args:
        fn: The akshare callable.
        host: Host bucket the function hits (sina / eastmoney / ths / ...).
        pages: Known page count (akshare boards paginate internally).

    Returns:
        Whatever ``fn`` returns. Errors propagate to the caller's handler,
        but the failed call is still metered.
    """
    # Bursts must respect the host cap too: space consecutive akshare calls
    # on the shared throttle (same bucket the HTTP paths wait on).
    _THROTTLE.wait(_politeness_bucket(host), host_min_interval(host))
    meter = get_active_meter()
    t0 = time.monotonic()
    try:
        result = fn(*args, **kwargs)
    except Exception:
        if meter is not None:
            meter.record(host, ok=False, elapsed_s=time.monotonic() - t0)
        raise
    if meter is not None:
        meter.record(host, ok=True, elapsed_s=time.monotonic() - t0)
    return result


def http_get_text(
    url: str,
    *,
    host_key: str,
    min_interval: float | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    retries: int = 3,
    backoff: float = 0.6,
) -> str:
    """Throttled GET returning the decoded body text; retries with backoff.

    Args:
        min_interval: Spacing override; None uses the section 9b host default.
            A caller override may be *slower* than the host cap but never
            faster — the cap floor is enforced here so bursts stay polite.

    Raises on the final attempt (callers wrap in try/except and convert to
    SourceStatus failed). Sessions never use the dead system proxy.
    """
    # Section 9b cap: clamp the caller's interval up to the host floor.
    floor = host_min_interval(host_key)
    if min_interval is None or min_interval < floor:
        min_interval = floor
    merged = dict(_SINA_HEADERS if "sina" in host_key else _TENCENT_HEADERS)
    if headers:
        merged.update(headers)
    meter = get_active_meter()
    last_exc: Exception | None = None
    for attempt in range(retries):
        _THROTTLE.wait(_politeness_bucket(host_key), min_interval)
        t0 = time.monotonic()
        try:
            resp = _session_for(host_key).get(
                url, params=params, headers=merged, timeout=timeout
            )
            resp.raise_for_status()
            if meter is not None:
                meter.record(host_key, ok=True, elapsed_s=time.monotonic() - t0)
            return resp.text
        except Exception as exc:  # noqa: BLE001 — retried, caller converts to status
            last_exc = exc
            if meter is not None:
                meter.record(host_key, ok=False, elapsed_s=time.monotonic() - t0)
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
    assert last_exc is not None
    raise last_exc


def http_get_json(
    url: str,
    *,
    host_key: str,
    min_interval: float | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    retries: int = 3,
    backoff: float = 0.6,
) -> Any:
    """Throttled GET returning parsed JSON (section 9b host default spacing)."""
    return json.loads(
        http_get_text(
            url,
            host_key=host_key,
            min_interval=min_interval,
            params=params,
            headers=headers,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
    )


# ---------------------------------------------------------------------------
# Unit parsers
# ---------------------------------------------------------------------------
_SINA_NODE_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
_SINA_COUNT_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeStockCount"
)
_SINA_PAGE_SIZE = 100


def sina_fetch_page(
    node: str, page: int, num: int = _SINA_PAGE_SIZE, fetch: Callable | None = None
) -> list[dict]:
    """One Sina Market_Center page as a list of row dicts.

    Args:
        node: Sina node name (``hs_a``, ``hs_bjs``).
        page: 1-based page number.
        num: Page size (Sina caps at 100).
        fetch: Injectable ``(url, params) -> str`` for tests.

    Returns:
        List of row dicts (Sina's own key names).
    """
    if fetch is None:
        text = http_get_text(
            _SINA_NODE_URL,
            host_key="sina",
            min_interval=0.22,
            params={
                "page": page,
                "num": num,
                "sort": "symbol",
                "asc": 1,
                "node": node,
                "symbol": "",
                "_s_r_a": "page",
            },
        )
    else:
        text = fetch(
            _SINA_NODE_URL,
            {
                "page": page,
                "num": num,
                "sort": "symbol",
                "asc": 1,
                "node": node,
                "symbol": "",
                "_s_r_a": "page",
            },
        )
    data = json.loads(text)
    if not isinstance(data, list):
        return []
    return data


def sina_count(node: str, fetch: Callable | None = None) -> int:
    """Sina node stock count (returns 0 when the node has no count)."""
    if fetch is None:
        text = http_get_text(_SINA_COUNT_URL, host_key="sina", params={"node": node})
    else:
        text = fetch(_SINA_COUNT_URL, {"node": node})
    try:
        value = json.loads(text.strip())
    except (ValueError, TypeError):
        return 0
    if isinstance(value, list):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_ths_money(text: str) -> float | None:
    """Parse a THS money string like ``'2.91亿'`` / ``'20.10万'`` into CNY.

    Args:
        text: Raw cell, e.g. ``'8.16亿'``, ``'2311.36万'``, ``'-'``, ``''``.

    Returns:
        CNY float, or None when the text carries no number.
    """
    if text is None:
        return None
    s = str(text).strip().replace(",", "")
    if not s or s in {"-", "--"}:
        return None
    mult = 1.0
    if s.endswith("亿"):
        mult = 1e8
        s = s[:-1]
    elif s.endswith("万"):
        mult = 1e4
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    """Best-effort float conversion (Sina returns strings, '-' for missing)."""
    if value is None:
        return None
    try:
        s = str(value).strip().replace(",", "")
        if s in {"", "-", "--", "None"}:
            return None
        return float(s)
    except ValueError:
        return None


def _sina_name_flags(name: str) -> tuple[bool, bool]:
    """(is_st, is_delisting) from the display name."""
    return ("ST" in name.upper()), ("退" in name)
