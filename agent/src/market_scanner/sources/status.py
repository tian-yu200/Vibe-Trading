"""SourceStatus dataclass shared by all market_scanner sources."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SourceStatus:
    """Outcome of one source fetch, per CONTRACT section 3.

    Attributes:
        name: Source identifier, e.g. ``"universe"``.
        status: ``ok`` / ``degraded`` / ``failed`` / ``skipped``.
        rows: Number of rows produced (0 when failed).
        latency_s: Wall-clock seconds of the fetch.
        error: Error text when ``failed`` (or partly ``degraded``).
        detail: Free-form extra info (e.g. failure counts, unit caveats).
        as_of: ISO timestamp of the data moment, when known.
        requests: Host request counts this source made (section 9b), e.g.
            ``{"sina": {"count": 60, "errors": 0}}``.
        effective_rps: Achieved requests/second across those requests.
    """

    name: str
    status: Literal["ok", "degraded", "failed", "skipped"] = "ok"
    rows: int = 0
    latency_s: float = 0.0
    error: str | None = None
    detail: str | None = None
    as_of: str | None = field(default=None)
    requests: dict[str, Any] = field(default_factory=dict)
    effective_rps: float | None = None

    def to_dict(self) -> dict:
        """JSON-serialisable dict for the ScanRun envelope."""
        return {
            "name": self.name,
            "status": self.status,
            "rows": self.rows,
            "latency_s": round(self.latency_s, 3),
            "error": self.error,
            "detail": self.detail,
            "as_of": self.as_of,
            "requests": self.requests,
            "effective_rps": self.effective_rps,
        }


def metered_status(name: str, hosts: list[str], latency_s: float, **kw) -> SourceStatus:
    """Build a SourceStatus with requests/effective_rps from the active meter.

    Args:
        name: Source name.
        hosts: Host buckets this source hit (e.g. ``["sina"]``).
        latency_s: Wall-clock latency of the source fetch.
        **kw: Remaining SourceStatus fields (status, rows, error, detail...).
    """
    meter = get_active_meter()
    stats = meter.to_dict() if meter is not None else {}
    requests = {h: stats[h] for h in hosts if h in stats}
    count = sum(v.get("count", 0) for v in requests.values())
    rps = round(count / max(latency_s, 1e-9), 2) if latency_s > 0 else None
    return SourceStatus(name=name, latency_s=latency_s, requests=requests,
                        effective_rps=rps, **kw)


def fail_status(name: str, exc: Exception, t0: float | None = None,
                hosts: list[str] | None = None) -> SourceStatus:
    """Build a failed SourceStatus from an exception (never raises)."""
    latency = time.monotonic() - t0 if t0 is not None else 0.0
    hosts = hosts or []
    try:
        from src.market_scanner.sources._common import get_active_meter
        meter = get_active_meter()
    except Exception:  # noqa: BLE001 — status building must never raise
        meter = None
    stats = meter.to_dict() if meter is not None else {}
    requests = {h: stats[h] for h in hosts if h in stats}
    count = sum(v.get("count", 0) for v in requests.values())
    rps = round(count / max(latency, 1e-9), 2) if latency > 0 else None
    return SourceStatus(
        name=name,
        status="failed",
        latency_s=latency,
        requests=requests,
        effective_rps=rps,
        rows=0,
        error=f"{type(exc).__name__}: {exc}"[:500],
    )
