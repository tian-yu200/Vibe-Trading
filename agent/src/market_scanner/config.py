"""ScanConfig — scanner settings with validation (CONTRACT section 4)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .models import DEFAULT_WEIGHTS, SUBSCORE_KEYS

logger = logging.getLogger(__name__)

_ALLOWED_KEYS = frozenset(
    {
        "top_n",
        "min_total_mv",
        "min_avg_amount_20d",
        "min_listing_days",
        "include_star",
        "include_chinext",
        "weights",
        "max_per_industry",
        "min_kline_bars",
        "kline_bars",
        "kline_workers",
    "kline_rate_per_sec",
    "kline_time_budget_s",
    "kline_stop_consecutive",
    "kline_stop_window",
    "kline_stop_min_error_rate",
    "kline_stop_min_elapsed_s",
    "kline_time_budget_cap_s",
    "kline_local_budget_s",
        "ai_enabled",
        "ai_review_pool",
        "ai_language",
        "ai_model",
        "ai_fallback_model",
        "ai_quota_wait_max_s",
        "ai_time_budget_s",
        "ai_call_timeout_s",
        "news_days",
        "min_coverage",
        "exclude_shrinking",
    }
)


@dataclass
class ScanConfig:
    """Scanner settings. Defaults are the CONTRACT section 4 table.

    All fields are validated on construction (and by :meth:`validate`), so a
    config object that exists is always usable.

    Attributes:
        top_n: Number of finalists after the industry cap (before AI review).
        min_total_mv: Minimum total market cap in CNY (micro-cap cut).
        min_avg_amount_20d: Minimum 20-day average kline amount in CNY.
        min_listing_days: Minimum days since listing.
        include_star: Include STAR Market (688/689) when True.
        include_chinext: Include ChiNext (300/301/302) when True.
        weights: Composite weights; keys exactly ``fundamental``/``trend``/
            ``flow``, summing to 1.0. Never renormalised at scoring time.
        max_per_industry: At most this many picks per industry.
        min_kline_bars: Minimum usable kline bars for the trend cut.
        kline_bars: Bars requested from the kline source.
        kline_workers: Worker threads for the kline source.
        kline_rate_per_sec: Scan-time kline refetch rate (requests/second).
        kline_time_budget_s: Wall-clock budget (s) for scan-time kline
            refetches; over budget, remaining symbols stay missing.
        kline_stop_consecutive: Consecutive transport/throttle failures that
            mark the price source unavailable and stop kline fetching (F4).
        kline_stop_window: Request window for the F4 outage rule.
        kline_stop_min_error_rate: Error fraction over the window that marks
            the source unavailable once kline_stop_min_elapsed_s have passed.
        kline_stop_min_elapsed_s: Minimum kline-stage wall time before the
            window rule can fire (avoids instant stops on tiny runs).
        kline_time_budget_cap_s: Hard cap (s) for the shortfall-scaled scan
            kline budget (F5: base budget grows with the remaining shortfall
            at ~1 req/s, never above this cap).
        kline_local_budget_s: Generous cap (s) for the scan's LOCAL kline
            phase (cache reads + appends). Local work never consumes the
            network budget (F10); leftovers become refetch needs.
        ai_enabled: Whether the AI review step should run when implemented.
        ai_review_pool: How many finalists the AI review considers.
        ai_language: Language for AI summaries/reasons.
        ai_model: Model name for the AI review; ``None`` uses the app default.
        ai_fallback_model: Optional fallback model; ``None`` disables fallback.
        ai_quota_wait_max_s: Max seconds to sleep on a near provider rate-limit.
        ai_time_budget_s: Wall-clock budget for the AI review (sleeps count).
        ai_call_timeout_s: Per-call wall-clock timeout for the AI review.
        news_days: Lookback window for per-stock news in the AI review.
        exclude_shrinking: Exclude stocks with negative revenue and recurring-
            profit growth after fundamentals are available.
        min_coverage: Coverage floor in [0, 1]. After scoring, the run fails
            when trend or fundamentals coverage (see pipeline coverage guard)
            falls below this ratio.
    """

    top_n: int = 20
    min_total_mv: float = 3e9
    min_avg_amount_20d: float = 5e7
    min_listing_days: int = 365
    include_star: bool = True
    include_chinext: bool = True
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    max_per_industry: int = 3
    min_kline_bars: int = 120
    kline_bars: int = 260
    kline_workers: int = 6
    kline_rate_per_sec: float = 1.0
    kline_time_budget_s: float = 600.0
    kline_stop_consecutive: int = 10
    kline_stop_window: int = 60
    kline_stop_min_error_rate: float = 0.5
    kline_stop_min_elapsed_s: float = 120.0
    kline_time_budget_cap_s: float = 5400.0
    kline_local_budget_s: float = 1800.0
    ai_enabled: bool = True
    ai_review_pool: int = 30
    ai_language: str = "en"
    ai_model: str | None = None
    ai_fallback_model: str | None = None
    ai_quota_wait_max_s: float = 300.0
    ai_time_budget_s: float = 900.0
    ai_call_timeout_s: float = 90.0
    news_days: int = 30
    min_coverage: float = 0.90
    exclude_shrinking: bool = True

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Raise ``ValueError`` when any field is out of contract range.

        Integral floats (``20.0``) are coerced to ``int`` for int fields so a
        JSON round-trip always lands back on the same types.
        """
        self.top_n = _require_int("top_n", self.top_n, minimum=1)
        _require_number("min_total_mv", self.min_total_mv, minimum=0.0)
        _require_number("min_avg_amount_20d", self.min_avg_amount_20d, minimum=0.0)
        self.min_listing_days = _require_int("min_listing_days", self.min_listing_days, minimum=0)
        _require_bool("include_star", self.include_star)
        _require_bool("include_chinext", self.include_chinext)
        _validate_weights(self.weights)
        self.max_per_industry = _require_int(
            "max_per_industry", self.max_per_industry, minimum=1
        )
        self.min_kline_bars = _require_int("min_kline_bars", self.min_kline_bars, minimum=1)
        self.kline_bars = _require_int("kline_bars", self.kline_bars, minimum=1)
        if self.kline_bars < self.min_kline_bars:
            raise ValueError(
                f"kline_bars ({self.kline_bars}) must be >= min_kline_bars "
                f"({self.min_kline_bars})"
            )
        self.kline_workers = _require_int("kline_workers", self.kline_workers, minimum=1)
        self.kline_rate_per_sec = _require_number(
            "kline_rate_per_sec", self.kline_rate_per_sec, minimum=0.1
        )
        self.kline_time_budget_s = _require_number(
            "kline_time_budget_s", self.kline_time_budget_s, minimum=0.0
        )
        self.kline_stop_consecutive = _require_int(
            "kline_stop_consecutive", self.kline_stop_consecutive, minimum=1
        )
        self.kline_stop_window = _require_int(
            "kline_stop_window", self.kline_stop_window, minimum=10
        )
        self.kline_stop_min_error_rate = _require_number(
            "kline_stop_min_error_rate", self.kline_stop_min_error_rate,
            minimum=0.0,
        )
        if self.kline_stop_min_error_rate > 1.0:
            raise ValueError(
                "kline_stop_min_error_rate must be <= 1.0, "
                f"got {self.kline_stop_min_error_rate}"
            )
        self.kline_stop_min_elapsed_s = _require_number(
            "kline_stop_min_elapsed_s", self.kline_stop_min_elapsed_s,
            minimum=0.0,
        )
        self.kline_time_budget_cap_s = _require_number(
            "kline_time_budget_cap_s", self.kline_time_budget_cap_s, minimum=0.0
        )
        if self.kline_time_budget_cap_s < self.kline_time_budget_s:
            raise ValueError(
                "kline_time_budget_cap_s "
                f"({self.kline_time_budget_cap_s}) must be >= "
                f"kline_time_budget_s ({self.kline_time_budget_s})"
            )
        self.kline_local_budget_s = _require_number(
            "kline_local_budget_s", self.kline_local_budget_s, minimum=0.0
        )
        _require_bool("ai_enabled", self.ai_enabled)
        self.ai_review_pool = _require_int("ai_review_pool", self.ai_review_pool, minimum=1)
        if not isinstance(self.ai_language, str) or not self.ai_language.strip():
            raise ValueError(
                f"ai_language must be a non-empty str, got {self.ai_language!r}"
            )
        self.ai_language = self.ai_language.strip().lower()
        self.ai_model = _optional_str("ai_model", self.ai_model)
        self.ai_fallback_model = _optional_str(
            "ai_fallback_model", self.ai_fallback_model
        )
        self.ai_quota_wait_max_s = _require_number(
            "ai_quota_wait_max_s", self.ai_quota_wait_max_s, minimum=0.0
        )
        self.ai_time_budget_s = _require_number(
            "ai_time_budget_s", self.ai_time_budget_s, minimum=0.0
        )
        self.ai_call_timeout_s = _require_number(
            "ai_call_timeout_s", self.ai_call_timeout_s, minimum=0.0
        )
        self.news_days = _require_int("news_days", self.news_days, minimum=1)
        _require_bool("exclude_shrinking", self.exclude_shrinking)
        _require_number("min_coverage", self.min_coverage, minimum=0.0)
        if self.min_coverage > 1.0:
            raise ValueError(
                f"min_coverage must be <= 1.0, got {self.min_coverage}"
            )
        # Defensive copy so caller mutation cannot bypass validation later.
        self.weights = {str(k): float(v) for k, v in self.weights.items()}

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict of every setting (fresh copies of dicts)."""
        return {
            "top_n": self.top_n,
            "min_total_mv": self.min_total_mv,
            "min_avg_amount_20d": self.min_avg_amount_20d,
            "min_listing_days": self.min_listing_days,
            "include_star": self.include_star,
            "include_chinext": self.include_chinext,
            "weights": dict(self.weights),
            "max_per_industry": self.max_per_industry,
            "min_kline_bars": self.min_kline_bars,
            "kline_bars": self.kline_bars,
            "kline_workers": self.kline_workers,
            "kline_rate_per_sec": self.kline_rate_per_sec,
            "kline_time_budget_s": self.kline_time_budget_s,
            "kline_stop_consecutive": self.kline_stop_consecutive,
            "kline_stop_window": self.kline_stop_window,
            "kline_stop_min_error_rate": self.kline_stop_min_error_rate,
            "kline_stop_min_elapsed_s": self.kline_stop_min_elapsed_s,
            "kline_time_budget_cap_s": self.kline_time_budget_cap_s,
        "kline_local_budget_s": self.kline_local_budget_s,
            "ai_enabled": self.ai_enabled,
            "ai_review_pool": self.ai_review_pool,
            "ai_language": self.ai_language,
            "ai_model": self.ai_model,
            "ai_fallback_model": self.ai_fallback_model,
            "ai_quota_wait_max_s": self.ai_quota_wait_max_s,
            "ai_time_budget_s": self.ai_time_budget_s,
            "ai_call_timeout_s": self.ai_call_timeout_s,
            "news_days": self.news_days,
            "min_coverage": self.min_coverage,
            "exclude_shrinking": self.exclude_shrinking,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ScanConfig:
        """Build a validated config; missing keys take defaults, extras raise."""
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError(f"ScanConfig.from_dict expects a dict, got {type(data).__name__}")
        unknown = sorted(set(data) - _ALLOWED_KEYS)
        if unknown:
            raise ValueError(f"unknown ScanConfig keys: {unknown}")
        payload = dict(data)
        if isinstance(payload.get("weights"), dict):
            payload["weights"] = dict(payload["weights"])
        return cls(**payload)


# --------------------------------------------------------------------------- #
# Field validators
# --------------------------------------------------------------------------- #


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value).__name__}")


def _optional_str(name: str, value: Any) -> str | None:
    """Accept ``None`` or a non-empty stripped string for an optional setting."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str or None, got {type(value).__name__}")
    stripped = value.strip()
    return stripped or None


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        # Accept integral floats (20.0) for JSON round-trips, reject the rest.
        if isinstance(value, float) and not isinstance(value, bool) and value.is_integer():
            value = int(value)
        else:
            raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _require_number(name: str, value: Any, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return float(value)


def _validate_weights(weights: Any) -> None:
    if not isinstance(weights, dict):
        raise ValueError(f"weights must be a dict, got {type(weights).__name__}")
    keys = set(weights)
    expected = set(SUBSCORE_KEYS)
    if keys != expected:
        raise ValueError(
            f"weights keys must be exactly {sorted(expected)}, got {sorted(keys)}"
        )
    total = 0.0
    for key in SUBSCORE_KEYS:
        value = weights[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"weights[{key!r}] must be a number, got {type(value).__name__}")
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"weights[{key!r}] must be finite, got {value}")
        if value < 0:
            raise ValueError(f"weights[{key!r}] must be >= 0, got {value}")
        total += float(value)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0, got {total}")
