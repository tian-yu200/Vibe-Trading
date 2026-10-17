"""Identifier helpers, funnel constants and envelope builders (CONTRACT §2/§5/§7).

Stdlib-only on purpose: config, exclusions, scoring and pipeline all import it,
so it must never pull in pandas or the source adapters.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Scoring vocabulary
# --------------------------------------------------------------------------- #

#: The three sub-score keys that make up the composite (CONTRACT §6).
SUBSCORE_KEYS: tuple[str, ...] = ("fundamental", "trend", "flow")

#: Contract defaults for the composite weights (§4). Never renormalised (§6).
DEFAULT_WEIGHTS: dict[str, float] = {
    "fundamental": 0.40,
    "trend": 0.35,
    "flow": 0.25,
}

# --------------------------------------------------------------------------- #
# Funnel stages (CONTRACT §5, in order)
# --------------------------------------------------------------------------- #

STAGE_UNIVERSE = "universe"

#: Ordered exclusion stages; each stock fails at most one of these.
EXCLUSION_STAGES: tuple[str, ...] = (
    "excluded:bse_unsupported",
    "excluded:st",
    "excluded:delisting",
    "excluded:suspended",
    "excluded:new_listing",
    "excluded:board_disabled",
    "excluded:micro_cap",
    "excluded:illiquid",
    "insufficient_data:liquidity",
    "insufficient_data:fundamentals",
    "excluded:shrinking",
    "insufficient_data:trend",
)

STAGE_SCORED = "scored"
STAGE_FINALISTS = "finalists"
STAGE_AI_DEMOTED = "ai_demoted"
STAGE_WATCHLIST = "watchlist"

#: Full ordered funnel, exclusions followed by the pipeline-owned tail.
FULL_FUNNEL_STAGES: tuple[str, ...] = (
    STAGE_UNIVERSE,
    *EXCLUSION_STAGES,
    STAGE_SCORED,
    STAGE_FINALISTS,
    STAGE_AI_DEMOTED,
    STAGE_WATCHLIST,
)

# --------------------------------------------------------------------------- #
# AI review placeholder (review.py lands in a later task)
# --------------------------------------------------------------------------- #

AI_STATUS_PENDING = "not_reviewed"
AI_NOTE_PENDING = "AI review not implemented yet"


def ai_pending(note: str = AI_NOTE_PENDING) -> dict[str, Any]:
    """Placeholder ``ai`` block for a pick that has not been AI-reviewed yet."""
    return {
        "status": AI_STATUS_PENDING,
        "verdict": None,
        "summary": None,
        "reasons": [],
        "red_flags": [],
        "news_checked": False,
        "note": note,
        "latency_s": 0.0,
        "attempts": 0,
        "outcome": None,
    }


# --------------------------------------------------------------------------- #
# Identifiers (CONTRACT §2)
# --------------------------------------------------------------------------- #

def normalize_code(code: Any) -> str:
    """Return the canonical 6-digit string code (zero-padded)."""
    return str(code).strip().zfill(6)


def exchange_from_code(code: Any) -> str:
    """Derive ``SH`` / ``SZ`` / ``BJ`` from a 6-digit A-share code."""
    code = normalize_code(code)
    head = code[0]
    if head in {"4", "8"} or code.startswith("92"):
        return "BJ"
    if head in {"0", "2", "3"}:
        return "SZ"
    return "SH"


def board_from_code(code: Any) -> str:
    """Derive the board tag (CONTRACT §2 table) from a 6-digit code."""
    code = normalize_code(code)
    prefix3 = code[:3]
    if prefix3 in {"688", "689"}:
        return "star"
    if prefix3 in {"600", "601", "603", "605"}:
        return "main_sh"
    if prefix3 in {"000", "001", "002", "003"}:
        return "main_sz"
    if prefix3 in {"300", "301", "302"}:
        return "chinext"
    if code.startswith(("4", "8", "92")):
        return "bse"
    # Defensive fallback for shapes outside the contract table (B-shares etc.).
    return {"SH": "main_sh", "SZ": "main_sz", "BJ": "bse"}[exchange_from_code(code)]


def symbol_from_code(code: Any) -> str:
    """Return the ``sh600519``-style symbol for a 6-digit code."""
    code = normalize_code(code)
    return f"{exchange_from_code(code).lower()}{code}"


# --------------------------------------------------------------------------- #
# Plain-language number formatting for reasons (CONTRACT §6)
# --------------------------------------------------------------------------- #

def fmt_cny(value: Any) -> str:
    """Format a CNY amount as ``+1.23亿`` / ``-4500万`` (signed)."""
    number = _finite(value)
    if number is None:
        return "n/a"
    sign = "+" if number >= 0 else "-"
    magnitude = abs(number)
    if magnitude >= 1e8:
        return f"{sign}{magnitude / 1e8:.2f}亿"
    if magnitude >= 1e4:
        return f"{sign}{magnitude / 1e4:.0f}万"
    return f"{sign}{magnitude:.0f}元"


def fmt_pct(value: Any, *, signed: bool = False, digits: int = 1) -> str:
    """Format a percent number (18.5 means 18.5%) as ``18.5%`` / ``+18.5%``."""
    number = _finite(value)
    if number is None:
        return "n/a"
    pattern = f"{{:{'+' if signed else ''}.{digits}f}}%"
    return pattern.format(number)


def fmt_ratio(value: Any, digits: int = 2) -> str:
    """Format a plain ratio such as a volume expansion multiplier."""
    number = _finite(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}x"


def make_funnel_entry(
    stage: str,
    count: int,
    removed: int,
    note: str | None = None,
) -> dict[str, Any]:
    """Build one funnel entry (CONTRACT §5)."""
    return {"stage": stage, "count": int(count), "removed": int(removed), "note": note}


def _finite(value: Any) -> float | None:
    """Coerce to a finite float; ``None`` for None/NaN/inf/garbage."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number
