"""AI review of scan finalists (CONTRACT section 8).

One stock per LLM call; the model sees ONLY the data pack we build (metrics,
sub-scores, rule reasons, news N1..Nn, announcements A1..An, red flags
R1..Rn, metric ids ``M:<field>``). Output is a strict JSON object; malformed
output gets exactly one retry with the validation error appended. A stock is
never dropped and this module never raises out of :func:`run_ai_review`.

Verdict effects (applied by the pipeline, see section 7):

* ``avoid``  -> the pick moves to ``ai_demoted`` (kept visible with reasons);
  the next pool candidates backfill up to ``top_n``.
* ``neutral``-> the pick stays, but its *final ordering* is penalised by
  :data:`NEUTRAL_RANK_PENALTY` points (rule score and sub-scores unchanged).
* A material red flag (regulatory / earnings_warning / holder_selldown /
  pledge_high) that the model does not address with a cited reason or red
  flag forces ``avoid`` with a note naming the flag ids.
* A SEVERE red flag (``csrc_investigation`` / ``audit_opinion`` /
  ``penalty_fraud`` / ``delisting_risk`` — ``sources/news.py`` stamps them
  ``severity='severe'``) forces ``avoid`` with a note of the form
  ``severe red flag R# (<type>) not resolved`` unless a reason or red flag
  both cites the flag AND states an explicit resolution (case closed,
  decision revoked, risk lifted, 已结案/已撤销/解除...).

LLM access goes through the app's own provider path
(``src.providers.chat.ChatLLM``), lazily imported; when no provider is
configured every pick stays ``not_reviewed`` with a note saying why. Tests
inject a plain ``callable(messages) -> str`` through ``run_scan(llm=...)``.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time as _time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable

from .config import ScanConfig
from .models import AI_NOTE_PENDING, AI_STATUS_PENDING, ai_pending

logger = logging.getLogger(__name__)

#: Per-call wall-clock budget for one LLM review (seconds).
AI_CALL_TIMEOUT_S = 90.0

#: Maximum concurrent LLM calls (flash-tier gateways dislike bursts).
MAX_AI_CONCURRENCY = 3

#: Extra seconds slept past a parsed quota reset before retrying once.
QUOTA_RETRY_GRACE_S = 5.0

#: Ordering penalty applied to ``neutral`` verdicts in the final pick order.
#: Documented contract: score/sub-scores stay untouched; only ranking shifts.
NEUTRAL_RANK_PENALTY = 5.0

VERDICTS: tuple[str, ...] = ("watch", "neutral", "avoid")

#: Failure kinds carried on a ``failed`` ai block (``ai["failure_kind"]``).
FAILURE_KINDS: tuple[str, ...] = ("quota", "invalid_json", "timeout", "error")

#: Red-flag types that force a demotion unless the model addresses them.
#: ``delisting_risk`` also belongs to :data:`SEVERE_RED_FLAG_TYPES`; the
#: severe rule is stricter and wins (it is skipped by the material note).
MATERIAL_RED_FLAG_TYPES: frozenset[str] = frozenset(
    {
        "regulatory",
        "earnings_warning",
        "holder_selldown",
        "pledge_high",
        "delisting_risk",
    }
)

#: SEVERE red-flag class (CONTRACT section 8, TODO_R2 item E): CSRC
#: investigation, qualified/adverse/disclaimer audit opinion, penalty or
#: fraud decisions, delisting risk. These require verdict ``avoid`` unless
#: the pack clearly shows the issue resolved AND the model cites the flag;
#: :func:`_finalise_ai_block` enforces it. ``sources/news.py`` derives its
#: ``severity='severe'`` label from this same set.
SEVERE_RED_FLAG_TYPES: frozenset[str] = frozenset(
    {
        "csrc_investigation",
        "audit_opinion",
        "penalty_fraud",
        "delisting_risk",
    }
)

#: Precise wording for the flow sub-score and flow metrics in the prompt and
#: in every pack: the numbers are THS (同花顺) net fund flow — not Eastmoney
#: main-force, not northbound/institutional money (the 002832 summary
#: mislabelled it as main-force inflow).
FLOW_LABEL: str = (
    "THS (同花顺) net fund flow - not Eastmoney main-force, "
    "not northbound/institutional"
)

_VALIDATORS: tuple[str, ...] = VERDICTS

#: Provider rate-limit detection. ``_RATE_LIMIT_RE`` matches messages like
#: ``预计 2026-10-13 00:55:59 重置后恢复`` (reset time is UTC).
_RESET_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"
    r"|(?:reset|重置)[^\d]{0,12}(\d{2}:\d{2}(?::\d{2})?)",
    re.IGNORECASE,
)
_RATE_LIMIT_RE = re.compile(
    r"rate.?limit|too many requests|429|quota|限流|频率|配额", re.IGNORECASE
)
_UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Data pack
# --------------------------------------------------------------------------- #


def build_data_pack(
    code: str,
    *,
    name: str | None = None,
    board: str | None = None,
    industry: str | None = None,
    score: float | None = None,
    sub_scores: dict[str, float] | None = None,
    metrics: dict[str, Any] | None = None,
    rule_reasons: Iterable[str] = (),
    news: list[dict] | None = None,
    announcements: list[dict] | None = None,
    red_flags: list[dict] | None = None,
) -> dict[str, Any]:
    """Build one stock's citable data pack (CONTRACT section 8).

    Item ids are assigned here and are unique within the pack: ``N1..Nn``
    for news, ``A1..An`` for announcements, ``R1..Rn`` for red flags. Metric
    ids are ``M:<field>`` over the non-None metrics keys. ``flow_label`` is
    fixed wording (:data:`FLOW_LABEL`) so the model cannot misname the flow
    input as Eastmoney main-force or northbound money.
    """
    news_items = [
        {**item, "id": f"N{i + 1}"} for i, item in enumerate(news or [])
    ]
    ann_items = [
        {**item, "id": f"A{i + 1}"} for i, item in enumerate(announcements or [])
    ]
    red_items = [
        {**item, "id": f"R{i + 1}"} for i, item in enumerate(red_flags or [])
    ]
    clean_metrics = {
        key: value
        for key, value in (metrics or {}).items()
        if value is not None and not _is_nan(value)
    }
    return {
        "code": code,
        "name": name,
        "board": board,
        "industry": industry,
        "score": score,
        "sub_scores": dict(sub_scores or {}),
        # Label the flow input so the model cannot misname it (TODO_R2 E).
        "flow_label": FLOW_LABEL,
        "rule_reasons": [str(r) for r in rule_reasons],
        "news": news_items,
        "announcements": ann_items,
        "red_flags": red_items,
        "metric_ids": [f"M:{key}" for key in sorted(clean_metrics)],
        "metrics": clean_metrics,
    }


def pack_cite_ids(pack: dict[str, Any]) -> set[str]:
    """Every citable id inside ``pack`` (N/A/R items plus M: metrics)."""
    ids: set[str] = set()
    for key in ("news", "announcements", "red_flags"):
        for item in pack.get(key) or []:
            item_id = item.get("id") if isinstance(item, dict) else None
            if item_id:
                ids.add(str(item_id))
    ids.update(str(m) for m in pack.get("metric_ids") or [])
    return ids


def pack_material_red_flags(pack: dict[str, Any]) -> list[dict]:
    """Red-flag items whose type demands a demotion unless addressed."""
    return [
        item
        for item in pack.get("red_flags") or []
        if isinstance(item, dict) and str(item.get("type")) in MATERIAL_RED_FLAG_TYPES
    ]


def _is_severe_flag(item: dict[str, Any]) -> bool:
    """True for the SEVERE class: explicit ``severity='severe'`` or type."""
    return (
        str(item.get("severity")) == "severe"
        or str(item.get("type")) in SEVERE_RED_FLAG_TYPES
    )


def pack_severe_red_flags(pack: dict[str, Any]) -> list[dict]:
    """Red-flag items that force ``avoid`` unless cited with a resolution."""
    return [
        item
        for item in pack.get("red_flags") or []
        if isinstance(item, dict) and _is_severe_flag(item)
    ]


def pack_has_checked_news(pack: dict[str, Any]) -> bool:
    """True when at least one news item or announcement was supplied."""
    return bool(pack.get("news")) or bool(pack.get("announcements"))


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


def build_messages(pack: dict[str, Any], *, language: str = "en") -> list[dict[str, str]]:
    """OpenAI-format messages for one stock review."""
    lang = "Chinese (Simplified)" if str(language).lower().startswith("zh") else "English"
    system = (
        "You are a research assistant maintaining a medium-term A-share "
        "watchlist (multi-week holding horizon). Review the ONE stock whose "
        "data pack is supplied in the user message.\n"
        "Rules:\n"
        "1. Use ONLY the supplied pack. Never invent facts, numbers or ids.\n"
        f"2. Write summary, reasons and red_flags text in {lang}.\n"
        "3. Every reason and every red flag MUST cite pack ids in `cites` "
        "(news N#, announcements A#, red flags R#, metrics M:<field>). "
        "Cites must exist in the pack.\n"
        "4. Verdict is one of: watch (worthy of the watchlist), neutral "
        "(mixed; keep but deprioritise), avoid (demote from the watchlist).\n"
        f'5. Fund flow wording: the flow sub-score and the flow metrics are '
        f'"{FLOW_LABEL}". Never describe them as Eastmoney main-force '
        "(主力资金) or as northbound/institutional money.\n"
        "6. Demote (avoid) on material red flags — regulatory probe/inquiry/"
        "penalty, earnings warning, major-holder sell-down, heavy pledging — "
        "unless the pack contains a cited counter-argument you also cite in "
        "a reason.\n"
        "7. SEVERE red flags require verdict avoid unless the pack clearly "
        "shows the issue resolved. SEVERE types: csrc_investigation (立案调查"
        "/立案告知书/立案通知/证监会…立案), audit_opinion (保留意见/否定意见/"
        "无法表示意见), penalty_fraud (行政处罚决定书/市场禁入/欺诈发行/"
        "财务造假), delisting_risk (退市风险/终止上市). To return watch or "
        "neutral for such a flag you MUST cite that flag's id in a reason or "
        "red_flag and state the explicit resolution (case closed / decision "
        "revoked / risk lifted — 已结案、已撤销、解除、整改完成...). An "
        "unresolved severe red flag is forced to avoid by the review code.\n"
        "8. Output ONLY the JSON object, no prose, no code fences:\n"
        '{"code": "...", "verdict": "watch|neutral|avoid", "confidence": 0.0,'
        ' "summary": "...", "reasons": [{"text": "...", "cites": ["M:roe"]}'
        '], "red_flags": [{"type": "...", "text": "...", "cites": ["R1"]}]}\n'
        "9. confidence is a number from 0 to 1."
    )
    user = (
        f"Review stock {pack.get('code')} ({pack.get('name')}). "
        "Data pack (JSON):\n"
        f"{json.dumps(pack, ensure_ascii=False, default=str)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Output parsing / validation
# --------------------------------------------------------------------------- #


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object found in ``text``; raise ValueError.

    Flash models like to wrap JSON in ```` ```json ```` fences or prepend a
    sentence — both are tolerated.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty LLM output")
    candidate = _strip_code_fence(text)
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
        raise ValueError("output JSON is not an object")
    except (json.JSONDecodeError, ValueError):
        pass
    obj = _first_balanced_object(text)
    if obj is None:
        raise ValueError("no JSON object found in LLM output")
    try:
        data = json.loads(obj)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("output JSON is not an object")
    return data


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    fence = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)
    match = fence.match(stripped)
    return match.group(1).strip() if match else stripped


def _first_balanced_object(text: str) -> str | None:
    """Scan for the first brace-balanced ``{...}`` outside strings."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def validate_review(
    pack: dict[str, Any], data: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Validate + sanitise one parsed reply against its pack.

    Returns ``(ai_block_skeleton, notes)`` where the skeleton carries the
    contract fields (verdict/summary/reasons/red_flags) but not status or
    news_checked (the caller fills those). Raises ``ValueError`` with a short
    message on schema errors — the caller retries once with that message.
    """
    verdict = data.get("verdict")
    if verdict not in _VALIDATORS:
        raise ValueError(
            f"verdict must be one of {_VALIDATORS}, got {verdict!r}"
        )
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence must be a number in [0, 1], got {confidence!r}")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence!r}")

    valid_ids = pack_cite_ids(pack)
    reasons: list[dict[str, Any]] = []
    for entry in data.get("reasons") or []:
        cleaned = _clean_cited_entry(entry, valid_ids)
        if cleaned is not None:
            reasons.append(cleaned)
    red_flags: list[dict[str, Any]] = []
    for entry in data.get("red_flags") or []:
        cleaned = _clean_cited_entry(entry, valid_ids)
        if cleaned is not None:
            red_flags.append(cleaned)
    summary = data.get("summary")
    summary = str(summary).strip() if summary is not None else None
    return {
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "summary": summary or None,
        "reasons": reasons,
        "red_flags": red_flags,
    }, []


def _clean_cited_entry(entry: Any, valid_ids: set[str]) -> dict[str, Any] | None:
    """Normalise one reason/red-flag entry; None when unusable or uncitable."""
    if not isinstance(entry, dict):
        return None
    text = entry.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    raw_cites = entry.get("cites") or []
    if isinstance(raw_cites, str):
        raw_cites = [raw_cites]
    cites = [str(c) for c in raw_cites]
    if cites and any(c not in valid_ids for c in cites):
        return None  # unknown cites: drop the whole entry (contract section 8)
    cleaned: dict[str, Any] = {"text": text.strip(), "cites": cites}
    if "type" in entry:
        cleaned["type"] = str(entry.get("type"))
    return cleaned


# --------------------------------------------------------------------------- #
# Provider rate-limit (429 / quota) detection
# --------------------------------------------------------------------------- #


class QuotaError(Exception):
    """Provider rate-limit with a parsed UTC reset time (or None if unknown)."""

    def __init__(self, message: str, reset_at: datetime | None = None) -> None:
        super().__init__(message)
        self.reset_at = reset_at


def _exc_text(exc: BaseException) -> str:
    """Flatten an exception (and its ``__cause__``/``__context__`` chain)."""
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " | ".join(parts)


def detect_quota_error(exc: BaseException) -> QuotaError | None:
    """Detect a provider rate-limit (HTTP 429 / RateLimitError / quota text).

    The reset time in a message like ``预计 2026-10-13 00:55:59 重置后恢复``
    is UTC; it is normalised to an aware UTC datetime by the caller.
    Returns ``None`` when ``exc`` is not a rate-limit.
    """
    text = _exc_text(exc)
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    retry_after = getattr(exc, "retry_after_s", None)
    is_limit = (
        isinstance(exc, TimeoutError) is False
        and str(getattr(exc, "status_code", "") or "") == "429"
    ) or bool(_RATE_LIMIT_RE.search(text))
    if not is_limit and not (retry_after is not None or status == 429):
        return None
    reset_at: datetime | None = None
    if retry_after is not None:
        try:
            reset_at = _time_to_utc(_time.time() + float(retry_after))
        except (TypeError, ValueError):
            reset_at = None
    if reset_at is None:
        reset_at = _parse_reset_utc(text)
    return QuotaError(f"rate limited (status={status}): {text[:300]}", reset_at)


def _parse_reset_utc(text: str) -> datetime | None:
    """Pull a reset timestamp out of a provider message (UTC)."""
    for match in _RESET_RE.finditer(text):
        stamp = match.group(1) or match.group(2)
        if not stamp:
            continue
        parsed = _coerce_utc(stamp)
        if parsed is not None:
            return parsed
    return None


def _coerce_utc(stamp: str) -> datetime | None:
    """Parse ``YYYY-MM-DD HH:MM:SS`` (or ``HH:MM``/``HH:MM:SS``) as UTC."""
    stamp = stamp.strip().replace("T", " ")
    try:
        if len(stamp) <= 8 and ":" in stamp:  # bare HH:MM[:SS] -> today's UTC date
            parts = [int(p) for p in stamp.split(":")]
            hour = parts[0]
            minute = parts[1] if len(parts) > 1 else 0
            second = parts[2] if len(parts) > 2 else 0
            now = _time_to_utc(_time.time())
            return now.replace(hour=hour, minute=minute, second=second, microsecond=0)
        return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_UTC)
    except ValueError:
        return None


def _time_to_utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=_UTC)


def quota_note(reset_at: datetime | None) -> str:
    """Short human note for a quota failure (``review failed: quota (resets HH:MM)``)."""
    if reset_at is None:
        return "review failed: quota"
    return f"review failed: quota (resets {reset_at.astimezone().strftime('%H:%M')})"


def _failed_ai(kind: str, note: str) -> dict[str, Any]:
    block = ai_pending()
    block["status"] = "failed"
    block["failure_kind"] = kind
    block["note"] = note[:200]
    return block


def _finish_outcome(
    outcome: ReviewOutcome,
    *,
    started: float,
    attempts: int,
    kind: str,
) -> ReviewOutcome:
    """Attach stable call telemetry to one review result."""
    outcome.ai["latency_s"] = round(max(0.0, _time.monotonic() - started), 3)
    outcome.ai["attempts"] = int(attempts)
    outcome.ai["outcome"] = kind
    return outcome


# --------------------------------------------------------------------------- #
# One-stock review
# --------------------------------------------------------------------------- #


@dataclass
class ReviewOutcome:
    """Result of reviewing one stock."""

    code: str
    ai: dict[str, Any]
    forced: bool = False
    error: str | None = None
    quota: QuotaError | None = None


def review_one(
    llm: Callable[[list[dict[str, str]]], str],
    pack: dict[str, Any],
    *,
    language: str = "en",
    timeout: float | None = None,
    config: ScanConfig | None = None,
    sleep: Callable[[float], None] | None = None,
    deadline: float | None = None,
) -> ReviewOutcome:
    """Review one pack; never raises. One retry on invalid output.

    A provider rate-limit (``QuotaError``) is retried ONCE after sleeping to
    the reset (+grace) when the reset is within ``config.ai_quota_wait_max_s``
    and the run still has AI time budget left; otherwise the stock fails with
    ``failure_kind='quota'`` and the outcome carries the ``quota`` marker so
    the batch can stop hammering the provider.
    """
    code = str(pack.get("code"))
    if timeout is None:
        timeout = (
            float(config.ai_call_timeout_s)
            if config is not None
            else AI_CALL_TIMEOUT_S
        )
    messages = build_messages(pack, language=language)
    started = _time.monotonic()
    sleep_fn = sleep or _time.sleep
    last_error: str | None = None
    quota: QuotaError | None = None
    quota_retried = False
    attempts = 0
    for attempt in range(3):
        attempts += 1
        try:
            raw = _call_llm(llm, messages, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — a stock failure never crashes a scan
            qerr = detect_quota_error(exc)
            if qerr is not None and not quota_retried and _quota_wait_allowed(
                qerr, config, deadline
            ):
                quota_retried = True
                quota = qerr
                wait = _quota_wait_s(qerr)
                logger.warning(
                    "AI review rate-limited for %s; sleeping %.1fs until reset",
                    code,
                    wait,
                )
                sleep_fn(wait)
                continue
            if qerr is not None:
                note = quota_note(qerr.reset_at)
                return _finish_outcome(ReviewOutcome(
                    code=code,
                    ai=_failed_ai("quota", note),
                    error=str(exc)[:200],
                    quota=qerr,
                ), started=started, attempts=attempts, kind="quota")
            last_error = _exc_text(exc)
            logger.warning("AI review call failed for %s: %s", code, last_error)
            kind = "timeout" if isinstance(exc, TimeoutError) else "error"
            return _finish_outcome(ReviewOutcome(
                code=code,
                ai=_failed_ai(kind, f"AI review failed: {last_error}"),
                error=last_error,
            ), started=started, attempts=attempts, kind=kind)
        try:
            data = extract_json_object(raw)
            skeleton, _notes = validate_review(pack, data)
        except ValueError as exc:
            last_error = str(exc)
            if attempt == 0:
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw or ""},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply was rejected: {exc}. "
                            "Reply again with ONLY the corrected JSON object."
                        ),
                    },
                ]
                continue
            return _finish_outcome(ReviewOutcome(
                code=code,
                ai=_failed_ai("invalid_json", f"AI review failed: {last_error}"),
                error=last_error,
            ), started=started, attempts=attempts, kind="invalid_json")
        ai, forced, note = _finalise_ai_block(pack, skeleton)
        if note:
            logger.debug("AI review for %s applied note: %s", code, note)
        return _finish_outcome(
            ReviewOutcome(code=code, ai=ai, forced=forced),
            started=started,
            attempts=attempts,
            kind="ok",
        )
    failed = _failed_ai("error", f"AI review failed: {(last_error or 'unknown error')}")
    return _finish_outcome(
        ReviewOutcome(code=code, ai=failed, error=last_error, quota=quota),
        started=started,
        attempts=attempts,
        kind="error",
    )


def _quota_wait_s(quota: QuotaError) -> float:
    """Seconds to sleep until the reset (+grace)."""
    if quota.reset_at is None:
        return 0.0
    wait = (quota.reset_at - _time_to_utc(_time.time())).total_seconds()
    return max(0.0, wait) + QUOTA_RETRY_GRACE_S


def _quota_wait_allowed(
    quota: QuotaError, config: ScanConfig | None, deadline: float | None
) -> bool:
    """True when sleeping to the reset fits the wait cap and the time budget."""
    if quota.reset_at is None:
        return False
    wait = _quota_wait_s(quota)
    wait_max = float(getattr(config, "ai_quota_wait_max_s", 300.0) or 0.0)
    if wait > wait_max:
        return False
    if deadline is not None and _time.time() + wait > deadline:
        return False
    return True


def _call_llm(
    llm: Callable[[list[dict[str, str]]], str],
    messages: list[dict[str, str]],
    *,
    timeout: float,
) -> str:
    """Invoke the (possibly slow) completion callable under a timeout."""
    if timeout <= 0:
        return str(llm(messages) or "")
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(llm, messages)
    try:
        result = str(future.result(timeout=timeout) or "")
    except FutureTimeoutError as exc:
        future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"LLM call timed out after {timeout:.0f}s") from exc
    else:
        pool.shutdown(wait=True)
        return result


def resolve_evidence(pack: dict[str, Any], cited_ids: set[str]) -> list[dict[str, Any]]:
    """Resolve cite ids to human-readable evidence rows (audit gap fix).

    Args:
        pack: One stock's data pack (see :func:`build_data_pack`).
        cited_ids: Ids cited in reasons/red_flags (``N3``, ``A1``, ``R2``,
            ``M:roe``...).

    Returns:
        A list ordered by id, one entry per cited id that exists in the pack:
        ``{id, kind (news|announcement|red_flag|metric), date, title_or_text,
        source, url}`` for pack items, and ``{id, kind: 'metric', name,
        value}`` for metrics.
    """
    evidence: list[dict[str, Any]] = []
    kind_by_key = {"news": "news", "announcements": "announcement",
                   "red_flags": "red_flag"}
    for key, kind in kind_by_key.items():
        for item in pack.get(key) or []:
            item_id = str(item.get("id") or "")
            if item_id not in cited_ids:
                continue
            evidence.append({
                "id": item_id,
                "kind": kind,
                "date": item.get("date"),
                "title_or_text": item.get("title") or item.get("text"),
                "source": item.get("source"),
                "url": item.get("url"),
            })
    metrics = pack.get("metrics") or {}
    for metric_id in pack.get("metric_ids") or []:
        if str(metric_id) not in cited_ids:
            continue
        name = str(metric_id)[2:]  # strip the 'M:' prefix
        evidence.append({
            "id": str(metric_id),
            "kind": "metric",
            "name": name,
            "value": metrics.get(name),
        })
    evidence.sort(key=lambda e: e["id"])
    return evidence


#: Explicit-resolution phrases. A SEVERE red flag only survives a
#: watch/neutral verdict when a citing reason/red_flag carries one of these
#: (case closed, decision revoked, risk lifted, 已结案/已撤销/解除...).
_RESOLUTION_RE = re.compile(
    r"已结案|结案|销案|撤销|撤回|终止(?:调查|立案|核查|审查)|"
    r"解除|整改(?:完成|完毕|到位)|已完成整改|影响消除|消除|恢复正常|取消|"
    r"了结|终结|无保留意见|"
    r"resolved|cleared|lifted|withdrawn|dismissed|closed|settled|completed",
    re.IGNORECASE,
)

#: Negation words that sit right in front of a resolution phrase and cancel
#: it (``尚未结案``, ``未解除``, ``case not yet closed``).
_RESOLUTION_NEGATION_RE = re.compile(
    r"未|尚未|仍未|没有|不存在|并非|still\s+(?:open|ongoing)|not\s+(?:yet\s+)?",
    re.IGNORECASE,
)


def _has_explicit_resolution(text: str) -> bool:
    """True when ``text`` states an explicit, un-negated resolution."""
    if not text:
        return False
    for match in _RESOLUTION_RE.finditer(text):
        window = text[max(0, match.start() - 8): match.start()]
        if _RESOLUTION_NEGATION_RE.search(window):
            continue  # e.g. '尚未结案', '风险未解除'
        return True
    return False


def _severe_is_resolved(item: dict[str, Any], skeleton: dict[str, Any]) -> bool:
    """True when a reason/red_flag cites ``item`` AND states a resolution."""
    flag_id = str(item.get("id") or "")
    if not flag_id:
        return False
    entries = list(skeleton.get("reasons") or []) + list(skeleton.get("red_flags") or [])
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cites = {str(c) for c in entry.get("cites") or []}
        if flag_id in cites and _has_explicit_resolution(str(entry.get("text") or "")):
            return True
    return False


def _finalise_ai_block(
    pack: dict[str, Any], skeleton: dict[str, Any]
) -> tuple[dict[str, Any], bool, str | None]:
    """Apply news_checked + the forced-avoid rules; return (ai, forced, note).

    Two forcing rules apply when the model answered ``watch``/``neutral``:

    * SEVERE first — every severe flag must be cited *with an explicit
      resolution*, else ``avoid`` with
      ``severe red flag R# (<type>) not resolved`` per flag.
    * material — every remaining (non-severe) material flag must at least be
      cited, else ``avoid`` with ``red flag R#,... not addressed by model``.
    """
    checked = pack_has_checked_news(pack)
    notes: list[str] = []
    if not checked:
        notes.append("no news checked")

    verdict = skeleton["verdict"]
    forced = False
    if verdict in ("watch", "neutral"):
        addressed = _cited_ids(skeleton)
        severe_items = pack_severe_red_flags(pack)
        severe_ids = {str(item.get("id")) for item in severe_items}
        forced_notes: list[str] = []
        for item in severe_items:
            if _severe_is_resolved(item, skeleton):
                continue
            forced_notes.append(
                f"severe red flag {item.get('id')} ({item.get('type')}) not resolved"
            )
        uncited = [
            str(item.get("id"))
            for item in pack_material_red_flags(pack)
            if str(item.get("id")) not in addressed
            and str(item.get("id")) not in severe_ids
        ]
        if uncited:
            forced_notes.append(
                f"red flag {', '.join(uncited)} not addressed by model"
            )
        if forced_notes:
            verdict = "avoid"
            forced = True
            notes.extend(forced_notes)

    ai = ai_pending()
    ai["status"] = "reviewed"
    ai["verdict"] = verdict
    ai["summary"] = skeleton.get("summary")
    ai["reasons"] = skeleton.get("reasons", [])
    ai["red_flags"] = skeleton.get("red_flags", [])
    ai["news_checked"] = checked
    ai["note"] = "; ".join(notes) if notes else None
    # Evidence rows: every cited id resolved back to what it refers to.
    ai["evidence"] = resolve_evidence(pack, _cited_ids(skeleton))
    return ai, forced, ai["note"]


def _cited_ids(skeleton: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for entry in list(skeleton.get("reasons") or []) + list(skeleton.get("red_flags") or []):
        for cite in entry.get("cites") or []:
            ids.add(str(cite))
    return ids


# --------------------------------------------------------------------------- #
# Batch review
# --------------------------------------------------------------------------- #


@dataclass
class AIReviewSummary:
    """Run-level outcome of one AI review pass."""

    status: str  # reviewed | partial | not_reviewed
    note: str
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    reviewed: int = 0
    failed: int = 0
    demoted: int = 0
    forced: int = 0
    quota_failed: int = 0
    ai_stats: dict[str, Any] = field(default_factory=dict)


def ai_stats_from_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate call telemetry from a set of per-stock AI blocks."""
    latencies = sorted(
        float(ai.get("latency_s") or 0.0)
        for ai in results.values()
        if int(ai.get("attempts") or 0) > 0
    )
    outcome_counts = {kind: 0 for kind in ("ok", "timeout", "quota", "invalid_json", "error")}
    for ai in results.values():
        outcome = ai.get("outcome")
        if outcome in outcome_counts:
            outcome_counts[outcome] += 1

    def percentile(q: float) -> float:
        if not latencies:
            return 0.0
        index = max(0, min(len(latencies) - 1, math.ceil(q * len(latencies)) - 1))
        return round(latencies[index], 3)

    return {
        "count": len(latencies),
        "p50_latency_s": percentile(0.50),
        "p90_latency_s": percentile(0.90),
        "max_latency_s": round(max(latencies, default=0.0), 3),
        "outcomes": outcome_counts,
    }


def summarize_ai_results(
    results: dict[str, dict[str, Any]],
    *,
    note: str | None = None,
    forced: int = 0,
) -> AIReviewSummary:
    """Build a summary for a complete set of per-stock AI blocks."""
    reviewed = sum(1 for ai in results.values() if ai.get("status") == "reviewed")
    failed = sum(1 for ai in results.values() if ai.get("status") == "failed")
    demoted = sum(1 for ai in results.values() if ai.get("verdict") == "avoid")
    quota_failed = sum(
        1 for ai in results.values() if ai.get("failure_kind") == "quota"
    )
    if reviewed and not failed:
        status = "reviewed"
    elif reviewed and failed:
        status = "partial"
    else:
        status = "not_reviewed"
    if note is None:
        quota_part = f", {quota_failed} quota" if quota_failed else ""
        note = (
            f"AI review: {reviewed} reviewed, {failed} failed{quota_part}, "
            f"{demoted} demoted, {forced} forced"
        )
    return AIReviewSummary(
        status=status,
        note=note,
        results=results,
        reviewed=reviewed,
        failed=failed,
        demoted=demoted,
        forced=forced,
        quota_failed=quota_failed,
        ai_stats=ai_stats_from_results(results),
    )


def run_ai_review(
    codes: list[str],
    *,
    packs_by_code: dict[str, dict[str, Any]],
    llm: Callable[[list[dict[str, str]]], str] | None,
    config: ScanConfig,
    timeout: float | None = None,
    sleep: Callable[[float], None] | None = None,
    now: Callable[[], float] | None = None,
) -> AIReviewSummary:
    """Review every pool stock; never raises, never drops a stock.

    A quota (rate-limit) failure with a far reset stops further LLM calls for
    the run: the remaining picks are marked ``failed``/``quota`` without
    hammering the provider, and ``quota_failed`` counts them.

    Args:
        codes: Pool codes in rule-rank order.
        packs_by_code: Pre-built data pack per code.
        llm: Completion callable; ``None`` skips the review entirely.
        config: Scan settings (``ai_language``, model, quota/budget knobs).
        timeout: Per-call budget in seconds.
        sleep: Injectable sleeper (tests pass a fake).
        now: Injectable monotonic/epoch clock for the AI time budget.

    Returns:
        :class:`AIReviewSummary` with per-stock ``ai`` blocks and counts.
    """
    if not codes:
        return AIReviewSummary(
            status="not_reviewed",
            note="no candidates to review",
            ai_stats=ai_stats_from_results({}),
        )
    if llm is None:
        pending = {code: ai_pending() for code in codes}
        return AIReviewSummary(
            status="not_reviewed",
            note=f"AI review skipped: {AI_NOTE_PENDING}",
            results=pending,
            ai_stats=ai_stats_from_results(pending),
        )

    language = config.ai_language
    effective_timeout = (
        float(config.ai_call_timeout_s) if timeout is None else float(timeout)
    )
    results: dict[str, dict[str, Any]] = {}
    forced_count = 0
    # AI time budget covers the whole pass (including any quota sleeps).
    clock = now or _time.time
    started = clock()
    budget = float(getattr(config, "ai_time_budget_s", 900.0) or 0.0)
    deadline = started + budget if budget > 0 else None
    quota_stop: QuotaError | None = None

    def _work(code: str) -> ReviewOutcome:
        return review_one(
            llm,
            packs_by_code[code],
            language=language,
            timeout=effective_timeout,
            config=config,
            sleep=sleep,
            deadline=deadline,
        )

    # Sequential once a quota failure has stopped the run: remaining picks get
    # the same quota failure without any further provider calls.
    def _run_all() -> list[ReviewOutcome]:
        nonlocal quota_stop
        outcomes: list[ReviewOutcome] = []
        for code in codes:
            if quota_stop is not None:
                note = quota_note(quota_stop.reset_at)
                outcomes.append(
                    ReviewOutcome(
                        code=code, ai=_failed_ai("quota", note), quota=quota_stop
                    )
                )
                continue
            outcome = _work(code)
            if outcome.quota is not None:
                quota_stop = outcome.quota
            outcomes.append(outcome)
        return outcomes

    def _work_guarded(code: str) -> ReviewOutcome:
        # Thread-safe quota gate: once one worker sees a far-reset quota
        # failure, later submissions short-circuit without a provider call.
        if quota_stop is not None:
            note = quota_note(quota_stop.reset_at)
            return ReviewOutcome(code=code, ai=_failed_ai("quota", note), quota=quota_stop)
        return _work(code)

    workers = max(1, min(MAX_AI_CONCURRENCY, len(codes)))
    if workers == 1:
        outcomes = _run_all()
    else:
        # Submit at most `workers` codes up front, then refill one slot at a
        # time as results come back. ``pool.map`` would hand every code to a
        # worker before this thread can observe the first quota failure, so the
        # ``quota_stop`` gate in ``_work_guarded`` would never stop the batch.
        outcomes_by_code: dict[str, ReviewOutcome] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            in_flight = {}
            idx = 0
            while idx < len(codes) and len(in_flight) < workers:
                code = codes[idx]
                in_flight[pool.submit(_work_guarded, code)] = code
                idx += 1
            while in_flight:
                done, _pending = wait(
                    in_flight, return_when=FIRST_COMPLETED
                )
                for fut in done:
                    code = in_flight.pop(fut)
                    outcome = fut.result()
                    if outcome.quota is not None and quota_stop is None:
                        quota_stop = outcome.quota
                    outcomes_by_code[code] = outcome
                    # Refill the freed slot; _work_guarded short-circuits it if
                    # the run has already been stopped by a quota failure.
                    if idx < len(codes):
                        nxt = codes[idx]
                        in_flight[pool.submit(_work_guarded, nxt)] = nxt
                        idx += 1
        outcomes = [outcomes_by_code[code] for code in codes]

    for outcome in outcomes:
        results[outcome.code] = outcome.ai
        if outcome.forced:
            forced_count += 1

    return summarize_ai_results(results, forced=forced_count)


# --------------------------------------------------------------------------- #
# Default provider path
# --------------------------------------------------------------------------- #


def build_default_llm_callable(
    timeout: float | None = None,
    *,
    model: str | None = None,
    fallback_model: str | None = None,
) -> Callable[[list[dict[str, str]]], str]:
    """Completion callable backed by the app's own ``ChatLLM`` provider path.

    ``model``/``fallback_model`` are passed into ``ChatLLM(model_name=...)``;
    ``None`` keeps the app default (from ``agent\\.env``). When a fallback is
    configured, a quota/model error on the primary retries once on the
    fallback model.

    Raises whatever ``ChatLLM`` construction raises (typically a missing
    provider configuration); the pipeline converts that into a
    ``not_reviewed`` run status with a note, never a crash.
    """
    effective_timeout = AI_CALL_TIMEOUT_S if timeout is None else float(timeout)
    client = _construct_chat_llm(model)
    if fallback_model:
        fallback_client = _construct_chat_llm(fallback_model)
        return _wrap_chat_llm_with_fallback(
            client, fallback_client, effective_timeout, fallback_model
        )
    return _wrap_chat_llm(client, effective_timeout)


def resolve_llm_callable(
    llm: Any, timeout: float | None = None
) -> Callable[[list[dict[str, str]]], str] | None:
    """Coerce an injected ``llm`` into ``callable(messages) -> str``.

    Accepts ``None`` (skip), a plain callable (used as-is, preserving the
    test-injection path), or a ``ChatLLM`` client (bound to ``.chat``). The
    raw ``ChatLLM`` object is NOT callable — passing it through unadapted is
    the exact failure that turned every review into
    ``TypeError: 'ChatLLM' object is not callable`` (run 246483edc829).
    """
    effective_timeout = AI_CALL_TIMEOUT_S if timeout is None else float(timeout)
    if llm is None:
        return None
    if isinstance(llm, str):
        return None
    try:
        from src.providers.chat import ChatLLM  # lazy: needs the agent runtime env
    except Exception:  # noqa: BLE001 — provider optional; adapt by duck-typing
        ChatLLM = ()  # type: ignore[assignment]
    if ChatLLM and isinstance(llm, ChatLLM):
        return _wrap_chat_llm(llm, effective_timeout)
    if callable(llm):
        return llm
    chat = getattr(llm, "chat", None)
    if callable(chat):
        return _wrap_chat_llm(llm, effective_timeout)
    return None


def _construct_chat_llm(model: str | None = None) -> Any:
    """Build the app's ChatLLM client (lazy import).

    ``model=None`` uses the app default from ``agent\\.env``.
    """
    from src.providers.chat import ChatLLM  # lazy: needs the agent runtime env

    return ChatLLM(model_name=model)


def _wrap_chat_llm(
    client: Any, timeout: float
) -> Callable[[list[dict[str, str]]], str]:
    """Adapt a ChatLLM client to ``callable(messages) -> str``.

    ``.chat`` accepts a ``timeout: int`` kwarg and returns an ``LLMResponse``;
    the review layer only wants the text.
    """
    timeout_s = int(timeout) if timeout else None

    def _complete(messages: list[dict[str, str]]) -> str:
        response = client.chat(messages, timeout=timeout_s)
        if getattr(response, "content_filter_triggered", False):
            raise RuntimeError("provider content filter triggered")
        return response.content or ""

    return _complete


def _wrap_chat_llm_with_fallback(
    primary: Any,
    fallback: Any,
    timeout: float,
    fallback_model: str,
) -> Callable[[list[dict[str, str]]], str]:
    """Adapt primary+fallback clients; one fallback retry on quota/model error."""
    timeout_s = int(timeout) if timeout else None

    def _complete(messages: list[dict[str, str]]) -> str:
        try:
            response = primary.chat(messages, timeout=timeout_s)
            if getattr(response, "content_filter_triggered", False):
                raise RuntimeError("provider content filter triggered")
            return response.content or ""
        except Exception as exc:  # noqa: BLE001 — one fallback retry, then re-raise
            if detect_quota_error(exc) is None and not _is_model_error(exc):
                raise
            logger.warning(
                "primary model failed (%s); retrying once on fallback %s",
                type(exc).__name__,
                fallback_model,
            )
            response = fallback.chat(messages, timeout=timeout_s)
            if getattr(response, "content_filter_triggered", False):
                raise RuntimeError("provider content filter triggered")
            return response.content or ""

    return _complete


def _is_model_error(exc: BaseException) -> bool:
    """True for errors that indicate the configured model is unavailable."""
    text = _exc_text(exc).lower()
    return any(
        marker in text
        for marker in ("model", "not found", "does not exist", "invalid")
    )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _is_nan(value: Any) -> bool:
    try:
        return value != value
    except Exception:  # noqa: BLE001 — non-numeric sentinels are fine
        return False


def aggregate_source_status(
    name: str, statuses: list[Any]
) -> Any:
    """Collapse per-stock SourceStatus rows into one run-level status.

    Per-stock failures are isolated (contract section 8): the aggregate is
    ``degraded`` when some stocks failed, ``failed`` when all failed,
    ``ok``/``skipped`` otherwise, with a detail line counting failures.
    """
    from .pipeline import SourceStatus  # local import avoids a cycle at load

    if not statuses:
        return SourceStatus(name=name, status="skipped", detail="no stocks requested")
    kinds = {getattr(s, "status", "ok") for s in statuses}
    rows = sum(int(getattr(s, "rows", 0) or 0) for s in statuses)
    failed = sum(1 for s in statuses if getattr(s, "status", "ok") == "failed")
    degraded = sum(1 for s in statuses if getattr(s, "status", "ok") == "degraded")
    total = len(statuses)
    if failed == total:
        kind = "failed"
    elif failed or degraded:
        kind = "degraded"
    elif kinds == {"skipped"}:
        kind = "skipped"
    else:
        kind = "ok"
    detail = f"{failed + degraded}/{total} stocks failed" if (failed or degraded) else None
    latency = max((float(getattr(s, "latency_s", 0.0) or 0.0) for s in statuses), default=0.0)
    # as_of: the newest item date across the per-stock statuses (data date,
    # not fetch time) — None when no per-stock status carried one.
    as_ofs = [getattr(s, "as_of", None) for s in statuses if getattr(s, "as_of", None)]
    as_of = max(as_ofs) if as_ofs else None
    return SourceStatus(
        name=name, status=kind, rows=rows, latency_s=latency,
        detail=detail, as_of=as_of,
    )


__all__ = [
    "AI_CALL_TIMEOUT_S",
    "AIReviewSummary",
    "FAILURE_KINDS",
    "FLOW_LABEL",
    "MATERIAL_RED_FLAG_TYPES",
    "MAX_AI_CONCURRENCY",
    "NEUTRAL_RANK_PENALTY",
    "QUOTA_RETRY_GRACE_S",
    "QuotaError",
    "ReviewOutcome",
    "SEVERE_RED_FLAG_TYPES",
    "VERDICTS",
    "aggregate_source_status",
    "build_data_pack",
    "build_default_llm_callable",
    "build_messages",
    "detect_quota_error",
    "extract_json_object",
    "pack_cite_ids",
    "pack_has_checked_news",
    "pack_material_red_flags",
    "pack_severe_red_flags",
    "quota_note",
    "resolve_llm_callable",
    "review_one",
    "run_ai_review",
    "validate_review",
]
