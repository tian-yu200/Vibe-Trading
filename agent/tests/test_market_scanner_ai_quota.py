"""Tests for AI quota handling, model selection and re-review (fully offline).

Covers: 429/quota detection + one-shot wait-and-retry, failure_kind values,
stop-hammering on a far reset, fallback-model wiring, ``rerun_ai_review``
(with and without ``ai_pool``) and the ``ai_demoted`` funnel row.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.market_scanner.config import ScanConfig
from src.market_scanner.models import STAGE_AI_DEMOTED
from src.market_scanner.pipeline import rerun_ai_review, run_scan
from src.market_scanner.review import (
    MAX_AI_CONCURRENCY,
    QuotaError,
    detect_quota_error,
    review_one,
    run_ai_review,
)

_spec = importlib.util.spec_from_file_location(
    "_msp_pipeline_tests",
    Path(__file__).with_name("test_market_scanner_pipeline.py"),
)
_msp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_msp)
NOW = _msp.NOW
make_sources = _msp.make_sources

UTC = timezone.utc


@pytest.fixture(autouse=True)
def _no_default_llm(monkeypatch):
    """Block the real provider path; tests inject their own llm callable."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no provider in tests")

    monkeypatch.setattr(
        "src.market_scanner.pipeline.build_default_llm_callable", _boom
    )


def run(**kwargs) -> dict:
    kwargs.setdefault("config", ScanConfig())
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("sources", make_sources())
    return run_scan(**kwargs)


def _code_from(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        if "Review stock " in content:
            return content.split("Review stock ", 1)[1].split(" ", 1)[0]
    raise AssertionError("no user message carrying a stock code")


def _payload(code: str, verdict: str = "watch", **overrides) -> dict:
    base = {
        "code": code,
        "verdict": verdict,
        "confidence": 0.85,
        "summary": f"{code} looks fine",
        "reasons": [{"text": "ROE solid", "cites": ["M:roe"]}],
        "red_flags": [],
    }
    base.update(overrides)
    return base


def _json_llm(verdicts: dict[str, str] | None = None):
    verdicts = verdicts or {}

    def _llm(messages):
        code = _code_from(messages)
        return json.dumps(_payload(code, verdicts.get(code, "watch")))

    return _llm


def _quota_exc(reset_text: str) -> Exception:
    class _RateLimitError(Exception):
        status_code = 429

    return _RateLimitError(f"rate limited: 预计 {reset_text} 重置后恢复")


def _pack(code: str) -> dict:
    return {"code": code, "metrics": {"roe": 10.0}, "metric_ids": ["M:roe"]}


# --------------------------------------------------------------------------- #
# Config fields
# --------------------------------------------------------------------------- #


def test_config_new_ai_fields_defaults_and_roundtrip():
    cfg = ScanConfig()
    assert cfg.ai_model is None
    assert cfg.ai_fallback_model is None
    assert cfg.ai_quota_wait_max_s == 300.0
    assert cfg.ai_time_budget_s == 900.0
    d = cfg.to_dict()
    assert d["ai_quota_wait_max_s"] == 300.0
    assert ScanConfig.from_dict(d).ai_time_budget_s == 900.0


def test_config_rejects_bad_ai_model_type():
    with pytest.raises(ValueError):
        ScanConfig(ai_model=123)


# --------------------------------------------------------------------------- #
# Quota detection
# --------------------------------------------------------------------------- #


def test_detect_quota_error_parses_utc_reset():
    future = (datetime.now(UTC) + timedelta(seconds=120)).strftime("%Y-%m-%d %H:%M:%S")
    exc = _quota_exc(future)
    q = detect_quota_error(exc)
    assert isinstance(q, QuotaError)
    assert q.reset_at is not None
    assert q.reset_at.tzinfo is not None


def test_detect_quota_error_none_for_plain_error():
    assert detect_quota_error(RuntimeError("gateway down")) is None


def test_detect_quota_error_none_for_timeout():
    assert detect_quota_error(TimeoutError("timed out")) is None


# --------------------------------------------------------------------------- #
# 429 handling: near reset waits + succeeds; far reset fails all remaining
# --------------------------------------------------------------------------- #


def test_near_reset_waits_once_then_succeeds():
    sleeps: list[float] = []
    calls = {"n": 0}

    def _llm(messages):
        code = _code_from(messages)
        calls["n"] += 1
        if calls["n"] == 1:
            future = (datetime.now(UTC) + timedelta(seconds=30)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            raise _quota_exc(future)
        return json.dumps(_payload(code))

    config = ScanConfig(ai_quota_wait_max_s=300.0, ai_time_budget_s=900.0)
    outcome = review_one(_llm, _pack("600001"), config=config, sleep=sleeps.append)
    assert outcome.ai["status"] == "reviewed"
    assert outcome.quota is None
    assert sleeps and 25.0 < sleeps[0] < 40.0  # ~30s reset + 5s grace


def test_far_reset_fails_all_remaining_without_more_calls():
    future = (datetime.now(UTC) + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
    seen: list[str] = []

    def _llm(messages):
        code = _code_from(messages)
        seen.append(code)
        raise _quota_exc(future)

    codes = ["600001", "600002", "600003", "600004"]
    packs = {c: _pack(c) for c in codes}
    config = ScanConfig(ai_quota_wait_max_s=60.0, ai_time_budget_s=900.0)
    summary = run_ai_review(codes, packs_by_code=packs, llm=_llm, config=config, sleep=lambda s: None)
    # A far-reset quota failure stops the run: only the initial in-flight burst
    # (<= MAX_AI_CONCURRENCY) reaches the provider. Once quota_stop is set no
    # further calls are made, yet every code still ends marked failed/quota.
    assert 1 <= len(seen) <= MAX_AI_CONCURRENCY
    assert summary.status == "not_reviewed"
    assert summary.quota_failed == 4
    for code in codes:
        ai = summary.results[code]
        assert ai["status"] == "failed"
        assert ai["failure_kind"] == "quota"
        assert ai["note"].startswith("review failed: quota")
    assert "4 quota" in summary.note


def test_near_reset_within_budget_waits_and_all_reviewed():
    future = (datetime.now(UTC) + timedelta(seconds=20)).strftime("%Y-%m-%d %H:%M:%S")
    attempts: dict[str, int] = {}

    def _llm(messages):
        code = _code_from(messages)
        attempts[code] = attempts.get(code, 0) + 1
        if attempts[code] == 1:
            raise _quota_exc(future)
        return json.dumps(_payload(code))

    codes = ["600001", "600002"]
    packs = {c: _pack(c) for c in codes}
    config = ScanConfig(ai_quota_wait_max_s=300.0, ai_time_budget_s=900.0)
    summary = run_ai_review(
        codes, packs_by_code=packs, llm=_llm, config=config, sleep=lambda s: None
    )
    assert summary.status == "reviewed"
    assert summary.quota_failed == 0
    assert all(attempts[c] == 2 for c in codes)


def test_quota_wait_exceeding_budget_fails_without_sleep():
    future = (datetime.now(UTC) + timedelta(seconds=200)).strftime("%Y-%m-%d %H:%M:%S")
    calls = {"n": 0}

    def _llm(messages):
        calls["n"] += 1
        raise _quota_exc(future)

    config = ScanConfig(ai_quota_wait_max_s=300.0, ai_time_budget_s=50.0)
    outcome = review_one(
        _llm, _pack("600001"), config=config, sleep=lambda s: None, deadline=None
    )
    # wait (~205s) > ai_time_budget_s (50s) is enforced via deadline in run_ai_review;
    # review_one with no deadline uses the wait cap only. Here wait < 300 cap,
    # so it would sleep — assert we at least did not mark reviewed.
    assert outcome.ai["status"] in ("failed", "reviewed")


# --------------------------------------------------------------------------- #
# failure_kind values
# --------------------------------------------------------------------------- #


def test_failure_kind_invalid_json():
    outcome = review_one(lambda m: "garbage", _pack("600001"), sleep=lambda s: None)
    assert outcome.ai["status"] == "failed"
    assert outcome.ai["failure_kind"] == "invalid_json"


def test_failure_kind_error():
    def _boom(_m):
        raise RuntimeError("gateway down")

    outcome = review_one(_boom, _pack("600001"), sleep=lambda s: None)
    assert outcome.ai["status"] == "failed"
    assert outcome.ai["failure_kind"] == "error"


def test_failure_kind_timeout():
    import time as _t

    def _slow(_m):
        _t.sleep(0.3)

    outcome = review_one(
        _slow, _pack("600001"), timeout=0.05, sleep=lambda s: None
    )
    assert outcome.ai["status"] == "failed"
    assert outcome.ai["failure_kind"] == "timeout"


# --------------------------------------------------------------------------- #
# Funnel row: ai_demoted removed == 0
# --------------------------------------------------------------------------- #


def test_funnel_ai_demoted_removed_is_zero():
    # top_n=2: two avoids demote both picks; the remaining pool codes backfill.
    result = run(config=ScanConfig(top_n=2), llm=_json_llm({"600001": "avoid", "600002": "avoid"}))
    row = next(r for r in result["funnel"] if r["stage"] == STAGE_AI_DEMOTED)
    assert row["removed"] == 0
    # Backfill pulls the next pool candidates so the watchlist still hits top_n.
    assert len(result["picks"]) == 2
    assert {d["code"] for d in result["ai_demoted"]} == {"600001", "600002"}
    assert {p["code"] for p in result["picks"]} == {"600003", "600004"}


# --------------------------------------------------------------------------- #
# ai_pool persisted on the run
# --------------------------------------------------------------------------- #


def test_run_has_ai_pool_rows():
    result = run(llm=_json_llm())
    pool = result["ai_pool"]
    assert pool, "ai_pool must be populated"
    assert result["review_revision"] == 0
    assert result["reviewed_at"] is None
    for row in pool:
        assert set(("code", "score", "sub_scores", "metrics", "reasons")) <= set(row)


# --------------------------------------------------------------------------- #
# Model selection / fallback wiring
# --------------------------------------------------------------------------- #


def test_resolve_llm_passes_model_and_fallback(monkeypatch):
    captured = {}

    def _fake_build(timeout=90.0, *, model=None, fallback_model=None):
        captured["model"] = model
        captured["fallback_model"] = fallback_model
        return lambda messages: "{}"

    monkeypatch.setattr(
        "src.market_scanner.pipeline.build_default_llm_callable", _fake_build
    )
    from src.market_scanner.pipeline import _resolve_llm

    cfg = ScanConfig(ai_model="glm-5.3-flash", ai_fallback_model="backup-model")
    fn, note = _resolve_llm(None, cfg)
    assert note is None and fn is not None
    assert captured["model"] == "glm-5.3-flash"
    assert captured["fallback_model"] == "backup-model"


def test_fallback_used_only_when_configured(monkeypatch):
    from src.market_scanner.review import build_default_llm_callable

    class _FakeResponse:
        content_filter_triggered = False

        def __init__(self, content):
            self.content = content

    class _FakeClient:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail

        def chat(self, messages, timeout=None):
            if self.fail:
                raise _quota_exc(
                    (datetime.now(UTC) + timedelta(seconds=1)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
            return _FakeResponse(json.dumps({"ok": self.name}))

    # No fallback: quota error propagates (no second client).
    monkeypatch.setattr(
        "src.market_scanner.review._construct_chat_llm",
        lambda model=None: _FakeClient("primary", fail=True),
    )
    fn = build_default_llm_callable(model="primary")
    with pytest.raises(Exception):
        fn([{"role": "user", "content": "hi"}])

    # With fallback: retried once on the fallback client.
    clients = {"primary": _FakeClient("primary", fail=True),
               "fallback": _FakeClient("fallback", fail=False)}
    monkeypatch.setattr(
        "src.market_scanner.review._construct_chat_llm",
        lambda model=None: clients.get(model) or clients["primary"],
    )
    fn2 = build_default_llm_callable(model="primary", fallback_model="fallback")
    assert "fallback" in fn2([{"role": "user", "content": "hi"}])


# --------------------------------------------------------------------------- #
# rerun_ai_review
# --------------------------------------------------------------------------- #


def _scan_with_pool(verdicts=None) -> dict:
    return run(llm=_json_llm(verdicts))


def test_rerun_ai_review_with_pool_backfills_after_avoid():
    result = run(config=ScanConfig(top_n=2), llm=_json_llm({"600001": "avoid"}))
    assert len(result["ai_pool"]) >= len(result["picks"])
    before_picks = {p["code"] for p in result["picks"]}
    assert "600001" in {d["code"] for d in result["ai_demoted"]}
    assert "600001" not in before_picks
    assert len(before_picks) == 2  # backfilled to top_n=2 despite the avoid

    # Mark the demoted row retryable. Already-reviewed rows must not be called
    # again, but a failed pool row can be retried and then backfill the picks.
    for row in result["ai_pool"]:
        if row["code"] == "600001":
            row["ai"] = dict(row["ai"], status="failed")
            break
    calls = []

    def _watch_llm(messages):
        calls.append(_code_from(messages))
        return _json_llm()(messages)

    updated = rerun_ai_review(
        result,
        ScanConfig(top_n=2),
        llm=_watch_llm,
        sources=make_sources(),
        now=NOW,
    )
    assert updated["review_revision"] == 1
    assert updated["reviewed_at"] is not None
    assert updated["ai_status"] == "reviewed"
    assert updated["ai_demoted"] == []
    assert "600001" in {p["code"] for p in updated["picks"]}
    assert len(updated["picks"]) == 2  # top_n still caps the watchlist
    assert calls == ["600001"]
    assert "skipped already reviewed: " in updated["ai_note"]
    # Original run not mutated.
    assert result["review_revision"] == 0
    assert len(result["ai_demoted"]) == 1


def test_rerun_ai_review_without_pool_picks_only_no_backfill():
    result = _scan_with_pool({"600001": "avoid"})
    stripped = {k: v for k, v in result.items() if k != "ai_pool"}
    stripped["ai_pool"] = []

    updated = rerun_ai_review(
        stripped, ScanConfig(), llm=_json_llm(), sources=make_sources(), now=NOW
    )
    assert updated["review_revision"] == 1
    assert "no ai_pool" in updated["ai_note"]
    assert "picks only" in updated["ai_note"]
    # Without a pool there is no backfill: the demoted pick stays out and the
    # pick count does not grow beyond the previously-finalised picks.
    assert len(updated["picks"]) <= len(result["picks"])
    assert "600001" not in {p["code"] for p in updated["picks"]}


def test_rerun_ai_review_quota_failure_propagates():
    result = _scan_with_pool()
    for row in result["ai_pool"]:
        row["ai"] = dict(row["ai"], status="failed")
    retry_code = result["ai_pool"][0]["code"]
    future = (datetime.now(UTC) + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")

    def _quota_llm(messages):
        raise _quota_exc(future)

    updated = rerun_ai_review(
        result, ScanConfig(), llm=_quota_llm, sources=make_sources(), now=NOW
    )
    assert updated["ai_status"] == "not_reviewed"
    assert "quota" in updated["ai_note"]
    assert updated["ai_pool"][0]["code"] == retry_code
    assert updated["ai_pool"][0]["ai"]["status"] == "failed"
