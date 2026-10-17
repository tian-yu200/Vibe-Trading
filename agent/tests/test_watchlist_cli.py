"""Offline tests for the watchlist CLI (fake service, no network)."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import pytest

from cli.commands import watchlist as wl_cli


class FakeService:
    """Minimal WatchlistService stand-in with canned data."""

    def __init__(self) -> None:
        self.run = {
            "run_id": "r1",
            "trade_date": "2026-10-12",
            "started_at": "2026-10-12T18:00:00",
            "finished_at": "2026-10-12T18:05:00",
            "status": "ok",
            "trigger": "cli",
            "duration_s": 300.0,
            "intraday": False,
            "ai_status": "reviewed",
            "ai_note": None,
            "config": {"top_n": 20},
            "funnel": [
                {"stage": "universe", "count": 5572, "removed": 0, "note": None},
                {"stage": "scored", "count": 120, "removed": 5452, "note": None},
            ],
            "coverage": {
                "passed_exclusions": 130,
                "insufficient": 10,
                "scored": 120,
                "trend_scored": 120,
                "flow_real": 100,
                "flow_neutral": 20,
            },
            "sources": [
                {
                    "name": "universe",
                    "status": "ok",
                    "rows": 5572,
                    "latency_s": 28.0,
                    "detail": "hs_a+hs_bjs",
                },
                {
                    "name": "kline",
                    "status": "ok",
                    "rows": 300,
                    "latency_s": 45.4,
                    "detail": "amount estimated from close*volume*100 for 300 symbols",
                },
            ],
            "picks": [
                {
                    "code": "600519",
                    "name": "贵州茅台",
                    "board": "main_sh",
                    "industry": "白酒",
                    "rank": 1,
                    "score": 88.5,
                    "change": "new",
                    "sub_scores": {"fundamental": 90, "trend": 85, "flow": 88},
                    "reasons": ["ROE 30.5% (top 5%)"],
                    "flags": [],
                    "ai": {
                        "status": "reviewed",
                        "verdict": "watch",
                        "summary": "基本面扎实",
                        "reasons": [{"text": "ROE高", "cites": ["M:roe"]}],
                        "red_flags": [],
                    },
                },
            ],
            "ai_demoted": [
                {
                    "code": "000002",
                    "name": "万科A",
                    "rank": 5,
                    "score": 60,
                    "ai": {"verdict": "avoid", "summary": "监管红旗"},
                },
            ],
            "requalified": [
                {
                    "code": "300750",
                    "name": "宁德时代",
                    "rank": 2,
                    "score": 85,
                    "sub_scores": {},
                    "reasons": [],
                },
            ],
            "scores": [],
        }
        self.view = {
            "disclaimer": "Research only, not investment advice.",
            "stale": {
                "is_stale": True,
                "latest_trading_day": "2026-10-12",
                "last_scan_trade_date": None,
                "last_scan_at": None,
            },
            "latest_run": {
                "run_id": "r1",
                "trade_date": "2026-10-12",
                "finished_at": "2026-10-12T18:05:00",
                "status": "ok",
                "ai_status": "reviewed",
                "ai_note": None,
                "duration_s": 300.0,
                "trigger": "cli",
                "funnel": [],
                "sources": [],
            },
            "items": [
                {
                    "code": "600519",
                    "name": "贵州茅台",
                    "board": "main_sh",
                    "industry": "白酒",
                    "origin": "auto",
                    "change": "new",
                    "rank": 1,
                    "score": 88.5,
                    "sub_scores": {},
                    "reasons": [],
                    "flags": [],
                    "ai": None,
                    "added_at": "t",
                    "note": None,
                    "scores_available": True,
                    "unscored_reason": None,
                }
            ],
            "dropped": [
                {
                    "code": "000002",
                    "name": "万科A",
                    "prev_rank": 5,
                    "reason": "excluded:st",
                }
            ],
            "requalified": [
                {
                    "code": "300750",
                    "name": "宁德时代",
                    "rank": 2,
                    "score": 85,
                    "removed_at": "t",
                    "label": "qualifies again - you removed it",
                }
            ],
            "ai_demoted": [],
        }
        self.settings = {
            "scan": {"top_n": 20},
            "schedule": {
                "enabled": False,
                "time": "18:00",
                "timezone": "Asia/Shanghai",
                "catch_up": True,
            },
        }
        self.calls: list[str] = []

    def run_scan(self, *, trigger, config_overrides=None, progress=None, llm=None):
        self.calls.append(
            f"scan trigger={trigger} overrides={config_overrides} noai={llm is None}"
        )
        return {
            "run_id": "r1",
            "trade_date": "2026-10-12",
            "status": "ok",
            "ai_status": "reviewed",
            "duration_s": 300.0,
            "funnel": self.run["funnel"],
            "sources": self.run["sources"],
            "counts": {
                "new": 1,
                "kept": 0,
                "dropped": 0,
                "requalified": 1,
                "ai_demoted": 1,
            },
        }

    def get_run(self, run_id):
        return self.run if run_id == "r1" else None

    def get_watchlist(self):
        return self.view

    def add_manual(self, code, note=None):
        self.calls.append(f"add {code}")
        return {"code": code, "name": None, "origin": "manual", "state": "active"}

    def remove(self, code):
        self.calls.append(f"remove {code}")
        return {"code": code, "state": "removed"}

    def restore(self, code):
        self.calls.append(f"restore {code}")
        return {"code": code, "state": "active"}

    def list_runs(self, limit=20):
        return [self.view["latest_run"]]

    def get_settings(self):
        return self.settings

    def update_settings(self, patch):
        self.calls.append(f"settings {patch}")
        if "schedule" in patch:
            self.settings["schedule"].update(patch["schedule"])
        return self.settings

    def invalidate_run(self, run_id, *, reason=None):
        self.calls.append(f"invalidate {run_id}")
        if run_id == "missing":
            raise ValueError(f"run not found: {run_id}")
        return {
            "run_id": run_id,
            "status": "failed",
            "failure_reason": reason or "invalidated manually",
            "restored": True,
            "restored_from": "r0",
            "restored_counts": {"added": 1, "removed": 2},
        }


@pytest.fixture()
def fake_svc(monkeypatch: pytest.MonkeyPatch) -> FakeService:
    svc = FakeService()
    monkeypatch.setattr(wl_cli, "_default_service", lambda: svc)
    monkeypatch.setattr(wl_cli, "_default_llm", lambda: None)  # no LLM in tests
    return svc


def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_scan_prints_blocks_and_reconcile(fake_svc: FakeService, capsys) -> None:
    args = _args(
        top=None, exclude_star=False, exclude_chinext=False, no_ai=True, json=False
    )
    rc = wl_cli._cmd_scan(args, fake_svc)
    out = capsys.readouterr().out
    assert rc == 0
    assert "FUNNEL" in out
    assert "COVERAGE" in out
    assert "reconcile: 130 - 10 = 120 = 120 = 100 + 20  HOLDS" in out
    assert "SOURCES" in out and "kline" in out
    assert "PICKS" in out and "贵州茅台" in out
    assert "AI-DEMOTED" in out
    assert "REQUALIFIED" in out
    assert "status=ok" in out and "total runtime" in out
    assert fake_svc.calls and "noai=True" in fake_svc.calls[0]


def _real_service_run(run_id: str, codes: list[str]) -> dict:
    picks = [
        {
            "code": code,
            "name": f"Stock {code}",
            "board": "main_sh",
            "industry": "test",
            "rank": rank,
            "score": 100.0 - rank,
            "sub_scores": {"fundamental": 80, "trend": 70, "flow": 60},
            "reasons": [],
            "flags": [],
            "ai": {"status": "not_reviewed"},
        }
        for rank, code in enumerate(codes, 1)
    ]
    return {
        "run_id": run_id,
        "trade_date": "2026-10-12",
        "started_at": "2026-10-12T18:00:00",
        "finished_at": "2026-10-12T18:01:00",
        "status": "ok",
        "trigger": "cli",
        "duration_s": 1.0,
        "intraday": False,
        "ai_status": "not_reviewed",
        "ai_note": None,
        "config": {"top_n": 20},
        "funnel": [],
        "sources": [],
        "picks": picks,
        "scores": [
            {
                "code": code,
                "name": f"Stock {code}",
                "board": "main_sh",
                "industry": "test",
                "score": 100.0 - rank,
                "fundamental": 80,
                "trend": 70,
                "flow": 60,
                "exclusion_reason": None,
            }
            for rank, code in enumerate(codes, 1)
        ],
        "ai_demoted": [],
        "requalified": [],
        "pinned": [],
    }


class _CliCalendar:
    def is_trading_day(self, day: _dt.date) -> bool:
        return day.weekday() < 5

    def previous_trading_day(self, day: _dt.date) -> _dt.date:
        day -= _dt.timedelta(days=1)
        while day.weekday() >= 5:
            day -= _dt.timedelta(days=1)
        return day

    def latest_completed_trading_day(self, now: _dt.datetime) -> _dt.date:
        if now.weekday() < 5 and now.time() >= _dt.time(15, 0):
            return now.date()
        return self.previous_trading_day(now.date())


def test_scan_applies_real_service_changes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from src.watchlist.service import WatchlistService
    from src.watchlist.store import WatchlistStore

    runs = iter(
        [
            _real_service_run("r1", ["600001", "600002"]),
            _real_service_run("r2", ["600001", "600003"]),
        ]
    )
    svc = WatchlistService(
        store=WatchlistStore(path=tmp_path / "watchlist.sqlite3"),
        scan_fn=lambda config, **kwargs: next(runs),
        calendar_loader=lambda: _CliCalendar(),
        now_fn=lambda: _dt.datetime(2026, 10, 12, 18, 0),
    )
    monkeypatch.setattr(wl_cli, "_default_llm", lambda: None)
    args = _args(
        top=None, exclude_star=False, exclude_chinext=False, no_ai=True, json=False
    )

    wl_cli._cmd_scan(args, svc)
    capsys.readouterr()
    rc = wl_cli._cmd_scan(args, svc)
    out = capsys.readouterr().out

    assert rc == 0
    assert "new      600003" in out
    assert "kept     600001" in out


def test_review_applies_real_service_changes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from src.watchlist.service import WatchlistService
    from src.watchlist.store import WatchlistStore

    predecessor = _real_service_run("r0", ["600001", "600002"])
    original = _real_service_run("r1", ["600001", "600004"])
    updated = _real_service_run("r1", ["600001", "600003"])
    scan_runs = iter([predecessor, original])
    svc = WatchlistService(
        store=WatchlistStore(path=tmp_path / "watchlist.sqlite3"),
        scan_fn=lambda config, **kwargs: next(scan_runs),
        rerun_fn=lambda run, config, **kwargs: updated,
        calendar_loader=lambda: _CliCalendar(),
        now_fn=lambda: _dt.datetime(2026, 10, 12, 18, 0),
    )
    svc.run_scan(trigger="cli")
    svc.run_scan(trigger="cli")
    monkeypatch.setattr(wl_cli, "_default_llm", lambda: object())
    args = _args(run_id=None, json=False)

    rc = wl_cli._cmd_review(args, svc)
    out = capsys.readouterr().out

    assert rc == 0
    assert "new      600003" in out
    assert "kept     600001" in out


def test_scan_prints_requests_and_cache_coverage(fake_svc: FakeService, capsys) -> None:
    fake_svc.run["requests"] = {
        "sina": {"count": 60, "errors": 0, "elapsed_s": 30.0, "effective_rps": 2.0},
        "tencent": {
            "count": 120,
            "errors": 1,
            "elapsed_s": 120.0,
            "effective_rps": 1.0,
        },
        "_totals": {
            "count": 180,
            "errors": 1,
            "elapsed_s": 150.0,
            "effective_rps": 1.2,
        },
    }
    fake_svc.run["coverage"] = {
        "passed_exclusions": 130,
        "insufficient": 10,
        "scored": 120,
        "trend_scored": 120,
        "flow_real": 100,
        "flow_neutral": 20,
        "kline_full_history": 120,
    }
    fake_svc.run["sources"] = [
        {
            "name": "klines",
            "status": "ok",
            "rows": 300,
            "latency_s": 120.0,
            "detail": "... cache_hit=20 appended_from_snapshot=280 ...",
        },
    ]
    args = _args(
        top=None, exclude_star=False, exclude_chinext=False, no_ai=True, json=False
    )
    wl_cli._cmd_scan(args, fake_svc)
    out = capsys.readouterr().out
    assert "REQUESTS" in out
    assert "sina" in out and "tencent" in out
    assert "TOTALS" in out
    assert (
        "kline cache coverage: stocks that passed exclusions AND have "
        "full history 120 / passed_exclusions 130"
    ) in out


def test_scan_json_valid(fake_svc: FakeService, capsys) -> None:
    args = _args(
        top=None, exclude_star=False, exclude_chinext=False, no_ai=False, json=True
    )
    rc = wl_cli._cmd_scan(args, fake_svc)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert payload["run_id"] == "r1"
    assert payload["counts"]["new"] == 1


def test_scan_overrides_passed(fake_svc: FakeService, capsys) -> None:
    args = _args(
        top=5, exclude_star=True, exclude_chinext=True, no_ai=False, json=False
    )
    # Force the LLM factory to succeed so --no-ai=False actually yields a
    # non-None llm (in the default test env ChatLLM() raises -> None -> the
    # scan runs without AI, which the noai=True assertion covers elsewhere).
    monkey_llm = object()
    import cli.commands.watchlist as wl_mod

    original = wl_mod._default_llm
    wl_mod._default_llm = lambda: monkey_llm
    try:
        wl_cli._cmd_scan(args, fake_svc)
    finally:
        wl_mod._default_llm = original
    call = fake_svc.calls[0]
    assert "'top_n': 5" in call
    assert "'include_star': False" in call
    assert "'include_chinext': False" in call
    assert "noai=False" in call


def test_list_disclaimer_and_stale(fake_svc: FakeService, capsys) -> None:
    args = _args()
    rc = wl_cli._cmd_list(args, fake_svc)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Research only, not investment advice." in out
    assert "STALE" in out
    assert "600519" in out
    assert "qualifies again - you removed it" in out


def test_add_remove_restore_dispatch(fake_svc: FakeService, capsys) -> None:
    wl_cli._cmd_add(_args(code="600519", note="n"), fake_svc)
    wl_cli._cmd_remove(_args(code="600519"), fake_svc)
    wl_cli._cmd_restore(_args(code="600519"), fake_svc)
    assert fake_svc.calls == ["add 600519", "remove 600519", "restore 600519"]


def test_add_invalid_code_exits_2(fake_svc: FakeService, capsys) -> None:
    def boom(code, note=None):
        raise ValueError("invalid A-share code: 'abc'")

    fake_svc.add_manual = boom
    rc = wl_cli._cmd_add(_args(code="abc", note=None), fake_svc)
    assert rc == 2


def test_schedule_on_off_status(fake_svc: FakeService, capsys) -> None:
    wl_cli._cmd_schedule(_args(schedule_cmd="on", time="19:30"), fake_svc)
    assert fake_svc.settings["schedule"]["enabled"] is True
    assert fake_svc.settings["schedule"]["time"] == "19:30"
    wl_cli._cmd_schedule(_args(schedule_cmd="off", time=None), fake_svc)
    assert fake_svc.settings["schedule"]["enabled"] is False
    out = capsys.readouterr().out
    assert "schedule enabled=False" in out


def test_add_subparser_parses() -> None:
    # add_subparser attaches 'watchlist' (with its own sub-subcommands) to the
    # TOP-LEVEL subparsers, exactly like _legacy.py does.
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    wl_cli.add_subparser(sub)
    args = parser.parse_args(
        ["watchlist", "scan", "--no-ai", "--top", "5", "--exclude-star"]
    )
    assert args.watchlist_cmd == "scan"
    assert args.no_ai is True
    assert args.top == 5
    assert args.exclude_star is True
    args2 = parser.parse_args(["watchlist", "add", "600519", "--note", "x"])
    assert args2.code == "600519"
    args3 = parser.parse_args(["watchlist", "schedule", "on", "--time", "19:00"])
    assert args3.schedule_cmd == "on"
    args4 = parser.parse_args(
        ["watchlist", "invalidate", "abc123", "--reason", "why"]
    )
    assert args4.watchlist_cmd == "invalidate"
    assert args4.run_id == "abc123"
    assert args4.reason == "why"


# --------------------------------------------------------------------------- #
# Coverage-guard failure surfacing (list / history / invalidate)
# --------------------------------------------------------------------------- #
def test_list_shows_last_failed_run_with_reason(
    fake_svc: FakeService, capsys
) -> None:
    fake_svc.view["last_failed_run"] = {
        "run_id": "r2",
        "trade_date": "2026-10-12",
        "finished_at": "2026-10-12T18:15:00",
        "status": "failed",
        "ai_status": "not_reviewed",
        "trigger": "schedule",
        "failure_reason": "insufficient coverage (trend 12.0% < 90%)",
    }
    rc = wl_cli._cmd_list(_args(), fake_svc)
    out = capsys.readouterr().out
    assert rc == 0
    assert "latest run r1" in out
    assert "last failed run r2" in out
    assert "reason=insufficient coverage (trend 12.0% < 90%)" in out


def test_history_prints_failure_reason(fake_svc: FakeService, capsys) -> None:
    fake_svc.view["latest_run"]["failure_reason"] = (
        "insufficient coverage (trend 12.0% < 90%)"
    )
    rc = wl_cli._cmd_history(_args(limit=5, json=False), fake_svc)
    out = capsys.readouterr().out
    assert rc == 0
    assert "reason=insufficient coverage (trend 12.0% < 90%)" in out


def test_invalidate_dispatch(fake_svc: FakeService, capsys) -> None:
    rc = wl_cli._cmd_invalidate(_args(run_id="r2", reason="why"), fake_svc)
    out = capsys.readouterr().out
    assert rc == 0
    assert "run r2 status=failed" in out
    assert "reason=why" in out
    assert "watchlist restored to r0" in out
    assert fake_svc.calls == ["invalidate r2"]


def test_invalidate_unknown_run_exits_2(fake_svc: FakeService, capsys) -> None:
    rc = wl_cli._cmd_invalidate(_args(run_id="missing", reason=None), fake_svc)
    out = capsys.readouterr().out
    assert rc == 2
    assert "ERROR" in out
