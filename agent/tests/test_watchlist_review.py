"""Offline tests for the watchlist AI re-review slice (service/store/CLI).

Covers: in-place update (no seq reorder, timestamps untouched),
review_revision increments, refusal of non-latest/unknown/failed runs,
single-flight exclusion both ways (scan vs review), quota-failed picks
fixed by a re-review, and manual/suppression rules on the recomputed view.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import threading
from pathlib import Path

import pytest

from src.watchlist.service import ScanAlreadyRunning, WatchlistService
from src.watchlist.store import WatchlistStore


# --------------------------------------------------------------------------- #
# helpers (mirrors test_watchlist_service.py)
# --------------------------------------------------------------------------- #
class FakeCalendar:
    def __init__(self, day: _dt.date) -> None:
        self.day = day

    def is_trading_day(self, d: _dt.date) -> bool:
        return d.weekday() < 5

    def previous_trading_day(self, d: _dt.date) -> _dt.date:
        d = d - _dt.timedelta(days=1)
        while d.weekday() >= 5:
            d -= _dt.timedelta(days=1)
        return d

    def latest_completed_trading_day(self, now: _dt.datetime) -> _dt.date:
        if now.date().weekday() < 5 and now.time() >= _dt.time(15, 0):
            return now.date()
        return self.previous_trading_day(now.date())


def _pick(
    code: str,
    rank: int,
    *,
    ai: dict | None = None,
    name: str | None = None,
) -> dict:
    return {
        "code": code,
        "name": name or f"股票{code}",
        "board": "main_sh",
        "industry": "测试",
        "rank": rank,
        "score": 100 - rank,
        "sub_scores": {"fundamental": 80, "trend": 70, "flow": 60},
        "reasons": ["r1"],
        "flags": [],
        "ai": ai if ai is not None else {"status": "not_reviewed"},
    }


def _quota_ai(code: str) -> dict:
    return {
        "status": "failed",
        "failure_kind": "quota",
        "note": "review failed: quota (resets 20:00)",
        "verdict": None,
    }


def _reviewed_ai() -> dict:
    return {"status": "reviewed", "verdict": "watch", "summary": "ok"}


def _scan_run(
    run_id: str,
    picks: list[str],
    *,
    status: str = "ok",
    trade_date: str = "2026-10-12",
    ai: dict | None = None,
    ai_status: str = "not_reviewed",
    ai_note: str | None = None,
    seq_started_at: str = "2026-10-12T18:00:00",
) -> dict:
    picks_list = [
        _pick(c, i + 1, ai=(ai or {}).get(c)) for i, c in enumerate(picks)
    ]
    scores = [
        {
            "code": c,
            "name": f"股票{c}",
            "board": "main_sh",
            "industry": "测试",
            "score": 100 - i,
            "fundamental": 80,
            "trend": 70,
            "flow": 60,
            "exclusion_reason": None,
        }
        for i, c in enumerate(picks)
    ]
    return {
        "run_id": run_id,
        "trade_date": trade_date,
        "started_at": seq_started_at,
        "finished_at": "2026-10-12T18:05:00",
        "status": status,
        "trigger": "cli",
        "duration_s": 1.0,
        "intraday": False,
        "ai_status": ai_status,
        "ai_note": ai_note,
        "config": {"top_n": 20},
        "funnel": [
            {"stage": "scored", "count": len(scores), "removed": 0, "note": None}
        ],
        "sources": [{"name": "universe", "status": "ok", "rows": 10}],
        "picks": picks_list,
        "ai_demoted": [],
        "requalified": [],
        "pinned": [],
        "scores": scores,
        "ai_pool": [dict(p) for p in picks_list],
        "review_revision": 0,
        "reviewed_at": None,
    }


def _reviewed_run(run: dict, picks: list[dict], **overrides) -> dict:
    """A pipeline-style re-review output: same envelope, new picks/AI state."""
    updated = dict(run)
    updated.update(
        {
            "picks": picks,
            "ai_status": "reviewed",
            "ai_note": None,
            "review_revision": int(run.get("review_revision") or 0) + 1,
            "reviewed_at": "2026-10-12T19:00:00",
        }
    )
    updated.update(overrides)
    return updated


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "watchlist.sqlite3"


@pytest.fixture()
def fake_now() -> _dt.datetime:
    return _dt.datetime(2026, 10, 12, 18, 0, 0)


def _service(
    store_path: Path,
    scan_fn,
    fake_now: _dt.datetime,
    rerun_fn=None,
) -> WatchlistService:
    return WatchlistService(
        store=WatchlistStore(path=store_path),
        scan_fn=scan_fn,
        rerun_fn=rerun_fn,
        calendar_loader=lambda: FakeCalendar(_dt.date(2026, 10, 12)),
        now_fn=lambda: fake_now,
    )


# --------------------------------------------------------------------------- #
# service: in-place update + revision
# --------------------------------------------------------------------------- #
def test_rerun_updates_in_place_and_bumps_revision(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    def scan_fn(config, **kw):
        return _scan_run(
            "r1",
            ["600519", "000001"],
            ai={"600519": _quota_ai("600519"), "000001": _quota_ai("000001")},
            ai_status="partial",
            ai_note="review failed: quota (resets 20:00)",
        )

    def rerun_fn(run, config, **kw):
        picks = [_pick("600519", 1, ai=_reviewed_ai()), _pick("000001", 2, ai=_reviewed_ai())]
        return _reviewed_run(run, picks)

    svc = _service(store_path, scan_fn, fake_now, rerun_fn=rerun_fn)
    svc.run_scan(trigger="cli")
    before = svc.store.get_run("r1")

    summary = svc.rerun_ai_review(trigger="cli")
    assert summary["run_id"] == "r1"
    assert summary["ai_status"] == "reviewed"
    assert summary["review_revision"] == 1
    assert summary["reviewed_at"] == "2026-10-12T19:00:00"

    after = svc.store.get_run("r1")
    # same run, same scan identity — only the AI fields moved
    assert after["started_at"] == before["started_at"]
    assert after["finished_at"] == before["finished_at"]
    assert after["duration_s"] == before["duration_s"]
    assert after["ai_status"] == "reviewed"
    assert after["review_revision"] == 1
    assert {p["code"]: p["pick"]["ai"]["status"] for p in svc.store.get_picks("r1")} == {
        "600519": "reviewed",
        "000001": "reviewed",
    }
    # no history reorder: still exactly one run, same seq position
    runs = svc.store.list_runs(limit=10)
    assert [r["run_id"] for r in runs] == ["r1"]

    # second re-review increments again
    summary2 = svc.rerun_ai_review("r1", trigger="cli")
    assert summary2["review_revision"] == 2


def test_rerun_first_run_keeps_all_new_labels(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """No predecessor: the diff stays 'new' (not 'kept') after a re-review."""

    def scan_fn(config, **kw):
        return _scan_run("r1", ["600519", "000001"])

    def rerun_fn(run, config, **kw):
        picks = [_pick("600519", 1, ai=_reviewed_ai()), _pick("000001", 2, ai=_reviewed_ai())]
        return _reviewed_run(run, picks)

    svc = _service(store_path, scan_fn, fake_now, rerun_fn=rerun_fn)
    svc.run_scan(trigger="cli")
    summary = svc.rerun_ai_review(trigger="cli")
    assert summary["counts"]["new"] == 2
    assert summary["counts"]["kept"] == 0
    assert summary["counts"]["dropped"] == 0
    view = svc.get_watchlist()
    assert all(i["change"] == "new" for i in view["items"])


def test_rerun_diffs_against_the_run_before(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """Kept/dropped labels compare vs the PREVIOUS run, not this run itself."""
    calls = {"n": 0}

    def scan_fn(config, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _scan_run("r1", ["600519"])
        return _scan_run("r2", ["600519", "000001"], seq_started_at="2026-10-12T18:10:00")

    def rerun_fn(run, config, **kw):
        # 000001 leaves picks; 600519 stays
        return _reviewed_run(run, [_pick("600519", 1, ai=_reviewed_ai())])

    svc = _service(store_path, scan_fn, fake_now, rerun_fn=rerun_fn)
    svc.run_scan(trigger="cli")
    svc.run_scan(trigger="cli")
    summary = svc.rerun_ai_review(trigger="cli")
    assert summary["counts"]["kept"] == 1  # 600519 was a pick in r1
    assert summary["counts"]["new"] == 0
    assert summary["counts"]["dropped"] == 0  # 000001 was never a pick in r1
    view = svc.get_watchlist()
    # 000001 left picks: its auto item is cleaned up (diff churn vs the
    # original scan), 600519 stays
    assert {i["code"] for i in view["items"]} == {"600519"}


# --------------------------------------------------------------------------- #
# service: refusals
# --------------------------------------------------------------------------- #
def test_rerun_refuses_unknown_failed_and_non_latest(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    calls = {"n": 0}

    def scan_fn(config, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            return _scan_run(
                "r2", [], status="failed", seq_started_at="2026-10-12T18:10:00"
            )
        if calls["n"] == 3:
            return _scan_run("r3", ["600519"], seq_started_at="2026-10-12T18:20:00")
        return _scan_run("r1", ["600519"])

    def rerun_fn(run, config, **kw):
        return _reviewed_run(run, list(run.get("picks") or []))

    svc = _service(store_path, scan_fn, fake_now, rerun_fn=rerun_fn)
    with pytest.raises(ValueError, match="no successful run"):
        svc.rerun_ai_review()
    with pytest.raises(ValueError, match="unknown"):
        svc.rerun_ai_review("nope")

    svc.run_scan(trigger="cli")  # r1 ok (latest)
    assert svc.rerun_ai_review("r1", trigger="cli")["run_id"] == "r1"

    svc.run_scan(trigger="cli")  # r2 failed
    with pytest.raises(ValueError, match="not a successful"):
        svc.rerun_ai_review("r2")

    svc.run_scan(trigger="cli")  # r3 ok, latest
    assert svc.rerun_ai_review("r3", trigger="cli")["run_id"] == "r3"
    with pytest.raises(ValueError, match="not the latest successful"):
        svc.rerun_ai_review("r1", trigger="cli")


def test_rerun_accepts_partial_as_latest_success(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """Boss fix: success = ok|partial. A partial latest run is re-reviewable
    by default; an older ok run is refused as not-latest."""
    calls = {"n": 0}

    def scan_fn(config, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _scan_run("r1", ["600519"])
        return _scan_run(
            "r2", ["600519", "000001"], status="partial",
            seq_started_at="2026-10-12T18:10:00",
            ai_status="partial",
            ai_note="review failed: quota (resets 20:00)",
        )

    def rerun_fn(run, config, **kw):
        return _reviewed_run(run, list(run.get("picks") or []))

    svc = _service(store_path, scan_fn, fake_now, rerun_fn=rerun_fn)
    svc.run_scan(trigger="cli")  # r1 ok
    svc.run_scan(trigger="cli")  # r2 partial (latest)
    summary = svc.rerun_ai_review(trigger="cli")  # default -> r2
    assert summary["run_id"] == "r2"
    assert summary["review_revision"] == 1
    with pytest.raises(ValueError, match="not the latest successful"):
        svc.rerun_ai_review("r1", trigger="cli")


def test_rerun_diffs_against_partial_predecessor(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    """A partial previous run is the diff baseline (ok|partial success —
    an ok-only predecessor lookup would have returned None -> all 'new')."""
    calls = {"n": 0}

    def scan_fn(config, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _scan_run("r1", ["600519"], status="partial")
        return _scan_run(
            "r2", ["600519", "000001"], seq_started_at="2026-10-12T18:10:00"
        )

    def rerun_fn(run, config, **kw):
        return _reviewed_run(run, list(run.get("picks") or []))

    svc = _service(store_path, scan_fn, fake_now, rerun_fn=rerun_fn)
    svc.run_scan(trigger="cli")  # r1 partial
    svc.run_scan(trigger="cli")  # r2 ok (latest)
    summary = svc.rerun_ai_review("r2", trigger="cli")
    assert summary["counts"]["kept"] == 1  # 600519 was a pick in partial r1
    assert summary["counts"]["new"] == 1  # 000001 is new vs r1


# --------------------------------------------------------------------------- #
# service: single-flight, both directions
# --------------------------------------------------------------------------- #
def test_rerun_and_scan_exclude_each_other(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    release = threading.Event()
    rerun_started = threading.Event()

    def scan_fn(config, **kw):
        return _scan_run("r1", ["600519"])

    def slow_rerun(run, config, **kw):
        rerun_started.set()
        release.wait(timeout=5)
        return _reviewed_run(run, list(run.get("picks") or []))

    svc = _service(store_path, scan_fn, fake_now, rerun_fn=slow_rerun)
    svc.run_scan(trigger="cli")

    t = threading.Thread(target=lambda: svc.rerun_ai_review(trigger="cli"), daemon=True)
    t.start()
    assert rerun_started.wait(timeout=5)
    with pytest.raises(ScanAlreadyRunning):
        svc.run_scan(trigger="cli")
    with pytest.raises(ScanAlreadyRunning):
        svc.rerun_ai_review(trigger="cli")
    assert svc.is_scanning() is True
    release.set()
    t.join(timeout=5)
    assert svc.is_scanning() is False

    # reverse: a running scan blocks a re-review
    scan_started = threading.Event()

    def slow_scan(config, **kw):
        scan_started.set()
        release.wait(timeout=5)
        return _scan_run("r2", ["600519"])

    svc._scan_fn = slow_scan
    release.clear()
    t2 = threading.Thread(target=lambda: svc.run_scan(trigger="cli"), daemon=True)
    t2.start()
    assert scan_started.wait(timeout=5)
    with pytest.raises(ScanAlreadyRunning):
        svc.rerun_ai_review(trigger="cli")
    release.set()
    t2.join(timeout=5)


# --------------------------------------------------------------------------- #
# service: quota-failed picks fixed + manual/suppression rules
# --------------------------------------------------------------------------- #
def test_rerun_fixes_quota_failed_picks(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    def scan_fn(config, **kw):
        return _scan_run(
            "r1",
            ["600519", "000001"],
            ai={"600519": _quota_ai("600519"), "000001": _quota_ai("000001")},
            ai_status="partial",
            ai_note="review failed: quota (resets 20:00)",
        )

    def rerun_fn(run, config, **kw):
        picks = [
            _pick("600519", 1, ai=_reviewed_ai()),
            _pick("000001", 2, ai=_reviewed_ai()),
        ]
        return _reviewed_run(run, picks)

    svc = _service(store_path, scan_fn, fake_now, rerun_fn=rerun_fn)
    svc.run_scan(trigger="cli")
    view = svc.get_watchlist()
    assert view["latest_run"]["ai_status"] == "partial"
    assert all(i["ai"]["failure_kind"] == "quota" for i in view["items"])

    svc.rerun_ai_review(trigger="cli")
    view = svc.get_watchlist()
    assert view["latest_run"]["ai_status"] == "reviewed"
    assert view["latest_run"]["ai_note"] is None
    assert all(i["ai"]["status"] == "reviewed" for i in view["items"])


def test_rerun_honours_manual_pins_and_suppressed_codes(
    store_path: Path, fake_now: _dt.datetime
) -> None:
    captured: dict = {}

    def scan_fn(config, **kw):
        return _scan_run("r1", ["600519", "000001", "300750"])

    def rerun_fn(run, config, **kw):
        captured["pinned"] = set(kw.get("pinned") or set())
        captured["suppressed"] = set(kw.get("suppressed") or set())
        picks = [
            _pick("600519", 1, ai=_reviewed_ai()),
            _pick("300750", 2, ai=_reviewed_ai()),
            _pick("688981", 3, ai=_reviewed_ai()),  # backfilled pick
        ]
        demoted = [_pick("000001", 4, ai={"status": "reviewed", "verdict": "avoid"})]
        return _reviewed_run(run, picks, ai_demoted=demoted)

    svc = _service(store_path, scan_fn, fake_now, rerun_fn=rerun_fn)
    svc.run_scan(trigger="cli")
    svc.remove("000001")  # suppressed
    svc.add_manual("603000", note="手动")  # manual pin

    summary = svc.rerun_ai_review(trigger="cli")
    assert captured["pinned"] == {"603000"}
    assert captured["suppressed"] == {"000001"}

    view = svc.get_watchlist()
    codes = {i["code"]: i for i in view["items"]}
    # suppressed code never re-adding; manual pin untouched; backfilled pick added
    assert "000001" not in codes
    assert codes["603000"]["origin"] == "manual"
    assert codes["603000"]["change"] == "manual"
    assert {"600519", "300750", "688981"} <= set(codes)
    assert summary["counts"]["ai_demoted"] == 1


# --------------------------------------------------------------------------- #
# store: update_run_review / last_successful_run_before
# --------------------------------------------------------------------------- #
def test_store_update_run_review_preserves_seq_and_rewrites_picks(
    tmp_path: Path,
) -> None:
    store = WatchlistStore(path=tmp_path / "wl.sqlite3")
    r1 = _scan_run("r1", ["600519"])
    r2 = _scan_run("r2", ["000001"], seq_started_at="2026-10-12T18:10:00")
    store.save_run(r1, changes={"new": [("600519", 1, 99.0, r1["picks"][0])]})
    store.save_run(
        r2, changes={"new": [("000001", 1, 99.0, r2["picks"][0])]}
    )
    order_before = [r["run_id"] for r in store.list_runs(limit=10)]

    updated = _reviewed_run(
        r2, [_pick("000001", 1, ai=_reviewed_ai()), _pick("600519", 2, ai=_reviewed_ai())]
    )
    changes = {
        "new": [("600519", 2, 98.0, updated["picks"][1])],
        "kept": [("000001", 1, 99.0, updated["picks"][0])],
    }
    store.update_run_review(updated, changes=changes)

    assert [r["run_id"] for r in store.list_runs(limit=10)] == order_before
    after = store.get_run("r2")
    assert after["ai_status"] == "reviewed"
    assert after["review_revision"] == 1
    assert after["started_at"] == r2["started_at"]  # untouched
    picks = {p["code"]: p["change"] for p in store.get_picks("r2")}
    assert picks == {"000001": "kept", "600519": "new"}


def test_store_last_successful_run_before(tmp_path: Path) -> None:
    store = WatchlistStore(path=tmp_path / "wl.sqlite3")
    assert store.last_successful_run_before("r1") is None
    store.save_run(_scan_run("r1", ["600519"]))
    store.save_run(
        _scan_run("r2", [], status="failed", seq_started_at="2026-10-12T18:10:00")
    )
    store.save_run(
        _scan_run("r3", ["000001"], seq_started_at="2026-10-12T18:20:00")
    )
    before_r3 = store.last_successful_run_before("r3")
    assert before_r3 is not None and before_r3["run_id"] == "r1"  # r2 failed
    assert store.last_successful_run_before("r1") is None


# --------------------------------------------------------------------------- #
# CLI: watchlist review [RUN_ID] [--json]
# --------------------------------------------------------------------------- #
def _cli_args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture()
def cli_fake(monkeypatch: pytest.MonkeyPatch):
    from cli.commands import watchlist as wl_cli

    class FakeSvc:
        def __init__(self) -> None:
            self.error: Exception | None = None
            self.calls: list[tuple] = []

        def rerun_ai_review(self, run_id, *, trigger, progress=None, llm=None):
            self.calls.append((run_id, trigger))
            if self.error:
                raise self.error
            if progress:
                progress("news", 0.3)
                progress("done", 1.0)
            return {
                "run_id": run_id or "r1",
                "ai_status": "reviewed",
                "ai_note": None,
                "duration_s": 2.5,
                "review_revision": 1,
                "reviewed_at": "2026-10-12T19:00:00",
                "counts": {"new": 0, "kept": 1, "dropped": 0},
            }

        def get_run(self, run_id):
            run = _scan_run(run_id, ["600519"], ai={"600519": _reviewed_ai()})
            run["ai_status"] = "reviewed"
            run["review_revision"] = 1
            run["reviewed_at"] = "2026-10-12T19:00:00"
            run["picks"][0]["reasons"] = ["ROE top"]
            run["picks"][0]["ai"] = _reviewed_ai()
            return run

    fake = FakeSvc()
    monkeypatch.setattr(wl_cli, "_default_service", lambda: fake)
    monkeypatch.setattr(wl_cli, "_default_llm", lambda: None)
    return wl_cli, fake


def test_cli_review_prints_picks_and_summary(cli_fake, capsys) -> None:
    wl_cli, fake = cli_fake
    args = _cli_args(run_id=None, json=False)
    rc = wl_cli._cmd_review(args, fake)
    out = capsys.readouterr().out
    assert rc == 0
    assert fake.calls == [(None, "cli")]
    assert "PICKS" in out
    assert "600519" in out
    assert "ai_status=reviewed" in out
    assert "review_revision=1" in out
    assert "reviewed_at=2026-10-12T19:00:00" in out


def test_cli_review_json_includes_counts(cli_fake, capsys) -> None:
    wl_cli, fake = cli_fake
    args = _cli_args(run_id="r1", json=True)
    rc = wl_cli._cmd_review(args, fake)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["run_id"] == "r1"
    assert payload["counts"]["kept"] == 1


def test_cli_review_value_error_exits_2(cli_fake, capsys) -> None:
    wl_cli, fake = cli_fake
    fake.error = ValueError("run r1 is not the latest successful run")
    rc = wl_cli._cmd_review(_cli_args(run_id="r1", json=False), fake)
    assert rc == 2
    assert "ERROR" in capsys.readouterr().out


def test_cli_review_already_running_exits_3(cli_fake, capsys) -> None:
    wl_cli, fake = cli_fake
    fake.error = ScanAlreadyRunning("a scan or review is already running")
    rc = wl_cli._cmd_review(_cli_args(run_id=None, json=False), fake)
    assert rc == 3
    assert "ERROR" in capsys.readouterr().out


def test_cli_review_registered_in_parser() -> None:
    import argparse as _argparse

    from cli.commands import watchlist as wl_cli

    parser = _argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="root", required=True)
    wl_cli.add_subparser(sub)
    args = parser.parse_args(["watchlist", "review", "r1", "--json"])
    assert args.watchlist_cmd == "review"
    assert args.run_id == "r1" and args.json is True
    args2 = parser.parse_args(["watchlist", "review"])
    assert args2.run_id is None
