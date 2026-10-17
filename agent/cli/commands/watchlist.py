"""``vibe-trading watchlist`` subcommand: scan | list | add | remove | restore |
forget | history | show | schedule | review | backfill (CONTRACT section 11).

All output goes through this module; the service is injectable for tests via
``_service_factory`` monkeypatching or the ``--json`` plumbing. Chinese names
print on Windows consoles: stdout is re-wrapped as UTF-8 with errors=replace.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from typing import Any

_DISCLAIMER = "Research only, not investment advice."


def _cli_home(context: str) -> dict[str, Any] | None:
    """F1: pin-check before any watchlist subcommand runs.

    Calls pi's ``home.require_home`` (never creates anything) and returns
    ``home_fields()`` for headers. On a pin mismatch the message goes to
    stderr and the process exits non-zero via ``SystemExit(2)``.
    """
    try:
        from src.watchlist.home import home_fields, require_home
    except ImportError:
        return None
    try:
        require_home(context)
    except Exception as exc:  # noqa: BLE001 — HomeMismatch matched by name
        if type(exc).__name__ == "HomeMismatch":
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from exc
        raise
    try:
        return dict(home_fields())
    except Exception:  # noqa: BLE001
        return None


def _print_home_header(home: dict[str, Any] | None, what: str) -> None:
    """One-line resolved-home header for scan/backfill/status output."""
    if home:
        print(f"home={home.get('home')} pid={home.get('pid')} ({what})")


def _ensure_utf8_stdout() -> None:
    """Windows consoles default to GBK; force UTF-8 with replacement."""
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except (AttributeError, ValueError):
        pass  # already wrapped (tests) or detached


def _default_service():
    from src.watchlist.service import WatchlistService

    return WatchlistService()


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------
def _print_funnel(funnel: list[dict] | None) -> None:
    print("FUNNEL")
    print(f"  {'stage':<28}{'count':>10}{'removed':>10}")
    for stage in funnel or []:
        count = stage.get("count")
        removed = stage.get("removed")
        print(
            f"  {stage.get('stage', '?'):<28}"
            f"{count if count is not None else '-':>10}"
            f"{removed if removed is not None else '-':>10}"
        )


def _print_coverage(run: dict) -> None:
    coverage = run.get("coverage")
    derived = False
    if not coverage:
        derived = True
        scores = run.get("scores", [])
        scored_rows = [
            s
            for s in scores
            if s.get("exclusion_reason") is None and s.get("score") is not None
        ]
        passed = len(scored_rows) + len(
            [s for s in scores if s.get("exclusion_reason")]
        )
        flow_real = sum(1 for s in scored_rows if (s.get("flow") is not None))
        coverage = {
            "passed_exclusions": passed,
            "insufficient": len(scores) - len(scored_rows),
            "scored": len(scored_rows),
            "trend_scored": sum(1 for s in scored_rows if s.get("trend") is not None),
            "flow_real": flow_real,
            "flow_neutral": len(scored_rows) - flow_real,
        }
    print(
        "COVERAGE"
        + ("  (derived from scores — pipeline did not report)" if derived else "")
    )
    print(
        f"  passed_exclusions={coverage.get('passed_exclusions')}  "
        f"insufficient={coverage.get('insufficient')}  "
        f"scored={coverage.get('scored')}"
    )
    print(
        f"  trend_scored={coverage.get('trend_scored')}  "
        f"flow_real={coverage.get('flow_real')}  "
        f"flow_neutral={coverage.get('flow_neutral')}"
    )
    try:
        pe = coverage.get("passed_exclusions") or 0
        ins = coverage.get("insufficient") or 0
        sc = coverage.get("scored") or 0
        tr = coverage.get("trend_scored") or 0
        fr = coverage.get("flow_real") or 0
        fn = coverage.get("flow_neutral") or 0
        ok = pe - ins == sc == tr == fr + fn
        print(
            f"  reconcile: {pe} - {ins} = {sc} = {tr} = {fr} + {fn}  "
            f"{'HOLDS' if ok else 'MISMATCH'}"
        )
    except (TypeError, ValueError):
        print("  reconcile: n/a")


def _print_sources(sources: list[dict] | None) -> None:
    print("SOURCES")
    print(f"  {'name':<16}{'status':<10}{'rows':>8}{'lat_s':>8}{'as_of':<24}  detail")
    for s in sources or []:
        detail = s.get("detail") or s.get("error") or ""
        print(
            f"  {s.get('name', '?'):<16}{s.get('status', '?'):<10}"
            f"{s.get('rows', 0):>8}{s.get('latency_s', 0):>8.1f}"
            f"{str(s.get('as_of') or '-'):<24}  {str(detail)[:70]}"
        )


def _print_picks(run: dict) -> None:
    picks = run.get("picks", [])
    print("PICKS")
    print(
        f"  {'rk':>3}{'chg':<9}{'code':<8}{'name':<12}{'board':<9}"
        f"{'industry':<14}{'score':>7} {'F':>5} {'T':>5} {'Fl':>5}  AI"
    )
    for p in picks:
        sub = p.get("sub_scores") or {}
        ai = p.get("ai") or {}
        print(
            f"  {p.get('rank', 0):>3}{_change_word(p):<9}{p['code']:<8}"
            f"{(p.get('name') or ''):<12}{(p.get('board') or ''):<9}"
            f"{(p.get('industry') or ''):<14}{p.get('score') or 0:>7.1f} "
            f"{sub.get('fundamental') or 0:>5.0f} {sub.get('trend') or 0:>5.0f} "
            f"{sub.get('flow') or 0:>5.0f}  {ai.get('verdict') or '-'}"
        )
    print()
    for p in picks:
        reasons = p.get("reasons") or []
        if reasons:
            print(f"  [{p['code']} {p.get('name')}]")
            for r in reasons:
                print(f"    - {r}")
        ai = p.get("ai") or {}
        for ev in ai.get("evidence") or []:
            if ev.get("kind") == "metric":
                print(f"    [{ev['id']}] metric {ev.get('name')} = {ev.get('value')}")
            else:
                date = ev.get("date") or "?"
                title = ev.get("title_or_text") or "?"
                source = ev.get("source") or "?"
                print(f"    [{ev['id']}] {date} {title} ({source})")
        if ai.get("summary"):
            print(f"    AI: {ai['summary']}")
            for r in ai.get("reasons") or []:
                cites = ",".join(r.get("cites", []))
                print(f"    AI-  {r.get('text')} [{cites}]")


def _print_extras(run: dict) -> None:
    demoted = run.get("ai_demoted") or []
    if demoted:
        print("AI-DEMOTED")
        for d in demoted:
            print(
                f"  {d.get('code')} {d.get('name')} -> {d.get('ai', {}).get('verdict')}"
                f": {(d.get('ai') or {}).get('summary', '')}"
            )
    requalified = run.get("requalified") or []
    if requalified:
        print("REQUALIFIED (qualifies again - you removed it)")
        for r in requalified:
            print(
                f"  {r.get('code')} {r.get('name')} rank={r.get('rank')} "
                f"score={r.get('score')}"
            )


def _print_run_summary(run: dict, total_s: float | None = None) -> None:
    print(
        f"run {run.get('run_id')} trade_date={run.get('trade_date')} "
        f"status={run.get('status')} ai_status={run.get('ai_status')} "
        f"ai_note={run.get('ai_note') or '-'}"
    )
    if run.get("failure_reason"):
        print(f"failure_reason: {run['failure_reason']}")
    if total_s is not None:
        print(f"total runtime: {total_s:.1f}s")


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def _print_requests(run: dict) -> None:
    requests = run.get("requests") or {}
    if not requests:
        return
    print("REQUESTS  (host, count, errors, wall span s, busy s, effective req/s)")
    for host, stat in requests.items():
        if host == "_totals":
            continue
        print(
            f"  {host:<24}{stat.get('count', 0):>8}{stat.get('errors', 0):>7}"
            f"{stat.get('elapsed_s', 0):>10.2f}{stat.get('busy_s', 0):>9.2f}"
            f"{stat.get('effective_rps', 0):>10.2f}"
        )
    totals = requests.get("_totals") or {}
    if totals:
        print(
            f"  {'TOTALS':<24}{totals.get('count', 0):>8}{totals.get('errors', 0):>7}"
            f"{totals.get('elapsed_s', 0):>10.2f}{totals.get('busy_s', 0):>9.2f}"
            f"{totals.get('effective_rps', 0):>10.2f}"
        )
    # kline cache coverage line (section 9b): from the pipeline's coverage
    # block — stocks that passed exclusions AND have full history.
    coverage = run.get("coverage") or {}
    if coverage.get("passed_exclusions") is not None:
        print(
            f"  kline cache coverage: stocks that passed exclusions AND have "
            f"full history {coverage.get('kline_full_history', 0)} / "
            f"passed_exclusions {coverage['passed_exclusions']}"
        )


def _print_kline_reasons(run: dict | None) -> None:
    """F7: per-symbol kline outcome counts + up to 5 example codes each."""
    reasons = (run or {}).get("kline_reasons")
    counts = (reasons or {}).get("counts") or {}
    if not reasons or not any(counts.values()):
        return
    try:
        from src.market_scanner.sources.kline import KLINE_REASONS as _ORDER
    except ImportError:
        _ORDER = tuple(sorted(counts))
    examples = (reasons or {}).get("examples") or {}
    print("KLINE-REASONS  (per-symbol fetch outcomes; counts by cause)")
    for key in _ORDER:
        n = counts.get(key, 0)
        if not n:
            continue
        codes = examples.get(key) or []
        shown = ",".join(codes[:5]) + ("..." if len(codes) > 5 else "")
        print(f"  {key:<28}{n:>8}  {shown}")


def _change_word(row: dict) -> str:
    """ASCII change marker (Windows console loses Unicode glyphs)."""
    raw = row.get("change")
    if raw in ("new", "kept", "manual", "restored"):
        return str(raw)
    if raw:
        return str(raw)
    return "-" if row.get("rank") is None else "kept"


def _apply_changes_to_picks(run: dict, changes: dict | None) -> dict:
    """Stamp the service-computed change labels onto each pick.

    ``pipeline._make_pick`` never fills ``pick['change']`` -- the diff lives
    in the store (scan_picks.change) and in the service's ``counts``. Rendering
    straight from the run payload therefore printed every pick as ``kept``
    (the ``_change_word`` fallback), hiding new entries in scan/review output.
    ``changes`` is the service's ``{"new": [...], "kept": [...]}`` mapping for
    this run; ``None`` leaves the run untouched (older runs, --json).
    """
    if not changes:
        return run
    labels: dict[str, str] = {}
    for kind in ("new", "kept"):
        for row in changes.get(kind, []) or []:
            if isinstance(row, dict):
                code = row.get("code")
            else:
                code = row[0]
            if code:
                labels[code] = kind
    if not labels:
        return run
    picks = run.get("picks")
    if not isinstance(picks, list) or not picks:
        return run
    merged = dict(run)
    merged["picks"] = [
        {**p, "change": labels.get(p.get("code"), p.get("change") or "kept")}
        for p in picks
    ]
    return merged


def _cmd_scan(args: argparse.Namespace, svc) -> int:
    home = _cli_home("watchlist scan")
    overrides: dict[str, Any] = {}
    if args.top is not None:
        overrides["top_n"] = args.top
    if args.exclude_star:
        overrides["include_star"] = False
    if args.exclude_chinext:
        overrides["include_chinext"] = False

    def progress(stage: str, fraction: float) -> None:
        print(f"  ... {stage} {fraction * 100:5.1f}%")

    t0 = _now()
    llm = None if args.no_ai else _default_llm()
    summary = svc.run_scan(
        trigger="cli",
        config_overrides=overrides or None,
        progress=progress,
        llm=llm,
    )
    total_s = (_now() - t0).total_seconds()

    run = svc.get_run(summary["run_id"]) or {}
    if args.json:
        payload = dict(run)
        payload["counts"] = summary.get("counts")
        payload["changes"] = summary.get("changes")
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    print()
    _print_home_header(home, "watchlist scan")
    _print_funnel(run.get("funnel"))
    _print_coverage(run)
    _print_sources(run.get("sources"))
    _print_kline_reasons(run)
    _print_requests(run)
    _print_picks(_apply_changes_to_picks(run, summary.get("changes")))
    _print_extras(run)
    _print_run_summary(run, total_s)
    return 0


def _cmd_review(args: argparse.Namespace, svc) -> int:
    """Re-run only the AI review for the latest (or one given) successful run."""
    _cli_home("watchlist review")

    def progress(stage: str, fraction: float) -> None:
        if not args.json:  # keep --json output machine-readable
            print(f"  ... {stage} {fraction * 100:5.1f}%")

    t0 = _now()
    llm = _default_llm()
    run_id = getattr(args, "run_id", None)
    from src.watchlist.service import ScanAlreadyRunning

    try:
        summary = svc.rerun_ai_review(
            run_id, trigger="cli", progress=progress, llm=llm
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    except ScanAlreadyRunning as exc:
        print(f"ERROR: {exc}")
        return 3
    total_s = (_now() - t0).total_seconds()

    run = svc.get_run(summary["run_id"]) or {}
    if args.json:
        payload = dict(run)
        payload["counts"] = summary.get("counts")
        payload["changes"] = summary.get("changes")
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    print()
    _print_picks(_apply_changes_to_picks(run, summary.get("changes")))
    _print_extras(run)
    _print_run_summary(run, total_s)
    print(
        f"review_revision={run.get('review_revision') or 1} "
        f"reviewed_at={run.get('reviewed_at') or '-'}"
    )
    return 0


def _default_llm():
    """The configured ChatLLM when available; None means skip the AI review."""
    try:
        from src.providers.chat import ChatLLM

        return ChatLLM()
    except Exception:  # noqa: BLE001 — no LLM configured: scan without AI
        return None


def _cmd_list(args: argparse.Namespace, svc) -> int:
    home = _cli_home("watchlist list")
    view = svc.get_watchlist()
    if getattr(args, "json", False):
        print(json.dumps(view, ensure_ascii=False, indent=2, default=str))
        return 0
    _print_home_header(home, "watchlist list")
    print(_DISCLAIMER)
    if view["stale"]["is_stale"]:
        print(f"STALE: no successful scan for {view['stale']['latest_trading_day']}")
    latest = view.get("latest_run")
    if latest:
        print(
            f"latest run {latest['run_id']} {latest['trade_date']} "
            f"status={latest['status']}"
        )
    last_failed = view.get("last_failed_run")
    if last_failed:
        print(
            f"last failed run {last_failed['run_id']} {last_failed['trade_date']} "
            f"status=failed reason={last_failed.get('failure_reason') or '-'}"
        )
    print()
    hdr = (
        f"  {'rk':>3}{'chg':<9}{'code':<8}{'name':<12}{'org':<7}"
        f"{'score':>7}  scores?"
    )
    print(hdr)
    for item in view["items"]:
        score_txt = (
            f"{item['score']:>7.1f}" if item.get("score") is not None else "     -"
        )
        scores_txt = (
            "ok" if item["scores_available"] else (item.get("unscored_reason") or "-")
        )
        print(
            f"  {item.get('rank') or 0:>3}{_change_word(item):<9}{item['code']:<8}"
            f"{(item.get('name') or ''):<12}{item['origin']:<7}{score_txt}  "
            f"{scores_txt}"
        )
    if view["requalified"]:
        print("\nREQUALIFIED (qualifies again - you removed it)")
        for r in view["requalified"]:
            print(
                f"  {r['code']} {r.get('name')} rank={r.get('rank')} "
                f"score={r.get('score')}"
            )
    if view["dropped"]:
        print("\nDROPPED (vs previous run)")
        for d in view["dropped"]:
            print(f"  {d['code']} {d.get('name')} prev_rank={d.get('prev_rank')} "
                  f"reason={d.get('reason')}")
            verdict = d.get("ai_verdict")
            reason = d.get("ai_reason")
            if verdict or reason:
                cites = d.get("ai_cites") or []
                cite_txt = f" [{', '.join(cites)}]" if cites else ""
                verdict_txt = verdict or "-"
                reason_txt = f" - {reason}" if reason else ""
                print(f"      AI: {verdict_txt}{reason_txt}{cite_txt}")
    return 0


def _cmd_add(args: argparse.Namespace, svc) -> int:
    _cli_home("watchlist add")
    try:
        item = svc.add_manual(args.code, note=args.note)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"added {item['code']} ({item.get('name') or 'unknown name'}) "
        f"origin={item['origin']}"
    )
    return 0


def _cmd_remove(args: argparse.Namespace, svc) -> int:
    _cli_home("watchlist remove")
    svc.remove(args.code)
    print(f"removed {args.code} (suppressed; will not be auto re-added)")
    return 0


def _cmd_restore(args: argparse.Namespace, svc) -> int:
    _cli_home("watchlist restore")
    try:
        item = svc.restore(args.code)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"restored {args.code} (change shown as 'restored')")
    return 0


def _cmd_forget(args: argparse.Namespace, svc) -> int:
    """F9: delete a manual test pin entirely (no suppression left behind)."""
    _cli_home("watchlist forget")
    try:
        item = svc.forget_manual(args.code)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"forgot {item.get('code')} ({item.get('name') or 'unknown name'}) "
        f"origin=manual (row deleted, no suppression left)"
    )
    return 0


def _cmd_history(args: argparse.Namespace, svc) -> int:
    _cli_home("watchlist history")
    runs = svc.list_runs(limit=args.limit)
    if getattr(args, "json", False):
        print(json.dumps(runs, ensure_ascii=False, indent=2, default=str))
        return 0
    for run in runs:
        line = (
            f"{run['run_id']} {run.get('trade_date')} status={run['status']} "
            f"ai={run.get('ai_status')} {run.get('duration_s')}s "
            f"trigger={run.get('trigger')}"
        )
        if run.get("failure_reason"):
            line += f" reason={run['failure_reason']}"
        print(line)
    return 0


def _cmd_invalidate(args: argparse.Namespace, svc) -> int:
    _cli_home("watchlist invalidate")
    try:
        result = svc.invalidate_run(args.run_id, reason=args.reason)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"run {result['run_id']} status=failed "
        f"reason={result['failure_reason']}"
    )
    if result.get("restored"):
        print(
            f"watchlist restored to {result.get('restored_from') or 'empty'} "
            f"(added={result['restored_counts']['added']} "
            f"removed={result['restored_counts']['removed']})"
        )
    else:
        print("watchlist unchanged (run was not the latest successful run)")
    return 0


def _cmd_show(args: argparse.Namespace, svc) -> int:
    _cli_home("watchlist show")
    run = svc.get_run(args.run_id)
    if run is None:
        print(f"run not found: {args.run_id}")
        return 1
    print(json.dumps(run, ensure_ascii=False, indent=2, default=str))
    return 0


def _cmd_schedule(args: argparse.Namespace, svc) -> int:
    home = _cli_home("watchlist schedule")
    if args.schedule_cmd == "on":
        patch: dict[str, Any] = {"schedule": {"enabled": True}}
        if args.time:
            patch["schedule"]["time"] = args.time
        svc.update_settings(patch)
    elif args.schedule_cmd == "off":
        svc.update_settings({"schedule": {"enabled": False}})
    settings = svc.get_settings()["schedule"]
    _print_home_header(home, "watchlist schedule")
    print(
        f"schedule enabled={settings['enabled']} time={settings['time']} "
        f"timezone={settings['timezone']} catch_up={settings['catch_up']}"
    )
    if args.schedule_cmd == "status":
        from src.watchlist.scheduler import scheduler_log_path

        print(f"schedule log_path={scheduler_log_path()}")
    return 0


# ---------------------------------------------------------------------------
# registration (called from cli._legacy)
# ---------------------------------------------------------------------------
def add_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the ``watchlist`` subcommand tree."""
    parser = subparsers.add_parser("watchlist", help="Watchlist: scan/list/manage")
    sub = parser.add_subparsers(dest="watchlist_cmd", required=True)

    scan = sub.add_parser("scan", help="Run one full scan now")
    scan.add_argument("--no-ai", action="store_true", help="Skip the AI review step")
    scan.add_argument("--top", type=int, default=None, help="Override top_n")
    scan.add_argument("--exclude-star", action="store_true", help="Exclude STAR board")
    scan.add_argument("--exclude-chinext", action="store_true", help="Exclude ChiNext")
    scan.add_argument("--json", action="store_true", help="Print the full run JSON")
    scan.set_defaults(func=_cmd_scan)

    lst = sub.add_parser("list", help="Show the current watchlist view")
    lst.add_argument("--json", action="store_true", help="Print the view as JSON")
    lst.set_defaults(func=_cmd_list)

    add = sub.add_parser("add", help="Manually add a code (pinned)")
    add.add_argument("code")
    add.add_argument("--note", default=None)
    add.set_defaults(func=_cmd_add)

    remove = sub.add_parser("remove", help="Remove a code (suppressed)")
    remove.add_argument("code")
    remove.set_defaults(func=_cmd_remove)

    forget = sub.add_parser(
        "forget",
        help="Delete a manual add entirely (F9 clean undo; refuses auto rows)",
    )
    forget.add_argument("code")
    forget.set_defaults(func=_cmd_forget)

    restore = sub.add_parser("restore", help="Restore a removed code")
    restore.add_argument("code")
    restore.set_defaults(func=_cmd_restore)

    hist = sub.add_parser("history", help="Run history")
    hist.add_argument("--limit", type=int, default=20)
    hist.add_argument("--json", action="store_true", help="Print history as JSON")
    hist.set_defaults(func=_cmd_history)

    inv = sub.add_parser(
        "invalidate", help="Mark a stored run failed and restore the watchlist"
    )
    inv.add_argument("run_id")
    inv.add_argument(
        "--reason",
        default=None,
        help="Failure reason (default: re-evaluated coverage or a standard note)",
    )
    inv.set_defaults(func=_cmd_invalidate)

    show = sub.add_parser("show", help="Print one run's JSON")
    show.add_argument("run_id")
    show.add_argument("--json", action="store_true", help="Print as JSON (default)")
    show.set_defaults(func=_cmd_show)

    review = sub.add_parser(
        "review",
        help="Re-run only the AI review for the latest successful run "
        "(or RUN_ID); never re-scans the market",
    )
    review.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="Run to re-review (default: the latest successful run)",
    )
    review.add_argument(
        "--json", action="store_true", help="Print the updated run JSON"
    )
    review.set_defaults(func=_cmd_review)

    sched = sub.add_parser("schedule", help="Schedule on|off|status")
    sched.add_argument("schedule_cmd", choices=["on", "off", "status"])
    sched.add_argument("--time", default=None, help="Fire time HH:MM (with 'on')")
    sched.set_defaults(func=_cmd_schedule)

    backfill = sub.add_parser(
        "backfill",
        help="Fetch missing/stale kline histories for the current universe",
    )
    backfill.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Kline refetch rate, requests/second (default 1.0)",
    )
    backfill.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max symbols to warm this run (default all)",
    )
    backfill.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the shortfall only; send no requests",
    )
    backfill.set_defaults(func=_cmd_backfill)
    return parser


def dispatch(args: argparse.Namespace) -> int:
    """Entry point from cli._legacy: run the chosen watchlist subcommand."""
    _ensure_utf8_stdout()
    try:
        svc = _default_service()
    except Exception as exc:  # noqa: BLE001 — HomeMismatch matched by name
        if type(exc).__name__ == "HomeMismatch":
            print(str(exc), file=sys.stderr)
            return 2
        raise
    return args.func(args, svc)


def _backfill_log_path(home: dict[str, Any] | None) -> Any:
    """`<home>/watchlist/backfill.log` (falls back to the runtime root)."""
    from pathlib import Path as _Path

    if home and home.get("home"):
        return _Path(home["home"]) / "watchlist" / "backfill.log"
    from src.config.paths import get_runtime_root

    return get_runtime_root() / "watchlist" / "backfill.log"


def _append_backfill_log(path: Any, line: str) -> None:
    """Append one progress line (best-effort; warns instead of failing)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        print(f"WARNING: backfill log unwritable ({path}): {exc}")


def _cmd_backfill(args: argparse.Namespace, svc=None, *,
                  universe_fn: Any | None = None,
                  calendar_fn: Any | None = None,
                  cache: Any | None = None,
                  warm_runner: Any | None = None) -> int:
    """F5c: assess the kline shortfall, then warm the cache (resumable).

    Missing, stale and gaps of any length are filled via
    ``assess_shortfall``/``warm_cache`` (tail-fill + full refetch); the job
    lock (kind="backfill") is held while warming and progress lands in
    ``<home>/watchlist/backfill.log``. ``--dry-run`` prints the shortfall
    only and sends no kline requests.
    """
    import datetime as _dt

    home = _cli_home("watchlist backfill")
    _print_home_header(home, "watchlist backfill")
    svc = svc or _default_service()

    from src.market_scanner.sources import universe as universe_mod

    fetch_universe = universe_fn or universe_mod.fetch_universe
    uni, uni_status = fetch_universe()
    if uni is None or getattr(uni, "empty", True):
        print("ERROR: universe fetch failed — cannot backfill")
        return 1
    print(f"universe: {uni_status.rows} rows ({uni_status.latency_s:.0f}s)")

    # early exclusions (no AI, no fundamentals): drop BSE/ST/delisted/suspended
    from src.market_scanner.sources.kline_cache import KlineCache

    df = uni
    df = df[df["board"] != "bse"]
    df = df[~df["is_st"] & ~df["is_delisting"] & ~df["is_suspended"]]
    symbols = [str(s) for s in df["symbol"].tolist() if s]
    print(f"after early exclusions: {len(symbols)} symbols")

    from src.market_scanner.sources.calendar import load_calendar

    if calendar_fn is not None:
        cal, _cal_status = calendar_fn()
    else:
        cal, _cal_status = load_calendar()
    now = _now()
    try:
        trade_date = cal.latest_completed_trading_day(now)
    except Exception:  # noqa: BLE001 — calendar must not kill backfill
        trade_date = now.date()
        print(f"WARNING: trading calendar unavailable; using {trade_date}")

    cache = cache or KlineCache()

    from src.market_scanner.sources.kline import assess_shortfall, warm_cache

    short = assess_shortfall(symbols, trade_date, calendar=cal, cache=cache)
    rate = min(float(getattr(args, "rate", 1.0) or 1.0), 1.0)
    print(
        f"shortfall for trade_date={trade_date}: missing={short.missing} "
        f"gap={short.gap} stale={short.stale} "
        f"appendable={short.appendable} est_requests={short.est_requests} "
        f"(~{short.est_requests / max(rate, 0.1) / 60:.0f} min at "
        f"{rate} req/s)"
    )
    if getattr(args, "dry_run", False):
        print("dry run: no requests sent")
        return 0
    max_n = getattr(args, "max", None)
    if max_n:
        print(f"capped to --max {max_n}")

    acquire = getattr(svc, "acquire_job", None)
    if acquire is None:
        print("ERROR: the watchlist service has no job lock (outdated)")
        return 3
    from src.watchlist.service import ScanAlreadyRunning

    try:
        handle = acquire("backfill", owner="cli")
    except ScanAlreadyRunning as exc:
        print(f"ERROR: {exc}")
        return 3

    log_path = _backfill_log_path(home)
    _append_backfill_log(
        log_path,
        f"{_dt.datetime.now().isoformat(timespec='seconds')} backfill "
        f"started trade_date={trade_date} symbols={len(symbols)} "
        f"est_requests={short.est_requests} "
        f"est_seconds={round(short.est_seconds, 1)} rate={rate} max={max_n}")

    total_holder: dict[str, int] = {"n": 0}

    def progress(done: int, total: int) -> None:
        total_holder["n"] = total
        if done % 25 == 0 or done >= total:
            remaining = max(total - done, 0)
            print(
                f"  {done}/{total} (~{remaining / max(rate, 0.1) / 60:.0f} "
                f"min left)", flush=True)
        if done == total or done % 100 == 0:
            _append_backfill_log(
                log_path,
                f"{_dt.datetime.now().isoformat(timespec='seconds')} "
                f"backfill progress {done}/{total}")

    run = warm_runner or warm_cache
    try:
        if warm_runner is None:
            # Production path: full engine wiring (calendar, cache, --max).
            result = warm_cache(
                symbols, trade_date, rate_per_sec=rate, progress=progress,
                lock_check=handle.check, stop=None, calendar=cal,
                cache=cache, max_symbols=max_n)
        else:
            # Test-double protocol: (symbols, trade_date, *, rate_per_sec,
            # progress, lock_check, stop, max_symbols).
            result = run(
                symbols, trade_date, rate_per_sec=rate, progress=progress,
                lock_check=handle.check, stop=None, max_symbols=max_n)
    except Exception as exc:  # noqa: BLE001 — never strand the job lock
        handle.release()
        print(f"ERROR: backfill aborted: {type(exc).__name__}: {exc}")
        _append_backfill_log(
            log_path,
            f"{_dt.datetime.now().isoformat(timespec='seconds')} backfill "
            f"aborted error={type(exc).__name__}: {exc}"[:300])
        return 1
    try:
        detail = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    except Exception:  # noqa: BLE001
        detail = {}
    finally:
        handle.release()
    _append_backfill_log(
        log_path,
        f"{_dt.datetime.now().isoformat(timespec='seconds')} backfill "
        f"finished trade_date={trade_date} " +
        " ".join(f"{k}={detail.get(k)}" for k in (
            "attempted", "filled", "failed", "skipped_known_missing",
            "requests", "duration_s", "stopped_unavailable", "aborted")))
    print(
        f"backfill done: attempted={detail.get('attempted')} "
        f"filled={detail.get('filled')} failed={detail.get('failed')} "
        f"requests={detail.get('requests')} "
        f"duration_s={detail.get('duration_s')}")
    if detail.get("aborted"):
        print(f"backfill aborted: {detail.get('error') or 'lock lost'}")
        return 1
    if detail.get("stopped_unavailable"):
        print("backfill stopped: price source unavailable "
              f"({detail.get('error') or 'outage'}); resume by re-running")
        return 1
    remaining = (detail.get("attempted") or 0) - (detail.get("filled") or 0)
    print(f"remaining after this run: {remaining} (resume by re-running)")
    return 0


def _now():
    import datetime as _dt

    return _dt.datetime.now()
