import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import {
  AlertTriangle,
  Bot,
  CalendarClock,
  ChevronDown,
  ChevronRight,
  Database,
  Eye,
  Loader2,
  Plus,
  RefreshCw,
  RotateCcw,
  Trash2,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import {
  api,
  ApiError,
  isHomeMismatchError,
  type WatchlistAi,
  type WatchlistDropped,
  type WatchlistEvidence,
  type WatchlistItem,
  type WatchlistRequalified,
  type WatchlistRunSummary,
  type WatchlistScanStatus,
  type WatchlistScheduleStatus,
  type WatchlistSettings,
  type WatchlistSourceStatus,
  type WatchlistStale,
  type WatchlistView,
} from "@/lib/api";

const fieldClass =
  "w-full rounded-md border bg-background px-3 py-2 text-sm outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20 disabled:cursor-not-allowed disabled:opacity-60";
const labelClass = "text-sm font-medium";
const hintClass = "text-xs text-muted-foreground";

const SCAN_POLL_MS = 2_000;

/** Board → short tag. Unknown boards fall back to the raw value. */
function boardTag(board?: string | null): { label: string; tone: string } {
  switch (board) {
    case "main_sh":
    case "SH":
      return { label: "Main SH", tone: "bg-muted text-muted-foreground" };
    case "main_sz":
    case "SZ":
      return { label: "Main SZ", tone: "bg-muted text-muted-foreground" };
    case "star":
    case "科创板":
      return { label: "STAR", tone: "bg-primary/10 text-primary" };
    case "chinext":
    case "创业板":
      return { label: "ChiNext", tone: "bg-warning/10 text-warning" };
    default:
      return { label: board ?? "—", tone: "bg-muted text-muted-foreground" };
  }
}

function subScore(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(1);
}

/** "review failed: quota (resets HH:MM)" → { isQuota, time }. */
function parseAiNote(note?: string | null): { isQuota: boolean; time: string | null } {
  if (!note) return { isQuota: false, time: null };
  const match = note.match(/resets (\d{1,2}:\d{2})/);
  return { isQuota: /review failed: quota/i.test(note), time: match ? match[1] : null };
}

/** Localized quota-failure line; null when the note is not a quota failure. */
function quotaNoteLabel(note: string | null | undefined, t: TFunction): string | null {
  const parsed = parseAiNote(note);
  if (!parsed.isQuota) return null;
  return t("watchlist.aiQuotaNote", { time: parsed.time ?? "—" });
}

function formatDateTime(value: string | null | undefined, locale: string): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(locale, { dateStyle: "medium", timeStyle: "short" });
}

function formatSeconds(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value.toFixed(1)}s`;
}

function StatusPill({
  label,
  tone,
}: {
  label: string;
  tone: "success" | "danger" | "warning" | "neutral" | "primary";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-2 py-0.5 text-xs font-medium",
        tone === "success" && "bg-success/10 text-success",
        tone === "danger" && "bg-danger/10 text-danger",
        tone === "warning" && "bg-warning/10 text-warning",
        tone === "primary" && "bg-primary/10 text-primary",
        tone === "neutral" && "bg-muted text-muted-foreground",
      )}
    >
      {label}
    </span>
  );
}

function changeBadge(change: WatchlistItem["change"], t: TFunction): ReactNode {
  switch (change) {
    case "new":
      return <StatusPill label={t("watchlist.changeNew")} tone="success" />;
    case "kept":
      return <StatusPill label={t("watchlist.changeKept")} tone="neutral" />;
    case "restored":
      return <StatusPill label={t("watchlist.changeRestored")} tone="primary" />;
    case "manual":
      return <StatusPill label={t("watchlist.changeManual")} tone="warning" />;
    default:
      return null;
  }
}

function aiBadge(ai: WatchlistAi | null | undefined, t: TFunction): ReactNode {
  if (!ai || ai.status === "not_reviewed") {
    return <StatusPill label={t("watchlist.aiNotReviewed")} tone="neutral" />;
  }
  if (ai.status === "failed") {
    // Quota failures render distinctly (warning, not danger): they resolve
    // themselves at the provider reset time and a re-review can fix them.
    if (ai.failure_kind === "quota") {
      return <StatusPill label={t("watchlist.aiQuota")} tone="warning" />;
    }
    return <StatusPill label={t("watchlist.aiFailed")} tone="danger" />;
  }
  switch (ai.verdict) {
    case "watch":
      return <StatusPill label={t("watchlist.aiWatch")} tone="success" />;
    case "neutral":
      return <StatusPill label={t("watchlist.aiNeutral")} tone="warning" />;
    case "avoid":
      return <StatusPill label={t("watchlist.aiAvoid")} tone="danger" />;
    default:
      return <StatusPill label={t("watchlist.aiReviewed")} tone="primary" />;
  }
}

/** "AI: avoid - <reason> [N1, A2]" for a stock the AI demoted into the dropped list. */
function droppedAiLabel(d: WatchlistDropped, t: TFunction): string {
  const verdict = d.ai_verdict ?? "";
  const reason = d.ai_reason ?? "";
  const head = reason
    ? t("watchlist.droppedAi", { verdict, reason })
    : t("watchlist.droppedAiVerdictOnly", { verdict });
  const cites = d.ai_cites ?? [];
  if (cites.length === 0) return head;
  return `${head} [${cites.join(", ")}]`;
}
function staleBanner(stale: WatchlistStale | null | undefined, t: TFunction): ReactNode {
  if (!stale?.is_stale) return null;
  return (
    <div
      role="status"
      className="flex items-start gap-2 rounded-lg border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-warning"
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div>
        <p className="font-medium">{t("watchlist.staleTitle")}</p>
        <p className={hintClass}>
          {t("watchlist.staleBody", {
            latestDay: stale.latest_trading_day ?? "—",
            lastScanDay: stale.last_scan_trade_date ?? "—",
          })}
        </p>
      </div>
    </div>
  );
}

/** F1: loud red banner when the API refused with a HomeMismatch 503. The
 * message is shown verbatim (it names the expected + resolved home and the
 * fix); Run scan / Re-run review are disabled while it is present. */
function homeMismatchBanner(message: string, t: TFunction): ReactNode {
  return (
    <div
      role="alert"
      data-testid="watchlist-home-mismatch"
      className="flex items-start gap-3 rounded-lg border-2 border-danger bg-danger/15 px-4 py-3 text-sm text-danger"
    >
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden />
      <div className="space-y-1">
        <p className="font-semibold">{t("watchlist.homeMismatchTitle")}</p>
        <p data-testid="watchlist-home-mismatch-message" className="font-mono text-xs break-words">
          {message}
        </p>
      </div>
    </div>
  );
}

function sourceStatusLabel(
  status: WatchlistSourceStatus["status"],
  t: TFunction,
): string {
  switch (status) {
    case "ok":
      return t("watchlist.sourceOk");
    case "degraded":
      return t("watchlist.sourceDegraded");
    case "failed":
      return t("watchlist.sourceFailed");
    default:
      return status;
  }
}

/** Short date for an evidence item; "—" when absent. */
function evidenceDate(value: string | null | undefined, locale: string): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(locale, { dateStyle: "medium" });
}

/**
 * One resolved citation. The id badge matches a `cites` entry elsewhere in the
 * AI block; the title becomes a link only when the run supplied a url.
 */
function evidenceRow(e: WatchlistEvidence, locale: string, t: TFunction): ReactNode {
  const title = e.title_or_text ?? e.name ?? "—";
  const kindLabel = t(`watchlist.evidenceKind_${e.kind}` as never, { defaultValue: e.kind });
  return (
    <li key={e.id} className="flex flex-wrap items-baseline gap-2 text-sm">
      <span
        className="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-xs text-primary"
        data-testid={`evidence-id-${e.id}`}
      >
        {e.id}
      </span>
      <span className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">{kindLabel}</span>
      <span className="text-xs text-muted-foreground">{evidenceDate(e.date, locale)}</span>
      <span className="min-w-0 flex-1 text-muted-foreground">
        {e.url ? (
          <a
            href={e.url}
            target="_blank"
            rel="noreferrer"
            className="underline underline-offset-2 hover:text-foreground"
          >
            {title}
          </a>
        ) : (
          title
        )}
      </span>
      {e.source && <span className="text-xs text-muted-foreground">{e.source}</span>}
      {e.value != null && <span className="text-xs tabular-nums text-muted-foreground">{String(e.value)}</span>}
    </li>
  );
}

/**
 * Watchlist page: latest market-scan run, the auto/manual picks, the diff
 * sections (dropped / requalified / AI-demoted), a run-now action with live
 * progress, and the schedule card.
 *
 * Research-only surface: every view carries the disclaimer line and no action
 * here places an order or changes a mandate.
 */
export function Watchlist() {
  const { t, i18n } = useTranslation();
  const locale = i18n.language;

  const [view, setView] = useState<WatchlistView | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  // F1: the API's HomeMismatch 503 (message verbatim). When set, the page
  // shows a loud red banner and disables Run scan / Re-run review.
  const [homeMismatch, setHomeMismatch] = useState<string | null>(null);

  const [scanStatus, setScanStatus] = useState<WatchlistScanStatus | null>(null);
  const [scanStarting, setScanStarting] = useState(false);
  const [reviewStarting, setReviewStarting] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const scanPollRef = useRef<number | null>(null);

  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [funnelOpen, setFunnelOpen] = useState(false);

  const [removeTarget, setRemoveTarget] = useState<WatchlistItem | null>(null);
  const [removing, setRemoving] = useState(false);

  const [addCode, setAddCode] = useState("");
  const [addNote, setAddNote] = useState("");
  const [adding, setAdding] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);

  const [runs, setRuns] = useState<WatchlistRunSummary[]>([]);
  const [runsOpen, setRunsOpen] = useState(false);

  const [schedule, setSchedule] = useState<WatchlistScheduleStatus | null>(null);
  const [settings, setSettings] = useState<WatchlistSettings | null>(null);
  const [scheduleTime, setScheduleTime] = useState("18:00");
  const [scheduleTz, setScheduleTz] = useState("Asia/Shanghai");
  const [scheduleSaving, setScheduleSaving] = useState(false);
  const [scheduleEnabledSaving, setScheduleEnabledSaving] = useState(false);

  const refreshView = useCallback(async () => {
    try {
      const data = await api.getWatchlist();
      setView(data.watchlist);
      setLoadError(null);
      setHomeMismatch(null);
    } catch (error) {
      if (isHomeMismatchError(error)) {
        setHomeMismatch(error.message);
        setLoadError(null);
      } else {
        setLoadError(error instanceof ApiError ? error.message : String(error));
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshSchedule = useCallback(async () => {
    try {
      const [sched, st] = await Promise.all([api.getWatchlistSchedule(), api.getWatchlistSettings()]);
      setSchedule(sched);
      setSettings(st.settings);
      setScheduleTime(sched.time ?? st.settings.schedule.time ?? "18:00");
      setScheduleTz(sched.timezone ?? st.settings.schedule.timezone ?? "Asia/Shanghai");
    } catch {
      // Schedule card is non-critical; the main view already surfaced errors.
    }
  }, []);

  useEffect(() => {
    void refreshView();
    void refreshSchedule();
  }, [refreshView, refreshSchedule]);

  const stopScanPoll = useCallback(() => {
    if (scanPollRef.current !== null) {
      window.clearInterval(scanPollRef.current);
      scanPollRef.current = null;
    }
  }, []);

  const pollScan = useCallback(async () => {
    try {
      const status = await api.getWatchlistScanStatus();
      setScanStatus(status);
      if (!status.running) {
        stopScanPoll();
        await refreshView();
        if (status.error) setScanError(status.error);
      }
    } catch (error) {
      // Transient poll failure: keep polling; the next tick retries. A home
      // mismatch is not transient, so surface it and stop.
      if (isHomeMismatchError(error)) {
        setHomeMismatch(error.message);
        stopScanPoll();
      }
    }
  }, [refreshView, stopScanPoll]);

  const startScanPolling = useCallback(() => {
    stopScanPoll();
    scanPollRef.current = window.setInterval(() => void pollScan(), SCAN_POLL_MS);
  }, [pollScan, stopScanPoll]);

  useEffect(() => stopScanPoll, [stopScanPoll]);

  async function handleRunScan() {
    setScanStarting(true);
    setScanError(null);
    try {
      await api.startWatchlistScan();
      const status = await api.getWatchlistScanStatus();
      setScanStatus(status);
      if (status.running) startScanPolling();
      else await refreshView();
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        toast.error(t("watchlist.scanAlreadyRunning"));
        const status = await api.getWatchlistScanStatus();
        setScanStatus(status);
        if (status.running) startScanPolling();
      } else if (isHomeMismatchError(error)) {
        setHomeMismatch(error.message);
      } else {
        setScanError(error instanceof ApiError ? error.message : String(error));
      }
    } finally {
      setScanStarting(false);
    }
  }

  async function handleRerunReview() {
    setReviewStarting(true);
    setScanError(null);
    try {
      await api.startWatchlistReview(latestRun?.run_id ?? undefined);
      toast.success(t("watchlist.reviewStarted"));
      const status = await api.getWatchlistScanStatus();
      setScanStatus(status);
      if (status.running) startScanPolling();
      else await refreshView();
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        toast.error(t("watchlist.reviewAlreadyRunning"));
        const status = await api.getWatchlistScanStatus();
        setScanStatus(status);
        if (status.running) startScanPolling();
      } else if (isHomeMismatchError(error)) {
        setHomeMismatch(error.message);
      } else {
        setScanError(error instanceof ApiError ? error.message : String(error));
      }
    } finally {
      setReviewStarting(false);
    }
  }

  async function handleRemove() {
    if (!removeTarget) return;
    setRemoving(true);
    try {
      await api.removeWatchlistItem(removeTarget.code);
      setRemoveTarget(null);
      await refreshView();
    } catch (error) {
      if (isHomeMismatchError(error)) setHomeMismatch(error.message);
      toast.error(error instanceof ApiError ? error.message : String(error));
    } finally {
      setRemoving(false);
    }
  }

  async function handleRestore(code: string) {
    try {
      await api.restoreWatchlistItem(code);
      await refreshView();
    } catch (error) {
      if (isHomeMismatchError(error)) setHomeMismatch(error.message);
      toast.error(error instanceof ApiError ? error.message : String(error));
    }
  }

  async function handleAdd(event: React.FormEvent) {
    event.preventDefault();
    setAddError(null);
    const code = addCode.trim();
    if (!code) {
      setAddError(t("watchlist.addCodeRequired"));
      return;
    }
    setAdding(true);
    try {
      await api.addWatchlistItem(code, addNote.trim() || null);
      setAddCode("");
      setAddNote("");
      await refreshView();
      toast.success(t("watchlist.addSuccess", { code }));
    } catch (error) {
      if (isHomeMismatchError(error)) setHomeMismatch(error.message);
      setAddError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setAdding(false);
    }
  }

  async function toggleScheduleEnabled(next: boolean) {
    setScheduleEnabledSaving(true);
    try {
      const updated = await api.updateWatchlistSettings({ schedule: { enabled: next } });
      setSettings(updated.settings);
      await refreshSchedule();
    } catch (error) {
      if (isHomeMismatchError(error)) setHomeMismatch(error.message);
      toast.error(error instanceof ApiError ? error.message : String(error));
    } finally {
      setScheduleEnabledSaving(false);
    }
  }

  async function saveScheduleTime(event: React.FormEvent) {
    event.preventDefault();
    setScheduleSaving(true);
    try {
      const updated = await api.updateWatchlistSettings({
        schedule: { time: scheduleTime, timezone: scheduleTz },
      });
      setSettings(updated.settings);
      await refreshSchedule();
      toast.success(t("watchlist.scheduleSaved"));
    } catch (error) {
      if (isHomeMismatchError(error)) setHomeMismatch(error.message);
      toast.error(error instanceof ApiError ? error.message : String(error));
    } finally {
      setScheduleSaving(false);
    }
  }

  async function loadRuns() {
    if (runsOpen) {
      setRunsOpen(false);
      return;
    }
    try {
      const body = await api.listWatchlistRuns(20);
      setRuns(body.runs);
      setRunsOpen(true);
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : String(error));
    }
  }

  function toggleExpanded(code: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(code)) next.delete(code);
      else next.add(code);
      return next;
    });
  }

  const latestRun = view?.latest_run ?? null;
  const scanning = scanStatus?.running === true || scanStarting || reviewStarting;
  // Boss rule: success = ok | partial — a partial run is still reviewable;
  // only a failed run blocks the re-review button and the gate below.
  const runReviewable = latestRun?.status === "ok" || latestRun?.status === "partial";
  // The latest run "needs review" when its AI pass is missing/partial or any
  // pick failed the review (quota/timeout/invalid_json/error).
  const needsReview =
    runReviewable &&
    (latestRun?.ai_status === "not_reviewed" ||
      latestRun?.ai_status === "partial" ||
      (view?.items ?? []).some((i) => i.ai?.failure_kind != null) ||
      (view?.ai_demoted ?? []).some((i) => i.ai?.failure_kind != null));

  function runSummaryStrip(): ReactNode {
    if (!latestRun) {
      return (
        <p className={hintClass} data-testid="watchlist-no-run">
          {t("watchlist.noRuns")}
        </p>
      );
    }
    const failedSources = (latestRun.sources ?? []).filter((s) => s.status === "failed" || s.status === "degraded");
    const aiLabel =
      latestRun.ai_status === "reviewed"
        ? t("watchlist.aiReviewed")
        : latestRun.ai_status === "partial"
          ? t("watchlist.aiPartial")
          : t("watchlist.aiNotReviewed");
    return (
      <div className="space-y-1.5" data-testid="watchlist-run-summary">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          {latestRun.intraday ? (
            <span
              className="rounded bg-warning/15 px-2 py-0.5 font-medium text-warning"
              data-testid="watchlist-intraday-banner"
            >
              {t("watchlist.intradayBanner")}
            </span>
          ) : (
            <span className="font-medium" data-testid="watchlist-data-as-of">
              {t("watchlist.dataAsOf", { date: latestRun.trade_date ?? "—" })}
            </span>
          )}
          <span className={hintClass}>
            {t("watchlist.finishedAt", { when: formatDateTime(latestRun.finished_at, locale) })}
          </span>
          <StatusPill
            label={t(`watchlist.trigger_${latestRun.trigger ?? "cli"}` as never)}
            tone="neutral"
          />
          {latestRun.status === "ok" ? (
            <StatusPill label={t("watchlist.statusOk")} tone="success" />
          ) : (
            <StatusPill
              label={latestRun.status === "partial" ? t("watchlist.statusPartial") : t("watchlist.statusFailed")}
              tone={latestRun.status === "partial" ? "warning" : "danger"}
            />
          )}
          <StatusPill label={aiLabel} tone={latestRun.ai_status === "reviewed" ? "primary" : "neutral"} />
          <span className={hintClass}>{formatSeconds(latestRun.duration_s)}</span>
          {runReviewable && (
            <button
              type="button"
              onClick={() => void handleRerunReview()}
              disabled={scanning || homeMismatch != null}
              data-testid="watchlist-rerun-review"
              className={cn(
                "ml-auto inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60",
                needsReview && "border-warning/40 text-warning",
              )}
            >
              {scanning ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
              ) : (
                <Bot className="h-3.5 w-3.5" aria-hidden />
              )}
              {scanning ? t("watchlist.reviewing") : t("watchlist.rerunReview")}
            </button>
          )}
        </div>
        {failedSources.length > 0 && (
          <p className="text-xs text-warning" data-testid="watchlist-failed-sources">
            {t("watchlist.failedSources", {
              sources: failedSources.map((s) => `${s.name} (${sourceStatusLabel(s.status, t)})`).join(", "),
            })}
          </p>
        )}
        {latestRun.ai_note &&
          (quotaNoteLabel(latestRun.ai_note, t) ? (
            <p
              className="flex items-center gap-1.5 text-xs text-warning"
              data-testid="watchlist-ai-quota-note"
            >
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden />
              {quotaNoteLabel(latestRun.ai_note, t)}
            </p>
          ) : (
            <p className={hintClass}>{latestRun.ai_note}</p>
          ))}
      </div>
    );
  }

  function funnelSection(): ReactNode {
    // The API serves funnel stages as objects {stage, count, removed, note}
    // (never tuples); every numeric access tolerates missing fields.
    const funnel = latestRun?.funnel ?? [];
    const coverage = latestRun?.coverage ?? null;
    if (funnel.length === 0) return null;
    const maxCount = Math.max(...funnel.map((e) => e.count ?? 0), 1);
    return (
      <section className="rounded-lg border bg-card" aria-label={t("watchlist.funnelTitle")}>
        <button
          type="button"
          onClick={() => setFunnelOpen((v) => !v)}
          className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium"
          aria-expanded={funnelOpen}
        >
          <span className="flex items-center gap-2">
            <Database className="h-4 w-4 text-muted-foreground" aria-hidden />
            {t("watchlist.funnelTitle")}
          </span>
          {funnelOpen ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" aria-hidden />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" aria-hidden />
          )}
        </button>
        {funnelOpen && (
          <div className="space-y-3 border-t px-4 py-3">
            <ul className="space-y-1.5">
              {funnel.map((entry) => (
                <li key={entry.stage}>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="w-40 shrink-0 truncate text-muted-foreground">{entry.stage}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded bg-muted">
                      <div
                        className="h-full rounded bg-primary/60"
                        style={{ width: `${((entry.count ?? 0) / maxCount) * 100}%` }}
                      />
                    </div>
                    <span className="w-12 text-right tabular-nums">{entry.count ?? 0}</span>
                    {(entry.removed ?? 0) > 0 && (
                      <span
                        className="w-12 text-right tabular-nums text-muted-foreground"
                        data-testid={`watchlist-funnel-removed-${entry.stage}`}
                      >
                        −{entry.removed}
                      </span>
                    )}
                  </div>
                  {entry.note && (
                    <p className={hintClass} data-testid={`watchlist-funnel-note-${entry.stage}`}>
                      {entry.note}
                    </p>
                  )}
                </li>
              ))}
            </ul>
            {coverage && (
              <p className={hintClass} data-testid="watchlist-coverage">
                {t("watchlist.coverageLine", {
                  trendScored: coverage.trend_scored ?? "—",
                  flowReal: coverage.flow_real ?? "—",
                  flowNeutral: coverage.flow_neutral ?? "—",
                  klineFull: coverage.kline_full_history ?? "—",
                  passed: coverage.passed_exclusions ?? "—",
                })}
              </p>
            )}
            {(latestRun?.sources ?? []).length > 0 && (
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-muted-foreground">
                    <th className="pb-1 font-medium">{t("watchlist.colSource")}</th>
                    <th className="pb-1 font-medium">{t("watchlist.colStatus")}</th>
                    <th className="pb-1 font-medium">{t("watchlist.colAsOf")}</th>
                    <th className="pb-1 font-medium">{t("watchlist.colNote")}</th>
                  </tr>
                </thead>
                <tbody>
                  {(latestRun?.sources ?? []).map((s) => (
                    <tr key={s.name} className="border-t">
                      <td className="py-1">{s.name}</td>
                      <td className="py-1">{sourceStatusLabel(s.status, t)}</td>
                      <td className="py-1 text-muted-foreground" data-testid={`watchlist-as-of-${s.name}`}>
                        {s.as_of ?? "—"}
                      </td>
                      <td className="py-1 text-muted-foreground">{s.detail ?? s.note ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </section>
    );
  }

  function detailRow(item: WatchlistItem): ReactNode {
    if (!expanded.has(item.code)) return null;
    const ai = item.ai;
    return (
      <tr key={`${item.code}-detail`}>
        <td colSpan={9} className="bg-muted/30 px-4 py-3">
          <div className="space-y-3 text-sm">
            {item.reasons && item.reasons.length > 0 && (
              <div>
                <p className="mb-1 font-medium">{t("watchlist.reasonsTitle")}</p>
                <ul className="list-disc space-y-0.5 pl-5 text-muted-foreground">
                  {item.reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            )}
            {ai?.summary && (
              <div>
                <p className="mb-1 flex items-center gap-1.5 font-medium">
                  <Bot className="h-4 w-4 text-primary" aria-hidden />
                  {t("watchlist.aiSummaryTitle")}
                </p>
                <p className="text-muted-foreground">{ai.summary}</p>
              </div>
            )}
            {ai && ai.reasons && ai.reasons.length > 0 && (
              <div>
                <p className="mb-1 font-medium">{t("watchlist.aiReasonsTitle")}</p>
                <ul className="list-disc space-y-0.5 pl-5 text-muted-foreground">
                  {ai.reasons.map((r, i) => (
                    <li key={i}>
                      {r.text}
                      {r.cites && r.cites.length > 0 && (
                        <span className="ml-1 text-xs text-primary">[{r.cites.join(", ")}]</span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {ai && ai.evidence && ai.evidence.length > 0 && (
              <div data-testid="watchlist-evidence">
                <p className="mb-1 font-medium">{t("watchlist.evidenceTitle")}</p>
                <ul className="space-y-1">
                  {ai.evidence.map((e) => evidenceRow(e, locale, t))}
                </ul>
              </div>
            )}
            {ai && ai.red_flags && ai.red_flags.length > 0 && (
              <div>
                <p className="mb-1 flex items-center gap-1.5 font-medium text-danger">
                  <XCircle className="h-4 w-4" aria-hidden />
                  {t("watchlist.redFlagsTitle")}
                </p>
                <ul className="list-disc space-y-0.5 pl-5 text-muted-foreground">
                  {ai.red_flags.map((r, i) => (
                    <li key={i}>
                      {r.text}
                      {r.cites && r.cites.length > 0 && (
                        <span className="ml-1 text-xs text-primary">[{r.cites.join(", ")}]</span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {ai && ai.news_checked === false && (
              <p className={hintClass}>{ai.note ?? t("watchlist.noNewsChecked")}</p>
            )}
            {ai && ai.status === "failed" && ai.note && (
              <p
                className={cn(quotaNoteLabel(ai.note, t) ? "text-warning" : hintClass)}
                data-testid={
                  quotaNoteLabel(ai.note, t) ? `watchlist-item-quota-note-${item.code}` : undefined
                }
              >
                {quotaNoteLabel(ai.note, t) ?? ai.note}
              </p>
            )}
            {item.note && <p className={hintClass}>{t("watchlist.noteLabel", { note: item.note })}</p>}
          </div>
        </td>
      </tr>
    );
  }

  function renderItemRow(item: WatchlistItem): ReactNode {
    const board = boardTag(item.board);
    return (
      <div key={item.code} data-testid={`watchlist-item-${item.code}`}>
        <div className="flex flex-wrap items-center gap-2 px-4 py-3 text-sm">
          <button
            type="button"
            onClick={() => toggleExpanded(item.code)}
            aria-expanded={expanded.has(item.code)}
            aria-label={t("watchlist.expandAria", { code: item.code })}
            className="rounded p-1 text-muted-foreground transition hover:bg-muted"
          >
            {expanded.has(item.code) ? (
              <ChevronDown className="h-4 w-4" aria-hidden />
            ) : (
              <ChevronRight className="h-4 w-4" aria-hidden />
            )}
          </button>
          <span className="w-8 tabular-nums text-muted-foreground">{item.rank ?? "—"}</span>
          {changeBadge(item.change, t)}
          <span className="w-16 font-mono">{item.code}</span>
          <span className="min-w-0 flex-1 truncate font-medium">{item.name}</span>
          <span className={cn("rounded px-1.5 py-0.5 text-xs", board.tone)}>{board.label}</span>
          {item.industry && <span className="hidden text-xs text-muted-foreground sm:inline">{item.industry}</span>}
          {item.scores_available ? (
            <>
              <span className="w-12 text-right font-medium tabular-nums">{subScore(item.score)}</span>
              <span className="flex gap-1 text-xs tabular-nums">
                <span className="rounded bg-muted px-1">F {subScore(item.sub_scores?.fundamental)}</span>
                <span className="rounded bg-muted px-1">T {subScore(item.sub_scores?.trend)}</span>
                <span className="rounded bg-muted px-1">Fl {subScore(item.sub_scores?.flow)}</span>
              </span>
            </>
          ) : (
            <span className="w-40 text-right text-xs text-muted-foreground">
              {item.unscored_reason ?? t("watchlist.unscored")}
            </span>
          )}
          {aiBadge(item.ai, t)}
          <button
            type="button"
            onClick={() => setRemoveTarget(item)}
            aria-label={t("watchlist.removeAria", { code: item.code, name: item.name })}
            className="rounded p-1 text-muted-foreground transition hover:bg-muted hover:text-danger"
          >
            <Trash2 className="h-4 w-4" aria-hidden />
          </button>
        </div>
        {detailRow(item)}
      </div>
    );
  }

  function droppedSection(): ReactNode {
    const dropped = view?.dropped ?? [];
    if (dropped.length === 0) return null;
    return (
      <section className="rounded-lg border bg-card" aria-label={t("watchlist.droppedTitle")}>
        <h2 className="border-b px-4 py-3 text-sm font-medium">{t("watchlist.droppedTitle")}</h2>
        <ul className="divide-y">
          {dropped.map((d: WatchlistDropped) => (
            <li key={d.code} className="flex items-center gap-3 px-4 py-2 text-sm">
              <span className="w-16 font-mono text-muted-foreground">{d.code}</span>
              <span className="flex-1">{d.name}</span>
              {d.ai_verdict && (
                <span className={hintClass} data-testid={`watchlist-dropped-ai-${d.code}`}>
                  {droppedAiLabel(d, t)}
                </span>
              )}
              {d.prev_rank != null && (
                <span className={hintClass}>{t("watchlist.prevRank", { rank: d.prev_rank })}</span>
              )}
              {d.reason && <span className="text-xs text-danger">{d.reason}</span>}
            </li>
          ))}
        </ul>
      </section>
    );
  }

  function requalifiedSection(): ReactNode {
    const requalified = view?.requalified ?? [];
    if (requalified.length === 0) return null;
    return (
      <section className="rounded-lg border bg-card" aria-label={t("watchlist.requalifiedTitle")}>
        <h2 className="border-b px-4 py-3 text-sm font-medium">{t("watchlist.requalifiedTitle")}</h2>
        <ul className="divide-y">
          {requalified.map((r: WatchlistRequalified) => (
            <li key={r.code} className="flex items-center gap-3 px-4 py-2 text-sm">
              <span className="w-16 font-mono text-muted-foreground">{r.code}</span>
              <span className="flex-1">{r.name}</span>
              {r.rank != null && <span className={hintClass}>#{r.rank}</span>}
              {r.score != null && <span className="tabular-nums">{subScore(r.score)}</span>}
              <span className={cn(hintClass, "hidden sm:inline")}>{r.label ?? t("watchlist.requalifiedLabel")}</span>
              <button
                type="button"
                onClick={() => void handleRestore(r.code)}
                aria-label={t("watchlist.restoreAria", { code: r.code, name: r.name })}
                className="inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs transition hover:bg-muted"
              >
                <RotateCcw className="h-3 w-3" aria-hidden />
                {t("watchlist.restore")}
              </button>
            </li>
          ))}
        </ul>
      </section>
    );
  }

  function aiDemotedSection(): ReactNode {
    const demoted = view?.ai_demoted ?? [];
    if (demoted.length === 0) return null;
    return (
      <section className="rounded-lg border bg-card" aria-label={t("watchlist.aiDemotedTitle")}>
        <h2 className="border-b px-4 py-3 text-sm font-medium">{t("watchlist.aiDemotedTitle")}</h2>
        <ul className="divide-y">
          {demoted.map((item) => (
            <li key={item.code} className="flex items-center gap-3 px-4 py-2 text-sm">
              <span className="w-16 font-mono text-muted-foreground">{item.code}</span>
              <span className="flex-1">{item.name}</span>
              {aiBadge(item.ai, t)}
              {item.ai?.red_flags && item.ai.red_flags.length > 0 && (
                <span className="truncate text-xs text-danger">
                  {item.ai.red_flags.map((f) => f.text).join("; ")}
                </span>
              )}
            </li>
          ))}
        </ul>
      </section>
    );
  }

  function scheduleCard(): ReactNode {
    if (!schedule && !settings) return null;
    const enabled = schedule?.enabled ?? settings?.schedule.enabled ?? false;
    return (
      <section className="rounded-lg border bg-card p-4" aria-label={t("watchlist.scheduleTitle")}>
        <div className="mb-3 flex items-center gap-2">
          <CalendarClock className="h-4 w-4 text-primary" aria-hidden />
          <h2 className="text-sm font-medium">{t("watchlist.scheduleTitle")}</h2>
        </div>
        <div className="space-y-3">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={enabled}
              disabled={scheduleEnabledSaving}
              onChange={(e) => void toggleScheduleEnabled(e.target.checked)}
              data-testid="watchlist-schedule-enabled"
            />
            {t("watchlist.scheduleEnabled")}
          </label>
          <form onSubmit={saveScheduleTime} className="flex flex-wrap items-end gap-3">
            <div className="space-y-1">
              <label htmlFor="watchlist-schedule-time" className={labelClass}>
                {t("watchlist.scheduleTime")}
              </label>
              <input
                id="watchlist-schedule-time"
                type="time"
                value={scheduleTime}
                onChange={(e) => setScheduleTime(e.target.value)}
                className={cn(fieldClass, "w-32")}
                disabled={!enabled}
              />
            </div>
            <div className="space-y-1">
              <label htmlFor="watchlist-schedule-tz" className={labelClass}>
                {t("watchlist.scheduleTimezone")}
              </label>
              <input
                id="watchlist-schedule-tz"
                value={scheduleTz}
                onChange={(e) => setScheduleTz(e.target.value)}
                className={cn(fieldClass, "w-44 font-mono")}
                disabled={!enabled}
              />
            </div>
            <button
              type="submit"
              disabled={!enabled || scheduleSaving}
              className="rounded-md border px-3 py-2 text-sm transition hover:bg-muted disabled:opacity-50"
            >
              {scheduleSaving ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : t("watchlist.scheduleSave")}
            </button>
          </form>
          <div className="space-y-1 text-xs text-muted-foreground">
            <p>{t("watchlist.scheduleNext", { when: formatDateTime(schedule?.next_run_at, locale) })}</p>
            <p>{t("watchlist.scheduleLastFired", { when: formatDateTime(schedule?.last_fired_at, locale) })}</p>
            {schedule?.last_result && <p>{t("watchlist.scheduleLastResult", { result: schedule.last_result })}</p>}
          </div>
        </div>
      </section>
    );
  }

  function runsCard(): ReactNode {
    return (
      <section className="rounded-lg border bg-card" aria-label={t("watchlist.runHistoryTitle")}>
        <button
          type="button"
          onClick={() => void loadRuns()}
          className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium"
          aria-expanded={runsOpen}
        >
          <span>{t("watchlist.runHistoryTitle")}</span>
          {runsOpen ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" aria-hidden />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" aria-hidden />
          )}
        </button>
        {runsOpen && (
          <div className="border-t">
            {runs.length === 0 ? (
              <p className="p-4 text-sm text-muted-foreground">{t("watchlist.noRuns")}</p>
            ) : (
              <ul className="divide-y">
                {runs.map((run) => (
                  <li key={run.run_id} className="flex flex-wrap items-center gap-2 px-4 py-2 text-sm">
                    <span className="font-mono text-xs text-muted-foreground">{run.run_id.slice(0, 8)}</span>
                    <span>
                      {run.trade_date ?? "—"}
                      {run.intraday ? (
                        <span
                          className="ml-1 rounded bg-warning/15 px-1 text-xs text-warning"
                          data-testid={`watchlist-run-intraday-${run.run_id.slice(0, 8)}`}
                        >
                          {t("watchlist.intradayBanner")}
                        </span>
                      ) : null}
                    </span>
                    <span className={hintClass}>{formatDateTime(run.finished_at, locale)}</span>
                    {run.status === "ok" ? (
                      <StatusPill label={t("watchlist.statusOk")} tone="success" />
                    ) : run.status === "partial" ? (
                      <StatusPill label={t("watchlist.statusPartial")} tone="warning" />
                    ) : (
                      <StatusPill label={t("watchlist.statusFailed")} tone="danger" />
                    )}
                    <span className={cn(hintClass, "ml-auto")}>{formatSeconds(run.duration_s)}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </section>
    );
  }

  const items = view?.items ?? [];

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Eye className="h-6 w-6 text-primary" aria-hidden />
          <div>
            <h1 className="text-xl font-semibold">{t("watchlist.title")}</h1>
            <p className={hintClass}>{t("watchlist.subtitle")}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => void handleRunScan()}
          disabled={scanning || homeMismatch != null}
          data-testid="watchlist-run-scan"
          className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {scanning ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <RefreshCw className="h-4 w-4" aria-hidden />}
          {scanning ? t("watchlist.scanning") : t("watchlist.runScan")}
        </button>
      </header>

      <p className="text-xs text-muted-foreground" data-testid="watchlist-disclaimer">
        {t("watchlist.disclaimer")}
      </p>

      {staleBanner(view?.stale, t)}

      {homeMismatch && homeMismatchBanner(homeMismatch, t)}

      {scanError && (
        <p role="alert" className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
          {scanError}
        </p>
      )}

      {scanning && scanStatus && (
        <section
          className="rounded-lg border bg-card p-4"
          data-testid={scanStatus.kind === "review" ? "watchlist-review-progress" : "watchlist-scan-progress"}
        >
          <div className="mb-2 flex items-center gap-2 text-sm font-medium">
            <Loader2 className="h-4 w-4 animate-spin text-primary" aria-hidden />
            {scanStatus.stage
              ? t(`watchlist.stage_${scanStatus.stage}` as never, { defaultValue: scanStatus.stage })
              : scanStatus.kind === "review"
                ? t("watchlist.reviewing")
                : t("watchlist.scanning")}
          </div>
          <div className="h-2 overflow-hidden rounded bg-muted">
            <div
              className="h-full rounded bg-primary transition-all"
              style={{ width: `${Math.round((scanStatus.progress ?? 0) * 100)}%` }}
            />
          </div>
        </section>
      )}

      {loading ? (
        <div className="flex h-56 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" aria-hidden />
        </div>
      ) : loadError && !view ? (
        <div className="space-y-2 rounded-lg border bg-card p-6 text-sm">
          <p className="text-danger">{loadError}</p>
          <button
            type="button"
            onClick={() => {
              setLoading(true);
              void refreshView();
            }}
            className="rounded-md border px-3 py-1.5 transition hover:bg-muted"
          >
            {t("watchlist.retry")}
          </button>
        </div>
      ) : (
        <>
          <section className="rounded-lg border bg-card p-4">{runSummaryStrip()}</section>

          {funnelSection()}

          <form onSubmit={handleAdd} className="flex flex-wrap items-end gap-3 rounded-lg border bg-card p-4">
            <div className="space-y-1">
              <label htmlFor="watchlist-add-code" className={labelClass}>
                {t("watchlist.addCode")}
              </label>
              <input
                id="watchlist-add-code"
                value={addCode}
                onChange={(e) => setAddCode(e.target.value)}
                placeholder="600000"
                maxLength={6}
                className={cn(fieldClass, "w-32 font-mono")}
                data-testid="watchlist-add-code"
              />
            </div>
            <div className="min-w-48 flex-1 space-y-1">
              <label htmlFor="watchlist-add-note" className={labelClass}>
                {t("watchlist.addNote")}
              </label>
              <input
                id="watchlist-add-note"
                value={addNote}
                onChange={(e) => setAddNote(e.target.value)}
                className={fieldClass}
              />
            </div>
            <button
              type="submit"
              disabled={adding}
              data-testid="watchlist-add-submit"
              className="inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm transition hover:bg-muted disabled:opacity-50"
            >
              {adding ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Plus className="h-4 w-4" aria-hidden />}
              {t("watchlist.add")}
            </button>
            {addError && (
              <p role="alert" className="w-full text-sm text-danger" data-testid="watchlist-add-error">
                {addError}
              </p>
            )}
          </form>

          <section className="rounded-lg border bg-card" aria-label={t("watchlist.itemsTitle")}>
            <h2 className="border-b px-4 py-3 text-sm font-medium">{t("watchlist.itemsTitle")}</h2>
            {items.length === 0 ? (
              <p className="p-8 text-center text-sm text-muted-foreground">{t("watchlist.empty")}</p>
            ) : (
              <ul className="divide-y">{items.map((item) => renderItemRow(item))}</ul>
            )}
          </section>

          {droppedSection()}
          {requalifiedSection()}
          {aiDemotedSection()}
          {scheduleCard()}
          {runsCard()}
        </>
      )}

      <ConfirmDialog
        open={removeTarget !== null}
        title={t("watchlist.removeConfirmTitle", { name: removeTarget?.name ?? "", code: removeTarget?.code ?? "" })}
        description={t("watchlist.removeConfirmBody")}
        confirmLabel={t("watchlist.removeConfirm")}
        cancelLabel={t("layout.cancel")}
        tone="destructive"
        confirmDisabled={removing}
        onConfirm={() => void handleRemove()}
        onCancel={() => setRemoveTarget(null)}
      />
    </div>
  );
}
