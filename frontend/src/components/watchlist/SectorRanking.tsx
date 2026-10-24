import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { LayoutGrid, Loader2, Plus } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import {
  api,
  ApiError,
  isHomeMismatchError,
  type WatchlistRunSummary,
  type WatchlistSector,
  type WatchlistSectorLevel,
  type WatchlistSectorRanking,
  type WatchlistSectorStock,
} from "@/lib/api";
import { StatusPill, boardTag, subScore } from "./badges";

const hintClass = "text-xs text-muted-foreground";
const SECTOR_LIMIT = 10;
const RUN_CHOICES = 20;

interface SectorRankingProps {
  /** Run id of the page's current view; a new scan refetches "latest". */
  latestRunId: string | null;
  /** Codes active on the watchlist right now (badge, and no "Add" button). */
  watchingCodes: ReadonlySet<string>;
  /** Called after a stock was added, so the page reloads its list. */
  onAdded: () => Promise<void> | void;
  /** True while the API refuses with a home mismatch: adding is disabled. */
  disabled?: boolean;
}

/**
 * "By sector" view: the top 10 scored stocks of each Shenwan sector for one
 * scan (latest by default). Rule score only — the AI review covers the top
 * picks, not these lists, and the section says so. Read-only apart from the
 * per-row "Add", which is the same manual add as the form above.
 */
export function SectorRanking({ latestRunId, watchingCodes, onAdded, disabled = false }: SectorRankingProps) {
  const { t } = useTranslation();

  const [level, setLevel] = useState<WatchlistSectorLevel>("l1");
  // null = follow the latest successful run.
  const [runId, setRunId] = useState<string | null>(null);
  const [ranking, setRanking] = useState<WatchlistSectorRanking | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [runs, setRuns] = useState<WatchlistRunSummary[]>([]);
  const [adding, setAdding] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const body = await api.getWatchlistSectors({
        runId: runId ?? undefined,
        level,
        limit: SECTOR_LIMIT,
      });
      setRanking(body.ranking);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [runId, level]);

  // latestRunId is a dependency so "latest" refreshes after a new scan.
  useEffect(() => {
    void load();
  }, [load, latestRunId]);

  useEffect(() => {
    let cancelled = false;
    api
      .listWatchlistRuns(RUN_CHOICES)
      .then((body) => {
        if (!cancelled) setRuns(body.runs.filter((r) => r.status === "ok" || r.status === "partial"));
      })
      .catch(() => {
        // The run picker is optional; "latest" still works without it.
      });
    return () => {
      cancelled = true;
    };
  }, [latestRunId]);

  const sectors = ranking?.sectors ?? [];
  // Keep the chosen sector across reloads when it still exists; otherwise
  // fall back to the strongest one (the API lists strongest first).
  const current: WatchlistSector | null =
    sectors.find((s) => s.name === selected) ?? sectors[0] ?? null;

  async function handleAdd(stock: WatchlistSectorStock) {
    setAdding(stock.code);
    try {
      await api.addWatchlistItem(stock.code, null);
      toast.success(t("watchlist.addSuccess", { code: stock.code }));
      await onAdded();
    } catch (err) {
      if (isHomeMismatchError(err)) toast.error(err.message);
      else toast.error(err instanceof ApiError ? err.message : String(err));
    } finally {
      setAdding(null);
    }
  }

  function levelButton(value: WatchlistSectorLevel, label: string) {
    const active = level === value;
    return (
      <button
        type="button"
        onClick={() => {
          setLevel(value);
          setSelected(null);
        }}
        aria-pressed={active}
        data-testid={`watchlist-sectors-level-${value}`}
        className={cn(
          "px-2.5 py-1 text-xs font-medium transition",
          active ? "bg-primary text-primary-foreground" : "hover:bg-muted",
        )}
      >
        {label}
      </button>
    );
  }

  function stockRow(stock: WatchlistSectorStock) {
    const board = boardTag(stock.board);
    const watching = watchingCodes.has(stock.code);
    return (
      <li
        key={stock.code}
        data-testid={`watchlist-sector-row-${stock.code}`}
        className="flex flex-wrap items-center gap-2 px-4 py-2.5 text-sm"
      >
        <span className="w-6 tabular-nums text-muted-foreground">{stock.sector_rank}</span>
        <span className="w-16 font-mono">{stock.code}</span>
        <span className="min-w-0 flex-1 truncate font-medium">{stock.name ?? "—"}</span>
        <span className={cn("rounded px-1.5 py-0.5 text-xs", board.tone)}>{board.label}</span>
        <span className="hidden w-20 truncate text-xs text-muted-foreground sm:inline-block">
          {level === "l1" ? stock.industry : stock.sector}
        </span>
        <span className="w-12 text-right font-medium tabular-nums">{subScore(stock.score)}</span>
        <span className="flex gap-1 text-xs tabular-nums">
          <span className="rounded bg-muted px-1">F {subScore(stock.sub_scores?.fundamental)}</span>
          <span className="rounded bg-muted px-1">T {subScore(stock.sub_scores?.trend)}</span>
          <span className="rounded bg-muted px-1">Fl {subScore(stock.sub_scores?.flow)}</span>
        </span>
        <span className={cn(hintClass, "w-24 text-right tabular-nums")}>
          {t("watchlist.sectorMarketRank", { rank: stock.market_rank })}
        </span>
        {/* Fixed width so score / sub-score / rank columns line up whether a
            row shows badges or an Add button. */}
        <span className="flex w-40 shrink-0 items-center justify-end gap-1.5">
          {stock.in_picks && (
            <StatusPill label={t("watchlist.sectorInPicks", { n: ranking?.picks_count ?? 0 })} tone="primary" />
          )}
          {watching ? (
            <StatusPill label={t("watchlist.sectorWatching")} tone="success" />
          ) : (
            <button
              type="button"
              onClick={() => void handleAdd(stock)}
              disabled={disabled || adding !== null}
              aria-label={t("watchlist.sectorAddAria", { code: stock.code, name: stock.name ?? stock.code })}
              data-testid={`watchlist-sector-add-${stock.code}`}
              className="inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs transition hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            >
              {adding === stock.code ? (
                <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
              ) : (
                <Plus className="h-3 w-3" aria-hidden />
              )}
              {t("watchlist.add")}
            </button>
          )}
        </span>
      </li>
    );
  }

  function body() {
    if (loading && !ranking) {
      return (
        <div className="flex h-24 items-center justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" aria-hidden />
        </div>
      );
    }
    if (error && !ranking) {
      return (
        <div className="space-y-2 p-4 text-sm">
          <p className="text-danger" data-testid="watchlist-sectors-error">
            {error}
          </p>
          <button
            type="button"
            onClick={() => void load()}
            className="rounded-md border px-3 py-1.5 transition hover:bg-muted"
          >
            {t("watchlist.retry")}
          </button>
        </div>
      );
    }
    if (!ranking?.run || sectors.length === 0) {
      return (
        <p className="p-4 text-sm text-muted-foreground" data-testid="watchlist-sectors-empty">
          {ranking?.run ? t("watchlist.sectorEmpty") : t("watchlist.noRuns")}
        </p>
      );
    }
    return (
      <>
        <div className="space-y-2 px-4 pt-3">
          <p className={hintClass} data-testid="watchlist-sectors-note">
            {t("watchlist.sectorNote", {
              scored: ranking.scored_total,
              date: ranking.run.trade_date ?? "—",
            })}
          </p>
          <div
            className="flex max-h-40 flex-wrap gap-1.5 overflow-y-auto pb-1"
            role="group"
            aria-label={t("watchlist.sectorPickerAria")}
          >
            {sectors.map((s) => {
              const active = s.name === current?.name;
              return (
                <button
                  key={s.name}
                  type="button"
                  onClick={() => setSelected(s.name)}
                  aria-pressed={active}
                  data-testid={`watchlist-sector-chip-${s.name}`}
                  className={cn(
                    "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs transition",
                    active ? "border-primary bg-primary/10 text-primary" : "hover:bg-muted",
                  )}
                >
                  {s.name}
                  <span className="tabular-nums text-muted-foreground">{s.count}</span>
                </button>
              );
            })}
          </div>
        </div>
        {current && (
          <ul className="mt-1 divide-y border-t" data-testid="watchlist-sector-rows">
            {current.top.map((stock) => stockRow(stock))}
          </ul>
        )}
      </>
    );
  }

  return (
    <section
      className="rounded-lg border bg-card"
      aria-label={t("watchlist.sectorTitle", { n: SECTOR_LIMIT })}
      data-testid="watchlist-sectors"
    >
      <div className="flex flex-wrap items-center gap-2 border-b px-4 py-3">
        <LayoutGrid className="h-4 w-4 text-primary" aria-hidden />
        <h2 className="text-sm font-medium">{t("watchlist.sectorTitle", { n: SECTOR_LIMIT })}</h2>
        {loading && ranking && <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" aria-hidden />}
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <div
            className="inline-flex overflow-hidden rounded-md border"
            role="group"
            aria-label={t("watchlist.sectorLevelAria")}
          >
            {levelButton("l1", t("watchlist.sectorLevelL1"))}
            {levelButton("l2", t("watchlist.sectorLevelL2"))}
          </div>
          <select
            value={runId ?? ""}
            onChange={(e) => {
              setRunId(e.target.value || null);
              setSelected(null);
            }}
            aria-label={t("watchlist.sectorRunAria")}
            data-testid="watchlist-sectors-run"
            className="rounded-md border bg-background px-2 py-1 text-xs"
          >
            <option value="">{t("watchlist.sectorRunLatest")}</option>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {`${r.trade_date ?? "—"} · ${r.run_id.slice(0, 8)}`}
              </option>
            ))}
          </select>
        </div>
      </div>
      {body()}
    </section>
  );
}
