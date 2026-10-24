import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { Watchlist } from "../Watchlist";
import realPayload from "./fixtures/watchlist.json";

const { api, toast } = vi.hoisted(() => ({
  api: {
    getWatchlist: vi.fn(),
    addWatchlistItem: vi.fn(),
    removeWatchlistItem: vi.fn(),
    restoreWatchlistItem: vi.fn(),
    startWatchlistScan: vi.fn(),
    startWatchlistReview: vi.fn(),
    getWatchlistScanStatus: vi.fn(),
    listWatchlistRuns: vi.fn(),
    getWatchlistRun: vi.fn(),
    getWatchlistSettings: vi.fn(),
    updateWatchlistSettings: vi.fn(),
    getWatchlistSchedule: vi.fn(),
    getWatchlistSectors: vi.fn(),
  },
  toast: { error: vi.fn(), success: vi.fn() },
}));

vi.mock("@/lib/api", () => {
  class ApiError extends Error {
    status: number;
    code?: string;
    constructor(message: string, status: number, code?: string) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.code = code;
    }
  }
  return {
    api,
    ApiError,
    isHomeMismatchError: (error: unknown): boolean =>
      error instanceof ApiError && error.status === 503,
  };
});

vi.mock("sonner", () => ({ toast }));

function makeItem(overrides = {}) {
  return {
    code: "600001",
    name: "Example Co",
    board: "main_sh",
    industry: "Banks",
    origin: "auto",
    change: "new",
    rank: 1,
    score: 82.5,
    sub_scores: { fundamental: 80, trend: 85, flow: 82.5 },
    reasons: ["Strong ROE"],
    flags: [],
    ai: {
      status: "reviewed",
      verdict: "watch",
      summary: "Solid momentum.",
      reasons: [{ text: "clean trend", cites: ["M:roe"] }],
      red_flags: [],
      news_checked: true,
    },
    scores_available: true,
    ...overrides,
  };
}

function makeView(overrides = {}) {
  return {
    disclaimer: "Research only, not investment advice.",
    stale: { is_stale: false },
    latest_run: {
      run_id: "run-1",
      trade_date: "2026-10-12",
      finished_at: "2026-10-12T18:05:00Z",
      status: "ok",
      ai_status: "reviewed",
      duration_s: 12.3,
      trigger: "cli",
      funnel: [
        { stage: "universe", count: 100, removed: 0, note: null },
        { stage: "excluded:illiquid", count: 20, removed: 80, note: null },
        { stage: "scored", count: 20, removed: 0, note: null },
      ],
      sources: [],
      coverage: null,
    },
    items: [makeItem()],
    dropped: [],
    requalified: [],
    ai_demoted: [],
    ...overrides,
  };
}

function mockAll(view = makeView()) {
  api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: view });
  api.getWatchlistScanStatus.mockResolvedValue({ running: false });
  api.getWatchlistSettings.mockResolvedValue({
    status: "ok",
    settings: {
      scan: {},
      schedule: { enabled: false, time: "18:00", timezone: "Asia/Shanghai" },
    },
  });
  api.getWatchlistSchedule.mockResolvedValue({
    enabled: false,
    time: "18:00",
    timezone: "Asia/Shanghai",
  });
  api.listWatchlistRuns.mockResolvedValue({ status: "ok", runs: [] });
  api.getWatchlistSectors.mockResolvedValue({ status: "ok", ranking: EMPTY_SECTORS });
}

const EMPTY_SECTORS = { run: null, level: "l1", limit: 10, scored_total: 0, picks_count: 0, sectors: [] };

beforeEach(() => {
  vi.clearAllMocks();
  mockAll();
});

describe("Watchlist page", () => {
  it("renders the disclaimer line", async () => {
    render(<Watchlist />);
    const disclaimer = await screen.findByTestId("watchlist-disclaimer");
    expect(disclaimer).toHaveTextContent("Research only, not investment advice.");
  });

  it("renders the latest-run summary strip", async () => {
    render(<Watchlist />);
    const summary = await screen.findByTestId("watchlist-run-summary");
    expect(summary).toHaveTextContent("Data as of 2026-10-12 close");
    expect(summary).toHaveTextContent("ok");
  });

  it("shows a stale banner when the view is stale", async () => {
    api.getWatchlist.mockResolvedValue({
      status: "ok",
      watchlist: makeView({ stale: { is_stale: true, latest_trading_day: "2026-10-12" } }),
    });
    render(<Watchlist />);
    expect(await screen.findByText("Results may be stale")).toBeInTheDocument();
  });

  it(
    "starts a scan, polls status, then reloads when finished",
    async () => {
      api.startWatchlistScan.mockResolvedValue({ job_id: "j1" });
      // First poll after start: still running (shows the progress panel).
      // Later polls: finished, which triggers the reload.
      api.getWatchlistScanStatus
        .mockResolvedValueOnce({ running: true, job_id: "j1", stage: "scoring", progress: 0.5 })
        .mockResolvedValue({ running: false, job_id: "j1" });

      const user = userEvent.setup();
      render(<Watchlist />);
      await screen.findByTestId("watchlist-run-summary");

      await user.click(screen.getByTestId("watchlist-run-scan"));

      // The first poll surfaces the running state.
      await screen.findByTestId("watchlist-scan-progress", {}, { timeout: 5000 });

      // The page reloads once a poll reports the scan is no longer running.
      await waitFor(() => {
        expect(api.getWatchlist.mock.calls.length).toBeGreaterThanOrEqual(2);
      }, { timeout: 15000 });
    },
    20000,
  );

  it("surfaces a 409 as an already-running toast", async () => {
    const { ApiError } = await import("@/lib/api");
    api.startWatchlistScan.mockRejectedValue(new ApiError("scan already running", 409));

    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");

    await user.click(screen.getByTestId("watchlist-run-scan"));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith("A scan is already running.");
    }, { timeout: 5000 });
  }, 10000);

  it("removes an item through the ConfirmDialog", async () => {
    api.removeWatchlistItem.mockResolvedValue({ status: "ok", item: makeItem() });
    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-item-600001");

    await user.click(screen.getByRole("button", { name: /Remove Example Co/ }));

    const confirm = await screen.findByRole("button", { name: "Remove" });
    await user.click(confirm);

    await waitFor(() => {
      expect(api.removeWatchlistItem).toHaveBeenCalledWith("600001");
    }, { timeout: 5000 });
  }, 10000);

  it("restores a requalified item", async () => {
    api.getWatchlist.mockResolvedValue({
      status: "ok",
      watchlist: makeView({
        requalified: [
          {
            code: "600002",
            name: "Requal Co",
            rank: 3,
            score: 70,
            label: "qualifies again — you removed it",
          },
        ],
      }),
    });
    api.restoreWatchlistItem.mockResolvedValue({ status: "ok" });
    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByText("Qualifies again");

    await user.click(screen.getByRole("button", { name: /Restore Requal Co/ }));

    await waitFor(() => {
      expect(api.restoreWatchlistItem).toHaveBeenCalledWith("600002");
    }, { timeout: 5000 });
  }, 10000);

  it("toggles the schedule enabled flag via PUT settings", async () => {
    api.updateWatchlistSettings.mockResolvedValue({
      status: "ok",
      settings: {
        scan: {},
        schedule: { enabled: true, time: "18:00", timezone: "Asia/Shanghai" },
      },
    });
    const user = userEvent.setup();
    render(<Watchlist />);
    const toggle = await screen.findByTestId("watchlist-schedule-enabled");

    await user.click(toggle);

    await waitFor(() => {
      expect(api.updateWatchlistSettings).toHaveBeenCalledWith({
        schedule: { enabled: true },
      });
    }, { timeout: 5000 });
  }, 10000);

  it("renders the AI evidence block with id badge, date, title link, and source", async () => {
    api.getWatchlist.mockResolvedValue({
      status: "ok",
      watchlist: makeView({
        items: [
          makeItem({
            ai: {
              status: "reviewed",
              verdict: "watch",
              summary: "Solid momentum.",
              reasons: [{ text: "clean trend", cites: ["N1"] }],
              red_flags: [],
              news_checked: true,
              evidence: [
                {
                  id: "N1",
                  kind: "news",
                  date: "2026-10-01",
                  title_or_text: "Q3 beat",
                  source: "Eastmoney",
                  url: "https://example.com/n1",
                },
              ],
            },
          }),
        ],
      }),
    });
    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-item-600001");
    await user.click(screen.getByRole("button", { name: "Details for 600001" }));

    const block = await screen.findByTestId("watchlist-evidence");
    expect(block).toHaveTextContent("Evidence");
    expect(screen.getByTestId("evidence-id-N1")).toBeInTheDocument();
    expect(block).toHaveTextContent("Q3 beat");
    expect(block).toHaveTextContent("Eastmoney");
    const link = screen.getByRole("link", { name: "Q3 beat" });
    expect(link).toHaveAttribute("href", "https://example.com/n1");
  }, 10000);

  it("omits the evidence block for older runs without evidence", async () => {
    // makeItem's ai has no `evidence` (older run).
    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-item-600001");
    await user.click(screen.getByRole("button", { name: "Details for 600001" }));

    await screen.findByText("AI summary");
    expect(screen.queryByTestId("watchlist-evidence")).not.toBeInTheDocument();
  }, 10000);

  it("renders the as-of column for each source", async () => {
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: makeView({
      latest_run: {
        run_id: "run-1", trade_date: "2026-10-12", status: "ok",
        ai_status: "reviewed", trigger: "cli", finished_at: "2026-10-12T18:05:00Z",
        funnel: [
          { stage: "universe", count: 100, removed: 0, note: null },
          { stage: "scored", count: 20, removed: 80, note: null },
        ],
        sources: [
          { name: "universe", status: "ok", as_of: "2026-10-12" },
          { name: "kline", status: "ok", as_of: "2026-10-12" },
        ],
      },
    }) });
    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    // the sources table lives inside the (collapsed) funnel section
    await user.click(screen.getByRole("button", { name: "Scan funnel & sources" }));
    expect(screen.getByText("As of")).toBeInTheDocument();
    expect(screen.getByTestId("watchlist-as-of-universe")).toHaveTextContent("2026-10-12");
    expect(screen.getByTestId("watchlist-as-of-kline")).toHaveTextContent("2026-10-12");
  });

  it("renders an em dash in the as-of column when a source has no as_of", async () => {
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: makeView({
      latest_run: {
        run_id: "run-1", trade_date: "2026-10-12", status: "ok",
        ai_status: "reviewed", trigger: "cli", finished_at: "2026-10-12T18:05:00Z",
        funnel: [
          { stage: "universe", count: 100, removed: 0, note: null },
          { stage: "scored", count: 20, removed: 80, note: null },
        ],
        sources: [{ name: "universe", status: "ok" }],
      },
    }) });
    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    await user.click(screen.getByRole("button", { name: "Scan funnel & sources" }));
    expect(screen.getByTestId("watchlist-as-of-universe")).toHaveTextContent("—");
  });

  it("shows the intraday banner instead of the data-as-of line", async () => {
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: makeView({
      latest_run: {
        ...makeView().latest_run,
        intraday: true,
      },
    }) });
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    expect(screen.getByTestId("watchlist-intraday-banner")).toHaveTextContent(
      "Intraday data - not a closing snapshot",
    );
    expect(screen.queryByTestId("watchlist-data-as-of")).not.toBeInTheDocument();
  });

  it("re-runs the AI review for the latest run and polls", async () => {
    api.startWatchlistReview.mockResolvedValue({ job_id: "j2" });
    api.getWatchlistScanStatus
      .mockResolvedValueOnce({ running: true, job_id: "j2", kind: "review", stage: "news", progress: 0.3 })
      .mockResolvedValue({ running: false, job_id: "j2", kind: "review" });

    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");

    await user.click(screen.getByTestId("watchlist-rerun-review"));

    await waitFor(() => {
      expect(api.startWatchlistReview).toHaveBeenCalledWith("run-1");
    }, { timeout: 5000 });
    expect(toast.success).toHaveBeenCalledWith("AI review re-run started.");
    // the shared progress panel reflects the review kind
    await screen.findByTestId("watchlist-review-progress", {}, { timeout: 5000 });
  }, 10000);

  it("surfaces a 409 on re-run as a review-already-running toast", async () => {
    const { ApiError } = await import("@/lib/api");
    api.startWatchlistReview.mockRejectedValue(new ApiError("review already running", 409));

    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");

    await user.click(screen.getByTestId("watchlist-rerun-review"));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith("A scan or review is already running.");
    }, { timeout: 5000 });
  }, 10000);

  it("renders the run-level quota note distinctly with the reset time", async () => {
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: makeView({
      latest_run: {
        ...makeView().latest_run,
        ai_status: "partial",
        ai_note: "review failed: quota (resets 20:00)",
      },
    }) });
    render(<Watchlist />);
    const note = await screen.findByTestId("watchlist-ai-quota-note");
    expect(note).toHaveTextContent("review failed: quota (resets 20:00)");
  });

  it("renders a non-quota ai_note without the quota styling", async () => {
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: makeView({
      latest_run: {
        ...makeView().latest_run,
        ai_note: "AI review skipped",
      },
    }) });
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    expect(screen.queryByTestId("watchlist-ai-quota-note")).not.toBeInTheDocument();
  });

  it("renders a quota-failed pick with the quota pill and note", async () => {
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: makeView({
      latest_run: {
        ...makeView().latest_run,
        ai_status: "partial",
        ai_note: "review failed: quota (resets 20:00)",
      },
      items: [
        makeItem({
          ai: {
            status: "failed",
            failure_kind: "quota",
            note: "review failed: quota (resets 20:00)",
          },
        }),
      ],
    }) });
    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-item-600001");
    // distinct quota pill (warning tone), not the generic danger "AI failed"
    expect(screen.getByText("quota failed")).toBeInTheDocument();
    expect(screen.queryByText("AI failed")).not.toBeInTheDocument();
    // expanding shows the localized quota note
    await user.click(screen.getByRole("button", { name: "Details for 600001" }));
    const note = await screen.findByTestId("watchlist-item-quota-note-600001");
    expect(note).toHaveTextContent("review failed: quota (resets 20:00)");
  }, 10000);

  it("keeps the re-run button disabled while a scan is running", async () => {
    api.startWatchlistScan.mockResolvedValue({ job_id: "j1" });
    api.getWatchlistScanStatus
      .mockResolvedValueOnce({ running: true, job_id: "j1", kind: "scan", stage: "scoring", progress: 0.5 })
      .mockResolvedValue({ running: false, job_id: "j1", kind: "scan" });

    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");

    const rerun = screen.getByTestId("watchlist-rerun-review");
    expect(rerun).not.toBeDisabled();
    await user.click(screen.getByTestId("watchlist-run-scan"));
    await screen.findByTestId("watchlist-scan-progress", {}, { timeout: 5000 });
    expect(screen.getByTestId("watchlist-rerun-review")).toBeDisabled();
  }, 10000);

  it("offers the re-run button for a partial run (success = ok|partial)", async () => {
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: makeView({
      latest_run: { ...makeView().latest_run, status: "partial", ai_status: "partial" },
    }) });
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    expect(screen.getByTestId("watchlist-rerun-review")).toBeInTheDocument();
  });

  it("hides the re-run button when the latest run failed", async () => {
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: makeView({
      latest_run: { ...makeView().latest_run, status: "failed", ai_status: "not_reviewed" },
    }) });
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    expect(screen.queryByTestId("watchlist-rerun-review")).not.toBeInTheDocument();
  });

  it("renders the coverage line from latest_run.coverage", async () => {
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: makeView({
      latest_run: {
        ...makeView().latest_run,
        coverage: {
          passed_exclusions: 130,
          insufficient: 10,
          scored: 120,
          trend_scored: 118,
          flow_real: 100,
          flow_neutral: 20,
          kline_full_history: 125,
        },
      },
    }) });
    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    await user.click(screen.getByRole("button", { name: "Scan funnel & sources" }));
    expect(screen.getByTestId("watchlist-coverage")).toHaveTextContent(
      "trend-scored 118, flow real 100 + neutral 20, kline cache coverage 125/130",
    );
  });

  it("hides the coverage line when the run carries no coverage", async () => {
    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    await user.click(screen.getByRole("button", { name: "Scan funnel & sources" }));
    expect(screen.queryByTestId("watchlist-coverage")).not.toBeInTheDocument();
  });

  it("renders the AI reason for a dropped entry the AI demoted", async () => {
    const view = { ...realPayload.watchlist };
    const dropped = [...view.dropped];
    dropped[0] = {
      ...dropped[0],
      ai_verdict: "avoid",
      ai_reason: "valuation stretched after the spike",
      ai_cites: ["N1", "A2"],
    };
    view.dropped = dropped;
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: view });
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    expect(screen.getByTestId("watchlist-dropped-ai-002039")).toHaveTextContent(
      "AI: avoid - valuation stretched after the spike [N1, A2]",
    );
    // The AI line comes before the pre-existing rank and reason text.
    const row = screen.getByTestId("watchlist-dropped-ai-002039").closest("li");
    const aiIndex = row!.textContent!.indexOf("AI: avoid");
    const reasonIndex = row!.textContent!.indexOf("rank fell to 29");
    expect(aiIndex).toBeGreaterThanOrEqual(0);
    expect(aiIndex).toBeLessThan(reasonIndex);
  });

  it("renders no AI line for a dropped entry without AI fields", async () => {
    api.getWatchlist.mockResolvedValue({
      status: "ok",
      watchlist: { ...realPayload.watchlist },
    });
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    // The untreated fixture keeps every dropped row AI-free.
    realPayload.watchlist.dropped.forEach((d) => {
      expect(screen.queryByTestId(`watchlist-dropped-ai-${d.code}`)).not.toBeInTheDocument();
    });
    expect(screen.queryByText(/AI:/)).not.toBeInTheDocument();
  });

  it("shows a loud HomeMismatch banner verbatim and disables the scan buttons", async () => {
    const { ApiError } = await import("@/lib/api");
    const message =
      "Vibe-Trading watchlist refuses to run (api): this process resolved home " +
      "C:\\Users\\<user>\\.vibe-trading but the pinned home is " +
      "E:\\vibe-trading\\home\\.vibe-trading. the resolved home has no HOME_ID " +
      "marker. Start via serve-8765.sh / source env.sh, or set VIBE_TRADING_HOME.";
    api.getWatchlist.mockRejectedValue(new ApiError(message, 503));
    render(<Watchlist />);

    const banner = await screen.findByTestId("watchlist-home-mismatch");
    // The title is i18n'd; the body must be the API message verbatim.
    expect(banner).toHaveTextContent("Watchlist disabled: wrong runtime home");
    expect(screen.getByTestId("watchlist-home-mismatch-message")).toHaveTextContent(message);
    expect(banner).toHaveAttribute("role", "alert");

    // Run scan / Re-run review are disabled while the mismatch is present.
    expect(screen.getByTestId("watchlist-run-scan")).toBeDisabled();
  });

  it("disables Re-run review too when a 503 arrives with a view loaded", async () => {
    const { ApiError } = await import("@/lib/api");
    api.startWatchlistScan.mockRejectedValue(new ApiError("pinned home mismatch", 503));
    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    // The run strip offers Re-run review while a view is loaded.
    expect(screen.getByTestId("watchlist-rerun-review")).toBeEnabled();

    await user.click(screen.getByTestId("watchlist-run-scan"));

    await screen.findByTestId("watchlist-home-mismatch");
    expect(screen.getByTestId("watchlist-rerun-review")).toBeDisabled();
  });
});
describe("Watchlist page: by-sector section", () => {
  function bankRow(code: string, name: string, rank: number, score: number, inPicks: boolean) {
    return {
      code,
      name,
      board: "main_sh",
      industry: "银行Ⅱ",
      sector: "银行",
      score,
      sub_scores: { fundamental: score, trend: score, flow: score },
      sector_rank: rank,
      market_rank: rank,
      in_picks: inPicks,
      watching: inPicks,
    };
  }

  function bankRanking() {
    return {
      run: { run_id: "run-1", trade_date: "2026-10-12", status: "ok", finished_at: null, ai_status: "reviewed", intraday: false },
      level: "l1",
      limit: 10,
      scored_total: 2,
      picks_count: 1,
      sectors: [
        {
          name: "银行",
          count: 2,
          leader_score: 82.5,
          top: [bankRow("600001", "Example Co", 1, 82.5, true), bankRow("600036", "Other Bank", 2, 70, false)],
        },
      ],
    };
  }

  it("sits under the watchlist and badges the stocks already on it", async () => {
    api.getWatchlistSectors.mockResolvedValue({ status: "ok", ranking: bankRanking() });
    render(<Watchlist />);
    const section = await screen.findByTestId("watchlist-sectors");
    const list = screen.getByRole("region", { name: "Watchlist" });
    expect(list.compareDocumentPosition(section) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    const onList = await within(section).findByTestId("watchlist-sector-row-600001");
    expect(onList).toHaveTextContent("Watching");
    expect(within(section).getByTestId("watchlist-sector-add-600036")).toBeEnabled();
  });

  it("reloads the page's watchlist after adding from a sector", async () => {
    api.getWatchlistSectors.mockResolvedValue({ status: "ok", ranking: bankRanking() });
    api.addWatchlistItem.mockResolvedValue({ status: "ok", item: { code: "600036" } });
    const user = userEvent.setup();
    render(<Watchlist />);
    await user.click(await screen.findByTestId("watchlist-sector-add-600036"));

    await waitFor(() => expect(api.getWatchlist).toHaveBeenCalledTimes(2));
    expect(api.addWatchlistItem).toHaveBeenCalledWith("600036", null);
  });
});
