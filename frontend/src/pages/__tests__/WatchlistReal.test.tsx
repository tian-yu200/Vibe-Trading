import { render, screen } from "@testing-library/react";
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
  },
  toast: { error: vi.fn(), success: vi.fn() },
}));

vi.mock("@/lib/api", () => ({
  api,
  ApiError: class ApiError extends Error {
    status: number;
    code?: string;
    constructor(message: string, status: number, code?: string) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.code = code;
    }
  },
  isHomeMismatchError: (error: unknown): boolean =>
    error instanceof Error && (error as { status?: number }).status === 503,
}));

vi.mock("sonner", () => ({ toast }));

/** Deep clone of the REAL /api/watchlist payload (M2 run ee122cc9). */
function realView() {
  return JSON.parse(JSON.stringify(realPayload)).watchlist;
}

function mockAll(view: unknown) {
  api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: view });
  api.getWatchlistScanStatus.mockResolvedValue({ running: false });
  api.getWatchlistSettings.mockResolvedValue({
    status: "ok",
    settings: {
      scan: {},
      schedule: { enabled: true, time: "18:00", timezone: "Asia/Shanghai" },
    },
  });
  api.getWatchlistSchedule.mockResolvedValue({
    enabled: true,
    time: "18:00",
    timezone: "Asia/Shanghai",
  });
  api.listWatchlistRuns.mockResolvedValue({ status: "ok", runs: [] });
}

function itemRowTestIds(): string[] {
  return screen
    .getAllByTestId(/^watchlist-item-\d{6}$/)
    .map((el) => el.getAttribute("data-testid") ?? "");
}

beforeEach(() => {
  vi.clearAllMocks();
  mockAll(realView());
});

describe("Watchlist page with the REAL /api/watchlist payload", () => {
  it("renders disclaimer, 20 rows, funnel stages, an as-of cell, a timeout row, and the coverage line", async () => {
    const user = userEvent.setup();
    render(<Watchlist />);

    expect(await screen.findByTestId("watchlist-disclaimer")).toHaveTextContent(
      "Research only, not investment advice.",
    );
    await screen.findByTestId("watchlist-item-301087");
    expect(itemRowTestIds()).toHaveLength(20);

    await user.click(screen.getByRole("button", { name: "Scan funnel & sources" }));
    // Funnel stages render as objects (this exact shape crashed the page before).
    expect(screen.getByText("excluded:illiquid")).toBeInTheDocument();
    expect(screen.getByText("insufficient_data:liquidity")).toBeInTheDocument();
    expect(screen.getByTestId("watchlist-as-of-kline")).toHaveTextContent("2026-10-13");
    expect(screen.getByTestId("watchlist-coverage")).toHaveTextContent(
      "trend-scored 3676, flow real 3672 + neutral 4, kline cache coverage 3676/3676",
    );

    // A 'review failed: timeout'-style row: expand a failed pick, read its note.
    const failed = realPayload.watchlist.items.find((i) => i.ai?.status === "failed");
    expect(failed).toBeDefined();
    await user.click(screen.getByRole("button", { name: `Details for ${failed!.code}` }));
    // The expanded row (same testid container) carries the failure note.
    expect(screen.getByTestId(`watchlist-item-${failed!.code}`).textContent).toMatch(
      /timed out after 90s/,
    );
  }, 15000);

  it("shows the stale marker when the view is stale", async () => {
    const view = realView();
    view.stale = { ...view.stale, is_stale: true };
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: view });
    render(<Watchlist />);
    expect(await screen.findByText("Results may be stale")).toBeInTheDocument();
  });

  it("renders a quota-failed pick distinctly", async () => {
    const view = realView();
    const failed = view.items.find((i) => i.ai?.status === "failed");
    failed.ai = {
      ...failed.ai,
      failure_kind: "quota",
      note: "review failed: quota (resets 20:00)",
    };
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: view });
    const user = userEvent.setup();
    render(<Watchlist />);
    await screen.findByTestId(`watchlist-item-${failed.code}`);
    expect(screen.getByText("quota failed")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: `Details for ${failed.code}` }));
    expect(
      await screen.findByTestId(`watchlist-item-quota-note-${failed.code}`),
    ).toHaveTextContent("review failed: quota (resets 20:00)");
  }, 10000);

  it("shows the intraday banner instead of the data-as-of line", async () => {
    const view = realView();
    view.latest_run = { ...view.latest_run, intraday: true };
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: view });
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    expect(screen.getByTestId("watchlist-intraday-banner")).toBeInTheDocument();
    expect(screen.queryByTestId("watchlist-data-as-of")).not.toBeInTheDocument();
  });

  it("renders the empty state when the watchlist has no items", async () => {
    const view = realView();
    view.items = [];
    api.getWatchlist.mockResolvedValue({ status: "ok", watchlist: view });
    render(<Watchlist />);
    await screen.findByTestId("watchlist-run-summary");
    expect(screen.getByText("No stocks on the watchlist yet.")).toBeInTheDocument();
  });
});
