import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { SectorRanking } from "../SectorRanking";
import realL1 from "./fixtures/sectors_l1.json";
import realL2 from "./fixtures/sectors_l2.json";

// Fixtures are REAL /api/watchlist/sectors responses (run 1d9095ad, trade
// date 2026-10-21), captured through the route on a copy of the live DB.

const { api, toast } = vi.hoisted(() => ({
  api: {
    getWatchlistSectors: vi.fn(),
    listWatchlistRuns: vi.fn(),
    addWatchlistItem: vi.fn(),
  },
  toast: { error: vi.fn(), success: vi.fn() },
}));

vi.mock("@/lib/api", () => {
  class ApiError extends Error {
    status: number;
    constructor(message: string, status: number) {
      super(message);
      this.name = "ApiError";
      this.status = status;
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

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

const RUNS = [
  { run_id: "1d9095ad79c24f3094553b4920643006", trade_date: "2026-10-21", status: "partial" },
  { run_id: "deadbeef00000000000000000000aaaa", trade_date: "2026-10-21", status: "failed" },
  { run_id: "31adfde8a2ee4f7cbea8b8ca0bd1410a", trade_date: "2026-10-20", status: "partial" },
];

/** The 20 codes that were on the live watchlist when the fixture was taken. */
function liveWatching(): Set<string> {
  return new Set(
    realL1.ranking.sectors.flatMap((s) => s.top).filter((s) => s.watching).map((s) => s.code),
  );
}

function renderIt(props: Partial<Parameters<typeof SectorRanking>[0]> = {}) {
  const onAdded = vi.fn();
  const utils = render(
    <SectorRanking latestRunId="1d9095ad79c24f3094553b4920643006" watchingCodes={liveWatching()} onAdded={onAdded} {...props} />,
  );
  return { ...utils, onAdded };
}

function rowCodes(): string[] {
  return within(screen.getByTestId("watchlist-sector-rows"))
    .getAllByTestId(/^watchlist-sector-row-\d{6}$/)
    .map((el) => (el.getAttribute("data-testid") ?? "").slice(-6));
}

beforeEach(() => {
  vi.clearAllMocks();
  api.getWatchlistSectors.mockImplementation(async (params?: { level?: string }) =>
    clone(params?.level === "l2" ? realL2 : realL1),
  );
  api.listWatchlistRuns.mockResolvedValue({ status: "ok", runs: RUNS });
  api.addWatchlistItem.mockResolvedValue({ status: "ok", item: { code: "603068" } });
});

describe("SectorRanking with the REAL sector payload", () => {
  it("shows the not-AI-reviewed note, 31 sector chips and the strongest sector's top 10", async () => {
    renderIt();
    const note = await screen.findByTestId("watchlist-sectors-note");
    expect(note).toHaveTextContent("Ranked by rule score only, not AI-reviewed.");
    expect(note).toHaveTextContent("2805 scored stocks");
    expect(note).toHaveTextContent("trade date 2026-10-21");

    expect(screen.getAllByTestId(/^watchlist-sector-chip-/)).toHaveLength(31);
    const first = screen.getByTestId("watchlist-sector-chip-医药生物");
    expect(first).toHaveAttribute("aria-pressed", "true");
    expect(first).toHaveTextContent("239");

    expect(api.getWatchlistSectors).toHaveBeenCalledWith({ runId: undefined, level: "l1", limit: 10 });
    const codes = rowCodes();
    expect(codes).toHaveLength(10);
    expect(codes[0]).toBe("600267");

    const top = screen.getByTestId("watchlist-sector-row-600267");
    expect(top).toHaveTextContent("海正药业");
    expect(top).toHaveTextContent("化学制药");
    expect(top).toHaveTextContent("#1 overall");
    expect(top).toHaveTextContent("Top 20");
    expect(top).toHaveTextContent("Watching");
    expect(within(top).queryByRole("button")).toBeNull();
  });

  it("switches sector on a chip click and offers Add for stocks not on the watchlist", async () => {
    const user = userEvent.setup();
    renderIt();
    await user.click(await screen.findByTestId("watchlist-sector-chip-电子"));

    expect(screen.getByTestId("watchlist-sector-chip-电子")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("watchlist-sector-chip-医药生物")).toHaveAttribute("aria-pressed", "false");
    const codes = rowCodes();
    expect(codes.slice(0, 3)).toEqual(["603893", "002351", "603068"]);

    const outsider = screen.getByTestId("watchlist-sector-row-603068");
    expect(outsider).toHaveTextContent("博通集成");
    expect(outsider).not.toHaveTextContent("Top 20");
    expect(outsider).not.toHaveTextContent("Watching");
    expect(within(outsider).getByRole("button", { name: "Add 博通集成 (603068) to the watchlist" })).toBeEnabled();
  });

  it("adds a stock with the existing manual add, then tells the page to reload", async () => {
    const user = userEvent.setup();
    const { onAdded } = renderIt();
    await user.click(await screen.findByTestId("watchlist-sector-chip-电子"));
    await user.click(screen.getByTestId("watchlist-sector-add-603068"));

    await waitFor(() => expect(onAdded).toHaveBeenCalledTimes(1));
    expect(api.addWatchlistItem).toHaveBeenCalledWith("603068", null);
    expect(toast.success).toHaveBeenCalledWith("Added 603068.");
  });

  it("marks a stock as watching once the page's watchlist contains it", async () => {
    const user = userEvent.setup();
    const { rerender, onAdded } = renderIt();
    await user.click(await screen.findByTestId("watchlist-sector-chip-电子"));
    expect(screen.getByTestId("watchlist-sector-add-603068")).toBeInTheDocument();

    const watching = new Set([...liveWatching(), "603068"]);
    rerender(
      <SectorRanking latestRunId="1d9095ad79c24f3094553b4920643006" watchingCodes={watching} onAdded={onAdded} />,
    );
    expect(screen.queryByTestId("watchlist-sector-add-603068")).toBeNull();
    expect(screen.getByTestId("watchlist-sector-row-603068")).toHaveTextContent("Watching");
  });

  it("surfaces an add failure as a toast and leaves the page alone", async () => {
    const user = userEvent.setup();
    const { ApiError } = await import("@/lib/api");
    api.addWatchlistItem.mockRejectedValue(new ApiError("BSE code unsupported", 400));
    const { onAdded } = renderIt();
    await user.click(await screen.findByTestId("watchlist-sector-chip-电子"));
    await user.click(screen.getByTestId("watchlist-sector-add-603068"));

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("BSE code unsupported"));
    expect(onAdded).not.toHaveBeenCalled();
  });

  it("disables Add while the page reports a home mismatch", async () => {
    const user = userEvent.setup();
    renderIt({ disabled: true });
    await user.click(await screen.findByTestId("watchlist-sector-chip-电子"));
    expect(screen.getByTestId("watchlist-sector-add-603068")).toBeDisabled();
  });

  it("switches to level-2 industries and shows each stock's sector", async () => {
    const user = userEvent.setup();
    renderIt();
    await screen.findByTestId("watchlist-sector-chip-医药生物");
    await user.click(screen.getByTestId("watchlist-sectors-level-l2"));

    await waitFor(() =>
      expect(api.getWatchlistSectors).toHaveBeenLastCalledWith({ runId: undefined, level: "l2", limit: 10 }),
    );
    expect(await screen.findByTestId("watchlist-sector-chip-化学制药")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getAllByTestId(/^watchlist-sector-chip-/)).toHaveLength(127);
    expect(screen.getByTestId("watchlist-sectors-level-l2")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("watchlist-sector-row-600267")).toHaveTextContent("医药生物");
  });

  it("offers only successful runs in the scan picker and reloads for the chosen one", async () => {
    const user = userEvent.setup();
    renderIt();
    await screen.findByTestId("watchlist-sectors-note");
    const picker = screen.getByTestId("watchlist-sectors-run") as HTMLSelectElement;
    await waitFor(() => expect(picker.options).toHaveLength(3)); // latest + 2 successful runs
    expect(picker.options[0].textContent).toBe("Latest scan");
    expect(picker.options[2].textContent).toBe("2026-10-20 · 31adfde8");

    await user.selectOptions(picker, "31adfde8a2ee4f7cbea8b8ca0bd1410a");
    await waitFor(() =>
      expect(api.getWatchlistSectors).toHaveBeenLastCalledWith({
        runId: "31adfde8a2ee4f7cbea8b8ca0bd1410a",
        level: "l1",
        limit: 10,
      }),
    );
  });

  it("reloads the latest ranking when the page reports a new run", async () => {
    const { rerender, onAdded } = renderIt();
    await screen.findByTestId("watchlist-sectors-note");
    expect(api.getWatchlistSectors).toHaveBeenCalledTimes(1);
    rerender(<SectorRanking latestRunId="new-run" watchingCodes={liveWatching()} onAdded={onAdded} />);
    await waitFor(() => expect(api.getWatchlistSectors).toHaveBeenCalledTimes(2));
  });

  it("shows the no-runs line on an empty store", async () => {
    api.getWatchlistSectors.mockResolvedValue({
      status: "ok",
      ranking: { run: null, level: "l1", limit: 10, scored_total: 0, picks_count: 0, sectors: [] },
    });
    renderIt({ latestRunId: null });
    expect(await screen.findByTestId("watchlist-sectors-empty")).toHaveTextContent("No scans have run yet.");
  });

  it("shows a load error with a working retry", async () => {
    const user = userEvent.setup();
    const { ApiError } = await import("@/lib/api");
    api.getWatchlistSectors.mockRejectedValueOnce(new ApiError("boom", 500));
    renderIt();
    expect(await screen.findByTestId("watchlist-sectors-error")).toHaveTextContent("boom");
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByTestId("watchlist-sectors-note")).toBeInTheDocument();
  });
});
