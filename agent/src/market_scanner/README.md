# A-share daily scanner and watchlist

Research tool that scans the Chinese A-share market once a day, ranks the
survivors, reviews the finalists with an LLM, and maintains a small watchlist.
**Research only, not investment advice.** No order placement anywhere in this
feature; the scan and the watchlist are read-only with respect to brokers.

All commands below assume the repo virtualenv:
`R\.venv\Scripts\python.exe` (PowerShell) or `R/.venv/Scripts/python.exe` (bash).

## What it does

One scan per trading day:

1. Pull the full A-share universe (Sina `hs_a` + `hs_bjs`, BSE included so it
   can be counted and excluded).
2. Apply ordered exclusions (ST, delisting, suspended, new listings, boards,
   micro caps, illiquid names, insufficient data).
3. Score every survivor on fundamental / trend / flow percentiles.
4. Keep the top `top_n` stocks (max 3 per industry).
5. Review the top 30 pool with news + announcements + structured red flags
   through the configured LLM.
6. Persist the run, diff it against the previous successful run, and update the
   watchlist items and view.

The run is a `ScanRun` envelope: `run_id, trade_date, status, ai_status,
duration_s, funnel, sources, picks, scores, changes, counts, news/announcements
per finalist, red flags, request metrics`.

## Pipeline and funnel stages

Each stock is counted once, at the first rule it fails:

1. `universe`
2. `excluded:bse_unsupported`
3. `excluded:st`
4. `excluded:delisting`
5. `excluded:suspended`
6. `excluded:new_listing` (listed < 365 days)
7. `excluded:board_disabled` (STAR / ChiNext can be turned off)
8. `excluded:micro_cap` (total market cap < 3 bn CNY)
9. `excluded:illiquid` (20-day average amount < 50 m CNY, from K-lines)
10. `insufficient_data:fundamentals` (no published report)
11. `insufficient_data:trend` (K-lines missing, failed or < `min_kline_bars`)
12. `scored`
13. `finalists` (top_n after the industry cap)
14. `ai_demoted`
15. `watchlist`

Every scored or excluded stock carries an `exclusion_reason` (None if scored).

## Scoring
`score = 0.40·fundamental + 0.35·trend + 0.25·flow` (weights configurable via
settings; **never renormalised**). Sub-scores are 0-100 cross-sectional
percentiles within the scored set.

- **fundamental** (0.40) = 0.35·quality + 0.25·growth + 0.25·valuation +
  0.15·leverage. Quality: ROE, gross margin, OCF/EPS (clipped 0-3). Growth:
  revenue/profit YoY, winsorised. Valuation: inverse PE (industry-relative when
  the industry group has >= 5 scored members) and inverse PB; loss-makers take
  the bottom slot. Financials/brokers are NOT penalised for leverage (neutral
  50 + `leverage_neutral_financial` flag).
- **trend** (0.35) = 0.30·MA alignment + 0.20·distance-to-250d-high +
  0.15·60d relative return vs CSI 300 + 0.15·120d relative return +
  0.10·volume expansion (20d/60d) + 0.10·20d momentum, minus an
  overheated-move penalty and an extreme-volatility penalty.
- **flow** (0.25) = percentile of the float-cap-scaled THS net-flow blend
  `0.2·flow_1d + 0.3·flow_3d + 0.5·flow_5d` plus a capped Dragon-Tiger
  net-buy bonus.

**Missing-data rules (no renormalisation):**
- Missing flow → flow = neutral **50**, flag `flow_missing`, reason line.
- Missing fundamentals or missing/short K-lines (< `min_kline_bars`, default
  120) → not eligible (`insufficient_data:fundamentals` / `insufficient_data:trend`).
- **Northbound holdings are info-only** (never scored) and appear in the AI
  evidence pack only.

## Data sources and caveats
| Signal | Source | Caveats |
|---|---|---|
| Universe snapshot | Sina `hs_a` + `hs_bjs` lists | Free unofficial endpoints; fields can change |
| Listing date / industry | akshare exchange name lists | — |
| Fundamentals | akshare quarterly indicators, latest published period per stock | BSE rows fetched but funnel-excluded |
| K-lines (qfq) | Tencent `fqkline` + SQLite cache + Sina snapshot append | **Eastmoney push2 hosts may be blocked on some networks**; Tencent is the working path here |
| Money flow | THS per-stock net flow (`stock_fund_flow_individual`) | Prompt + pack label it exactly: **`THS (同花顺) net fund flow - not Eastmoney main-force, not northbound/institutional`** (never "main force" / 主力, never northbound) |
| Dragon-Tiger | Eastmoney datacenter LHB | — |
| Northbound | semi-annual per-stock holdings | **info only, never scored** (daily disclosure stopped 2024-08) |
| News / announcements | Eastmoney search-api + np-anotice; Sina GBK fallback | summary-only for news; announcements have full text |
| Red flags | pledge ratio (EM), holder changes (EM datacenter), earnings forecasts (EM), title scan on announcements | material: `pledge_high`, `holder_selldown`, `earnings_warning`, `regulatory`; severe: `csrc_investigation`, `audit_opinion`, `penalty_fraud`, `delisting_risk`. Every flag carries `severity: "material" \| "severe"` |

## Politeness and metrics
Per-host sustained rates: Tencent K-lines <= 1 req/s (refetch and backfill),
Sina <= 2, Eastmoney datacenter/news/announcements <= 2, THS <= 2, LLM
concurrency <= 3. Overridable via `VIBE_TRADING_KLINE_RATE_PER_SEC`,
`VIBE_TRADING_SINA_RATE_PER_SEC`, `VIBE_TRADING_EASTMONEY_RATE_PER_SEC`,
`VIBE_TRADING_THS_RATE_PER_SEC` (declared in `agent/src/config/env_schema.py`).
Ten consecutive throttle-type errors (403/429/501, connection reset) **stop
that source for the run** (`throttle stop`), marked in the source status. Every
run records per-host request metrics
(`requests = {host: {count, errors, elapsed_s, effective_rps}}`); the CLI
prints a REQUESTS block.

## K-line cache and backfill
K-lines live in SQLite at `get_runtime_root()/market_scanner/cache/`. The
daily bar is appended from the Sina snapshot; a prev-close mismatch triggers a
full refetch of that stock only. The run reports cache coverage. Backfill is
resumable:

```powershell
R\.venv\Scripts\vibe-trading.exe watchlist backfill --rate 1.0 --max 500
```
- The top `ai_review_pool` (default 30) finalists get an LLM review via
  `ChatLLM`, configured from `agent/.env` (`LANGCHAIN_PROVIDER`,
  `LANGCHAIN_MODEL_NAME`, provider key + base URL; empty key => the scan
  finishes and picks are marked `not_reviewed` / "not AI-reviewed").
- The model sees only the supplied pack: metrics, sub-scores, rule reasons,
  news (N1..Nn), announcements (A1..An), red flags (R1..Rn) and a fixed
  `flow_label`. Every verdict must cite item ids; the pipeline strips
  uncited claims.
- **Flow wording (prompt + pack):** the flow sub-score and the flow metrics
  are labelled exactly
  `THS (同花顺) net fund flow - not Eastmoney main-force, not northbound/institutional`;
  the model must never call them main-force (主力) or northbound money.
- **Material red flags force demotion:** `regulatory`, `earnings_warning`,
  `holder_selldown`, `pledge_high` must be cited in a reason or red flag,
  otherwise the verdict is forced to `avoid` with a note naming the flag ids
  (`red flag R1.. not addressed by model`).
- **SEVERE red flags force `avoid` unless resolved:** the severe types are
  `csrc_investigation` (立案告知书 / 立案调查 / 立案通知[/书] / 证监会…立案 /
  中国证券监督管理委员会…立案), `audit_opinion` (保留意见 / 否定意见 /
  无法表示意见 in audit-report titles — a clean 无保留意见 is never flagged),
  `penalty_fraud` (行政处罚决定书 / 市场禁入 / 欺诈发行 / 财务造假) and
  `delisting_risk` (退市风险 / 终止上市). They carry `severity: "severe"`.
  The prompt requires `avoid` unless the pack clearly shows the issue
  resolved; the code enforces it — an unresolved severe flag (uncited, or
  cited without an explicit resolution such as 已结案 / 撤销 / 解除 /
  整改完成 / resolved / closed) forces `avoid` with note
  `severe red flag R# (<type>) not resolved`.
- Strict JSON output; invalid output gets **one retry** with the validation
  error appended. Stocks still failing are marked `review failed`
  (`ai_status`), never silently dropped.

## Watchlist rules
- Changes vs the previous successful run are labelled `new`, `kept` or
  `dropped`. `dropped` shows the new rank, the exclusion reason, or "no
  data in this run".
- **Manual adds are pinned** and never auto-removed.
- **Removed codes are suppressed** and never auto-re-added. If a suppressed
  code qualifies again it is listed as "qualifies again - you removed it"
  with a restore action.
- **Stale marker:** set when the latest completed trading day has no
  successful scan.

## CLI

```text
vibe-trading watchlist scan [--no-ai] [--top N] [--exclude-star] [--exclude-chinext] [--json]
vibe-trading watchlist list
vibe-trading watchlist add 600519 --note "manual pin"
vibe-trading watchlist remove 600519
vibe-trading watchlist restore 600519
vibe-trading watchlist history --limit 20
vibe-trading watchlist show <run_id>
vibe-trading watchlist schedule on --time 18:00 | off | status
vibe-trading watchlist backfill [--rate 1.0] [--max 500]
```
Examples (PowerShell: `R\.venv\Scripts\vibe-trading.exe`; bash: `R/.venv/Scripts/vibe-trading.exe`):
```text
vibe-trading watchlist scan --no-ai --top 10
vibe-trading watchlist list
```

## API
All routes require auth (`Depends(require_auth)`), mounted by `api_server.py`:
`GET /api/watchlist` (full view), `POST /api/watchlist/items` `{code, note?}`,
`DELETE /api/watchlist/items/{code}`, `POST /api/watchlist/items/{code}/restore`,
`POST /api/watchlist/scan` `{overrides?}` -> 202 `{job_id}` (409 if a scan is
running; runs in a background thread), `GET /api/watchlist/scan/status`
(`{running, job_id, stage, progress, started_at, finished_at, result, error}`),
`GET /api/watchlist/runs?limit=`, `GET /api/watchlist/runs/{run_id}`,
`GET|PUT /api/watchlist/settings`, `GET /api/watchlist/schedule`
(scheduler status merged with staleness).

## Scheduler
- An asyncio task started from the API-server lifespan: fires the scan at
  **18:00 Asia/Shanghai on trading days only** (trading calendar; weekday
  fallback honours the 15:00 close cutoff). **Off by default**.
- `VIBE_TRADING_WATCHLIST_SCHEDULE=1|0` overrides the stored setting.
- Catch-up on server start: if the fire time already passed on the latest
  completed trading day and there is no successful scan for that day, it
  runs once. Cross-process single-flight (lock file), `ScanAlreadyRunning`
  recorded, never crashes the server. **No OS-level scheduled tasks.**
  `watchlist schedule on --time 18:00` persists the setting.

## Scan settings (defaults)
| Setting | Default |
|---|---|
| top_n | 20 |
| min_total_mv | 3e9 (3 bn CNY) |
| min_avg_amount_20d | 5e7 (50 m CNY) |
| min_listing_days | 365 |
| include_star | True |
| include_chinext | True |
| weights | fundamental 0.40 / trend 0.35 / flow 0.25 |
| max_per_industry | 3 |
| min_kline_bars | 120 (kline_bars 260 fetched) |
| ai_enabled / ai_review_pool | True / 30 |
| news_days | 30 |

Change via `PUT /api/watchlist/settings` or the settings store; BSE is always
excluded as unsupported. STAR / ChiNext can be excluded per scan via CLI flags.

## Disclaimer
**Research only, not investment advice.** This feature places no orders and
never touches a broker. Data comes from free unofficial endpoints and can be
wrong, delayed or unavailable; verify anything before acting on it.
