"""Offline unit tests for market_scanner sources (CONTRACT section 3).

All HTTP/akshare calls are stubbed with small recorded fixtures under
``agent/tests/fixtures/market_scanner/``. No network access.
"""

from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.market_scanner.sources._common import (
    board_from_code,
    exchange_from_code,
    parse_ths_money,
)
from src.market_scanner.sources import calendar as cal_mod
from src.market_scanner.sources import flows as flows_mod
from src.market_scanner.sources import fundamentals as fund_mod
from src.market_scanner.sources import fundamentals as fund_cache_mod
from src.market_scanner.sources import kline as kline_mod
from src.market_scanner.sources import listing as listing_mod
from src.market_scanner.sources import universe as universe_mod
from src.market_scanner.sources.status import SourceStatus

FIXTURES = Path(__file__).parent / "fixtures" / "market_scanner"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Pure helpers: board mapping / units / THS parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "code,expected",
    [
        ("600519", "main_sh"), ("601318", "main_sh"), ("603000", "main_sh"), ("605111", "main_sh"),
        ("000001", "main_sz"), ("001979", "main_sz"), ("002594", "main_sz"), ("003816", "main_sz"),
        ("688981", "star"), ("689009", "star"),
        ("300750", "chinext"), ("301269", "chinext"), ("302132", "chinext"),
        ("920002", "bse"), ("430047", "bse"), ("833171", "bse"), ("871981", "bse"),
    ],
)
def test_board_mapping(code: str, expected: str) -> None:
    assert board_from_code(code) == expected


def test_exchange_mapping() -> None:
    assert exchange_from_code("600519") == "SH"
    assert exchange_from_code("300750") == "SZ"
    assert exchange_from_code("920002") == "BJ"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2.91亿", 2.91e8),
        ("20.10万", 20.10e4),
        ("-3.50亿", -3.50e8),
        ("2311.36万", 2311.36e4),
        ("1234", 1234.0),
        ("-", None),
        ("", None),
        ("N/A", None),
    ],
)
def test_parse_ths_money(text: str, expected: float | None) -> None:
    assert parse_ths_money(text) == expected


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def _sina_stub() -> "json.JSONDecoder":
    pages = _load("sina_universe_pages.json")
    calls = {"hs_a": 0, "hs_bjs": 0}

    def fetch(url: str, params: dict) -> str:
        node = params["node"]
        if "getHQNodeStockCount" in url:
            return {"hs_a": "6", "hs_bjs": "2"}.get(node, "[]")
        key = f"{node}_page{params['page']}"
        if key in pages:
            return json.dumps(pages[key])
        return "[]"

    return fetch


def test_fetch_universe_offline() -> None:
    df, st = universe_mod.fetch_universe(fetch=_sina_stub())
    assert st.status == "ok"
    assert st.rows == 8  # 6 hs_a + 2 hs_bjs
    assert list(df.columns) == [
        "symbol", "name", "exchange", "board", "price", "pct_chg", "volume",
        "amount", "pe", "pb", "total_mv", "float_mv", "turnover_rate",
        "is_st", "is_delisting", "is_suspended",
        # settlement (prev close) + OHLC: needed by the K-line snapshot-append
        # guard and the appended bar itself.
        "settlement", "open", "high", "low",
    ]
    assert df.index.name == "code"
    # BSE included
    assert "920002" in df.index and "430047" in df.index
    assert df.loc["920002", "board"] == "bse"
    assert df.loc["920002", "exchange"] == "BJ"
    # units: Sina mktcap 1e4 -> CNY (161010510.2088 * 1e4 = 1.6101e12)
    assert df.loc["600519", "total_mv"] == pytest.approx(161010510.2088 * 1e4)
    # ST / 退 / suspended flags
    assert bool(df.loc["601318", "is_st"]) is True
    assert bool(df.loc["000002", "is_st"]) is True  # *STExample
    assert bool(df.loc["000002", "is_delisting"]) is True  # name contains 退
    assert bool(df.loc["000002", "is_suspended"]) is True  # price 0.000
    assert bool(df.loc["600519", "is_suspended"]) is False
    # symbols
    assert df.loc["600519", "symbol"] == "sh600519"
    assert df.loc["920002", "symbol"] == "bj920002"


def test_fetch_universe_failure_returns_status_not_raise() -> None:
    def boom(url: str, params: dict) -> str:
        raise ConnectionError("RemoteDisconnected")

    df, st = universe_mod.fetch_universe(fetch=boom, max_pages=1)
    assert df.empty
    assert st.status == "failed"
    assert "ConnectionError" in (st.error or "")


def test_universe_degraded_when_one_node_fails() -> None:
    pages = _load("sina_universe_pages.json")

    def fetch(url: str, params: dict) -> str:
        node = params["node"]
        if "getHQNodeStockCount" in url:
            return {"hs_a": "6", "hs_bjs": "2"}.get(node, "[]")
        if node == "hs_bjs":
            raise TimeoutError("hs_bjs down")
        key = f"{node}_page{params['page']}"
        return json.dumps(pages.get(key, []))

    df, st = universe_mod.fetch_universe(fetch=fetch)
    assert st.status == "degraded"
    assert st.rows == 6
    assert "hs_bjs" in (st.error or "")


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
def _listing_fetchers() -> dict[str, Callable[[], pd.DataFrame]]:
    data = _load("listing_lists.json")
    return {
        "sh": lambda: pd.DataFrame(data["sh_main"] + data["sh_star"]),
        "sz": lambda: pd.DataFrame(data["sz"]),
        "bj": lambda: pd.DataFrame(data["bj"]),
    }


def test_fetch_listing_info_offline() -> None:
    df, st = listing_mod.fetch_listing_info(fetchers=_listing_fetchers())
    assert st.status == "ok"
    assert st.rows == 6  # 2 sh main + 1 star + 2 sz + 1 bj
    assert list(df.columns) == ["list_date", "industry", "exchange", "board"]
    assert df.loc["600519", "list_date"] == dt.date(2001, 8, 27)
    assert df.loc["600519", "exchange"] == "SH"
    # board is stamped from the code prefix (300/301/302 -> chinext,
    # 688/689 -> star), never from the vendor lists.
    assert df.loc["300750", "board"] == "chinext"
    assert df.loc["688981", "board"] == "star"
    assert df.loc["600519", "board"] == "main_sh"
    assert df.loc["000001", "board"] == "main_sz"
    assert df.loc["920002", "board"] == "bse"
    assert df.loc["920002", "exchange"] == "BJ"
    assert df.loc["920002", "industry"] == "汽车制造业"
    assert df.loc["300750", "list_date"] == dt.date(2018, 6, 11)
    # as_of: the listing lists carry a snapshot date; with no dates in the
    # fixture it falls back to fetch date (non-null).
    assert st.as_of


def test_fetch_listing_info_degraded_when_bj_fails() -> None:
    data = _load("listing_lists.json")
    fetchers = _listing_fetchers()

    def bj_boom() -> pd.DataFrame:
        raise ConnectionError("bj down")

    fetchers["bj"] = bj_boom
    df, st = listing_mod.fetch_listing_info(fetchers=fetchers)
    assert st.status == "degraded"
    assert st.rows == 5
    assert "920002" not in df.index


def test_fetch_listing_info_degraded_when_list_missing() -> None:
    """A silently missing exchange list (0 rows) must be degraded with the
    reason, not 'ok' (boss requirement #2); per-list counts always shown."""
    data = _load("listing_lists.json")
    fetchers = _listing_fetchers()
    fetchers["sh"] = lambda: pd.DataFrame()  # SH silently returns nothing

    df, st = listing_mod.fetch_listing_info(fetchers=fetchers)
    assert st.status == "degraded"
    assert "missing lists: sh" in (st.detail or "")
    assert "per-list" in (st.detail or "")


# --------------------------------------------------------------------------- #
# Fundamentals: per-stock fallback period
# --------------------------------------------------------------------------- #
def _fund_fetchers() -> dict[str, Callable]:
    data = _load("fundamentals_boards.json")
    import akshare  # noqa: F401 — module import only

    def yjbb(date: str, **kw) -> pd.DataFrame:
        if date == "20260630":
            return pd.DataFrame(data["yjbb_20260630"])
        if date == "20260331":
            return pd.DataFrame(data["yjbb_20260331"])
        return pd.DataFrame()

    def zcfz(date: str, **kw) -> pd.DataFrame:
        if date == "20260630":
            return pd.DataFrame(data["zcfz_20260630"])
        if date == "20260331":
            return pd.DataFrame(data["zcfz_20260331"])
        return pd.DataFrame()

    def xjll(date: str, **kw) -> pd.DataFrame:
        if date == "20260630":
            return pd.DataFrame(data["xjll_20260630"])
        return pd.DataFrame()

    return {"yjbb": yjbb, "zcfz": zcfz, "xjll": xjll}


def test_fetch_fundamentals_per_stock_fallback() -> None:
    fns = _fund_fetchers()
    df, st = fund_mod.fetch_fundamentals(
        dt.date(2026, 10, 13), yjbb_fn=fns["yjbb"], zcfz_fn=fns["zcfz"], xjll_fn=fns["xjll"],
    )
    assert st.rows == 3
    assert df.index.name == "code"
    # 600519 has 2026Q2 everywhere -> stays Q2
    assert df.loc["600519", "report_period"] == "2026Q2"
    assert df.loc["600519", "roe"] == pytest.approx(30.5)
    assert df.loc["600519", "debt_ratio"] == pytest.approx(20.1)
    assert df.loc["600519", "industry"] == "酿酒行业"
    # 000001 missing from Q2 yjbb -> falls back to Q1
    assert df.loc["000001", "report_period"] == "2026Q1"
    # 688981 has Q2
    assert df.loc["688981", "report_period"] == "2026Q2"
    assert df.loc["688981", "ocf_per_share"] == pytest.approx(2.2)


def test_fetch_fundamentals_all_boards_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(date: str, **kw) -> pd.DataFrame:
        raise ConnectionError("dc down")

    monkeypatch.setattr(fund_cache_mod, "_cache_path", lambda p, t: tmp_path / f"{p}_{t}.pkl")
    df, st = fund_mod.fetch_fundamentals(
        dt.date(2026, 10, 13), yjbb_fn=boom, zcfz_fn=boom, xjll_fn=boom,
    )
    assert df.empty
    assert st.status == "failed"
    assert "ConnectionError" in (st.error or "")


def test_fundamentals_future_period_never_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """as_of=2026-10-13 must not fetch 2026Q3/Q4 (period-end in the future)."""
    requested: list[str] = []

    def yjbb(date: str, **kw) -> pd.DataFrame:
        requested.append(f"yjbb:{date}")
        if date == "20260630":
            return pd.DataFrame(_load("fundamentals_boards.json")["yjbb_20260630"])
        return pd.DataFrame()

    monkeypatch.setattr(fund_cache_mod, "_cache_path", lambda p, t: tmp_path / f"{p}_{t}.pkl")
    df, st = fund_mod.fetch_fundamentals(
        dt.date(2026, 10, 13), yjbb_fn=yjbb, zcfz_fn=yjbb, xjll_fn=yjbb,
    )
    assert not any("20260930" in r or "20261231" in r for r in requested)
    assert st.status == "ok"
    assert df.loc["600519", "report_period"] == "2026Q2"


def test_fundamentals_empty_period_not_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Newest period published-empty -> skipped, status stays ok."""
    data = _load("fundamentals_boards.json")
    calls: list[str] = []

    def yjbb(date: str, **kw) -> pd.DataFrame:
        calls.append(date)
        if date == "20260630":
            return pd.DataFrame(data["yjbb_20260630"])
        return pd.DataFrame()  # 2026Q3 not published yet

    monkeypatch.setattr(fund_cache_mod, "_cache_path", lambda p, t: tmp_path / f"{p}_{t}.pkl")
    df, st = fund_mod.fetch_fundamentals(
        dt.date(2026, 10, 13), yjbb_fn=yjbb, zcfz_fn=yjbb, xjll_fn=yjbb,
    )
    assert st.status == "ok"  # empty period is normal, NOT degraded
    assert df.loc["600519", "report_period"] == "2026Q2"


def test_fundamentals_earlier_period_fill_only_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the newest period covers all stocks, earlier periods aren't fetched."""
    data = _load("fundamentals_boards.json")
    periods_seen: list[str] = []

    def yjbb(date: str, **kw) -> pd.DataFrame:
        periods_seen.append(date)
        if date == "20260630":
            # Covers all three stocks -> no earlier period needed.
            return pd.DataFrame(data["yjbb_20260630"] + [
                {"股票代码": "000001", "股票简称": "平安银行", "每股收益": "1.1",
                 "净资产收益率": "5.0", "每股净资产": "20.0", "所处行业": "银行"}
            ])
        return pd.DataFrame()

    monkeypatch.setattr(fund_cache_mod, "_cache_path", lambda p, t: tmp_path / f"{p}_{t}.pkl")
    df, st = fund_mod.fetch_fundamentals(
        dt.date(2026, 10, 13), yjbb_fn=yjbb, zcfz_fn=yjbb, xjll_fn=yjbb,
    )
    # yjbb is called once per (period × table); all three tables share the
    # stub, so the newest period shows 3 calls. The key assertion: after the
    # newest period covers everything, no earlier period adds coverage, so
    # 20260331 is walked once (3 calls) and 20251231+ are never touched.
    assert df.loc["000001", "report_period"] == "2026Q2"
    assert periods_seen.count("20260630") == 3
    assert periods_seen.count("20260331") == 3
    assert "20251231" not in periods_seen
    assert "20250930" not in periods_seen


def test_fundamentals_cache_hit_and_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call with a fresh cache hits no network; stale entry refetches."""
    data = _load("fundamentals_boards.json")
    calls = {"n": 0}

    def yjbb(date: str, **kw) -> pd.DataFrame:
        calls["n"] += 1
        if date == "20260630":
            return pd.DataFrame(data["yjbb_20260630"])
        return pd.DataFrame()

    cache_root = tmp_path / "cache"

    def cache_path(period: str, table: str) -> Path:
        return cache_root / f"{period}_{table}.pkl"

    monkeypatch.setattr(fund_cache_mod, "_cache_path", cache_path)

    fund_mod.fetch_fundamentals(dt.date(2026, 10, 13), yjbb_fn=yjbb, zcfz_fn=yjbb, xjll_fn=yjbb)
    cold_calls = calls["n"]
    assert cold_calls >= 1

    # Warm: everything cached -> zero additional network calls.
    fund_mod.fetch_fundamentals(dt.date(2026, 10, 13), yjbb_fn=yjbb, zcfz_fn=yjbb, xjll_fn=yjbb)
    assert calls["n"] == cold_calls

    # Stale: force the saved_at timestamp past the current-period TTL (12h).
    for pkl in cache_root.glob("20260630_*.pkl"):
        with pkl.open("rb") as fh:
            payload = pickle.load(fh)
        payload["saved_at"] -= 13 * 3600
        with pkl.open("wb") as fh:
            pickle.dump(payload, fh)
    fund_mod.fetch_fundamentals(dt.date(2026, 10, 13), yjbb_fn=yjbb, zcfz_fn=yjbb, xjll_fn=yjbb)
    assert calls["n"] > cold_calls  # stale -> refetched


# --------------------------------------------------------------------------- #
# Klines
# --------------------------------------------------------------------------- #
def _tencent_stub(klines: dict | None = None) -> Callable:
    data = klines if klines is not None else _load("tencent_klines.json")

    def fetch(url: str, params: dict) -> str:
        param = params["param"].split(",")[0]
        if param in data:
            return json.dumps(data[param])
        if param in data:
            return json.dumps(data[param])
        return json.dumps({"code": 0, "data": {}})

    return fetch


def test_fetch_klines_offline() -> None:
    frames, st = kline_mod.fetch_klines(
        ["sh600519", "sh000300"], fetch=_tencent_stub(), progress=None,
    )
    assert st.status == "ok"
    assert st.rows == 2
    assert "600519" in frames and "000300" in frames
    df = frames["600519"]
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert df["date"].is_monotonic_increasing
    assert df.iloc[-1]["close"] == pytest.approx(1288.00)
    # amount = close * volume * 100 (volume is lots, live-verified vs Sina)
    assert df.iloc[-1]["amount"] == pytest.approx(1288.00 * 31556 * 100)
    assert "amount estimated from close*volume*100" in (st.detail or "")


def test_kline_amount_fallback_and_7th_element() -> None:
    """6-field rows get close*volume*100; a 7th dict/None is never read as amount."""
    payload = {
        "code": 0,
        "data": {
            "sh600519": {
                "qfqday": [
                    # 6 fields only -> amount from close*volume*100
                    ["2026-10-09", "1262.00", "1283.00", "1290.00", "1261.00", "30000"],
                    # 7th element is a dict (quote metadata) -> ignored
                    ["2026-10-12", "1262.95", "1288.00", "1290.00", "1262.95", "31556",
                     {"name": "贵州茅台", "je": "should-not-leak"}],
                ]
            }
        },
    }
    df = kline_mod._parse_kline_payload(json.dumps(payload), "sh600519")
    assert len(df) == 2
    # dict 7th element must NOT appear as amount
    assert df.iloc[1]["amount"] == pytest.approx(1288.00 * 31556 * 100)
    assert df.iloc[0]["amount"] == pytest.approx(1283.00 * 30000 * 100)
    assert "should-not-leak" not in df.to_json()


def test_kline_rate_limiter_spacing_fake_clock() -> None:
    """min_interval spacing holds under a fake clock; no real sleeping."""
    lim = kline_mod.KlineRateLimiter(rate_per_sec=8.0)
    now = {"t": 1000.0}
    slept: list[float] = []

    def clock() -> float:
        return now["t"]

    def sleeper(d: float) -> None:
        slept.append(d)
        now["t"] += d

    fire_times = []
    for _ in range(10):
        lim.wait(clock=clock, sleeper=sleeper)
        fire_times.append(now["t"])
    # 8 req/s -> min_interval 0.125 s; 10 requests span >= 9 intervals
    assert fire_times[-1] - fire_times[0] >= 9 * 0.125 - 0.06
    # total sleep ~= spacing sum (jitter <= 0.05 each)
    assert sum(slept) <= 10 * (0.125 + 0.05) + 0.01


def test_kline_circuit_breaker_halves_rate() -> None:
    lim2 = kline_mod.KlineRateLimiter(rate_per_sec=8.0)
    trips = 0
    for i in range(60):
        ok = i < 39  # 39 ok, then 11 fails -> 11/50 = 22% > 20%
        if lim2.note_result(ok):
            trips += 1
    assert trips >= 1
    assert lim2.min_interval == pytest.approx(1.0 / 4.0)  # halved once (8 -> 4)
    # second window of failures halves again
    for _ in range(50):
        lim2.note_result(False)
    assert lim2.min_interval == pytest.approx(1.0 / 2.0)


def test_fetch_klines_breaker_noted_in_detail() -> None:
    """A failing limiter's trip lands in status.detail."""
    calls = {"n": 0}

    def flaky(url: str, params: dict) -> str:
        calls["n"] += 1
        raise ConnectionError("boom")

    lim = kline_mod.KlineRateLimiter(rate_per_sec=100.0)
    frames, st = kline_mod.fetch_klines(
        [f"sh6{i:05d}" for i in range(60)], fetch=flaky, limiter=lim, workers=6,
    )
    assert st.status == "failed"
    assert "breaker_trips=" in (st.detail or "")


def test_fetch_klines_progress_and_failure_count() -> None:
    calls = []

    def progress(done: int, total: int) -> None:
        calls.append((done, total))

    # sh000300 uses "day" (not qfqday) — still parsed; bj920002_error is empty -> failure
    frames, st = kline_mod.fetch_klines(
        ["sh600519", "bj920002_error", "sh000300"],
        fetch=_tencent_stub(), progress=progress,
    )
    assert calls[-1] == (3, 3)
    assert st.status == "degraded"
    assert "failed=1" in (st.detail or "")
    assert "bj920002_error" not in frames
    assert "600519" in frames and "000300" in frames


def test_fetch_klines_all_fail() -> None:
    def boom(url: str, params: dict) -> str:
        raise ConnectionError("net down")

    cache = kline_mod.KlineCache(
        path=Path(__import__("tempfile").mkdtemp()) / "k.sqlite3")
    frames, st = kline_mod.fetch_klines(["sh600519"], fetch=boom, cache=cache)
    assert frames == {}
    assert st.status == "failed"
    assert "missing=1" in (st.detail or "")


def test_fetch_benchmark_offline() -> None:
    df, st = kline_mod.fetch_benchmark("sh000300", fetch=_tencent_stub())
    assert st.status == "ok"
    assert df is not None and len(df) == 2
    assert df.iloc[-1]["close"] == pytest.approx(4230.55)


# --------------------------------------------------------------------------- #
# Flows
# --------------------------------------------------------------------------- #
def _ths_fetchers() -> dict[str, Callable[[], pd.DataFrame]]:
    data = _load("ths_flow.json")
    return {
        "即时": lambda: pd.DataFrame(data["ths_instant"]),
        "3日排行": lambda: pd.DataFrame(data["ths_3d"]),
        "5日排行": lambda: pd.DataFrame(data["ths_5d"]),
    }


def test_fetch_fund_flow_offline() -> None:
    df, st = flows_mod.fetch_fund_flow(fetchers=_ths_fetchers())
    assert st.status == "ok"
    assert "THS net flow, not Eastmoney main-force" in (st.detail or "")
    assert list(df.columns) == ["flow_1d", "pct_chg_1d", "flow_3d", "pct_chg_3d", "flow_5d", "pct_chg_5d"]
    # THS text parsed to CNY
    assert df.loc["600519", "flow_1d"] == pytest.approx(2000.00e4)
    assert df.loc["600519", "flow_3d"] == pytest.approx(-3.50e8)
    assert df.loc["600519", "flow_5d"] == pytest.approx(8.16e8)
    assert df.loc["600519", "pct_chg_3d"] == pytest.approx(-1.20)
    # stock only in 3日 gets flow_3d but NaN flow_1d/5d
    assert df.loc["300530", "flow_3d"] == pytest.approx(2311.36e4)
    assert pd.isna(df.loc["300530", "flow_1d"])


def test_fetch_fund_flow_failure() -> None:
    def boom() -> pd.DataFrame:
        raise ConnectionError("ths down")

    df, st = flows_mod.fetch_fund_flow(
        fetchers={"即时": boom, "3日排行": boom, "5日排行": boom},
    )
    assert df.empty
    assert st.status == "failed"


def test_fetch_dragon_tiger_offline() -> None:
    data = _load("flows_and_misc.json")

    def lhb_fn(start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame(data["lhb"])

    df, st = flows_mod.fetch_dragon_tiger(dt.date(2026, 10, 12), days=30, lhb_fn=lhb_fn)
    assert st.status == "ok"
    assert df.loc["600519", "lhb_count"] == 2
    assert df.loc["600519", "lhb_net_buy"] == pytest.approx(120000000.0 - 30000000.0)
    assert df.loc["600519", "lhb_last_date"] == "2026-10-12"
    assert "日涨幅偏离值达7%" in (df.loc["600519", "lhb_reasons"] or "")


def test_fetch_dragon_tiger_failure() -> None:
    def boom(start_date: str, end_date: str) -> pd.DataFrame:
        raise ConnectionError("lhb down")

    df, st = flows_mod.fetch_dragon_tiger(dt.date(2026, 10, 12), lhb_fn=boom)
    assert df.empty
    assert st.status == "failed"


def test_fetch_northbound_offline() -> None:
    data = _load("flows_and_misc.json")

    def fetch(url: str, params: dict) -> dict:
        return data["northbound"]

    df, st = flows_mod.fetch_northbound_info("2026-06-30", fetch=fetch)
    assert st.status == "ok"
    assert st.rows == 2
    assert df.loc["300750", "nb_hold_ratio"] == pytest.approx(20.28)
    assert df.loc["600519", "nb_hold_mv"] == pytest.approx(64400000000.0)
    assert df.loc["600519", "nb_report_date"] == "2026-06-30"


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
def test_calendar_from_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    days = _load("flows_and_misc.json")["calendar_days"]

    def hist_fn() -> pd.DataFrame:
        return pd.DataFrame({"trade_date": pd.to_datetime(days)})

    monkeypatch.setattr(cal_mod, "_cache_path", lambda: tmp_path / "trade_calendar.json")
    calendar, st = cal_mod.load_calendar(hist_fn=hist_fn, force_refresh=True)
    assert st.status == "ok"
    assert st.rows == len(days)
    assert calendar.is_trading_day(dt.date(2026, 10, 13)) is True
    assert calendar.is_trading_day(dt.date(2026, 10, 17)) is False  # Saturday
    # latest completed: 2026-10-13 16:00 -> today (trading + past 15:00)
    assert calendar.latest_completed_trading_day(dt.datetime(2026, 10, 13, 16, 0)) == dt.date(2026, 10, 13)
    # 2026-10-13 10:00 -> previous trading day 2026-10-12
    assert calendar.latest_completed_trading_day(dt.datetime(2026, 10, 13, 10, 0)) == dt.date(2026, 10, 12)
    # Sunday -> previous trading day Friday
    assert calendar.latest_completed_trading_day(dt.datetime(2026, 10, 18, 16, 0)) == dt.date(2026, 10, 16)


def test_calendar_cache_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    days = _load("flows_and_misc.json")["calendar_days"]
    cache = tmp_path / "trade_calendar.json"
    cache.write_text(json.dumps({"trading_days": days}), encoding="utf-8")
    monkeypatch.setattr(cal_mod, "_cache_path", lambda: cache)

    def hist_fn() -> pd.DataFrame:  # must NOT be called on a cache hit
        raise AssertionError("network called despite cache")

    calendar, st = cal_mod.load_calendar(hist_fn=hist_fn)
    # Fresh cache read is 'ok' (boss requirement #3): it came from a previous
    # successful Sina fetch; detail still says source=cache.
    assert st.status == "ok"
    assert st.detail and "cache" in (st.detail or "")
    assert calendar.is_trading_day(dt.date(2026, 10, 12)) is True


def test_calendar_weekday_fallback_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cal_mod, "_cache_path", lambda: tmp_path / "missing.json")

    def hist_fn() -> pd.DataFrame:
        raise ConnectionError("sina down")

    calendar, st = cal_mod.load_calendar(hist_fn=hist_fn, force_refresh=True)
    assert st.status == "degraded"
    assert "ConnectionError" in (st.error or "")
    # weekday fallback: Monday 2026-10-12 trades, Saturday does not
    assert calendar.is_trading_day(dt.date(2026, 10, 12)) is True
    assert calendar.is_trading_day(dt.date(2026, 10, 17)) is False


# --------------------------------------------------------------------------- #
# SourceStatus shape
# --------------------------------------------------------------------------- #
def test_source_status_to_dict() -> None:
    st = SourceStatus(name="x", status="failed", rows=0, latency_s=1.2345, error="boom")
    d = st.to_dict()
    assert d["name"] == "x" and d["status"] == "failed"
    assert d["latency_s"] == 1.234  # rounded to 3dp
    assert json.dumps(d, ensure_ascii=False)  # JSON-serialisable


# --------------------------------------------------------------------------- #
# as_of per source (boss requirement #5)
# --------------------------------------------------------------------------- #
def test_as_of_universe() -> None:
    df, st = universe_mod.fetch_universe(fetch=_sina_stub())
    assert st.as_of  # session date or intraday flag


def test_as_of_listing() -> None:
    df, st = listing_mod.fetch_listing_info(fetchers=_listing_fetchers())
    assert st.as_of  # list files date (fetch date)


def test_as_of_calendar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    days = _load("flows_and_misc.json")["calendar_days"]
    cache = tmp_path / "trade_calendar.json"
    cache.write_text(json.dumps({"trading_days": days}), encoding="utf-8")
    monkeypatch.setattr(cal_mod, "_cache_path", lambda: cache)
    calendar, st = cal_mod.load_calendar(hist_fn=lambda: (_ for _ in ()).throw(AssertionError()))
    assert st.as_of == days[-1]  # calendar range end


def test_as_of_fundamentals() -> None:
    fns = _fund_fetchers()
    df, st = fund_mod.fetch_fundamentals(
        dt.date(2026, 10, 13), yjbb_fn=fns["yjbb"], zcfz_fn=fns["zcfz"], xjll_fn=fns["xjll"],
    )
    assert st.as_of and "2026-06-30" in st.as_of  # report period(s) used


# --------------------------------------------------------------------------- #
# (2)+(3): news/announcements as_of + announcement source label
# --------------------------------------------------------------------------- #
def test_news_status_as_of_and_ann_source() -> None:
    import src.market_scanner.sources.news as news_mod

    ann_payload = json.dumps({"data": {"list": [
        {"notice_date": "2026-10-10 00:00:00", "title": "年报",
         "art_code": "AN1", "columns": [{"column_name": "年报"}]},
        {"notice_date": "2026-10-12 00:00:00", "title": "业绩预告",
         "art_code": "AN2", "columns": []},
    ]}})
    items, st = news_mod.fetch_announcements(
        "600519", days=30, fetch=lambda url, **kw: ann_payload)
    assert st.as_of == "2026-10-12"  # latest item date
    assert all(i.get("source") == "Eastmoney announcements" for i in items)
