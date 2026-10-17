"""F4: transport errors count toward the kline outage stop (offline).

Fake clock + fake fetchers only. No network, no I/O outside tmp_path.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from src.market_scanner.sources.kline import (
    KlineRateLimiter,
    fetch_klines,
    is_throttle_error,
    is_transport_error,
)
from src.market_scanner.sources.kline_cache import KlineCache


# ---------------------------------------------------------------- helpers ---
class _FakeClock:
    """Thread-safe fake clock (worker threads advance it per attempt)."""

    def __init__(self) -> None:
        import threading as _threading
        self.t = 1000.0
        self._lock = _threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.t

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.t += seconds


def _bars(dates, close=10.0):
    return pd.DataFrame([{
        "date": pd.Timestamp(d), "open": close, "high": close,
        "low": close, "close": close, "volume": 1000.0,
        "amount": close * 1000.0 * 100,
    } for d in dates])


def _payload(symbol: str, dates, close=10.0) -> str:
    rows = [[d, str(close), str(close), str(close), str(close), "1000"]
            for d in dates]
    return json.dumps({"code": 0, "data": {symbol: {"qfqday": rows}}})


def _cache(tmp_path: Path) -> KlineCache:
    return KlineCache(path=tmp_path / "k.sqlite3")


# ------------------------------------------------------- classification ----
def test_is_transport_error_classification():
    assert is_transport_error(TimeoutError("ReadTimeout: timed out")) is True
    assert is_transport_error(ConnectionError("no route")) is True
    assert is_transport_error(Exception("RemoteDisconnected: Remote end closed")) is True
    assert is_transport_error(ValueError("no kline rows for sh600000")) is False
    assert is_transport_error(Exception("boom")) is False
    # throttle-class stays throttle-class (checked first in the fetcher)
    assert is_transport_error(Exception("HTTP 501 Server Error")) is False
    assert is_transport_error(Exception("Connection reset by peer")) is False
    assert is_throttle_error(Exception("HTTP 501 Server Error")) is True


# ------------------------------------------------------- consecutive rule --
def test_ten_consecutive_transport_errors_stop():
    clock = _FakeClock()
    lim = KlineRateLimiter(rate_per_sec=100.0, clock=clock)
    for _ in range(9):
        out = lim.note_result(False, transport=True, elapsed_s=0.01)
        assert out["unavailable"] is False
    out = lim.note_result(False, transport=True, elapsed_s=0.01)
    assert out["unavailable"] is True
    assert lim.unavailable is True
    assert lim.throttle_stop is True
    assert lim.unavailable_reason is not None
    assert "consecutive" in lim.unavailable_reason


def test_mixed_throttle_and_transport_count_together():
    clock = _FakeClock()
    lim = KlineRateLimiter(rate_per_sec=100.0, clock=clock)
    for _ in range(5):
        lim.note_result(False, throttle=True, elapsed_s=0.01)
    for _ in range(5):
        lim.note_result(False, transport=True, elapsed_s=0.01)
    assert lim.unavailable is True


def test_success_resets_consecutive_transport_errors():
    clock = _FakeClock()
    lim = KlineRateLimiter(rate_per_sec=100.0, clock=clock)
    for _ in range(9):
        lim.note_result(False, transport=True, elapsed_s=0.01)
    lim.note_result(True, elapsed_s=0.01)
    lim.note_result(False, transport=True, elapsed_s=0.01)
    assert lim.unavailable is False
    assert lim.throttle_stop is False


def test_permanent_failures_do_not_stop():
    clock = _FakeClock()
    lim = KlineRateLimiter(rate_per_sec=100.0, clock=clock)
    for _ in range(60):
        out = lim.note_result(False, elapsed_s=0.01)
    assert out["unavailable"] is False
    assert lim.unavailable is False
    assert lim.throttle_stop is False


# ------------------------------------------------------- window rule -------
def test_window_rule_needs_elapsed_time_and_majority_errors():
    # 12 fails / 20 notes (60%) but only 40 s elapsed -> no stop; the same
    # mix spread over 140 s -> stop. Runs of fails stay under 3, so the
    # consecutive rule never fires in either leg.
    clock = _FakeClock()
    lim = KlineRateLimiter(rate_per_sec=100.0, stop_window=20,
                           stop_min_error_rate=0.5,
                           stop_min_elapsed_s=120.0, clock=clock)
    for i in range(20):
        clock.advance(2.0)
        ok = (i % 5) < 2
        lim.note_result(ok, transport=(not ok), elapsed_s=0.01)
    assert lim.unavailable is False
    clock2 = _FakeClock()
    lim2 = KlineRateLimiter(rate_per_sec=100.0, stop_window=20,
                            stop_min_error_rate=0.5,
                            stop_min_elapsed_s=120.0, clock=clock2)
    for i in range(20):
        clock2.advance(7.0)
        ok = (i % 5) < 2
        lim2.note_result(ok, transport=(not ok), elapsed_s=0.01)
    assert lim2.unavailable is True
    assert "errors over the last" in (lim2.unavailable_reason or "")


# ------------------------------------------------------- fetch behaviour ---
def test_storm_stops_fast_with_fake_clock(tmp_path):
    """200 symbols, every fetch a ReadTimeout: stop after ~10 attempts.

    The stub sleeps briefly per attempt so the stop logic (not thread
    scheduling) decides the bound — the submissions-race with instant
    stubs would let dozens of attempts through before the main thread
    drains the first completions.
    """
    import time as _time

    clock = _FakeClock()
    calls = {"n": 0}

    def flaky(url, params):
        calls["n"] += 1
        clock.advance(15.0)  # each attempt burns a 15 s timeout (fake)
        _time.sleep(0.005)
        raise TimeoutError("ReadTimeout: HTTPSConnectionPool timed out")

    trade_date = dt.date(2026, 10, 15)
    lim = KlineRateLimiter(rate_per_sec=100.0, clock=clock)
    frames, st = fetch_klines(
        [f"sh6{i:05d}" for i in range(200)], cache=_cache(tmp_path),
        trade_date=trade_date, fetch=flaky, limiter=lim, workers=1,
        now_fn=lambda: dt.datetime(2026, 10, 15, 20, 0),
    )
    assert st.unavailable is True
    assert st.status == "failed"
    assert st.error is not None and "price source unavailable" in st.error
    assert "transport outage" in (st.detail or "")
    # workers=1: the 10th consecutive failure stops; a queued item may
    # already have been dequeued, hence the [10, 15] bound.
    assert 10 <= calls["n"] <= 15
    # fake-clock storm length is bounded (~10 x 15 s), not ~200 x 15 s.
    assert clock.t - 1000.0 <= 300.0
    assert getattr(st, "kline_reasons")["counts"]["transport_error"] >= 10


def test_storm_stops_with_parallel_workers(tmp_path):
    """Same storm at workers=6: stops early instead of attempting all 200."""
    import time as _time

    clock = _FakeClock()
    calls = {"n": 0}

    def flaky(url, params):
        calls["n"] += 1
        clock.advance(15.0)
        _time.sleep(0.01)
        raise TimeoutError("ReadTimeout: HTTPSConnectionPool timed out")

    trade_date = dt.date(2026, 10, 15)
    lim = KlineRateLimiter(rate_per_sec=100.0, clock=clock)
    frames, st = fetch_klines(
        [f"sh6{i:05d}" for i in range(200)], cache=_cache(tmp_path),
        trade_date=trade_date, fetch=flaky, limiter=lim, workers=6,
        now_fn=lambda: dt.datetime(2026, 10, 15, 20, 0),
    )
    assert st.unavailable is True
    assert st.status == "failed"
    assert calls["n"] < 200


def test_window_rule_stops_a_slow_burn_storm(tmp_path):
    """Interleaved ok/fail (consecutive never hits 10) still stops via the
    >50%-over-60-once-120s rule. workers=1 keeps completions paced so the
    stop latches deterministically (instant stubs would let all 150 finish
    before the main thread drains 60 notes)."""
    import time as _time

    clock = _FakeClock()

    def flaky(url, params):
        calls.append(1)
        clock.advance(2.5)
        _time.sleep(0.005)
        idx = int(params["param"].split(",")[0][3:])
        if idx % 6 == 5:
            return _payload(f"sh{idx:06d}", ["2026-10-14", "2026-10-15"])
        raise TimeoutError("ReadTimeout: timed out")

    calls: list = []
    trade_date = dt.date(2026, 10, 15)
    lim = KlineRateLimiter(rate_per_sec=100.0, clock=clock)
    frames, st = fetch_klines(
        [f"sh{i:06d}" for i in range(150)], cache=_cache(tmp_path),
        trade_date=trade_date, fetch=flaky, limiter=lim, workers=1,
        now_fn=lambda: dt.datetime(2026, 10, 15, 20, 0),
    )
    assert st.unavailable is True
    assert st.status == "failed"
    assert "price source unavailable" in (st.error or "")
    assert len(calls) < 100  # stopped early, not all 150 attempted
