"""Offline tests for section 9b: per-host meter + politeness + REQUESTS block."""

from __future__ import annotations

import time

import pytest

from src.market_scanner.sources import _common
from src.market_scanner.sources._common import (
    HostMeter,
    host_min_interval,
    host_rate,
    meter_akshare_call,
    set_active_meter,
)


def test_host_rates_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rates come from the env schema (agent_tuning), not raw os.environ.
    monkeypatch.delenv("VIBE_TRADING_SINA_RATE_PER_SEC", raising=False)
    assert host_rate("sina") == 5.0
    assert host_rate("tencent") == 1.0
    assert host_rate("ths") == 5.0
    assert host_rate("datacenter-eastmoney") == 3.0
    monkeypatch.setenv("VIBE_TRADING_SINA_RATE_PER_SEC", "0.5")
    # get_env_config() is a cached singleton: reset so the patched env is seen
    import src.config.accessor as acc

    acc.reset_env_config()
    try:
        assert host_rate("sina") == 0.5
    finally:
        monkeypatch.delenv("VIBE_TRADING_SINA_RATE_PER_SEC", raising=False)
        acc.reset_env_config()
    # interval is 1/rate from the schema default (5.0)
    assert host_min_interval("sina") == 0.2


def test_host_rate_classifies_full_hostnames() -> None:
    """Any *.eastmoney.com host shares the Eastmoney cap; Sina full
    hostnames the Sina cap; unknown hosts cap at 5 req/s (section 9b)."""
    assert host_rate("search-api-web.eastmoney.com") == 3.0
    assert host_rate("np-anotice-stock.eastmoney.com") == 3.0
    assert host_rate("datacenter-web.eastmoney.com") == 3.0
    assert host_rate("vip.stock.finance.sina.com.cn") == 5.0
    assert host_rate("unknown.example.com") == 5.0
    assert host_min_interval("datacenter-web.eastmoney.com") == pytest.approx(
        1 / 3, rel=1e-6)


def test_http_get_text_clamps_caller_interval_to_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-provided min_interval below the host cap is clamped up:
    eastmoney callers can be slower than 1/3 s, never faster."""
    from src.market_scanner.sources import _common

    waits: list[tuple[str, float]] = []

    class _Rec:
        @staticmethod
        def wait(bucket, min_interval):
            waits.append((bucket, min_interval))

    monkeypatch.setattr(_common, "_THROTTLE", _Rec())
    monkeypatch.setattr(_common, "_session_for", lambda b: _FakeSession())
    _common.http_get_text(
        "https://datacenter-web.eastmoney.com/x",
        host_key="datacenter-web.eastmoney.com", min_interval=0.05)
    assert waits and waits[0][0] == "eastmoney"
    assert waits[0][1] == pytest.approx(1 / 3, rel=1e-6)
    # Caller override slower than the cap is preserved.
    waits.clear()
    _common.http_get_text(
        "https://datacenter-web.eastmoney.com/x",
        host_key="datacenter-web.eastmoney.com", min_interval=0.9)
    assert waits == [("eastmoney", 0.9)]


def test_akshare_burst_respects_cap_spacing() -> None:
    """Consecutive meter_akshare_call invocations on one bucket are spaced
    by the host cap (eastmoney: >= 1/3 s between calls -> 3 calls >= 2/3 s)."""
    floor = host_min_interval("eastmoney")
    t0 = time.monotonic()
    for _ in range(3):
        meter_akshare_call(lambda: None, host="eastmoney")
    elapsed = time.monotonic() - t0
    assert elapsed >= 2 * floor * 0.95  # jitter only ever pushes later
    assert elapsed < 2 * floor + 2.0  # sane upper bound


def test_meter_counts_and_effective_rps() -> None:
    meter = HostMeter()
    meter.record("sina", ok=True, elapsed_s=0.5)
    meter.record("sina", ok=True, elapsed_s=0.5)
    meter.record("sina", ok=False, elapsed_s=0.1)
    stats = meter.to_dict()
    assert stats["sina"]["count"] == 3
    assert stats["sina"]["errors"] == 1
    # wall span = first request start -> last request end. Back-to-back
    # zero-gap records: the span is the last request's own duration plus any
    # real-clock gaps between records — here bounded by [0.1, 1.2] s.
    assert 0.05 <= stats["sina"]["elapsed_s"] <= 1.2
    assert stats["sina"]["busy_s"] == pytest.approx(1.1, abs=0.05)
    assert stats["sina"]["effective_rps"] == pytest.approx(
        3 / stats["sina"]["elapsed_s"], rel=0.01)
    assert stats["_totals"]["count"] == 3


def test_meter_rps_uses_wall_span_not_busy_sum() -> None:
    """Concurrent requests: rps = count / wall span, not count / busy sum
    (Sina regression: 62 reqs / 26.5 s wall = 2.3 req/s, not 8.3)."""
    meter = HostMeter()
    # 4 "concurrent" 1-s requests all ending at the same moment -> wall span
    # ~1 s, busy 4 s; rps must be ~4, not ~1.
    for _ in range(4):
        meter.record("sina", ok=True, elapsed_s=1.0)
    stats = meter.to_dict()
    assert stats["sina"]["busy_s"] == pytest.approx(4.0, abs=0.05)
    assert stats["sina"]["elapsed_s"] == pytest.approx(1.0, abs=0.05)
    assert stats["sina"]["effective_rps"] == pytest.approx(4.0, rel=0.1)


def test_meter_per_run_isolation() -> None:
    m1 = HostMeter()
    set_active_meter(m1)
    m1.record("sina", ok=True, elapsed_s=1.0)
    m2 = HostMeter()
    set_active_meter(m2)
    m2.record("tencent", ok=True, elapsed_s=0.2)
    # m1 unchanged, m2 has only tencent
    assert "sina" not in m2.to_dict()
    assert "sina" in m1.to_dict()
    set_active_meter(None)
    assert _common.get_active_meter() is None


def test_meter_akshare_call_counts_success_and_error() -> None:
    meter = HostMeter()
    set_active_meter(meter)
    try:
        def ok_fn():
            return "x"

        def boom_fn():
            raise ValueError("boom")

        meter_akshare_call(ok_fn, host="ths")
        with pytest.raises(ValueError):
            meter_akshare_call(boom_fn, host="ths")
        stats = meter.to_dict()
        assert stats["ths"]["count"] == 2
        assert stats["ths"]["errors"] == 1
    finally:
        set_active_meter(None)


def test_meter_akshare_call_noop_without_meter() -> None:
    set_active_meter(None)

    def ok_fn():
        return "x"

    assert meter_akshare_call(ok_fn, host="sina") == "x"  # no meter: no crash


def test_http_get_text_feeds_meter(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_scanner.sources import _common

    meter = HostMeter()
    set_active_meter(meter)
    monkeypatch.setattr(_common, "host_min_interval", lambda h: 0.0)
    monkeypatch.setattr(_common, "_THROTTLE",
                        type("T", (), {"wait": staticmethod(lambda b, m: None)})())
    monkeypatch.setattr(_common, "_session_for", lambda b: _FakeSession())
    _common.http_get_text("https://vip.stock.finance.sina.com.cn/x", host_key="sina")
    stats = meter.to_dict()
    assert stats["sina"]["count"] == 1
    assert stats["sina"]["errors"] == 0


class _FakeSession:
    def get(self, url, params=None, headers=None, timeout=None):
        class R:
            status_code = 200
            text = "[]"

            def raise_for_status(self):
                pass

        return R()


def test_throttle_stop_per_host_generic() -> None:
    """10 consecutive throttle-type errors -> source stops for the run."""
    from src.market_scanner.sources.kline import KlineRateLimiter

    lim = KlineRateLimiter(rate_per_sec=100.0)
    for i in range(10):
        lim.note_result(False, throttle=True, elapsed_s=0.01)
    assert lim.throttle_stop is True
    # a success resets the consecutive counter
    lim2 = KlineRateLimiter(rate_per_sec=100.0)
    for _ in range(9):
        lim2.note_result(False, throttle=True, elapsed_s=0.01)
    lim2.note_result(True, elapsed_s=0.01)
    lim2.note_result(False, throttle=True, elapsed_s=0.01)
    assert lim2.throttle_stop is False
