"""F4/F5 config knobs for the kline outage stop and budget cap (offline)."""

from __future__ import annotations

import pytest

from src.market_scanner.config import ScanConfig


def test_stop_and_cap_defaults():
    cfg = ScanConfig()
    assert cfg.kline_stop_consecutive == 10
    assert cfg.kline_stop_window == 60
    assert cfg.kline_stop_min_error_rate == 0.5
    assert cfg.kline_stop_min_elapsed_s == 120.0
    assert cfg.kline_time_budget_cap_s == 5400.0
    assert cfg.kline_local_budget_s == 1800.0
    # pre-existing budget default unchanged
    assert cfg.kline_time_budget_s == 600.0


def test_stop_and_cap_round_trip():
    cfg = ScanConfig(kline_stop_consecutive=5, kline_stop_window=30,
                     kline_stop_min_error_rate=0.7,
                     kline_stop_min_elapsed_s=60.0,
                     kline_time_budget_cap_s=3600.0,
                     kline_time_budget_s=600.0,
                     kline_local_budget_s=900.0)
    again = ScanConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert again.to_dict()["kline_time_budget_cap_s"] == 3600.0
    assert again.to_dict()["kline_local_budget_s"] == 900.0
    with pytest.raises(ValueError, match="kline_local_budget_s"):
        ScanConfig(kline_local_budget_s=-1)


def test_stop_knobs_validate():
    with pytest.raises(ValueError, match="kline_stop_consecutive"):
        ScanConfig(kline_stop_consecutive=0)
    with pytest.raises(ValueError, match="kline_stop_window"):
        ScanConfig(kline_stop_window=5)
    with pytest.raises(ValueError, match="kline_stop_min_error_rate"):
        ScanConfig(kline_stop_min_error_rate=1.5)
    with pytest.raises(ValueError, match="kline_stop_min_elapsed_s"):
        ScanConfig(kline_stop_min_elapsed_s=-1)
    with pytest.raises(ValueError, match="kline_time_budget_cap_s"):
        ScanConfig(kline_time_budget_cap_s=-1)
    with pytest.raises(ValueError, match="kline_time_budget_cap_s"):
        # cap below the base budget makes no sense
        ScanConfig(kline_time_budget_s=600.0, kline_time_budget_cap_s=60.0)
