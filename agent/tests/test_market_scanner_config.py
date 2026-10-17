"""Unit tests for ScanConfig (CONTRACT section 4). Offline, no I/O."""

from __future__ import annotations

import pytest

from src.market_scanner.config import ScanConfig
from src.market_scanner.models import DEFAULT_WEIGHTS


def test_defaults_match_contract_table():
    cfg = ScanConfig()
    assert cfg.ai_call_timeout_s == 90.0
    assert cfg.top_n == 20
    assert cfg.min_total_mv == 3e9
    assert cfg.min_avg_amount_20d == 5e7
    assert cfg.min_listing_days == 365
    assert cfg.include_star is True
    assert cfg.include_chinext is True
    assert cfg.weights == DEFAULT_WEIGHTS
    assert cfg.weights == {"fundamental": 0.40, "trend": 0.35, "flow": 0.25}
    assert cfg.max_per_industry == 3
    assert cfg.min_kline_bars == 120
    assert cfg.kline_bars == 260
    assert cfg.kline_workers == 6
    assert cfg.ai_enabled is True
    assert cfg.ai_review_pool == 30
    assert cfg.ai_language == "en"
    assert cfg.news_days == 30
    assert cfg.min_coverage == 0.90
    assert cfg.exclude_shrinking is True


def test_to_dict_round_trips_through_from_dict():
    cfg = ScanConfig(top_n=7, include_star=False, max_per_industry=2)
    again = ScanConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert again.to_dict() == cfg.to_dict()


def test_ai_call_timeout_round_trips_and_validates():
    cfg = ScanConfig(ai_call_timeout_s=12.5)
    assert cfg.to_dict()["ai_call_timeout_s"] == 12.5
    assert ScanConfig.from_dict(cfg.to_dict()).ai_call_timeout_s == 12.5
    with pytest.raises(ValueError):
        ScanConfig(ai_call_timeout_s=-1)


def test_to_dict_returns_fresh_weights_copy():
    cfg = ScanConfig()
    payload = cfg.to_dict()
    payload["weights"]["flow"] = 0.99
    assert cfg.weights["flow"] == 0.25


def test_from_dict_copies_incoming_weights():
    incoming = {"fundamental": 0.5, "trend": 0.3, "flow": 0.2}
    cfg = ScanConfig.from_dict({"weights": incoming})
    incoming["flow"] = 0.0
    assert cfg.weights["flow"] == 0.2


def test_from_dict_missing_keys_take_defaults():
    cfg = ScanConfig.from_dict({"top_n": 5})
    assert cfg.top_n == 5
    assert cfg.min_total_mv == 3e9
    assert cfg.weights == DEFAULT_WEIGHTS


def test_from_dict_none_returns_defaults():
    assert ScanConfig.from_dict(None) == ScanConfig()


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown ScanConfig keys"):
        ScanConfig.from_dict({"top_n": 5, "min_volume": 1})


def test_rejects_non_positive_top_n():
    with pytest.raises(ValueError, match="top_n"):
        ScanConfig(top_n=0)
    with pytest.raises(ValueError, match="top_n"):
        ScanConfig(top_n=-3)


def test_rejects_bool_where_int_expected():
    with pytest.raises(ValueError, match="top_n must be an int"):
        ScanConfig(top_n=True)


def test_rejects_negative_thresholds():
    with pytest.raises(ValueError, match="min_total_mv"):
        ScanConfig(min_total_mv=-1)
    with pytest.raises(ValueError, match="min_avg_amount_20d"):
        ScanConfig(min_avg_amount_20d=-5)


def test_rejects_non_bool_toggles():
    with pytest.raises(ValueError, match="include_star"):
        ScanConfig(include_star=1)


def test_rejects_incomplete_weights():
    with pytest.raises(ValueError, match="weights keys"):
        ScanConfig(weights={"fundamental": 0.5, "trend": 0.5})


def test_ai_language_is_normalised_to_lowercase():
    cfg = ScanConfig(ai_language="  ZH-Hans ")
    assert cfg.ai_language == "zh-hans"


def test_ai_language_rejects_empty_or_non_string():
    with pytest.raises(ValueError, match="ai_language"):
        ScanConfig(ai_language="   ")
    with pytest.raises(ValueError, match="ai_language"):
        ScanConfig(ai_language=None)
    with pytest.raises(ValueError, match="ai_language"):
        ScanConfig(ai_language=5)


def test_ai_language_included_in_from_dict_round_trip():
    cfg = ScanConfig(ai_language="zh")
    again = ScanConfig.from_dict(cfg.to_dict())
    assert again.ai_language == "zh"


def test_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        ScanConfig(weights={"fundamental": 0.5, "trend": 0.3, "flow": 0.1})


def test_rejects_negative_weight():
    with pytest.raises(ValueError, match=">= 0"):
        ScanConfig(weights={"fundamental": 1.2, "trend": -0.1, "flow": -0.1})


def test_rejects_kline_bars_below_min_kline_bars():
    with pytest.raises(ValueError, match="kline_bars"):
        ScanConfig(min_kline_bars=120, kline_bars=60)


def test_accepts_integral_float_for_int_fields():
    cfg = ScanConfig.from_dict({"top_n": 20.0, "max_per_industry": 3.0})
    assert cfg.top_n == 20
    assert isinstance(cfg.top_n, int)
    assert cfg.max_per_industry == 3


def test_mutation_of_internal_weights_needs_revalidation_path():
    # validate() re-copies; calling it again after an external mutation keeps
    # the config consistent (defensive API, not a supported edit path).
    cfg = ScanConfig()
    cfg.weights["flow"] = 0.25  # unchanged value, still valid
    cfg.validate()
    assert cfg.weights == DEFAULT_WEIGHTS


def test_min_coverage_round_trips_and_validates():
    cfg = ScanConfig(min_coverage=0.75)
    assert cfg.min_coverage == 0.75
    again = ScanConfig.from_dict(cfg.to_dict())
    assert again.min_coverage == 0.75
    assert again.to_dict()["min_coverage"] == 0.75
    with pytest.raises(ValueError, match="min_coverage"):
        ScanConfig(min_coverage=-0.1)
    with pytest.raises(ValueError, match="min_coverage"):
        ScanConfig(min_coverage=1.5)


def test_exclude_shrinking_round_trips_and_validates():
    cfg = ScanConfig(exclude_shrinking=False)
    assert cfg.to_dict()["exclude_shrinking"] is False
    assert ScanConfig.from_dict(cfg.to_dict()).exclude_shrinking is False
    with pytest.raises(ValueError, match="exclude_shrinking"):
        ScanConfig(exclude_shrinking=1)
