from __future__ import annotations

import math

import pytest

from quantdesk.features.compute import (
    Candle,
    FEATURES_VERSION,
    atr,
    compute_feature_set,
    feature_id,
    funding_zscore,
    log_return,
    log_returns_series,
    oi_change_pct,
    realized_vol,
    sma,
    trend_state,
)


def make_candles(closes, highs=None, lows=None):
    highs = highs or closes
    lows = lows or closes
    return [
        Candle(
            close_time_iso=f"2026-01-01T{i:02d}:00:00+00:00",
            open=closes[i - 1] if i > 0 else closes[0],
            high=highs[i],
            low=lows[i],
            close=closes[i],
            volume=1.0,
        )
        for i in range(len(closes))
    ]


def test_feature_id_format():
    assert feature_id("BTC", "ret_24h") == f"btc_ret_24h@{FEATURES_VERSION}"
    assert feature_id("BTC", "ret_24h", "v2") == "btc_ret_24h@v2"


def test_log_return_exact_value():
    candles = make_candles([100.0, 110.0, 121.0])
    result = log_return(candles, 1)
    assert result == pytest.approx(math.log(121.0 / 110.0))


def test_log_return_insufficient_history_returns_none():
    candles = make_candles([100.0, 110.0])
    assert log_return(candles, 5) is None


def test_log_returns_series_matches_manual_computation():
    candles = make_candles([100.0, 110.0, 99.0])
    series = log_returns_series(candles)
    assert series == [
        pytest.approx(math.log(110.0 / 100.0)),
        pytest.approx(math.log(99.0 / 110.0)),
    ]


def test_realized_vol_exact_two_bar_case():
    # closes: 100 -> 110 -> 100. Two log returns: ln(1.1), ln(100/110)
    candles = make_candles([100.0, 110.0, 100.0])
    r1 = math.log(1.1)
    r2 = math.log(100.0 / 110.0)
    mean = (r1 + r2) / 2
    variance = ((r1 - mean) ** 2 + (r2 - mean) ** 2) / 2
    expected_unannualized = math.sqrt(variance)

    result = realized_vol(candles, 2, annualize=False)
    assert result == pytest.approx(expected_unannualized)

    result_annualized = realized_vol(candles, 2, annualize=True)
    assert result_annualized == pytest.approx(expected_unannualized * math.sqrt(24 * 365))


def test_realized_vol_insufficient_window_returns_none():
    candles = make_candles([100.0, 110.0])
    assert realized_vol(candles, 24) is None


def test_atr_exact_value():
    # 3 candles: bar0 close=100 (h=101,l=99); bar1 h=105,l=95,close=102; bar2 h=110,l=100,close=108
    candles = [
        Candle("t0", 100.0, 101.0, 99.0, 100.0),
        Candle("t1", 100.0, 105.0, 95.0, 102.0),
        Candle("t2", 102.0, 110.0, 100.0, 108.0),
    ]
    # TR for bar1: max(105-95, |105-100|, |95-100|) = max(10,5,5) = 10
    # TR for bar2: max(110-100, |110-102|, |100-102|) = max(10,8,2) = 10
    result = atr(candles, period=2)
    assert result == pytest.approx((10.0 + 10.0) / 2)


def test_atr_insufficient_period_returns_none():
    candles = make_candles([100.0, 101.0])
    assert atr(candles, period=14) is None


def test_atr_needs_at_least_two_candles():
    candles = make_candles([100.0])
    assert atr(candles) is None


def test_sma_exact_value():
    candles = make_candles([10.0, 20.0, 30.0])
    assert sma(candles, 3) == pytest.approx(20.0)


def test_sma_insufficient_history_returns_none():
    candles = make_candles([10.0, 20.0])
    assert sma(candles, 3) is None


def test_trend_state_up():
    # Rising prices well above both SMAs
    closes = [100.0 + i for i in range(168)]  # steadily rising
    candles = make_candles(closes)
    assert trend_state(candles, short_window_hours=24, long_window_hours=168) == "up"


def test_trend_state_down():
    closes = [300.0 - i for i in range(168)]  # steadily falling
    candles = make_candles(closes)
    assert trend_state(candles, short_window_hours=24, long_window_hours=168) == "down"


def test_trend_state_chop_flat_prices():
    closes = [100.0] * 168
    candles = make_candles(closes)
    assert trend_state(candles, short_window_hours=24, long_window_hours=168) == "chop"


def test_trend_state_insufficient_history_returns_none():
    candles = make_candles([100.0] * 10)
    assert trend_state(candles, short_window_hours=24, long_window_hours=168) is None


def test_funding_zscore_exact_value():
    history = [0.0001, 0.0002, 0.0003]
    mean = sum(history) / 3
    variance = sum((v - mean) ** 2 for v in history) / 3
    expected = (0.0003 - mean) / math.sqrt(variance)
    assert funding_zscore(history) == pytest.approx(expected)


def test_funding_zscore_zero_variance_returns_none():
    assert funding_zscore([0.0001, 0.0001, 0.0001]) is None


def test_funding_zscore_insufficient_history_returns_none():
    assert funding_zscore([0.0001]) is None


def test_oi_change_pct_exact_value():
    assert oi_change_pct(110.0, 100.0) == pytest.approx(10.0)
    assert oi_change_pct(90.0, 100.0) == pytest.approx(-10.0)


def test_oi_change_pct_zero_base_returns_none():
    assert oi_change_pct(100.0, 0.0) is None


def test_compute_feature_set_produces_versioned_ids():
    closes = [100.0 + i * 0.1 for i in range(200)]
    candles = make_candles(closes)
    features = compute_feature_set(
        "BTC",
        candles,
        funding_history=[0.0001, 0.0002, 0.0003],
        oi_now=110.0,
        oi_24h_ago=100.0,
    )
    expected_ids = {
        f"btc_ret_1h@{FEATURES_VERSION}",
        f"btc_ret_24h@{FEATURES_VERSION}",
        f"btc_ret_7d@{FEATURES_VERSION}",
        f"btc_rvol_24h@{FEATURES_VERSION}",
        f"btc_rvol_7d@{FEATURES_VERSION}",
        f"btc_atr_14@{FEATURES_VERSION}",
        f"btc_trend_state@{FEATURES_VERSION}",
        f"btc_funding_z@{FEATURES_VERSION}",
        f"btc_oi_change_24h_pct@{FEATURES_VERSION}",
    }
    assert expected_ids == set(features.keys())
    assert features[f"btc_oi_change_24h_pct@{FEATURES_VERSION}"] == pytest.approx(10.0)
