"""Deterministic, versioned feature computation (plan §3).

Pure functions only: no I/O, no network, no LLM calls. Inputs are plain
lists/dicts (or pandas where convenient) so the code is trivially
unit-testable with fixed inputs and exact expected outputs.

Every feature carries an id like ``"btc_ret_24h@v1"`` combining the
instrument, the feature name, and ``FEATURES_VERSION`` so a forecast can
always be tied to the exact feature code that produced it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

FEATURES_VERSION = "v1"

HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True)
class Candle:
    """Minimal OHLC candle used by feature functions (1h bars expected)."""

    close_time_iso: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @classmethod
    def from_row(cls, row: dict) -> "Candle":
        return cls(
            close_time_iso=row["close_time"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
        )


def feature_id(instrument_id: str, name: str, version: str = FEATURES_VERSION) -> str:
    """Build a versioned feature id, e.g. ``"btc_ret_24h@v1"``."""
    return f"{instrument_id.lower()}_{name}@{version}"


def _closes(candles: Sequence[Candle]) -> list[float]:
    return [c.close for c in candles]


def log_return(candles: Sequence[Candle], lookback_hours: int) -> float | None:
    """Log return over ``lookback_hours`` 1h candles, using the latest close
    vs. the close ``lookback_hours`` candles earlier.

    Returns None if there is not enough history.
    """
    closes = _closes(candles)
    if len(closes) <= lookback_hours:
        return None
    latest = closes[-1]
    past = closes[-1 - lookback_hours]
    if past <= 0 or latest <= 0:
        return None
    return math.log(latest / past)


def log_returns_series(candles: Sequence[Candle]) -> list[float]:
    """Per-bar log returns: r[i] = ln(close[i] / close[i-1])."""
    closes = _closes(candles)
    out = []
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        if prev <= 0 or cur <= 0:
            continue
        out.append(math.log(cur / prev))
    return out


def realized_vol(candles: Sequence[Candle], window_hours: int, *, annualize: bool = True) -> float | None:
    """Realized volatility (stdev of 1h log returns) over the trailing window.

    ``window_hours`` is the number of 1h bars of returns to use (e.g. 24 for
    a 24h window, 168 for 7d). Uses population stdev (ddof=0) for
    determinism on small fixed test inputs. Annualized by
    sqrt(HOURS_PER_YEAR) when ``annualize`` is True.
    """
    returns = log_returns_series(candles)
    if len(returns) < window_hours:
        return None
    window = returns[-window_hours:]
    mean = sum(window) / len(window)
    variance = sum((r - mean) ** 2 for r in window) / len(window)
    vol = math.sqrt(variance)
    if annualize:
        vol *= math.sqrt(HOURS_PER_YEAR)
    return vol


def atr(candles: Sequence[Candle], period: int = 14) -> float | None:
    """Average True Range over ``period`` 1h bars (simple moving average of
    true range, not Wilder-smoothed).

    True range for bar i (i>=1): max(high-low, |high-prev_close|,
    |low-prev_close|). The first bar has no previous close, so true range
    for it is simply high-low and is only used if there aren't enough bars.
    """
    if len(candles) < 2:
        return None
    true_ranges: list[float] = []
    for i in range(1, len(candles)):
        c = candles[i]
        prev_close = candles[i - 1].close
        tr = max(
            c.high - c.low,
            abs(c.high - prev_close),
            abs(c.low - prev_close),
        )
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return None
    window = true_ranges[-period:]
    return sum(window) / len(window)


def sma(candles: Sequence[Candle], period_hours: int) -> float | None:
    """Simple moving average of close over the trailing ``period_hours`` bars."""
    closes = _closes(candles)
    if len(closes) < period_hours:
        return None
    window = closes[-period_hours:]
    return sum(window) / len(window)


def trend_state(
    candles: Sequence[Candle],
    *,
    short_window_hours: int = 24,
    long_window_hours: int = 168,
    chop_band_pct: float = 0.002,
) -> str | None:
    """Classify trend as "up" / "down" / "chop" from px vs SMA(24h) and SMA(168h).

    "up" when price is above both SMAs (each SMA offset by more than
    ``chop_band_pct``), "down" when below both, otherwise "chop". Returns
    None if there isn't enough history for the long SMA.
    """
    closes = _closes(candles)
    if not closes:
        return None
    px = closes[-1]
    sma_short = sma(candles, short_window_hours)
    sma_long = sma(candles, long_window_hours)
    if sma_short is None or sma_long is None:
        return None

    def offset_pct(reference: float) -> float:
        if reference == 0:
            return 0.0
        return (px - reference) / reference

    off_short = offset_pct(sma_short)
    off_long = offset_pct(sma_long)

    if off_short > chop_band_pct and off_long > chop_band_pct:
        return "up"
    if off_short < -chop_band_pct and off_long < -chop_band_pct:
        return "down"
    return "chop"


def funding_zscore(funding_history: Sequence[float]) -> float | None:
    """Z-score of the most recent funding rate against its own trailing history.

    ``funding_history`` should be ordered oldest -> newest and include the
    current observation as the last element. Uses population stdev (ddof=0).
    Returns None with fewer than 2 observations or zero variance.
    """
    if len(funding_history) < 2:
        return None
    values = list(funding_history)
    current = values[-1]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    if variance == 0:
        return None
    stdev = math.sqrt(variance)
    return (current - mean) / stdev


def oi_change_pct(oi_now: float, oi_24h_ago: float) -> float | None:
    """Percent change in open interest over 24h."""
    if oi_24h_ago == 0:
        return None
    return (oi_now - oi_24h_ago) / oi_24h_ago * 100.0


def compute_feature_set(
    instrument_id: str,
    candles_1h: Sequence[Candle],
    *,
    funding_history: Sequence[float] | None = None,
    oi_now: float | None = None,
    oi_24h_ago: float | None = None,
) -> dict[str, float | str | None]:
    """Compute the full feature set for one instrument, keyed by versioned
    feature id (e.g. ``"btc_ret_24h@v1"``).
    """
    out: dict[str, float | str | None] = {}

    out[feature_id(instrument_id, "ret_1h")] = log_return(candles_1h, 1)
    out[feature_id(instrument_id, "ret_24h")] = log_return(candles_1h, 24)
    out[feature_id(instrument_id, "ret_7d")] = log_return(candles_1h, 24 * 7)

    out[feature_id(instrument_id, "rvol_24h")] = realized_vol(candles_1h, 24)
    out[feature_id(instrument_id, "rvol_7d")] = realized_vol(candles_1h, 24 * 7)

    out[feature_id(instrument_id, "atr_14")] = atr(candles_1h, 14)

    out[feature_id(instrument_id, "trend_state")] = trend_state(candles_1h)

    if funding_history is not None:
        out[feature_id(instrument_id, "funding_z")] = funding_zscore(funding_history)

    if oi_now is not None and oi_24h_ago is not None:
        out[feature_id(instrument_id, "oi_change_24h_pct")] = oi_change_pct(oi_now, oi_24h_ago)

    return out
