"""Deterministic volatility-scaled time-series momentum advisor (plan §7).

ts_momentum_baseline is the pre-registered benchmark every LLM advisor must
beat (config/contracts/crypto_trend_llm_v1.yaml: benchmark). It is fully
deterministic: same features in, same ForecastSignal out, always.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID, uuid4

from quantdesk.common.schemas import ForecastSignal

ADVISOR_ID = "ts_momentum_baseline"
ADVISOR_VERSION = "ts_momentum_baseline_v1"

# Dead zone: |z| below this -> flat. Avoids churning on noise-level momentum.
DEAD_ZONE_Z = 0.25

FORECAST_TARGET = "next_24h_excess_return_sign"
HORIZON = timedelta(hours=24)


def _vol_normalized_momentum_z(return_7d: float, realized_vol_7d: float) -> float:
    """Vol-normalized 7d momentum z-score.

    realized_vol_7d is expected as a 7-day realized volatility (fractional,
    e.g. 0.35 for 35%), already on the same horizon as return_7d. Guards
    against division by zero / degenerate vol readings.
    """
    if realized_vol_7d is None or realized_vol_7d <= 1e-9:
        return 0.0
    return return_7d / realized_vol_7d


def ts_momentum_baseline(
    *,
    instrument_id: str,
    venue: str,
    features: dict,
    generated_at: datetime,
    data_cutoff_at: datetime,
    snapshot_id: UUID,
    signal_id: UUID | None = None,
    expires_after: timedelta = timedelta(hours=4),
) -> ForecastSignal:
    """Compute a deterministic ForecastSignal from a feature dict.

    Expected feature keys (all floats):
      - "return_24h": trailing 24h return (fractional)
      - "return_7d": trailing 7d return (fractional)
      - "realized_vol_7d": trailing 7d realized volatility (fractional, annualized or not,
         as long as consistently produced upstream)
      - "trend_state": optional string, e.g. "trending" | "chop" | "unknown" (informational only)
    """
    return_7d = float(features["return_7d"])
    realized_vol_7d = float(features["realized_vol_7d"])

    z = _vol_normalized_momentum_z(return_7d, realized_vol_7d)

    if z > DEAD_ZONE_Z:
        action = "long"
    elif z < -DEAD_ZONE_Z:
        action = "short"
    else:
        action = "flat"

    evidence_feature_ids = ["return_7d", "realized_vol_7d"]
    if "return_24h" in features:
        evidence_feature_ids.append("return_24h")
    if "trend_state" in features:
        evidence_feature_ids.append("trend_state")

    if action == "flat":
        thesis = (
            f"Vol-normalized 7d momentum z={z:.3f} is within the "
            f"[-{DEAD_ZONE_Z}, {DEAD_ZONE_Z}] dead zone; no directional edge."
        )
        invalidation_condition = None
    else:
        direction = "positive" if action == "long" else "negative"
        thesis = (
            f"Vol-normalized 7d momentum z={z:.3f} is {direction} and exceeds the "
            f"dead zone threshold of {DEAD_ZONE_Z}; momentum continuation expected "
            f"over the next 24h."
        )
        invalidation_condition = (
            "Vol-normalized 7d momentum z crosses back through zero, or "
            f"|z| falls back within [-{DEAD_ZONE_Z}, {DEAD_ZONE_Z}]."
        )

    return ForecastSignal(
        signal_id=signal_id or uuid4(),
        advisor_id=ADVISOR_ID,
        advisor_version=ADVISOR_VERSION,
        generated_at=generated_at,
        data_cutoff_at=data_cutoff_at,
        expires_at=generated_at + expires_after,
        venue=venue,
        instrument_id=instrument_id,
        forecast_target=FORECAST_TARGET,
        horizon=HORIZON,
        action=action,
        raw_score=z,
        thesis=thesis,
        invalidation_condition=invalidation_condition,
        evidence_feature_ids=evidence_feature_ids,
        snapshot_id=snapshot_id,
        model_run_id=None,
    )
