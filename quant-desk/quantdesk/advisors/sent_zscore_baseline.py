"""sent_zscore_contrarian_baseline_v1: frozen deterministic sentiment benchmark.

FROZEN. All thresholds in this module (Z_THRESHOLD, DATA_HEALTH_MIN,
PROB_OFFSET) were frozen on 2026-07-12, before prospective data collection
began (see config/contracts/crypto_sentiment_llm_v1.yaml, which names this
advisor as the pre-registered benchmark). They must never be tuned,
including after seeing evaluation results — retuning would invalidate the
benchmark's role as an independent, pre-registered comparison point for
crypto_sentiment_llm_v1. Any change requires a new advisor_id, not an edit
to these constants.

Feature-key convention: this baseline reads two features per instrument
from `snapshot["features"]`, using the prefixed keys
"{instrument_id}:sent_mean_1h_zscore" and "{instrument_id}:spike_flag".
If `instrument_ids` has exactly one element and the prefixed keys are
absent, it falls back to the unprefixed keys "sent_mean_1h_zscore" and
"spike_flag" (single-instrument snapshots may omit the prefix).

Rule: requires `snapshot["data_health"] >= DATA_HEALTH_MIN` and both the
z-score and spike_flag to be present and non-null, else abstain/flat.
Contrarian logic: an extreme negative z-score co-occurring with a spike
is read as capitulation (crowd overly bearish) -> long; an extreme
positive z-score co-occurring with a spike is read as euphoria (crowd
overly bullish) -> short. Without a spike_flag, an extreme z-score alone
does not trigger (it is not distinguished from steady-state sentiment).

This baseline does not forecast magnitude: expected_excess_return_bps is
always None. Only direction and a small, symmetric probability_positive
offset from 0.5 (+/- PROB_OFFSET) are produced.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID, uuid4

from quantdesk.common.research_schemas import ResearchForecast

ADVISOR_ID = "sent_zscore_contrarian_baseline_v1"
ADVISOR_VERSION = "sent_zscore_contrarian_baseline_v1:2026-07-12"
HORIZON_HOURS = 24

# Frozen 2026-07-12. Do not tune.
Z_THRESHOLD = 2.0
DATA_HEALTH_MIN = 0.8
PROB_OFFSET = 0.06


def _feature_lookup(features: dict, instrument_id: str, key: str, *, single_instrument: bool) -> object:
    prefixed_key = f"{instrument_id}:{key}"
    if prefixed_key in features:
        return features[prefixed_key]
    if single_instrument and key in features:
        return features[key]
    return None


def run_sent_zscore_contrarian_baseline_v1(
    *,
    snapshot: dict,
    instrument_ids: list[str],
    snapshot_id: UUID,
    generated_at: datetime,
    data_cutoff_at: datetime,
) -> list[ResearchForecast]:
    features = snapshot.get("features", {}) or {}
    data_health = snapshot.get("data_health", 0)
    single_instrument = len(instrument_ids) == 1

    forecasts: list[ResearchForecast] = []
    for instrument_id in instrument_ids:
        z = _feature_lookup(features, instrument_id, "sent_mean_1h_zscore", single_instrument=single_instrument)
        spike_flag = _feature_lookup(features, instrument_id, "spike_flag", single_instrument=single_instrument)

        degraded = data_health < DATA_HEALTH_MIN or z is None or spike_flag is None
        if degraded:
            forecasts.append(
                ResearchForecast(
                    forecast_id=uuid4(),
                    advisor_id=ADVISOR_ID,
                    advisor_version=ADVISOR_VERSION,
                    generated_at=generated_at,
                    data_cutoff_at=data_cutoff_at,
                    instrument_id=instrument_id,
                    horizon=timedelta(hours=HORIZON_HOURS),
                    direction="flat",
                    abstain=True,
                    probability_positive=None,
                    expected_excess_return_bps=None,
                    confidence=0.5,
                    evidence_feature_ids=[],
                    snapshot_id=snapshot_id,
                    model_run_id=None,
                )
            )
            continue

        z_value = float(z)
        triggered_long = z_value <= -Z_THRESHOLD and bool(spike_flag)
        triggered_short = z_value >= Z_THRESHOLD and bool(spike_flag)

        evidence_feature_ids = [
            f"{instrument_id}:sent_mean_1h_zscore" if f"{instrument_id}:sent_mean_1h_zscore" in features
            else "sent_mean_1h_zscore",
            f"{instrument_id}:spike_flag" if f"{instrument_id}:spike_flag" in features else "spike_flag",
        ]

        if triggered_long:
            direction = "long"
            probability_positive = 0.5 + PROB_OFFSET
            confidence = 0.6
        elif triggered_short:
            direction = "short"
            probability_positive = 0.5 - PROB_OFFSET
            confidence = 0.6
        else:
            direction = "flat"
            probability_positive = 0.5
            confidence = 0.5

        forecasts.append(
            ResearchForecast(
                forecast_id=uuid4(),
                advisor_id=ADVISOR_ID,
                advisor_version=ADVISOR_VERSION,
                generated_at=generated_at,
                data_cutoff_at=data_cutoff_at,
                instrument_id=instrument_id,
                horizon=timedelta(hours=HORIZON_HOURS),
                direction=direction,
                abstain=False,
                probability_positive=probability_positive,
                expected_excess_return_bps=None,
                confidence=confidence,
                evidence_feature_ids=evidence_feature_ids,
                snapshot_id=snapshot_id,
                model_run_id=None,
            )
        )

    return forecasts
