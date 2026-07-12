"""crypto_sentiment_llm_v1 advisor: LLM-based social sentiment advisor (sentiment plan §9).

Research-only. Emits ResearchForecast, never ForecastSignal or OrderIntent,
and must never be imported by quantdesk.portfolio, quantdesk.risk, or
quantdesk.execution. Every actual LLM call attempt (never a budget-skip or
degraded-data skip) produces exactly one ModelProvenance record via an
injected `persist` callable, mirroring llm_trend.py's provenance pattern.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Callable
from uuid import UUID, uuid4

import httpx

from quantdesk.advisors.llm_trend import TEMPERATURE, _cost_usd, _openrouter_complete
from quantdesk.advisors.sentiment_prompts import (
    SENTIMENT_ADVISOR_SYSTEM_PROMPT,
    SENTIMENT_ADVISOR_USER_PROMPT,
    SENTIMENT_PROMPT_VERSION,
    sha256_hex,
)
from quantdesk.common.research_schemas import ResearchForecast
from quantdesk.common.schemas import ModelProvenance

ADVISOR_ID = "crypto_sentiment_llm_v1"
HORIZON = timedelta(hours=24)
PARSER_VERSION = "sentiment_llm_parser_v1"
DATA_HEALTH_MIN = 0.8


@dataclass
class SentimentLlmAdvisorConfig:
    advisor_model: str
    monthly_budget_usd: Decimal
    max_cost_per_decision_usd: Decimal
    provider: str = "openrouter"


def _abstain_forecast(
    *,
    instrument_id: str,
    generated_at: datetime,
    data_cutoff_at: datetime,
    snapshot_id: UUID,
    advisor_version: str,
    model_run_id: UUID | None,
) -> ResearchForecast:
    return ResearchForecast(
        forecast_id=uuid4(),
        advisor_id=ADVISOR_ID,
        advisor_version=advisor_version,
        generated_at=generated_at,
        data_cutoff_at=data_cutoff_at,
        instrument_id=instrument_id,
        horizon=HORIZON,
        direction="flat",
        abstain=True,
        probability_positive=None,
        expected_excess_return_bps=None,
        confidence=0.0,
        evidence_feature_ids=[],
        snapshot_id=snapshot_id,
        model_run_id=model_run_id,
    )


def _is_degraded(snapshot: dict) -> bool:
    if snapshot.get("data_health", 0) < DATA_HEALTH_MIN:
        return True
    features = snapshot.get("features", {}) or {}
    if not features or all(v is None for v in features.values()):
        return True
    return False


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    return json.loads(stripped)


def _parse_forecasts(payload: dict, *, valid_instrument_ids: list[str]) -> dict[str, dict[str, Any]]:
    forecasts = payload.get("forecasts")
    if not isinstance(forecasts, list):
        raise ValueError("missing 'forecasts' list")
    out: dict[str, dict[str, Any]] = {}
    for entry in forecasts:
        instrument_id = entry["instrument_id"]
        if instrument_id not in valid_instrument_ids:
            continue
        direction = entry["direction"]
        if direction not in ("long", "short", "flat"):
            raise ValueError(f"invalid direction: {direction!r}")
        abstain = bool(entry["abstain"])
        probability_positive = entry.get("probability_positive")
        expected_excess_return_bps = entry.get("expected_excess_return_bps")
        confidence = float(entry["confidence"])
        thesis = str(entry.get("thesis", ""))[:500]
        evidence_feature_ids = list(entry.get("evidence_feature_ids", []))
        out[instrument_id] = {
            "direction": direction,
            "abstain": abstain,
            "probability_positive": (
                float(probability_positive) if probability_positive is not None else None
            ),
            "expected_excess_return_bps": (
                float(expected_excess_return_bps) if expected_excess_return_bps is not None else None
            ),
            "confidence": confidence,
            "thesis": thesis,
            "evidence_feature_ids": evidence_feature_ids,
        }
    return out


def run_crypto_sentiment_llm_v1(
    *,
    snapshot: dict,
    instrument_ids: list[str],
    config: SentimentLlmAdvisorConfig,
    snapshot_id: UUID,
    generated_at: datetime | None = None,
    data_cutoff_at: datetime | None = None,
    monthly_spend: Callable[[], Decimal],
    persist: Callable[[ModelProvenance], None],
    api_key: str | None = None,
    http_client: httpx.Client | None = None,
) -> list[ResearchForecast]:
    generated_at = generated_at or datetime.now(timezone.utc)
    data_cutoff_at = data_cutoff_at or generated_at
    advisor_version = f"{SENTIMENT_PROMPT_VERSION}:{config.advisor_model}"

    # --- degraded-data short-circuit (before budget check, before any LLM call) ---
    if _is_degraded(snapshot):
        return [
            _abstain_forecast(
                instrument_id=instrument_id,
                generated_at=generated_at,
                data_cutoff_at=data_cutoff_at,
                snapshot_id=snapshot_id,
                advisor_version=advisor_version,
                model_run_id=None,
            )
            for instrument_id in instrument_ids
        ]

    # --- budget guard -------------------------------------------------
    current_spend = monthly_spend()
    if current_spend >= config.monthly_budget_usd or config.max_cost_per_decision_usd <= 0:
        return [
            _abstain_forecast(
                instrument_id=instrument_id,
                generated_at=generated_at,
                data_cutoff_at=data_cutoff_at,
                snapshot_id=snapshot_id,
                advisor_version=advisor_version,
                model_run_id=None,
            )
            for instrument_id in instrument_ids
        ]

    symbols = ", ".join(instrument_ids)
    system_prompt = SENTIMENT_ADVISOR_SYSTEM_PROMPT.format(symbols=symbols)
    user_prompt = SENTIMENT_ADVISOR_USER_PROMPT.format(
        samples_json=json.dumps(snapshot.get("samples", []), sort_keys=True, default=str),
        features_json=json.dumps(snapshot.get("features", {}), sort_keys=True, default=str),
        data_health=snapshot.get("data_health"),
    )
    system_prompt_hash = sha256_hex(system_prompt)
    user_prompt_hash = sha256_hex(user_prompt)

    openrouter_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    parsed: dict[str, dict[str, Any]] | None = None
    last_model_run_id: UUID | None = None

    for _attempt in range(2):  # initial attempt + one retry
        model_run_id = uuid4()
        request_at = datetime.now(timezone.utc)
        response_text, input_tokens, output_tokens = _openrouter_complete(
            model=config.advisor_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=openrouter_key,
            http_client=http_client,
        )
        response_at = datetime.now(timezone.utc)

        raw_response_hash = sha256_hex(response_text)
        cost_usd = _cost_usd(config.advisor_model, input_tokens, output_tokens)

        provenance = ModelProvenance(
            model_run_id=model_run_id,
            model_provider=config.provider,
            model_id=config.advisor_model,
            system_prompt_hash=system_prompt_hash,
            user_prompt_hash=user_prompt_hash,
            schema_version="1.0",
            temperature=TEMPERATURE,
            input_dataset_cutoff_timestamp=data_cutoff_at,
            request_at=request_at,
            response_at=response_at,
            raw_response_hash=raw_response_hash,
            parser_version=PARSER_VERSION,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        persist(provenance)
        last_model_run_id = model_run_id

        try:
            payload = _extract_json(response_text)
            parsed = _parse_forecasts(payload, valid_instrument_ids=instrument_ids)
            break
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            parsed = None
            continue

    forecasts: list[ResearchForecast] = []
    for instrument_id in instrument_ids:
        if parsed is not None and instrument_id in parsed:
            entry = parsed[instrument_id]
            forecasts.append(
                ResearchForecast(
                    forecast_id=uuid4(),
                    advisor_id=ADVISOR_ID,
                    advisor_version=advisor_version,
                    generated_at=generated_at,
                    data_cutoff_at=data_cutoff_at,
                    instrument_id=instrument_id,
                    horizon=HORIZON,
                    direction=entry["direction"],
                    abstain=entry["abstain"],
                    probability_positive=entry["probability_positive"],
                    expected_excess_return_bps=entry["expected_excess_return_bps"],
                    confidence=entry["confidence"],
                    evidence_feature_ids=entry["evidence_feature_ids"],
                    snapshot_id=snapshot_id,
                    model_run_id=last_model_run_id,
                )
            )
        else:
            forecasts.append(
                _abstain_forecast(
                    instrument_id=instrument_id,
                    generated_at=generated_at,
                    data_cutoff_at=data_cutoff_at,
                    snapshot_id=snapshot_id,
                    advisor_version=advisor_version,
                    model_run_id=last_model_run_id,
                )
            )
    return forecasts
