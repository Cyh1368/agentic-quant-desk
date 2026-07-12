"""crypto_trend_llm_v1 advisor: LLM-based trend/regime advisor (plan §7).

Shadow-mode only. Every call is budget-guarded and every call (even a
skipped one is not a call) produces exactly one ModelProvenance record via
an injected `persist` callable — this module never imports the ledger.
"""
from __future__ import annotations

import json
import os

import httpx
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Callable, Iterable
from uuid import UUID, uuid4

from anthropic import Anthropic

from quantdesk.advisors.prompts import (
    ADVISOR_SYSTEM_PROMPT,
    ADVISOR_USER_PROMPT,
    PROMPT_VERSION,
    sha256_hex,
)
from quantdesk.common.schemas import ForecastSignal, ModelProvenance

ADVISOR_ID = "crypto_trend_llm_v1"
FORECAST_TARGET = "next_24h_excess_return_sign"
HORIZON = timedelta(hours=24)
PARSER_VERSION = "llm_trend_parser_v1"

MAX_TOKENS = 1000
TEMPERATURE = 0.0

# USD per million tokens, keyed by model id. Extend as models are added to
# config/desk.yaml (llm.advisor_model). Unknown models cost $0 rather than
# raising, so a misconfigured PRICES table never blocks a shadow run — but
# callers should treat a $0 cost_usd for a real call as a signal to update
# this table.
PRICES: dict[str, dict[str, Decimal]] = {
    "claude-haiku-4-5-20251001": {
        "input": Decimal("1.00"),
        "output": Decimal("5.00"),
    },
    "claude-sonnet-5": {
        "input": Decimal("3.00"),
        "output": Decimal("15.00"),
    },
    # OpenRouter model ids (OpenRouter adds ~5% on top of provider list
    # price; rates below include that margin, rounded up).
    "anthropic/claude-haiku-4.5": {
        "input": Decimal("1.05"),
        "output": Decimal("5.25"),
    },
    "anthropic/claude-sonnet-5": {
        "input": Decimal("3.15"),
        "output": Decimal("15.75"),
    },
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _openrouter_complete(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    http_client: httpx.Client | None = None,
) -> tuple[str, int, int]:
    """One OpenAI-format chat completion against OpenRouter.

    Returns (response_text, input_tokens, output_tokens).
    """
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "usage": {"include": True},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    if http_client is not None:
        resp = http_client.post(f"{OPENROUTER_BASE_URL}/chat/completions", json=body, headers=headers)
    else:
        with httpx.Client(timeout=30.0) as c:
            resp = c.post(f"{OPENROUTER_BASE_URL}/chat/completions", json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage", {}) or {}
    return text, int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)


def _cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> Decimal:
    prices = PRICES.get(model_id)
    if prices is None:
        return Decimal("0")
    input_cost = (Decimal(input_tokens) / Decimal(1_000_000)) * prices["input"]
    output_cost = (Decimal(output_tokens) / Decimal(1_000_000)) * prices["output"]
    return input_cost + output_cost


def _flat_signal(
    *,
    instrument_id: str,
    venue: str,
    generated_at: datetime,
    data_cutoff_at: datetime,
    snapshot_id: UUID,
    advisor_version: str,
    model_run_id: UUID | None,
    thesis: str,
) -> ForecastSignal:
    return ForecastSignal(
        signal_id=uuid4(),
        advisor_id=ADVISOR_ID,
        advisor_version=advisor_version,
        generated_at=generated_at,
        data_cutoff_at=data_cutoff_at,
        expires_at=generated_at + timedelta(hours=4),
        venue=venue,
        instrument_id=instrument_id,
        forecast_target=FORECAST_TARGET,
        horizon=HORIZON,
        action="flat",
        raw_score=0.0,
        thesis=thesis,
        invalidation_condition=None,
        evidence_feature_ids=[],
        snapshot_id=snapshot_id,
        model_run_id=model_run_id,
    )


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response.

    Tolerates the model wrapping JSON in a markdown fence despite
    instructions not to; raises ValueError on any other malformed input.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return json.loads(stripped)


def _parse_forecasts(
    payload: dict,
    *,
    valid_instrument_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Validate and index the model's forecast list by instrument_id.

    Raises ValueError on any structural problem (missing keys, bad action,
    unknown instrument, wrong types) so the caller can trigger the retry ->
    flat-default path uniformly.
    """
    if not isinstance(payload, dict) or "forecasts" not in payload:
        raise ValueError("response missing top-level 'forecasts' key")
    forecasts = payload["forecasts"]
    if not isinstance(forecasts, list):
        raise ValueError("'forecasts' is not a list")

    valid_ids = set(valid_instrument_ids)
    out: dict[str, dict[str, Any]] = {}
    for item in forecasts:
        if not isinstance(item, dict):
            raise ValueError("forecast entry is not an object")
        instrument_id = item.get("instrument_id")
        action = item.get("action")
        raw_score = item.get("raw_score")
        if instrument_id not in valid_ids:
            raise ValueError(f"unknown instrument_id {instrument_id!r}")
        if action not in ("long", "short", "flat"):
            raise ValueError(f"invalid action {action!r}")
        if not isinstance(raw_score, (int, float)):
            raise ValueError("raw_score must be numeric")
        thesis = item.get("thesis", "")
        if not isinstance(thesis, str):
            raise ValueError("thesis must be a string")
        invalidation_condition = item.get("invalidation_condition")
        if invalidation_condition is not None and not isinstance(invalidation_condition, str):
            raise ValueError("invalidation_condition must be a string or null")
        evidence_feature_ids = item.get("evidence_feature_ids", [])
        if not isinstance(evidence_feature_ids, list) or not all(
            isinstance(x, str) for x in evidence_feature_ids
        ):
            raise ValueError("evidence_feature_ids must be a list of strings")
        out[instrument_id] = {
            "action": action,
            "raw_score": float(raw_score),
            "thesis": thesis[:500],
            "invalidation_condition": invalidation_condition,
            "evidence_feature_ids": evidence_feature_ids,
        }
    return out


@dataclass
class LlmTrendAdvisorConfig:
    advisor_model: str
    monthly_budget_usd: Decimal
    max_cost_per_decision_usd: Decimal
    timeframe: str = "4h"
    calibration_window: int = 100
    provider: str = "anthropic"        # "anthropic" | "openrouter"


def run_crypto_trend_llm_v1(
    *,
    snapshot: dict,
    instrument_ids: list[str],
    calibration_summary: dict,
    config: LlmTrendAdvisorConfig,
    snapshot_id: UUID,
    generated_at: datetime | None = None,
    data_cutoff_at: datetime | None = None,
    monthly_spend: Callable[[], Decimal],
    persist: Callable[[ModelProvenance], None],
    client: Anthropic | None = None,
    api_key: str | None = None,
) -> list[ForecastSignal]:
    """Run the crypto_trend_llm_v1 advisor for one decision cycle.

    Returns one ForecastSignal per instrument_id. Budget-exceeded and
    parse-failure-after-retry both degrade to flat signals; silence is
    always the safe default (plan §7).

    `persist` is called once per actual API call attempt with a
    ModelProvenance record (never for a budget-skip, since no call was
    made and nothing to attribute cost/tokens to).
    """
    generated_at = generated_at or datetime.now(timezone.utc)
    data_cutoff_at = data_cutoff_at or generated_at
    advisor_version = f"{PROMPT_VERSION}:{config.advisor_model}"

    # --- budget guard -------------------------------------------------
    current_spend = monthly_spend()
    if (
        current_spend >= config.monthly_budget_usd
        or config.max_cost_per_decision_usd <= 0
    ):
        return [
            _flat_signal(
                instrument_id=instrument_id,
                venue=snapshot.get("venue", "hyperliquid"),
                generated_at=generated_at,
                data_cutoff_at=data_cutoff_at,
                snapshot_id=snapshot_id,
                advisor_version=advisor_version,
                model_run_id=None,
                thesis="Monthly LLM budget exceeded; advisor skipped (quality flag: budget_exceeded).",
            )
            for instrument_id in instrument_ids
        ]

    symbols = ", ".join(instrument_ids)
    system_prompt = ADVISOR_SYSTEM_PROMPT.format(
        symbols=symbols,
        timeframe=config.timeframe,
        window=config.calibration_window,
    )
    user_prompt = ADVISOR_USER_PROMPT.format(
        snapshot_json=json.dumps(snapshot, sort_keys=True, default=str),
        calibration_summary_json=json.dumps(calibration_summary, sort_keys=True, default=str),
    )
    system_prompt_hash = sha256_hex(system_prompt)
    user_prompt_hash = sha256_hex(user_prompt)

    use_openrouter = config.provider == "openrouter" and client is None
    if use_openrouter:
        resolved_client = None
        openrouter_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    else:
        resolved_client = client or Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))

    parsed: dict[str, dict[str, Any]] | None = None
    last_model_run_id: UUID | None = None

    for attempt in range(2):  # initial attempt + one retry
        model_run_id = uuid4()
        request_at = datetime.now(timezone.utc)
        if use_openrouter:
            response_text, input_tokens, output_tokens = _openrouter_complete(
                model=config.advisor_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=openrouter_key,
            )
        else:
            response = resolved_client.messages.create(
                model=config.advisor_model,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            response_text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
            input_tokens = getattr(response.usage, "input_tokens", 0) or 0
            output_tokens = getattr(response.usage, "output_tokens", 0) or 0
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
        except (ValueError, json.JSONDecodeError):
            parsed = None
            continue

    signals: list[ForecastSignal] = []
    for instrument_id in instrument_ids:
        if parsed is not None and instrument_id in parsed:
            entry = parsed[instrument_id]
            signals.append(
                ForecastSignal(
                    signal_id=uuid4(),
                    advisor_id=ADVISOR_ID,
                    advisor_version=advisor_version,
                    generated_at=generated_at,
                    data_cutoff_at=data_cutoff_at,
                    expires_at=generated_at + timedelta(hours=4),
                    venue=snapshot.get("venue", "hyperliquid"),
                    instrument_id=instrument_id,
                    forecast_target=FORECAST_TARGET,
                    horizon=HORIZON,
                    action=entry["action"],
                    raw_score=entry["raw_score"],
                    thesis=entry["thesis"],
                    invalidation_condition=entry["invalidation_condition"],
                    evidence_feature_ids=entry["evidence_feature_ids"],
                    snapshot_id=snapshot_id,
                    model_run_id=last_model_run_id,
                )
            )
        else:
            signals.append(
                _flat_signal(
                    instrument_id=instrument_id,
                    venue=snapshot.get("venue", "hyperliquid"),
                    generated_at=generated_at,
                    data_cutoff_at=data_cutoff_at,
                    snapshot_id=snapshot_id,
                    advisor_version=advisor_version,
                    model_run_id=last_model_run_id,
                    thesis="LLM response failed to parse after retry; defaulted to flat.",
                )
            )
    return signals
