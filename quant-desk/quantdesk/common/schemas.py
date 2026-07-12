"""Versioned inter-component message schemas (plan §5).

Money and quantities are Decimal, never float. Every schema carries
provenance IDs so any order traces back through verdict -> intent ->
decision -> forecasts -> snapshot -> raw data.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints
from typing_extensions import Annotated

Thesis = Annotated[str, StringConstraints(max_length=500)]


class ForecastSignal(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    signal_id: UUID
    advisor_id: str                    # "crypto_trend_llm", "ts_momentum_baseline"
    advisor_version: str               # prompt hash or code version
    generated_at: datetime
    data_cutoff_at: datetime           # nothing after this informed the forecast
    expires_at: datetime
    venue: str
    instrument_id: str
    forecast_target: str               # e.g. "next_24h_excess_return_sign"
    horizon: timedelta
    action: Literal["long", "short", "flat"]
    raw_score: float                   # model's own scale
    calibrated_probability_positive: float | None = None
    expected_excess_return_bps: float | None = None
    uncertainty_bps: float | None = None
    invalidation_condition: str | None = None
    evidence_feature_ids: list[str] = Field(default_factory=list)
    evidence_event_ids: list[str] = Field(default_factory=list)
    thesis: Thesis = ""
    snapshot_id: UUID
    model_run_id: UUID | None = None   # None for deterministic advisors


class OrderIntent(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    intent_id: UUID                    # basis of the deterministic client order ID
    decision_id: UUID
    created_at: datetime
    expires_at: datetime               # stale intents are never executed
    venue: str
    account_id: str
    instrument_id: str
    instrument_type: Literal["spot", "perp"]
    side: Literal["buy", "sell"]
    effect: Literal["open", "increase", "reduce", "close"]
    quantity: Decimal
    quantity_unit: str                 # "BTC", "USD-notional" — explicit, always
    order_type: Literal["limit", "market", "stop_market", "stop_limit"]
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: Literal["GTC", "IOC", "ALO"]
    reduce_only: bool
    max_slippage_bps: int
    max_fee_bps: int
    snapshot_id: UUID
    risk_verdict_id: UUID | None = None


class RiskCheckResult(BaseModel):
    check_id: str
    passed: bool
    observed: Decimal
    threshold: Decimal
    unit: str
    check_code_version: str


class StressResult(BaseModel):
    scenario_id: str                   # "gap_down", "funding_spike", "venue_outage"
    executable_loss: Decimal
    loss_unit: str
    within_limit: bool
    detail: str = ""


class RiskVerdict(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    verdict_id: UUID
    intent_id: UUID
    evaluated_at: datetime
    hard_check_version: str
    portfolio_snapshot_id: UUID
    verdict: Literal["approve", "reject", "approve_reduced"]
    approved_quantity: Decimal
    hard_checks: list[RiskCheckResult]
    stress_results: list[StressResult] = Field(default_factory=list)
    llm_critic_result_id: UUID | None = None
    reason_codes: list[str] = Field(default_factory=list)


class ModelProvenance(BaseModel):
    """One record per LLM invocation (plan §5)."""
    model_run_id: UUID
    model_provider: str
    model_id: str
    model_version: str | None = None
    system_prompt_hash: str
    user_prompt_hash: str
    schema_version: str
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    input_dataset_cutoff_timestamp: datetime
    request_at: datetime
    response_at: datetime
    raw_response_hash: str
    parser_version: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = Decimal("0")


class Lineage(BaseModel):
    """Record-level lineage attached to every normalized market/event row (plan §3)."""
    event_time: datetime
    published_time: datetime | None = None
    provider_time: datetime | None = None
    ingested_time: datetime
    available_to_strategy_time: datetime
    source_id: str
    source_revision: str | None = None
    raw_payload_hash: str
    normalizer_version: str
    quality_flags: list[str] = Field(default_factory=list)
