"""Research-only forecast types (sentiment plan §8).

STRUCTURAL ISOLATION INVARIANT: the portfolio engine, risk engine, and
executor must never import this module. A ResearchForecast cannot be
converted into a ForecastSignal or OrderIntent by any code path; promotion
of a research advisor is a future human-made code change, not a config
flip. Import-graph tests enforce this.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ResearchForecast(BaseModel):
    research_only: Literal[True] = True
    schema_version: Literal["1.0"] = "1.0"
    forecast_id: UUID
    advisor_id: str
    advisor_version: str
    generated_at: datetime
    data_cutoff_at: datetime
    instrument_id: str
    horizon: timedelta
    direction: Literal["long", "short", "flat"]
    abstain: bool
    probability_positive: float | None = None   # calibrated head, may be None early
    expected_excess_return_bps: float | None = None
    confidence: float
    evidence_feature_ids: list[str] = Field(default_factory=list)
    snapshot_id: UUID
    model_run_id: UUID | None = None            # None for deterministic baselines
