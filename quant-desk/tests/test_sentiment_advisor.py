"""Tests for quantdesk.advisors.sentiment_llm.run_crypto_sentiment_llm_v1."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock
from uuid import uuid4

import quantdesk.advisors.sentiment_llm as sentiment_llm
from quantdesk.advisors.sentiment_llm import (
    ADVISOR_ID,
    HORIZON,
    SentimentLlmAdvisorConfig,
    run_crypto_sentiment_llm_v1,
)

GENERATED_AT = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


def _config() -> SentimentLlmAdvisorConfig:
    return SentimentLlmAdvisorConfig(
        advisor_model="anthropic/claude-haiku-4.5",
        monthly_budget_usd=Decimal("50"),
        max_cost_per_decision_usd=Decimal("1"),
        provider="openrouter",
    )


def _healthy_snapshot() -> dict:
    return {
        "features": {"BTC:sent_mean_1h_zscore": 1.2, "BTC:spike_flag": False},
        "samples": ["btc looking bullish today"],
        "data_health": 0.95,
    }


def test_degraded_data_short_circuits_without_llm_or_persist(monkeypatch):
    mock_complete = Mock()
    monkeypatch.setattr(sentiment_llm, "_openrouter_complete", mock_complete)
    monthly_spend = Mock(return_value=Decimal("0"))
    persist = Mock()

    snapshot = {"features": {}, "samples": [], "data_health": 0.1}

    forecasts = run_crypto_sentiment_llm_v1(
        snapshot=snapshot,
        instrument_ids=["BTC", "ETH"],
        config=_config(),
        snapshot_id=uuid4(),
        generated_at=GENERATED_AT,
        monthly_spend=monthly_spend,
        persist=persist,
    )

    assert len(forecasts) == 2
    for f in forecasts:
        assert f.abstain is True
        assert f.direction == "flat"
        assert f.probability_positive is None
        assert f.expected_excess_return_bps is None
        assert f.confidence == 0.0
        assert f.model_run_id is None
        assert f.advisor_id == ADVISOR_ID
        assert f.horizon == HORIZON
        assert f.research_only is True

    mock_complete.assert_not_called()
    monthly_spend.assert_not_called()
    persist.assert_not_called()


def test_all_null_features_short_circuits(monkeypatch):
    mock_complete = Mock()
    monkeypatch.setattr(sentiment_llm, "_openrouter_complete", mock_complete)
    monthly_spend = Mock(return_value=Decimal("0"))
    persist = Mock()

    snapshot = {"features": {"BTC:sent_mean_1h_zscore": None}, "samples": [], "data_health": 0.95}

    forecasts = run_crypto_sentiment_llm_v1(
        snapshot=snapshot,
        instrument_ids=["BTC"],
        config=_config(),
        snapshot_id=uuid4(),
        generated_at=GENERATED_AT,
        monthly_spend=monthly_spend,
        persist=persist,
    )

    assert len(forecasts) == 1
    assert forecasts[0].abstain is True
    mock_complete.assert_not_called()
    monthly_spend.assert_not_called()
    persist.assert_not_called()


def test_budget_exceeded_short_circuits_without_llm(monkeypatch):
    mock_complete = Mock()
    monkeypatch.setattr(sentiment_llm, "_openrouter_complete", mock_complete)
    monthly_spend = Mock(return_value=Decimal("100"))  # >= budget of 50
    persist = Mock()

    forecasts = run_crypto_sentiment_llm_v1(
        snapshot=_healthy_snapshot(),
        instrument_ids=["BTC"],
        config=_config(),
        snapshot_id=uuid4(),
        generated_at=GENERATED_AT,
        monthly_spend=monthly_spend,
        persist=persist,
    )

    assert len(forecasts) == 1
    assert forecasts[0].abstain is True
    assert forecasts[0].direction == "flat"
    assert forecasts[0].model_run_id is None
    mock_complete.assert_not_called()
    persist.assert_not_called()
    monthly_spend.assert_called()


def test_happy_path_returns_one_forecast_per_instrument(monkeypatch):
    valid_response = json.dumps(
        {
            "forecasts": [
                {
                    "instrument_id": "BTC",
                    "direction": "long",
                    "abstain": False,
                    "probability_positive": 0.62,
                    "expected_excess_return_bps": 15.0,
                    "confidence": 0.7,
                    "thesis": "Aggregate sentiment features show a mild bullish tilt.",
                    "evidence_feature_ids": ["BTC:sent_mean_1h_zscore"],
                },
                {
                    "instrument_id": "ETH",
                    "direction": "flat",
                    "abstain": False,
                    "probability_positive": 0.5,
                    "expected_excess_return_bps": None,
                    "confidence": 0.4,
                    "thesis": "No strong signal.",
                    "evidence_feature_ids": [],
                },
            ]
        }
    )
    mock_complete = Mock(return_value=(valid_response, 100, 50))
    monkeypatch.setattr(sentiment_llm, "_openrouter_complete", mock_complete)
    monthly_spend = Mock(return_value=Decimal("0"))
    persist = Mock()

    forecasts = run_crypto_sentiment_llm_v1(
        snapshot=_healthy_snapshot(),
        instrument_ids=["BTC", "ETH"],
        config=_config(),
        snapshot_id=uuid4(),
        generated_at=GENERATED_AT,
        monthly_spend=monthly_spend,
        persist=persist,
        api_key="fake-key",
    )

    assert len(forecasts) == 2
    by_id = {f.instrument_id: f for f in forecasts}
    assert by_id["BTC"].direction == "long"
    assert by_id["BTC"].abstain is False
    assert by_id["BTC"].probability_positive == 0.62
    assert by_id["ETH"].direction == "flat"

    for f in forecasts:
        assert f.advisor_id == ADVISOR_ID
        assert f.horizon == HORIZON
        assert f.research_only is True
        assert f.model_run_id is not None

    mock_complete.assert_called_once()
    persist.assert_called_once()


def test_malformed_json_both_attempts_abstains_and_persists_twice(monkeypatch):
    mock_complete = Mock(return_value=("not json at all", 10, 5))
    monkeypatch.setattr(sentiment_llm, "_openrouter_complete", mock_complete)
    monthly_spend = Mock(return_value=Decimal("0"))
    persist = Mock()

    forecasts = run_crypto_sentiment_llm_v1(
        snapshot=_healthy_snapshot(),
        instrument_ids=["BTC"],
        config=_config(),
        snapshot_id=uuid4(),
        generated_at=GENERATED_AT,
        monthly_spend=monthly_spend,
        persist=persist,
        api_key="fake-key",
    )

    assert len(forecasts) == 1
    assert forecasts[0].abstain is True
    assert forecasts[0].direction == "flat"
    assert forecasts[0].model_run_id is not None  # last attempt's model_run_id retained

    assert mock_complete.call_count == 2
    assert persist.call_count == 2
