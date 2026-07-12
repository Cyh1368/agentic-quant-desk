import json
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from quantdesk.advisors.llm_trend import (
    ADVISOR_ID,
    LlmTrendAdvisorConfig,
    run_crypto_trend_llm_v1,
)
from quantdesk.common.schemas import ModelProvenance


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _response(payload: dict, input_tokens: int = 100, output_tokens: int = 50):
    return SimpleNamespace(
        content=[_text_block(json.dumps(payload))],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def _config(**overrides):
    kwargs = dict(
        advisor_model="claude-haiku-4-5-20251001",
        monthly_budget_usd=Decimal("25"),
        max_cost_per_decision_usd=Decimal("0.05"),
    )
    kwargs.update(overrides)
    return LlmTrendAdvisorConfig(**kwargs)


def _run_kwargs(**overrides):
    kwargs = dict(
        snapshot={"venue": "hyperliquid", "instruments": {"BTC": {}, "ETH": {}}},
        instrument_ids=["BTC", "ETH"],
        calibration_summary={"window": 100, "n_scored": 0},
        config=_config(),
        snapshot_id=uuid4(),
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        data_cutoff_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        monthly_spend=lambda: Decimal("0"),
        persist=lambda record: None,
    )
    kwargs.update(overrides)
    return kwargs


VALID_PAYLOAD = {
    "forecasts": [
        {
            "instrument_id": "BTC",
            "action": "long",
            "raw_score": 0.8,
            "thesis": "Strong uptrend on 7d momentum.",
            "invalidation_condition": "Price closes below 50MA.",
            "evidence_feature_ids": ["return_7d"],
        },
        {
            "instrument_id": "ETH",
            "action": "flat",
            "raw_score": 0.0,
            "thesis": "No clear regime.",
            "invalidation_condition": None,
            "evidence_feature_ids": [],
        },
    ]
}


def test_happy_path_parses_and_returns_signals():
    client = FakeClient([_response(VALID_PAYLOAD)])
    persisted = []
    signals = run_crypto_trend_llm_v1(
        **_run_kwargs(client=client, persist=persisted.append)
    )
    assert len(signals) == 2
    btc = next(s for s in signals if s.instrument_id == "BTC")
    eth = next(s for s in signals if s.instrument_id == "ETH")
    assert btc.action == "long"
    assert btc.raw_score == pytest.approx(0.8)
    assert btc.advisor_id == ADVISOR_ID
    assert eth.action == "flat"
    assert len(persisted) == 1
    assert client.messages.calls[0]["max_tokens"] == 1000
    assert client.messages.calls[0]["temperature"] == 0


def test_provenance_record_completeness():
    client = FakeClient([_response(VALID_PAYLOAD, input_tokens=200, output_tokens=80)])
    persisted = []
    run_crypto_trend_llm_v1(**_run_kwargs(client=client, persist=persisted.append))
    assert len(persisted) == 1
    record = persisted[0]
    assert isinstance(record, ModelProvenance)
    assert record.model_provider == "anthropic"
    assert record.model_id == "claude-haiku-4-5-20251001"
    assert len(record.system_prompt_hash) == 64
    assert len(record.user_prompt_hash) == 64
    assert len(record.raw_response_hash) == 64
    assert record.input_tokens == 200
    assert record.output_tokens == 80
    assert record.cost_usd > 0
    assert record.request_at.tzinfo is not None
    assert record.response_at.tzinfo is not None
    assert record.response_at >= record.request_at


def test_parse_failure_then_retry_then_flat():
    bad_response = _response({"not_forecasts": []})
    # second attempt also malformed -> should default to flat after exactly one retry
    also_bad_response = _response({"forecasts": "not-a-list"})
    client = FakeClient([bad_response, also_bad_response])
    persisted = []
    signals = run_crypto_trend_llm_v1(**_run_kwargs(client=client, persist=persisted.append))
    assert len(client.messages.calls) == 2  # initial + one retry, no more
    assert len(persisted) == 2  # one provenance record per actual call
    assert all(s.action == "flat" for s in signals)
    assert {s.instrument_id for s in signals} == {"BTC", "ETH"}


def test_retry_recovers_on_second_attempt():
    bad_response = _response("not even an object")  # will raise JSONDecodeError on malformed text
    good_response = _response(VALID_PAYLOAD)
    # first response body is invalid JSON text
    bad_response.content = [_text_block("not json at all {{{")]
    client = FakeClient([bad_response, good_response])
    persisted = []
    signals = run_crypto_trend_llm_v1(**_run_kwargs(client=client, persist=persisted.append))
    assert len(client.messages.calls) == 2
    assert len(persisted) == 2
    btc = next(s for s in signals if s.instrument_id == "BTC")
    assert btc.action == "long"


def test_budget_guard_skips_call_when_over_monthly_budget():
    client = FakeClient([_response(VALID_PAYLOAD)])
    persisted = []
    signals = run_crypto_trend_llm_v1(
        **_run_kwargs(
            client=client,
            persist=persisted.append,
            monthly_spend=lambda: Decimal("25.01"),
        )
    )
    assert len(client.messages.calls) == 0
    assert len(persisted) == 0
    assert all(s.action == "flat" for s in signals)
    assert all("budget_exceeded" in s.thesis for s in signals)


def test_budget_guard_skips_when_max_cost_per_decision_is_zero():
    client = FakeClient([_response(VALID_PAYLOAD)])
    signals = run_crypto_trend_llm_v1(
        **_run_kwargs(
            client=client,
            config=_config(max_cost_per_decision_usd=Decimal("0")),
        )
    )
    assert len(client.messages.calls) == 0
    assert all(s.action == "flat" for s in signals)


def test_unknown_instrument_in_response_defaults_that_one_to_flat_only_via_full_reparse():
    # response references an instrument outside instrument_ids -> whole parse invalid -> retry -> flat
    payload = {
        "forecasts": [
            {"instrument_id": "DOGE", "action": "long", "raw_score": 1.0, "thesis": "x",
             "invalidation_condition": None, "evidence_feature_ids": []}
        ]
    }
    client = FakeClient([_response(payload), _response(payload)])
    signals = run_crypto_trend_llm_v1(**_run_kwargs(client=client))
    assert all(s.action == "flat" for s in signals)
    assert len(client.messages.calls) == 2
