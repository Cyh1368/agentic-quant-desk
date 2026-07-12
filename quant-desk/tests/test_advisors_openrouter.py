"""OpenRouter provider path: request shape, provenance, cost, parse flow."""
import json
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import httpx

from quantdesk.advisors import llm_trend
from quantdesk.advisors.llm_trend import (
    LlmTrendAdvisorConfig,
    _openrouter_complete,
    run_crypto_trend_llm_v1,
)

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


def _config() -> LlmTrendAdvisorConfig:
    return LlmTrendAdvisorConfig(
        advisor_model="anthropic/claude-haiku-4.5",
        monthly_budget_usd=Decimal("25"),
        max_cost_per_decision_usd=Decimal("0.05"),
        provider="openrouter",
    )


def test_openrouter_complete_parses_openai_format():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        text, tin, tout = _openrouter_complete(
            model="anthropic/claude-haiku-4.5",
            system_prompt="sys", user_prompt="usr",
            api_key="sk-or-test", http_client=client,
        )
    assert text == "hello" and (tin, tout) == (100, 20)
    assert captured["url"].endswith("/chat/completions")
    assert "openrouter.ai" in captured["url"]
    assert captured["auth"] == "Bearer sk-or-test"
    assert captured["body"]["messages"][0]["role"] == "system"


def test_run_advisor_openrouter_produces_signals_and_provenance(monkeypatch):
    llm_response = json.dumps({
        "forecasts": [{
            "instrument_id": "BTC", "action": "long", "raw_score": 0.7,
            "thesis": "uptrend", "invalidation_condition": "close below SMA168",
            "evidence_feature_ids": ["btc_ret_7d@v1"],
        }]
    })
    monkeypatch.setattr(
        llm_trend, "_openrouter_complete",
        lambda **kw: (llm_response, 500, 80),
    )
    provenance_records = []
    signals = run_crypto_trend_llm_v1(
        snapshot={"venue": "hyperliquid"}, instrument_ids=["BTC", "ETH"],
        calibration_summary={}, config=_config(), snapshot_id=uuid4(),
        generated_at=NOW, data_cutoff_at=NOW,
        monthly_spend=lambda: Decimal("0"),
        persist=provenance_records.append,
        api_key="sk-or-test",
    )
    assert len(signals) == 2
    by_id = {s.instrument_id: s for s in signals}
    assert by_id["BTC"].action == "long"
    assert by_id["ETH"].action == "flat"  # not in response -> safe default
    assert len(provenance_records) == 1
    prov = provenance_records[0]
    assert prov.model_provider == "openrouter"
    # 500 in @ $1.05/M + 80 out @ $5.25/M
    assert prov.cost_usd == Decimal("500") / 1_000_000 * Decimal("1.05") + \
        Decimal("80") / 1_000_000 * Decimal("5.25")


def test_budget_guard_applies_to_openrouter_too(monkeypatch):
    def boom(**kw):
        raise AssertionError("no call should be made when over budget")
    monkeypatch.setattr(llm_trend, "_openrouter_complete", boom)
    signals = run_crypto_trend_llm_v1(
        snapshot={}, instrument_ids=["BTC"], calibration_summary={},
        config=_config(), snapshot_id=uuid4(),
        generated_at=NOW, data_cutoff_at=NOW,
        monthly_spend=lambda: Decimal("25"),
        persist=lambda p: None, api_key="sk-or-test",
    )
    assert signals[0].action == "flat"
    assert "budget" in signals[0].thesis.lower()
