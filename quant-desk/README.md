# agentic-quant-desk

An auditable, deterministic quant platform with experimental LLM forecasters (v1 — shadow mode)

Built per `agentic-quant-desk-plan-v2.md`. **No LLM sits in the mandatory path for
closing, reducing, cancelling, reconciling, or protecting a position.**

## v1 scope
- BTC + ETH only, Hyperliquid market data, **shadow execution only** — no
  trading credential exists in this codebase yet (Gate C prerequisite).
- One deterministic baseline advisor (vol-scaled TS momentum) + one LLM
  advisor (`crypto_trend_llm_v1`, Claude Haiku via OpenRouter) under a
  pre-registered research contract, scored prospectively, with **zero order
  influence** until Gate D.
- Decision trail and P0 alerts posted to Discord via webhook.
- `desk.mode: dry | live` in `config/desk.yaml` (or `DESK_MODE` env) is the
  trading switch; `live` refuses to start until Gate C is passed and a live
  execution adapter exists.

## Layout
```
quantdesk/
  common/     schemas (Pydantic, Decimal money), config loader
  data/       immutable raw landing zone, Hyperliquid + Coinbase reference, snapshots
  features/   deterministic versioned feature computation
  advisors/   ts_momentum_baseline (deterministic), crypto_trend_llm (OpenRouter)
  portfolio/  calibration → reliability weighting → dedup → ERC sizing → intents
  risk/       hard risk engine, stress scenarios, halt state machine
  execution/  shadow fill simulator, order state machine, protection watchdog
  ledger/     single-writer SQLite (WAL), durable intent queue, cost log
  scoring/    prospective forecast scoring, weekly report
config/       desk.yaml (limits, budget), research contracts
```

## Setup
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env   # fill OPENROUTER_API_KEY (+ optional DISCORD_WEBHOOK_URL)
pytest
```

## Run
```bash
python -m quantdesk            # one full shadow decision cycle
python -m quantdesk.data       # data-pipeline smoke fetch
```

## Cost budget ($50/mo total)
- LLM: hard-capped at $25/mo in `config/desk.yaml` (`llm.monthly_budget_usd`);
  calls are skipped (advisor goes flat) when the ledger cost log exceeds it.
  Observed: ~$0.004/cycle via OpenRouter ≈ $0.80/mo at 6 cycles/day.
- Infra: $0 (runs on existing hardware); data: $0 (public endpoints).

## Governance gates (do not skip)
Phase 0 items in the plan — venue admission checklist from the execution
host, tax review, capital/threat model — are prerequisites for any live
trading. `desk.mode: live` is refused at startup in v1.

---

*Architecture, implementation, and integration by Claude (Fable 5), Anthropic —
orchestrating parallel subagents under the plan-v2 safety contract.*
