# agentic-quant-desk

An auditable, deterministic quant platform with experimental LLM forecasters,
currently live on Hyperliquid **testnet**.

Built per `agentic-quant-desk-plan-v2.md`. **No LLM sits in the mandatory path for
closing, reducing, cancelling, reconciling, or protecting a position.**

## Current status
- BTC + ETH, Hyperliquid, running 6x/day on a scheduled host (PM2).
- `desk.mode: live` on `network: testnet` — idempotent order submission,
  venue price rounding, stop attach with flatten-on-failure, and
  venue-truth reconciliation + watchdog are in place. Mainnet stays
  hard-blocked in code until Gate C sign-off.
- Deterministic baseline advisor (vol-scaled TS momentum), LLM trend
  advisor (`crypto_trend_llm_v1`, Claude Haiku via OpenRouter), and a
  Twitter sentiment advisor now all contribute at a provisional 0.3
  trading weight, promoted from shadow-only after prospective scoring.
- Decision trail and P0 alerts posted to Discord via webhook.
- `desk.mode: dry | live` in `config/desk.yaml` (or `DESK_MODE` env) is the
  trading switch; mainnet `live` refuses to start until Gate C is passed.

## Layout
```
quantdesk/
  common/     schemas (Pydantic, Decimal money), config loader
  data/       immutable raw landing zone, Hyperliquid + Coinbase reference, snapshots
  features/   deterministic versioned feature computation
  advisors/   ts_momentum_baseline (deterministic), crypto_trend_llm (OpenRouter), sentiment (Twitter)
  portfolio/  calibration → reliability weighting → dedup → ERC sizing → intents
  risk/       hard risk engine, stress scenarios, halt state machine
  execution/  shadow fill simulator + live testnet adapter, order state machine, protection watchdog
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
python -m quantdesk            # one full decision cycle (dry or live per config/env)
python -m quantdesk.data       # data-pipeline smoke fetch
python -m quantdesk.dashboard  # read-only web dashboard on http://localhost:8420
```

## Dashboard
`quantdesk/dashboard/` serves a read-only web UI (Flask, sqlite opened with
`mode=ro`) over the ledgers and snapshots: latest advisor signals, positions,
intents / risk verdicts / orders / fills, LLM spend, tweet-level sentiment with
links to the original posts on x.com, and 30-day history (hourly sentiment
chart, forecast/order/fill logs). Runs under PM2 as `qd-dashboard` (port 8420,
localhost-bound — reach it from another machine with
`ssh -L 8420:localhost:8420 <desk-host>`).

## Cost budget ($50/mo total)
- LLM: hard-capped at $25/mo in `config/desk.yaml` (`llm.monthly_budget_usd`);
  calls are skipped (advisor goes flat) when the ledger cost log exceeds it.
  Observed: ~$0.004/cycle via OpenRouter ≈ $0.80/mo at 6 cycles/day.
- Infra: $0 (runs on existing hardware); data: $0 (public endpoints).

## Governance gates (do not skip)
Phase 0 items in the plan — venue admission checklist from the execution
host, tax review, capital/threat model — are prerequisites for mainnet
trading. `desk.mode: live` on `network: mainnet` is refused at startup
until Gate C is passed.

---

*Architecture, implementation, and integration by Claude (Fable 5), Anthropic —
orchestrating parallel subagents under the plan-v2 safety contract.*
