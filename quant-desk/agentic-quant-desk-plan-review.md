# Evaluation of the Fable 5 Agentic Quant Desk Plan

**Reviewed document:** `agentic-quant-desk-plan.md`  
**Overall assessment:** **Good architectural direction, but not yet implementation-ready or capital-ready**  
**Indicative score:** **7.0/10 as a concept; 4.5/10 as an executable trading specification**

## Executive conclusion

The plan gets the most important architectural principle right: **LLMs should generate research opinions, while deterministic software controls sizing, risk, and execution.** The use of typed messages, one-way reductions by LLM reviewers, credential isolation, venue-native protective orders, staged rollout, and append-only decision logging are all sound.

However, the document remains a high-level design rather than a sufficiently specified quant-desk implementation. Its largest weaknesses are:

1. It does not define a rigorous process for determining whether any advisor has statistically defensible predictive value.
2. It overstates the usefulness of historical LLM replay and understates temporal leakage, model-version drift, and non-reproducibility.
3. It makes an incorrect or at least unsafe inference that possession of a US Social Security number likely creates US tax-reporting obligations.
4. It treats LLM risk approval as a required gate, creating an unnecessary availability and consistency dependency in a safety-critical path.
5. It lacks a sufficiently rigorous execution state machine, idempotency model, and failure-recovery specification.
6. It proposes initial risk limits that are too permissive for an experimental system.
7. It mixes heterogeneous markets—crypto perpetuals and prediction markets—before proving one complete operational loop.
8. It under-specifies data quality, point-in-time correctness, security, observability, testing, and model governance.

The recommended implementation is therefore a **deterministic trading platform with optional LLM research modules**, not an autonomous multi-agent desk in which LLM approval is necessary to operate.

---

## What the plan does well

### 1. Correct separation between judgment and authority

The statement **“LLMs propose, code disposes”** is the strongest part of the plan. The executor has no LLM dependency and is intended to be the only component with trading credentials. This substantially reduces prompt-injection and hallucination risk.

**Retain:**

- Executor as pure deterministic code.
- No withdrawal permission for the execution credential where the venue supports that separation.
- Hard risk limits that agents cannot edit.
- LLM reviewers permitted only to veto or reduce—not add exposure.

### 2. Typed inter-agent contracts

Pydantic validation is appropriate. Structured schemas prevent free-form prose from silently becoming a trade instruction.

**Retain, but expand:** schemas need versioning, timestamps, provenance, market identifiers, currency units, confidence calibration metadata, and idempotency fields.

### 3. Sensible staged deployment

Signal-only → paper → micro-live → small-live is the right general sequence. The warning against self-modifying prompts and limits is also correct.

### 4. Recognition of correlated “multiple agents”

The plan correctly notes that several agents reading the same data may represent one correlated opinion rather than independent evidence. Input diversity matters more than persona diversity.

### 5. Credential and process isolation

Separating agent, executor, and watchdog processes is directionally correct. The agent process should never be able to read the execution secret or directly invoke venue write endpoints.

---

# Major findings and recommended fixes

## Critical finding 1 — No formal definition or validation of “edge”

### Problem

The plan describes advisors, signals, features, and promotion stages, but it never defines how an advisor proves that its output contains predictive information beyond a benchmark. A positive paper-trading period is not sufficient. With several agents, symbols, horizons, prompts, and parameter choices, false discoveries become likely.

“Conviction” is accepted as a model-provided number, but no calibration process is specified. An LLM saying `0.82` does not make the forecast an 82% event.

### Recommended fix

Create an **Advisor Research Contract** for every advisor before it may influence orders:

```yaml
advisor_id: crypto_trend_v1
forecast_target: next_24h_excess_return_sign
universe: [BTC, ETH, SOL]
decision_times_utc: [00:00, 04:00, 08:00, 12:00, 16:00, 20:00]
horizon: 24h
benchmark: volatility_scaled_time_series_momentum
abstention_allowed: true
primary_metric: net_information_coefficient
secondary_metrics:
  - brier_score
  - calibration_error
  - net_pnl_after_costs
  - turnover
  - maximum_drawdown
minimum_sample_size: 200_decisions
promotion_rule: pre_registered_thresholds_only
```

For each advisor:

- Pre-register the target, horizon, universe, benchmark, cost model, and pass/fail thresholds.
- Compare against simple baselines: always-flat, buy-and-hold, random forecast with identical trade frequency, and a deterministic strategy using the same features.
- Use block bootstrap or time-series-aware confidence intervals.
- Correct for multiple testing when comparing many advisors or prompt variants.
- Measure probability calibration with Brier score/reliability curves, not self-declared conviction.
- Require economic significance after fees, spread, funding, adverse selection, and failed-order costs—not only statistical significance.

### Rationale

Without this layer, the desk is an elaborate execution platform for unvalidated opinions. The architecture may be safe yet still have negative expected value.

**Priority:** P0—must be completed before live trading.

---

## Critical finding 2 — Historical LLM replay is not equivalent to a valid backtest

### Problem

The plan says journaling gives replay “for free.” It does not. Replaying historical snapshots through a contemporary LLM is vulnerable to:

- Training-data knowledge of later outcomes.
- Changed model weights and provider behavior.
- Non-deterministic responses.
- Hidden system-prompt or safety-layer changes.
- Current knowledge leaking into interpretation of old news.
- Survivorship and revision bias in historical data feeds.

Caching by snapshot hash only makes repeated runs cheaper; it does not make the original forecast point-in-time valid.

### Recommended fix

Separate three distinct evaluation methods:

1. **Classical backtest:** only deterministic features and deterministic decision rules. This is the primary tool for testing sizing, portfolio logic, and execution assumptions.
2. **Prospective shadow test:** store each live LLM forecast before outcomes occur. This is the only reliable dataset for evaluating the LLM advisor itself.
3. **Historical LLM replay:** use only for exploratory analysis, clearly labeled as contaminated and never used alone for promotion.

Add the following to every LLM record:

```text
model_provider
model_id
model_snapshot/version
system_prompt_hash
user_prompt_hash
schema_version
temperature/top_p/seed where supported
input_dataset_cutoff_timestamp
request_timestamp
response_timestamp
raw_response_hash
parser_version
```

Promotion of an LLM advisor should depend primarily on **prospective, timestamped shadow forecasts**, ideally over enough independent market events—not merely two to four weeks.

### Rationale

A quant backtest must reproduce the information set available at the decision time. Modern LLMs cannot reliably be forced to “forget” later events.

**Priority:** P0.

---

## Critical finding 3 — The tax statement is unsafe and should be removed

### Problem

The plan states that, “As a US SSN holder you likely have US tax reporting obligations.” An SSN alone does not determine US tax residency or filing obligations. Relevant factors include citizenship, green-card status, the substantial-presence test, visa exemptions, income source, physical presence, and whether income is effectively connected with a US trade or business.

### Recommended replacement

> Possession of a US Social Security number does not by itself establish US tax residency or a US filing obligation. Determine US status using citizenship, green-card status, the substantial-presence test and any applicable visa exemptions, then determine the source and character of trading income. Taiwan tax treatment must be assessed separately. Obtain advice from a professional familiar with US–Taiwan cross-border taxation before live deployment.

### Rationale

This correction prevents the system design from embedding a false identity/tax assumption. The IRS defines nonresident-alien status using the green-card and substantial-presence tests, not possession of an SSN.

**Priority:** P0 for documentation and account setup.

---

## Critical finding 4 — Venue eligibility must be a deployment gate, not a caveat

### Problem

The plan reasonably warns that an SSN is insufficient to establish Kalshi access, but it still proposes prediction markets as a second v1 domain. It also assumes a MetaMask wallet implies practical access to Polymarket. Venue eligibility and geographic restrictions can change, and API order placement may be rejected by geolocation controls.

### Recommended fix

Create a formal **Venue Admission Checklist** before any adapter is enabled:

```text
[ ] User eligibility verified under current member agreement/terms
[ ] Country and IP geoblock checked from the actual execution host
[ ] Automated trading/API use permitted for intended account type
[ ] KYC/account status complete
[ ] Deposit and withdrawal paths tested with a minimal amount
[ ] Market-data and order-rate limits documented
[ ] Fee schedule captured and versioned
[ ] Position, liquidation, resolution, and dispute mechanics documented
[ ] Tax/export records obtainable
[ ] Emergency key revocation or account lock tested
```

The adapter should run a **startup compliance preflight** and refuse to trade when the venue’s official geoblock endpoint, account state, or terms configuration indicates ineligibility.

### Rationale

Eligibility is a binary production dependency, not an implementation footnote. Current official Polymarket documentation explicitly instructs builders to check geographic restrictions before order placement, and current Kalshi guidance points users to country restrictions in its Member Agreement.

**Priority:** P0 before venue integration.

---

## Critical finding 5 — The LLM risk manager should not be a required approval dependency

### Problem

`require_llm_risk_approval: true` makes safety-critical operation dependent on a probabilistic external service. An outage, timeout, malformed response, provider policy change, or inconsistent judgment can block risk-reducing actions. It can also create a false impression that LLM approval makes a trade safe.

The risk manager is described as having final decision power, but the actual final authority should be deterministic risk code. An LLM may provide an additional veto but should never be required to close, cancel, hedge, or reduce exposure.

### Recommended fix

Use this authority hierarchy:

1. **Deterministic risk engine:** final authority for all limits and mandatory actions.
2. **Risk-reducing actions:** always allowed when mechanically valid; never await LLM approval.
3. **LLM risk critic:** optional veto/reduction on new or increased exposure only.
4. **LLM unavailable or invalid:** default to no new exposure, while exits and protective actions continue normally.

Replace:

```yaml
require_llm_risk_approval: true
```

with:

```yaml
llm_risk_critic:
  enabled: true
  applies_to: exposure_increasing_actions_only
  timeout_seconds: 15
  failure_policy: reject_new_exposure
  may_block_exits: false
  may_increase_size: false
  may_widen_stop: false
```

### Rationale

Safety functions must be deterministic and available even when all AI services fail.

**Priority:** P0.

---

## Critical finding 6 — Initial risk limits are too permissive

### Problem

For an experimental system, 150% gross exposure, 100% net crypto beta, 25% single-position size, 3× leverage, and a 15% drawdown kill level are not conservative. The maximum order size of $500 also conflicts with the proposed $200–500 total micro-live account.

A daily halt that merely stops opening positions may leave losing exposure active. Conversely, automatically flattening at an arbitrary daily threshold can crystallize losses during transient volatility. The plan needs explicit mode transitions, not comments in YAML.

### Recommended v1 limits

For initial micro-live validation:

```yaml
account:
  max_gross_exposure_pct: 30
  max_net_directional_pct: 20
  max_single_position_pct: 10
  max_leverage_per_position: 1
  daily_loss_soft_halt_pct: 0.75
  daily_loss_hard_halt_pct: 1.25
  weekly_loss_hard_halt_pct: 2.5
  max_drawdown_kill_pct: 4
  human_restart_required_after_hard_halt: true
orders:
  max_order_size_pct_equity: 5
  max_orders_per_hour: 4
  max_cancel_replace_attempts: 2
  mandatory_protective_exit: true
```

Define exact behavior:

- **Soft halt:** prohibit exposure increases; allow closes/reductions.
- **Hard halt:** cancel all resting exposure-increasing orders, reduce or flatten according to predeclared policy, and require human restart.
- **Kill:** revoke trading session/key when possible and reconcile independently.

Raise limits only after operational—not P&L—criteria are met: zero duplicate orders, zero unprotected positions, complete reconciliation, bounded slippage, and successful recovery drills.

### Rationale

The first objective is to validate process integrity. High leverage and broad exposure obscure whether losses arise from strategy, implementation, or operational failure.

**Priority:** P0.

---

## High finding 7 — The order schema and execution state machine are insufficient

### Problem

`ProposedAction` lacks essential production fields. It does not identify venue, account, instrument type, quantity unit, order type, time-in-force, client order ID, expiry, decision timestamp, data cutoff, reduce-only status, or permitted price range. Floating-point values should not be used for money or quantities.

The execution description—place, wait, reprice, perhaps market—is too vague to prevent duplicate or contradictory orders after retries and restarts.

### Recommended fix

Use a versioned `OrderIntent` separate from research rationale:

```python
class OrderIntent(BaseModel):
    schema_version: Literal["1.0"]
    intent_id: UUID
    decision_id: UUID
    created_at: datetime
    expires_at: datetime
    venue: str
    account_id: str
    instrument_id: str
    instrument_type: Literal["spot", "perp", "prediction"]
    side: Literal["buy", "sell"]
    effect: Literal["open", "increase", "reduce", "close"]
    quantity: Decimal
    quantity_unit: str
    order_type: Literal["limit", "market", "stop_market", "stop_limit"]
    limit_price: Decimal | None
    stop_price: Decimal | None
    time_in_force: Literal["GTC", "IOC", "ALO"]
    reduce_only: bool
    max_slippage_bps: int
    max_fee_bps: int
    snapshot_id: UUID
    risk_verdict_id: UUID
```

Implement an explicit durable state machine:

```text
CREATED → RISK_APPROVED → SUBMITTING → ACKNOWLEDGED →
PARTIALLY_FILLED → FILLED → PROTECTED → CLOSED
                  ↘ REJECTED / EXPIRED / CANCELLED / UNKNOWN
```

Required properties:

- Deterministic client order IDs.
- Idempotent submission and retry.
- At-most-once intent consumption.
- Recovery from `UNKNOWN` by querying venue state before resubmission.
- Partial-fill policy.
- Cancel/replace race handling.
- Position and order reconciliation against venue truth.
- No assumption that protective orders are atomically attached unless the venue documents atomic behavior.
- Decimal arithmetic with venue tick/lot-size normalization.

### Rationale

Most automated-trading catastrophes occur in state handling, not signal generation.

**Priority:** P0 before testnet execution.

---

## High finding 8 — SQLite plus multiple containers is an avoidable reliability risk

### Problem

The plan proposes three containers but keeps the main operational store in SQLite. SQLite can be safe for limited workloads, but concurrent writers, network-mounted filesystems, backups during writes, and process restarts require careful handling. The plan does not specify a single-writer architecture, WAL mode, lock timeouts, or transaction boundaries.

### Recommended fix

Choose one of two explicit designs:

**Option A—simplest v1:** one process owns SQLite writes; all other components communicate through an internal queue. Enable WAL, foreign keys, busy timeout, transactional outbox, and periodic integrity checks.

**Option B—recommended production baseline:** PostgreSQL for decisions, orders, fills, risk events, and agent runs; Parquet/object storage for historical market data. Use migrations and database constraints to enforce uniqueness and state transitions.

Do not use a shared SQLite database file over a network filesystem.

### Rationale

The desk ledger is safety-critical. Concurrency behavior must be intentional rather than left to default library behavior.

**Priority:** P1.

---

## High finding 9 — Data governance is substantially under-specified

### Problem

`ingested_at` and `source` are not enough to guarantee point-in-time correctness. The plan omits event time, provider publication time, revision status, sequence gaps, timezone rules, corporate/event corrections, symbol mapping, and stale/degraded feed behavior.

News feeds also introduce prompt injection, licensing, duplication, and timestamp ambiguity.

### Recommended fix

Every market/event record should include:

```text
event_time
published_time
provider_time
ingested_time
available_to_strategy_time
source_id
source_revision
raw_payload_hash
normalizer_version
quality_flags
```

Add:

- UTC everywhere internally.
- Exchange sequence-number and WebSocket gap detection.
- Immutable raw-data landing zone.
- Point-in-time symbol and contract metadata.
- Feed-quality score and degraded mode.
- Cross-source price sanity checks before trading.
- News deduplication by canonical event, not only headline similarity.
- Prompt-boundary escaping and clear labeling of all news as untrusted data.
- Data-retention and licensing review for any paid feed.

### Rationale

A correct model operating on revised, stale, or future-visible data is still an invalid system.

**Priority:** P1.

---

## High finding 10 — Crypto and prediction markets should not share the initial live phase

### Problem

Crypto perpetuals and prediction contracts have materially different risk models:

- Perpetuals: funding, leverage, liquidation, continuous marks, stop execution.
- Prediction markets: binary/categorical payoff, event correlation, resolution criteria, oracle/dispute risk, expiry, dead capital, and legal/geographic constraints.

A generic `direction`, `stop_price`, and `take_profit_price` schema does not adequately represent both.

### Recommended fix

Limit v1 to **one venue and one instrument class**, preferably spot or unlevered/1× crypto exposure. Keep prediction-market advisors in research-only mode until a dedicated adapter and risk model exist.

Create market-specific schemas and risk engines:

- `PerpOrderIntent` with leverage, liquidation distance, funding exposure, mark/index divergence.
- `PredictionOrderIntent` with outcome token, maximum loss, resolution source, expiry, correlated-event group, settlement uncertainty.

### Rationale

The fastest path to reliability is a narrow vertical slice, not broad market coverage.

**Priority:** P1.

---

## High finding 11 — “Advisor sees its recent outcomes” can induce unstable feedback

### Problem

Showing each advisor only its last three outcomes is too small a sample and can promote recency bias, strategy chasing, and overreaction. It also makes output path-dependent and harder to reproduce.

The claim that this “measurably reduces overconfident repetition” is not supported within the plan.

### Recommended fix

Do not feed raw recent P&L into the trading prompt by default. Instead provide a slowly updated, deterministic calibration summary generated outside the LLM:

```json
{
  "evaluation_window": 200,
  "coverage": 0.34,
  "hit_rate": 0.54,
  "brier_score": 0.218,
  "calibration_by_bucket": {...},
  "regime_breakdown": {...},
  "status": "ACTIVE_WITH_SIZE_MULTIPLIER_0.5"
}
```

Use predeclared degradation rules. Do not let the agent change its own strategy merely because the last few calls lost.

### Rationale

Risk controls should respond to statistically meaningful degradation, not short streaks.

**Priority:** P1.

---

## High finding 12 — The orchestrator’s signal aggregation is too informal

### Problem

“Net conflicting directions” and multiply conviction by budget and volatility scaling are not enough. Convictions from different agents are not comparable, signals may duplicate the same information, and simple netting can discard useful disagreement information.

### Recommended fix

Convert every advisor output into a calibrated expected-return distribution or standardized score before aggregation:

```text
expected_excess_return
forecast_volatility
forecast_horizon
confidence_interval
probability_positive
model_reliability_weight
signal_correlation_cluster
```

Then use deterministic portfolio construction, for example:

1. Calibrate each advisor on prospective data.
2. Apply reliability weights with conservative shrinkage toward zero.
3. De-duplicate signals by feature lineage and empirical residual correlation.
4. Build covariance estimates with shrinkage and stress overrides.
5. Solve a constrained sizing problem or use equal-risk contribution with caps.
6. Apply turnover and cost penalties.
7. Round to venue lots and re-run all constraints after rounding.

The LLM orchestrator may explain or veto, but it should not perform numerical portfolio optimization.

### Rationale

Raw LLM conviction is not a common unit of expected return.

**Priority:** P1.

---

## Medium finding 13 — Protective orders and stop logic need market-specific treatment

### Problem

The plan treats native stops as if they guarantee protection. Stops can gap, slip, reject, trigger on different reference prices, or be invalidated by partial fills. Some venues implement separate orders rather than atomic brackets.

ATR-based minimum/maximum stop distance is also too simplistic across instruments and regimes.

### Recommended fix

- Document trigger source: mark, index, last, or oracle price.
- Verify protective coverage after each fill and partial fill.
- Run an independent “unprotected position” watchdog.
- Specify maximum time a position may remain unprotected, ideally zero where atomic brackets exist.
- Calculate risk from executable loss under stress, not stop distance alone.
- Include gap/slippage scenarios and venue outage scenarios.
- Maintain liquidation-distance constraints for derivatives.

### Rationale

A stop is an order instruction, not a guaranteed fill price.

**Priority:** P1.

---

## Medium finding 14 — Paper trading and testnet criteria are too weak

### Problem

Testnets often have unrealistic liquidity, fills, funding, and participant behavior. A paper engine that fills at mid-price will materially overstate performance. “Tracking error is explainable” is not a measurable promotion criterion.

### Recommended fix

Use shadow execution against live production order books without submitting orders. Simulate:

- Queue position or conservative taker execution.
- Spread crossing.
- Market impact as a function of displayed depth and participation rate.
- Cancel latency.
- Partial fills.
- Funding and fees from versioned schedules.
- Adverse selection after fills.

Predefine promotion thresholds such as:

```text
>= 99.9% loop completion
0 duplicate live intents
0 unprotected positions
100% ledger-to-venue reconciliation
p95 decision-to-submit latency within limit
modeled-vs-observed slippage error within threshold
minimum number of independent trades/events
no unresolved P0/P1 incidents
```

### Rationale

Operational promotion should be based on measurable reliability, not subjective review.

**Priority:** P1.

---

## Medium finding 15 — Security model needs additional controls

### Problem

Environment variables and container separation are not enough. Secrets can leak through process inspection, crash dumps, logs, backups, CI systems, or compromised dependencies. A Telegram kill command can itself become an attack surface.

### Recommended fix

- Use a secret manager or root-readable mounted secret rather than broad environment exposure where feasible.
- Restrict executor egress to venue endpoints and required infrastructure.
- Pin dependencies and generate an SBOM.
- Scan images and dependencies for vulnerabilities.
- Run containers as non-root with read-only filesystems where possible.
- Sign release artifacts and require reviewed deployment commits.
- Use allowlisted Telegram user/chat IDs, signed commands, replay protection, and a second confirmation for destructive commands where delay is acceptable.
- Maintain an out-of-band kill method independent of the VPS and Telegram.
- Redact secrets and personal/account information from all prompts and logs.
- Conduct key-revocation and compromised-host drills.

### Rationale

A system with order authority should be treated as a financial production service, even at small capital.

**Priority:** P1.

---

## Medium finding 16 — Observability and incident response are incomplete

### Problem

A heartbeat and Telegram notification do not provide enough operational visibility. There is no explicit service-level objective, metric taxonomy, alert severity, runbook, or incident record.

### Recommended fix

Track at minimum:

- Data freshness and gap counts.
- Agent latency, parse failures, abstention rate, and token cost.
- Decision counts by rejection reason.
- Order acknowledgments, rejects, cancels, partial fills, and unknown states.
- Position reconciliation differences.
- Protective-order coverage.
- Gross/net exposure, risk utilization, P&L, drawdown, and margin/liquidation distance.
- Slippage versus arrival, decision, and mid prices.
- Provider/API availability and rate-limit headroom.

Create P0/P1/P2 incident definitions and runbooks for:

- Duplicate order.
- Position without protection.
- Stale data trade.
- Venue/API outage.
- LLM outage.
- Database corruption.
- Account compromise.
- Reconciliation mismatch.

### Rationale

An unattended system must make failure visible before it becomes a financial loss.

**Priority:** P1.

---

## Medium finding 17 — Cost estimates are too confident and omit important costs

### Problem

The LLM estimate depends on provider, model, cache behavior, prompt size, retry rate, and cadence. The infrastructure estimate omits monitoring, backup, secret management, egress, data storage, exchange fees, funding, spread, slippage, tax software, and potentially professional compliance advice.

### Recommended fix

Provide a parameterized monthly budget instead of one range:

```text
LLM cost = calls/day × days × (input tokens × input rate + output tokens × output rate)
Data cost = subscriptions + historical backfill + licensing
Infra cost = compute + DB + object storage + monitoring + backups + egress
Trading friction = fees + spread + slippage + funding + borrow + failed-order cost
Governance = accounting/tax/compliance + security tooling
```

Log actual cost per advisor decision and cost per executed trade. Require the economic evaluation to include both trading friction and operating expenses.

### Rationale

A low-capital desk can be operationally successful yet economically unviable after fixed costs.

**Priority:** P2.

---

# Specific schema corrections

## Revised signal schema

```python
class ForecastSignal(BaseModel):
    schema_version: Literal["1.0"]
    signal_id: UUID
    advisor_id: str
    advisor_version: str
    generated_at: datetime
    data_cutoff_at: datetime
    expires_at: datetime
    venue: str
    instrument_id: str
    forecast_target: str
    horizon: timedelta
    action: Literal["long", "short", "flat"]
    raw_score: float
    calibrated_probability_positive: float | None
    expected_excess_return_bps: float | None
    uncertainty_bps: float | None
    invalidation_condition: str | None
    evidence_feature_ids: list[str]
    evidence_event_ids: list[str]
    thesis: constr(max_length=500)
    snapshot_id: UUID
    model_run_id: UUID
```

### Why this is better

- Defines exactly what is being forecast.
- Distinguishes raw model score from calibrated probability.
- Makes expiry and data cutoff explicit.
- Connects evidence to immutable data IDs.
- Supports audit and point-in-time evaluation.

## Revised risk verdict

```python
class RiskVerdict(BaseModel):
    schema_version: Literal["1.0"]
    verdict_id: UUID
    intent_id: UUID
    evaluated_at: datetime
    hard_check_version: str
    portfolio_snapshot_id: UUID
    verdict: Literal["approve", "reject", "approve_reduced"]
    approved_quantity: Decimal
    hard_checks: list[RiskCheckResult]
    stress_results: list[StressResult]
    llm_critic_result_id: UUID | None
    reason_codes: list[str]
```

### Why this is better

The original `checks_passed: dict[str, bool]` is not enough. Every check should preserve the observed value, threshold, unit, and code version.

---

# Recommended target architecture

```text
                    IMMUTABLE RAW DATA
                            │
                    NORMALIZATION/QA
                            │
             POINT-IN-TIME FEATURE & EVENT STORE
                            │
       ┌────────────────────┴────────────────────┐
       │                                         │
DETERMINISTIC MODELS                       LLM ADVISORS
       │                                   (research only)
       └────────────── FORECAST STORE ───────────┘
                            │
              CALIBRATION / RELIABILITY LAYER
                            │
              DETERMINISTIC PORTFOLIO ENGINE
                            │
                 HARD RISK & STRESS ENGINE
                            │
          OPTIONAL LLM VETO ON NEW EXPOSURE
                            │
                   DURABLE INTENT QUEUE
                            │
              DETERMINISTIC EXECUTION ENGINE
                            │
                         VENUE
                            │
              RECONCILIATION / INDEPENDENT
              POSITION & PROTECTION WATCHDOG
```

## Key authority rule

**No LLM is in the mandatory path for closing, reducing, cancelling, reconciling, or protecting a position.**

---

# Recommended narrower v1

## Scope

- One venue.
- One account or subaccount.
- BTC and ETH only; SOL added after stable operation.
- Spot or maximum 1× effective exposure initially.
- One deterministic baseline strategy.
- One LLM advisor in shadow mode.
- No prediction-market execution.
- No automatic prompt or parameter modification.

## Objective

The v1 objective should not be profit. It should be:

> Run continuously for at least 30 days with complete point-in-time records, no duplicate orders, no unprotected positions, exact venue reconciliation, reliable fail-safe behavior, and sufficient prospective forecasts to compare the LLM advisor against a deterministic baseline.

## Suggested development sequence

### Phase 0 — Governance and venue verification

- Verify legal/venue eligibility from Taiwan.
- Determine tax status with qualified advice.
- Define maximum capital at risk.
- Write threat model and incident policy.

### Phase 1 — Data and ledger

- Immutable raw capture.
- Point-in-time normalized store.
- Feature definitions and tests.
- PostgreSQL ledger or explicit single-writer SQLite architecture.
- Reconciliation model.

### Phase 2 — Deterministic shadow desk

- Baseline strategy.
- Portfolio and hard-risk engines.
- Shadow execution using live books.
- Observability and runbooks.

### Phase 3 — LLM research layer

- One advisor with a pre-registered research contract.
- Structured outputs and model-version logging.
- Prospective forecast scoring.
- No order influence.

### Phase 4 — Testnet and failure drills

- Idempotency and restart tests.
- Duplicate-message and timeout injection.
- Venue outage and stale-data drills.
- Key revocation and kill-switch tests.

### Phase 5 — Micro-live

- Very low limits.
- LLM still shadow-only or allowed only as a veto.
- Human restart after every hard halt.
- Promotion based on operational metrics.

### Phase 6 — Controlled LLM contribution

- Allow a validated advisor to adjust exposure only within a small preallocated risk sleeve.
- Compare against an identical control sleeve without LLM input.
- Expand only after prospective evidence.

---

# Recommended acceptance gates

## Gate A — Architecture complete

- [ ] LLM has no credential or write path.
- [ ] All message schemas are versioned.
- [ ] Durable intent queue and idempotency implemented.
- [ ] Point-in-time data lineage available.
- [ ] Deterministic risk engine has unit, property, and scenario tests.

## Gate B — Shadow-ready

- [ ] Venue adapter passes contract tests.
- [ ] Live order-book simulator includes conservative costs.
- [ ] Reconciliation runs continuously.
- [ ] Observability dashboard and alerts work.
- [ ] Incident runbooks are documented.

## Gate C — Micro-live-ready

- [ ] Venue and geographic eligibility verified.
- [ ] Tax/account classification reviewed.
- [ ] Thirty days of clean shadow operation.
- [ ] Zero duplicate intents and zero unprotected positions.
- [ ] Recovery and kill drills passed.
- [ ] Capital and limits independently reviewed.

## Gate D — LLM may influence size

- [ ] Prospective sample meets pre-registered minimum.
- [ ] Advisor beats deterministic and random-frequency baselines after costs.
- [ ] Calibration is acceptable.
- [ ] Result survives regime and block-bootstrap analysis.
- [ ] Advisor is confined to a capped experimental risk sleeve.

---

# Claims that should be revised in the original plan

| Original claim or implication | Evaluation | Recommended wording |
|---|---|---|
| An SSN likely creates US tax-reporting obligations | Incorrect/unsafe simplification | An SSN alone does not determine US tax residency or filing obligations. |
| Journaling gives historical replay “for free” | Overstated | Journaling enables mechanical replay; LLM historical replay remains contaminated and non-reproducible. |
| Hyperliquid is the strongest single v1 choice | Plausible but insufficiently justified | Select a venue only after formal eligibility, API, custody, outage, fee, liquidity, and operational review. |
| A MetaMask wallet opens Polymarket access | Incomplete | Wallet compatibility does not establish geographic or account eligibility. |
| LLM risk approval should be required | Unsafe dependency | LLM critique may reject new exposure but must never block risk reduction. |
| SQLite is enough until multiple writers exist | In tension with three-container architecture | Specify a single-writer SQLite design or use PostgreSQL from the start. |
| Two to four weeks per stage | Arbitrary | Use sample-size and operational-event gates, with minimum calendar durations only as secondary safeguards. |
| Native stops protect the account | Incomplete | Verify trigger, atomicity, fill, partial-fill, and gap behavior; monitor protection independently. |
| Advisor self-conditioning on three outcomes is beneficial | Unsupported and likely unstable | Use long-window, deterministic calibration summaries and predeclared degradation rules. |
| $20–80 monthly LLM cost and $35–175 total burn | Rough and potentially stale | Use a parameterized cost model and record actual cost per decision and trade. |

---

# Final recommendation

Proceed with the project, but **reframe it as an auditable deterministic quant platform with experimental LLM forecasters**. The current plan is strong enough to guide a prototype, but not precise enough to authorize live capital.

The most important fixes are:

1. Build a formal advisor-validation and calibration framework.
2. Treat prospective shadow forecasts—not historical LLM replay—as the primary evidence.
3. Remove the SSN-based tax inference and formally verify venue eligibility.
4. Make deterministic risk code the final authority; use LLM risk review only as an optional veto on new exposure.
5. Implement a durable, idempotent execution state machine and independent reconciliation.
6. Reduce initial exposure and drawdown limits substantially.
7. Narrow v1 to one venue, one instrument class, and one or two highly liquid assets.
8. Add point-in-time data lineage, model-version governance, security controls, observability, and measurable promotion gates.

With these corrections, the design can become a credible research and execution platform. Without them, it risks being operationally sophisticated while lacking validated alpha and production-grade failure handling.

---

## Verification notes

The following current external points were checked against official sources during this review:

- Hyperliquid publishes official API documentation and an official Python SDK; its API-wallet/nonces documentation should be used to verify the exact permissions and revocation model rather than relying on a summary statement.
- Polymarket publishes an official geoblock endpoint and states that orders from blocked regions are rejected.
- Kalshi states that trading outside the United States is subject to country restrictions and its Member Agreement.
- IRS guidance determines resident/nonresident-alien status using the green-card and substantial-presence tests; an SSN alone is not the determining criterion.

These items should be rechecked immediately before implementation because venue rules, APIs, fees, and geographic restrictions can change.
