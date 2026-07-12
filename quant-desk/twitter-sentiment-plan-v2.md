# Coding-Agent Prompt: Add Twitter/X Sentiment to the Quant Desk (v2)

**Revision note.** v2 incorporates an external review of the v1 prompt. Key changes: provider-neutral time-window fetching replaces the cursor contract (TwitterAPI.io advanced search must not be cursor-paginated); every query window carries explicit completeness status, and incomplete windows null the volume features; **equal/author-equal weighting replaces engagement weighting** in v1 (engagement accumulates over time and leaks the future); aggregation is author-first; the scorer's three class probabilities are persisted with `sentiment = p_positive − p_negative` and fully pinned versions; the seasonal baseline is 28-day median/MAD; shadow isolation becomes **structural** via a `ResearchForecast` type the portfolio engine cannot consume; and the research contract moves to overlap-aware statistics (120 days / 500 decisions / 100 non-overlapping outcomes). Guiding principle: *a reproducible pipeline is not yet a valid measurement — prove the measurement first, evaluate the advisor second.*

Paste everything below the line into the coding agent's session, adjusting bracketed values first.

---

## Task

Add Twitter/X sentiment as a new data source, a deterministic feature family, and one new **research-only** LLM advisor. Follow all framework conventions: immutable raw capture, point-in-time lineage, deterministic features, versioned schemas, untrusted-data handling. Do not modify the portfolio engine, risk engine, or executor. The sentiment advisor's isolation must be structural (see §8), not merely configured.

## 1. Provider and source interface

Create `pipeline/sources/twitter.py` for TwitterAPI.io, behind a provider-neutral interface. **Do not use cursor pagination for advanced search** — current provider documentation says bounded requests, not cursors. Use a continuation contract:

```python
class SourceContinuation(BaseModel):
    mode: Literal["cursor", "time_split", "none"]
    cursor: str | None = None
    next_start_time: datetime | None = None
    next_end_time: datetime | None = None

class TweetSource(Protocol):
    def fetch_window(self, query: str, start_time: datetime,
                     end_time: datetime,
                     continuation: SourceContinuation | None = None) -> FetchPage: ...
```

The advanced-search adapter **recursively splits event-time windows** when a request reaches the provider result limit; timeline endpoints (curated authors) may use cursors where the provider supports them. Acceptance test: a fixture 15-minute interval whose volume exceeds one page is retrieved completely, with no duplicated and no omitted boundary timestamps.

API key via the standard mounted-secret mechanism; it must never appear in logs, prompts, fixtures, exceptions, or ledger rows. Startup preflight validates the key and records the provider's advertised rate limits and pricing tier.

**Budget accounting is reservation-based:** before each call, estimate and atomically reserve the maximum charge against the remaining budget; execute; reconcile estimated vs. reported usage; release the unused reservation — so concurrent workers and retries cannot pierce a hard cap. Track per endpoint: `request_count, records_returned, credits_estimated, credits_reported, cost_estimated_usd, cost_reported_usd`. Config caps: `max_requests_per_day`, `max_tweets_per_cycle`, `max_monthly_usd`. Cap exhaustion stops fetching with a P2 alert and marks affected windows (never fails silently).

## 2. What to search

Cadence: every 15 minutes (`cadence_minutes` in `config/sentiment.yaml`).

```yaml
assets:
  BTC:
    cashtag_query: '($BTC OR bitcoin) lang:en -is:retweet'
    min_author_followers: 150
  ETH:
    cashtag_query: '($ETH OR ethereum) lang:en -is:retweet'
    min_author_followers: 150
curated_authors:
  # ~30-60 handles from [FILL IN — my handle list]; hot-reloadable config;
  # each handle carries a category: analyst | fund | exchange_status |
  # protocol_status | macro_news | regulator (per-category features are
  # deferred until sample size permits, but the label is captured now).
global_filters:
  languages: [en]
  exclude_retweets: true
  exclude_replies: false
  dedupe_lookback_hours: 48
```

Tag every tweet `source_class: broad | curated`. If the provider lacks a server-side operator, enforce it client-side in the normalizer and document that in the module docstring. There is no per-window tweet *cap* on correctness grounds — window splitting fetches everything or the window is marked incomplete; the budget layer is what bounds spend.

## 3. Query-window completeness

Every scheduled fetch writes a query-window record:

```yaml
query_window_id: / asset: / source_class:
requested_start: / requested_end:
coverage_start: / coverage_end:
fetched_count:
complete: true|false
truncation_reason: null | provider_limit | budget_cap | rate_limit |
                   timeout | schema_error | partial_window
fetch_attempts: / provider_latency_ms: / quality_flags: []
```

When `complete = false`, all **absolute** volume, author-count, and spike features for that window are null. A missing record is never interpreted as low social activity — that is the review's most important correctness rule.

## 4. Storage, revisions, governance

Standard lineage fields as elsewhere in the pipeline (`event_time` = tweet `created_at`, `published_time`, `ingested_time`, `available_to_strategy_time`, `source_id`, `source_revision`, `raw_payload_hash`, `normalizer_version`, `quality_flags`; all UTC). Additions from the review:

**Four text representations, never one destructively normalized string:** `raw_text_restricted` (landing zone, retention-governed), `normalized_text_for_dedup` (lowercase, handles/URLs/whitespace stripped), `model_input_text` (versioned transform: URLs→stable token, mentions→stable token, preserve cashtags/hashtags/emojis/negation/punctuation, Unicode-normalized, truncation recorded), `display_text_for_llm` (links/markdown/HTML/control characters stripped, truncated by **tokenizer tokens**, wrapped in the untrusted envelope).

**Edits and deletions:** `first_seen_at, last_seen_at, content_revision, post_edit_history_ids, is_deleted, deleted_observed_at`. Never overwrite content in place; features use the content revision available at `snapshot.data_cutoff_at`.

**Engagement is append-only observations, not columns on the tweet:** `(tweet_id, observation_time, tweet_age_minutes, likes, reposts, replies, views)`. v1 features do not use engagement at all (§6); the store exists so a later version can use engagement observed at a pre-registered lag (e.g., 60 minutes post-publication) without look-ahead. Present-day engagement must never be joined onto historical snapshots.

**Terms-review gate, fail-closed.** Replace the bare date with:

```yaml
terms_review:
  reviewed_at: / reviewed_by:
  provider_terms_url: / platform_terms_url:
  permitted_storage: / raw_retention_days:
  deletion_sync_required: / redistribution_allowed:
  commercial_use_allowed: / model_training_allowed:
  notes:
```

The source refuses to enable while required items are unresolved. Raw-payload retention and derived-feature retention are independently configurable; if `deletion_sync_required`, deletion observations propagate to the raw zone per the configured policy.

## 5. Dedup, spam, author independence

**Content vs. propagation, kept separate.** Near-duplicates (by `normalized_text_for_dedup` within `dedupe_lookback_hours`) collapse to one canonical item for sentiment purposes, but retain `canonical_text_hash, canonical_tweet_id, duplicate_count, duplicate_unique_authors, duplicate_first_seen_at, duplicate_last_seen_at`. Sentiment means use canonical content only; propagation counts feed only campaign/diffusion features.

**Spam flags, central exclusion policy.** The normalizer flags (never deletes): followers < threshold, account age < 30 days, > `max_tweets_per_author_per_window: 5`. One versioned exclusion policy module decides what feature code excludes — no scattered hard-coded filters.

**Author-first aggregation.** Five tweets from one account are not five observations:

```text
author_sentiment    = robust_mean(tweet sentiments for one author in window)
population_sentiment = robust_mean(author_sentiment across authors)
```

## 6. Sentiment scoring — deterministic, fully pinned

Local model, default `cardiffnlp/twitter-roberta-base-sentiment-latest`, CPU. Persist per tweet: `p_negative, p_neutral, p_positive, sentiment, scorer_confidence, scorer_version, preprocessor_version`, with `sentiment = p_positive − p_negative` and `scorer_confidence = max(p_neg, p_neu, p_pos)`. Pin: model repo + commit SHA, tokenizer commit SHA, transformers version, torch version, max sequence length, truncation policy, preprocessing version. Scoring failures → `quality_flags: [unscored]`, excluded from aggregates.

**Asset-target handling.** A generic sentiment model measures linguistic valence, not per-token direction. Record `asset_mentions` and `target_ambiguity`; tweets mentioning multiple configured assets ("ETH strong, BTC weak") route to a `multi_asset` population and are **excluded from asset-specific means**; asset aggregates use unambiguous mentions only.

**Scorer validation (human-in-the-loop task, before the advisor's forecasts count).** Build a stratified labeling set (~300 tweets) covering sarcasm, slang, emojis, price targets, "good tech / bad token," multi-asset comparisons, and incident notices; I will hand-label it. Produce a confusion matrix and calibration report; TweetEval performance is not assumed to transfer to crypto-financial language. Ship the harness and the sampling script; block Gate D (not development) on the report.

## 7. Feature family

`features/sentiment.py`, versioned and unit-tested, computed only from rows with `available_to_strategy_time <= data_cutoff_at`, current content revision, and no exclusion flags. **All means are author-equal-weighted (v1 has no engagement weighting).**

Volume/spike (null when any contributing window is incomplete): `tweet_volume_1h`, `tweet_volume_24h`, `volume_zscore` and `authors_zscore` against a **versioned robust seasonal baseline** — UTC 15-minute buckets, prior 28 calendar days, minimum 14 valid observations per bucket, center = median, scale = 1.4826 × MAD, fallback = exponentially weighted intraday baseline; persist `baseline_version/window/n/center/scale` alongside each z-score. `spike_flag`: volume_z > 3 AND authors_z > 2.

Sentiment: `sent_mean_1h`, `sent_mean_24h` (per population: broad, curated), `sent_dispersion_1h`, and `sent_delta_6h` defined exactly as `mean([t−1h, t]) − mean([t−7h, t−6h])`, null unless both windows meet minimum tweet and author counts.

Information-quality: `positive_fraction, negative_fraction, neutral_probability_mean, low_confidence_fraction, sentiment_entropy_mean, author_concentration_1h, effective_authors_1h, new_author_fraction_1h, median_author_age_days, top5_author_weight_share_1h`, plus campaign features from the propagation view.

`data_health` = `min(time_coverage, fetch_success_rate, completeness, scoring_success_rate, schema_valid_rate)`, all components persisted for diagnosis; below 0.8, sentiment features emit null with a quality flag.

Snapshot builder: add the compact `sentiment` block plus up to 5 sample tweets under strict selection rules — max one per author, fixed quotas across broad/curated, near-duplicate campaigns excluded, stratified rather than top-engagement (top-engagement selection over-represents sensational and manipulated posts), each as `display_text_for_llm` in the untrusted envelope. The advisor must run correctly with zero samples, and the harness supports a prospective features-only vs. features-plus-samples comparison (config flag per decision, logged in provenance).

## 8. Structural shadow isolation

The sentiment advisor emits a distinct type, not a zero-weighted `ForecastSignal`:

```python
class ResearchForecast(BaseModel):
    research_only: Literal[True] = True
    schema_version: Literal["1.0"]
    forecast_id: UUID
    advisor_id: str
    advisor_version: str
    generated_at: datetime
    data_cutoff_at: datetime
    instrument_id: str
    horizon: timedelta
    direction: Literal["long", "short", "flat"]
    abstain: bool
    probability_positive: float | None      # calibrated head, may be None early
    expected_excess_return_bps: float | None
    confidence: float
    evidence_feature_ids: list[str]
    snapshot_id: UUID
    model_run_id: UUID
```

The portfolio engine must not import `ResearchForecast` or accept it through any interface. Enforcement tests: (1) type-level — no code path can convert a `ResearchForecast` into an `OrderIntent` or `ForecastSignal`; (2) dependency-level — an import-graph test asserts no path from `pipeline/sources/twitter.py` or the sentiment advisor to executor code; (3) provenance — persisted reliability weight is zero and no `OrderIntent.source` chain can cite the sentiment advisor. Promotion to `TradableSignal`-emitting status is a future, human-made code change, not a config flip.

## 9. Research contract and benchmark

```yaml
advisor_id: crypto_sentiment_llm_v1
forecast_target: next_24h_excess_log_return
universe: [BTC, ETH]
decision_times_utc: [00:00, 04:00, 08:00, 12:00, 16:00, 20:00]
horizon: 24h
benchmark: sent_zscore_contrarian_baseline_v1   # frozen before data collection
abstention_allowed: true                        # coverage always reported
primary_metric:
  name: spearman_ic
  forecast: expected_excess_return_bps
  outcome: next_24h_excess_log_return
  uncertainty: stationary_block_bootstrap
secondary_metrics: [brier_score, log_loss, calibration_slope,
                    calibration_intercept, balanced_accuracy,
                    net_pnl_after_costs, turnover, coverage]
minimum_calendar_days: 120
minimum_decisions: 500
minimum_non_overlapping_outcomes: 100
overlap_adjustment: stationary_block_bootstrap_and_HAC
promotion_rule: pre_registered_thresholds_only
interim_analysis_at: [200_decisions]            # look, don't promote
```

Four-hourly decisions on a 24h horizon overlap sixfold, so all uncertainty is overlap-adjusted (HAC/Newey–West and stationary block bootstrap), effective sample size is reported, and non-overlapping daily cohorts run as a robustness check. Forecasts are scored only after their full outcome window closes. Results are reported by asset, source class, volatility regime, and calendar period.

**Frozen benchmark.** Implement `models/sent_zscore_contrarian_baseline.py` with exact, pre-registered thresholds, z-score source, health requirements, abstention conditions, and a probability output — written down before prospective collection starts, never tuned on the final evaluation period. It emits `ResearchForecast`s too.

**Incremental-value design.** The evaluation harness must support five comparisons: (1) deterministic sentiment benchmark, (2) sentiment LLM, (3) the existing non-sentiment model, (4) existing model + deterministic sentiment features, (5) existing model + LLM sentiment forecast. The question that gates promotion is whether sentiment adds information beyond trend, volatility, and funding — not whether it predicts returns in isolation.

Advisor prompt: keep the v1 skeleton (untrusted samples, aggregate-over-anecdote, contrarian-at-extremes and campaign cautions, flat on degraded data) and add: output `probability_positive` and optional `expected_excess_return_bps` per the schema; `abstain: true` is always available and never penalized in-prompt (coverage is measured outside).

## 10. Tests

Beyond the framework's standard suite: window splitting (no boundary gaps/duplicates on high-volume fixtures); completeness (unrecoverable truncation nulls absolute volume features); mutable engagement (later engagement observations never alter a historical snapshot); revision/deletion (edits never replace historical content versions); author independence (five posts by one author ≠ five authors); multi-asset ambiguity (co-mentions follow the documented exclusion policy); deterministic inference (pinned fixture texts reproduce probabilities within tolerance); schema drift (unknown raw fields retained, missing required fields quarantined); budget concurrency (parallel workers cannot exceed caps); shadow isolation (all three §8 tests); outcome timing (no scoring before the 24h window closes); overlap-aware evaluation (the registered uncertainty estimator is actually invoked).

Injection testing is split into three layers, because exact generative output equality is a brittle CI invariant: (1) deterministic prompt-construction tests — hostile fixture text is verifiably inside the untrusted envelope, escaped, token-truncated; (2) mock-model contract tests — parser and schema validation reject out-of-schema responses; (3) an offline temperature-zero red-team suite against the real model, run pre-release rather than in CI, with a rubric (no instruction-following, no schema violation) instead of string equality.

## 11. Acceptance criteria

Engineering: 48h on the VPS without unhandled exceptions; every query window has explicit completeness status; no secret in logs/prompts/fixtures/exceptions/ledger; adapter passes window-splitting, retry, and budget-concurrency tests; raw→normalized→scored→feature records share immutable lineage; historical snapshots provably unchanged by later engagement, edits, or deletions; model and tokenizer pinned to immutable revisions; the advisor emits only `ResearchForecast`; static and runtime tests prove no sentiment output can influence orders; budget reconciliation basis and interval defined, estimated-vs-reported within the defined tolerance.

Data quality: ≥95% of scheduled windows carry valid completeness metadata; ≥90% of otherwise-valid posts score successfully; nulling fires whenever health < threshold; duplicate propagation and author concentration are on the dashboard; broad and curated populations separately inspectable.

Research (before promotion is even considered): ≥120 calendar days, ≥500 decisions, ≥100 effectively non-overlapping outcomes; overlap-adjusted uncertainty; per-asset/source-class/regime/period breakdowns; benchmark frozen before evaluation; incremental-value comparisons run; coverage and abstention reported; no threshold selected on the final prospective period; scorer validation report (§6) complete.

Out of scope: any change to portfolio/risk/execution code; any promotion logic; engagement weighting (design later against the observation store); per-category curated features (labels captured, features deferred); historical backfill beyond 7 days.
