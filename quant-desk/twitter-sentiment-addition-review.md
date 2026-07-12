# Review of the Proposed Twitter/X Sentiment Addition

## Executive assessment

**Recommendation: approve the concept, but revise the implementation prompt before coding.**

The proposal is unusually disciplined for a social-sentiment trading component. It separates immutable ingestion, deterministic scoring, point-in-time features, a shadow-only LLM advisor, and a deterministic benchmark. It also correctly prohibits changes to portfolio, risk, and execution logic.

The main weaknesses are not in the high-level architecture. They are in provider compatibility, measurement validity, statistical evaluation, data governance, and enforcement of shadow isolation.

## Priority findings

| Priority | Finding | Recommended fix | Rationale |
|---|---|---|---|
| Critical | The required cursor-pagination interface conflicts with TwitterAPI.io advanced-search documentation. | Use a provider-neutral time-window continuation contract. Implement deterministic `since_time`/`until_time` splitting for advanced search; retain cursors only for endpoints that support them. | The current provider documentation says not to paginate advanced search and instead keep each bounded request below the result limit. |
| Critical | A 200-tweet cap can silently turn high-volume intervals into incomplete samples. | Record `complete`, `truncation_reason`, `fetched_count`, and coverage bounds for every query window. Null absolute volume features for incomplete windows. | A capped sample cannot support unbiased volume, author-count, or spike features. |
| Critical | Engagement-weighted sentiment is not point-in-time comparable. | Prefer equal weights in v1, or collect engagement at a fixed lag such as 60 minutes after publication. Store append-only engagement observations. | Likes and reposts accumulate over time. Unequal tweet age introduces look-ahead and survivorship bias. |
| Critical | “Reliability weight zero” is only a configuration control. | Emit a separate `ResearchForecast` type that is not accepted by orchestrator or executor interfaces. Add dependency and runtime tests proving no conversion to `OrderIntent`. | Shadow isolation should be structural, not dependent on one numeric field. |
| High | Generic document sentiment is not necessarily asset-specific sentiment. | Exclude or separately classify tweets mentioning multiple configured assets; add `asset_mentions` and `target_ambiguity`. | “ETH is strong but BTC is weak” cannot be safely assigned one scalar to both assets. |
| High | Tweet-level aggregation overstates independence. | Aggregate per author first, then across authors. Add author-concentration and effective-author metrics. | Five tweets from one account are not five independent observations. |
| High | The scalar mapping from the three-class model is undefined. | Persist `p_negative`, `p_neutral`, and `p_positive`; define `sentiment = p_positive - p_negative`; pin model and tokenizer commit SHAs. | Reproducibility requires exact mapping, immutable revisions, and preprocessing versions. |
| High | The seven-day same-hour z-score baseline is too underspecified and small. | Use a versioned robust seasonal baseline, preferably 28 days of UTC 15-minute buckets, median center, and MAD scale. | Seven observations per time bucket produce unstable z-scores. |
| High | The 24-hour forecasts overlap heavily. | Use HAC/Newey-West or block-bootstrap uncertainty, report effective sample size, and require calendar-duration and non-overlapping outcome minima. | Six four-hourly decisions share most of the same 24-hour return path. |
| High | “Net information coefficient” is undefined. | Pre-register the exact forecast variable, outcome, benchmark, correlation statistic, and confidence-interval method. | Evaluation cannot be reproduced from a metric name alone. |
| Medium | Copypasta collapse loses the distinction between content and propagation. | Maintain a canonical-content view and a propagation view with duplicate count and unique duplicating authors. | Repetition should not multiply sentiment evidence, but the campaign itself may be informative. |
| Medium | Data licensing and deletion obligations are unresolved. | Add a formal terms/retention record and support configurable raw-payload retention, edit revisions, and deletion observations. | “Immutable forever” may conflict with provider or platform obligations. |
| Medium | The prompt-injection integration test is potentially flaky. | Separate deterministic prompt-construction tests from an offline, temperature-zero model red-team suite. | Exact LLM output equality is not a reliable CI invariant. |

---

## 1. Provider and ingestion design

### 1.1 Revise the source interface

Replace the mandatory cursor contract with:

```python
class TweetSource(Protocol):
    def fetch_window(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        continuation: SourceContinuation | None = None,
    ) -> FetchPage:
        ...
```

```python
class SourceContinuation(BaseModel):
    mode: Literal["cursor", "time_split", "none"]
    cursor: str | None = None
    next_start_time: datetime | None = None
    next_end_time: datetime | None = None
```

The TwitterAPI.io advanced-search adapter should recursively split event-time windows when a request reaches the provider result limit. Timeline endpoints may use cursors if supported.

Add an acceptance test where a 15-minute fixture interval exceeds one page. Assert complete retrieval with no duplicate or omitted boundary timestamps.

### 1.2 Add query-window completeness

Every scheduled fetch should write a query-window record:

```yaml
query_window_id:
asset:
source_class:
requested_start:
requested_end:
coverage_start:
coverage_end:
fetched_count:
complete: true
truncation_reason: null
fetch_attempts:
provider_latency_ms:
quality_flags: []
```

Possible truncation reasons should include:

```text
provider_limit
budget_cap
rate_limit
timeout
schema_error
partial_window
```

When `complete=false`, absolute tweet-volume, author-count, and spike features should be null. Do not interpret missing records as low social activity.

### 1.3 Use reservation-based budget accounting

Before each external call:

1. Estimate and reserve the maximum charge.
2. Atomically reduce the remaining budget.
3. Execute the request.
4. Reconcile estimated and actual usage.
5. Release unused reservation.

This prevents concurrent workers or retries from exceeding a supposedly hard limit.

Track cost by endpoint and billable unit:

```text
request_count
records_returned
credits_estimated
credits_reported
cost_estimated_usd
cost_reported_usd
```

The “within 5%” acceptance criterion should only apply after the billing basis and comparison interval are defined.

---

## 2. Storage, revisions, and governance

### 2.1 Keep separate text representations

Do not reuse one destructively normalized string for every purpose. Store:

```text
raw_text_restricted
normalized_text_for_dedup
model_input_text
display_text_for_llm
```

The model-input transformation should be versioned and should generally:

- replace URLs with a stable token;
- replace mentions with a stable token;
- preserve cashtags, hashtags, emojis, negation, and punctuation;
- normalize Unicode;
- record truncation.

### 2.2 Support edits and deletions

Add:

```text
first_seen_at
last_seen_at
content_revision
post_edit_history_ids
is_deleted
deleted_observed_at
```

Never overwrite historical content in place. Feature computation must use the content revision available at `snapshot.data_cutoff_at`.

### 2.3 Expand the terms-review gate

A date alone is inadequate. Use:

```yaml
terms_review:
  reviewed_at:
  reviewed_by:
  provider_terms_url:
  platform_terms_url:
  permitted_storage:
  raw_retention_days:
  deletion_sync_required:
  redistribution_allowed:
  commercial_use_allowed:
  model_training_allowed:
  notes:
```

The source should fail closed when required items are unresolved. Raw retention and derived-feature retention should be independently configurable.

---

## 3. Deduplication, spam, and author independence

### 3.1 Preserve content and propagation separately

For near duplicates, retain:

```text
canonical_text_hash
canonical_tweet_id
duplicate_count
duplicate_unique_authors
duplicate_first_seen_at
duplicate_last_seen_at
```

Use one canonical content item in sentiment means. Use propagation counts only in campaign and diffusion features.

### 3.2 Aggregate by author first

Recommended calculation:

```text
author_sentiment =
    robust_mean(tweet sentiment for one author in the window)

population_sentiment =
    robust_mean(author sentiment across authors)
```

Add:

```text
author_concentration_1h
effective_authors_1h
top5_author_weight_share_1h
new_author_fraction_1h
```

The existing maximum of five tweets per author is useful as a guardrail, but it does not make those tweets independent observations.

### 3.3 Keep flagged rows, but define exclusion centrally

The normalizer should retain spam-like rows and assign explicit flags. Feature code should use one versioned exclusion policy rather than hard-coded filters scattered across modules.

---

## 4. Sentiment scoring

### 4.1 Define the model outputs

Persist:

```text
p_negative
p_neutral
p_positive
sentiment
scorer_confidence
scorer_version
preprocessor_version
```

Define:

```python
sentiment = p_positive - p_negative
scorer_confidence = max(p_negative, p_neutral, p_positive)
```

Pin:

- model repository;
- model commit SHA;
- tokenizer commit SHA;
- Transformers version;
- Torch version;
- maximum sequence length;
- truncation policy;
- preprocessing version.

### 4.2 Add asset-target handling

For v1:

- score asset-specific aggregates only when an asset mention is unambiguous;
- route BTC-and-ETH co-mentions to a `multi_asset` population or exclude them from asset-specific means;
- retain `asset_mentions`;
- flag `target_ambiguity`.

A generic sentiment model measures linguistic valence, not necessarily directional sentiment toward each token mentioned.

### 4.3 Validate on a crypto-specific labeled sample

Before trusting the scorer, manually label a stratified sample containing:

- sarcasm;
- slang;
- emojis;
- price targets;
- “good technology, bad token” statements;
- multi-asset comparisons;
- exchange or protocol incident notices.

Report confusion matrices and calibration. Do not assume performance on TweetEval transfers directly to crypto-financial language.

---

## 5. Engagement weighting

### Preferred v1

Use equal-weighted or author-equal-weighted sentiment.

### Later version

Store append-only engagement observations:

```text
tweet_id
observation_time
tweet_age_minutes
likes
reposts
replies
views
```

Use the observation closest to a pre-registered lag such as 60 minutes after publication.

Do not use present-day engagement for historical replay. That would leak future virality into past features.

---

## 6. Feature definitions

### 6.1 Use a robust, explicit seasonal baseline

Suggested definition:

```text
bucket: UTC 15-minute slot
lookback: prior 28 calendar days
minimum valid observations: 14
center: median
scale: 1.4826 × MAD
fallback: exponentially weighted intraday baseline
```

Persist:

```text
baseline_version
baseline_window_start
baseline_window_end
baseline_n
baseline_center
baseline_scale
```

### 6.2 Define `sent_delta_6h` exactly

Use:

```text
mean([t-1h, t]) - mean([t-7h, t-6h])
```

Require minimum valid tweet and author counts in both windows; otherwise return null.

### 6.3 Strengthen `data_health`

Calculate components:

```text
time_coverage
fetch_success_rate
completeness
latency_freshness
scoring_success_rate
schema_valid_rate
```

A conservative composite can be:

```python
data_health = min(
    time_coverage,
    fetch_success_rate,
    completeness,
    scoring_success_rate,
    schema_valid_rate,
)
```

Retain all components for diagnosis.

### 6.4 Add information-quality features

Recommended additions:

```text
positive_fraction
negative_fraction
neutral_probability_mean
low_confidence_fraction
sentiment_entropy_mean
author_concentration_1h
effective_authors_1h
new_author_fraction_1h
median_author_age_days
top5_author_weight_share_1h
```

Separate curated accounts into analysts, funds, exchanges, protocol-status accounts, macro news, and regulators when sample size permits.

---

## 7. Advisor and prompt design

The prompt correctly treats sample tweets as untrusted data. Strengthen it further:

- maximum one sample per author;
- fixed quotas for broad and curated populations;
- exclude near-duplicate campaigns;
- strip links, markdown, HTML, and control characters;
- truncate by tokenizer tokens rather than characters;
- permit the advisor to run with zero samples;
- run a prospective feature-only versus feature-plus-samples comparison.

Top-engagement-only selection is likely to overrepresent sensational or manipulated posts.

---

## 8. Structural shadow isolation

Create distinct schemas:

```python
class ResearchForecast(BaseModel):
    research_only: Literal[True] = True
    advisor_id: str
    probability_positive: float | None
    expected_excess_return_bps: float | None
    abstain: bool
    evidence_feature_ids: list[str]
```

```python
class TradableSignal(BaseModel):
    approved_for_portfolio: Literal[True] = True
    ...
```

The portfolio engine must not import or accept `ResearchForecast`.

Add tests that:

- no sentiment advisor output can instantiate `OrderIntent`;
- no dependency path exists from the sentiment module to executor code;
- the reliability weight is zero in persisted provenance;
- source signals cited by any order cannot include the sentiment advisor.

---

## 9. Research contract

### 9.1 Define the forecast output

Use a probability rather than only a sign:

```yaml
probability_positive: 0.0-1.0
expected_excess_return_bps: optional
direction: long | short | flat
abstain: true | false
confidence: 0.0-1.0
```

Always report prediction coverage. Otherwise a model can appear strong by abstaining on almost every difficult case.

### 9.2 Define the primary metric precisely

Replace `net_information_coefficient` with an explicit contract, for example:

```yaml
primary_metric:
  name: spearman_ic
  forecast: expected_excess_return_bps
  outcome: next_24h_excess_log_return
  uncertainty: stationary_block_bootstrap
```

For probability forecasts, make Brier score or log loss primary and report calibration slope, calibration intercept, balanced accuracy, and decision utility after costs.

### 9.3 Correct for overlapping outcomes

Four-hour decisions with a 24-hour horizon overlap sixfold. Use:

- HAC/Newey-West standard errors;
- stationary or moving-block bootstrap;
- non-overlapping daily cohorts as a robustness check;
- effective sample size;
- asset- and time-clustered uncertainty where appropriate.

Revise the earliest promotion review to:

```yaml
minimum_calendar_days: 120
minimum_decisions: 500
minimum_non_overlapping_outcomes: 100
```

Two hundred decisions can be an interim analysis point, not a standalone promotion threshold.

### 9.4 Freeze the benchmark

The deterministic contrarian rule must specify exact thresholds, z-score source, health requirements, abstention conditions, and output probability before prospective data collection begins.

Do not optimize benchmark thresholds on the same period used for final evaluation.

### 9.5 Test incremental value

Compare:

1. sentiment deterministic benchmark;
2. sentiment LLM;
3. existing non-sentiment model;
4. existing model plus deterministic sentiment features;
5. existing model plus LLM sentiment forecast.

The relevant question is whether sentiment adds information beyond trend, volatility, funding, and existing news—not merely whether it predicts returns in isolation.

---

## 10. Testing recommendations

Add these tests to the proposed suite:

1. **Window splitting:** high-volume fixtures are split without boundary gaps or duplicates.
2. **Completeness:** an unrecoverably truncated interval nulls absolute volume features.
3. **Mutable engagement:** later engagement observations never alter a historical snapshot.
4. **Revision/deletion:** edits do not replace the historical content version.
5. **Author independence:** five posts by one author do not become five independent authors.
6. **Multi-asset ambiguity:** BTC/ETH co-mentions follow the documented exclusion policy.
7. **Deterministic inference:** pinned fixture texts produce probabilities within tolerance.
8. **Schema drift:** unknown raw fields are retained; missing required fields are quarantined.
9. **Budget concurrency:** concurrent workers cannot exceed caps.
10. **Shadow isolation:** `ResearchForecast` cannot create or be converted to `OrderIntent`.
11. **Outcome timing:** forecasts are not scored until the full 24-hour outcome window closes.
12. **Overlap-aware evaluation:** the registered uncertainty estimator is used.

For injection testing, separate:

- deterministic prompt-construction and untrusted-envelope tests;
- mock-model contract tests;
- an offline real-model red-team suite at temperature zero.

Do not make exact generative output equality a brittle CI requirement.

---

## 11. Revised acceptance criteria

### Engineering

- 48 hours without unhandled exceptions.
- Every query window has explicit completeness status.
- No secret appears in logs, prompts, fixtures, exceptions, or ledger rows.
- Provider adapter passes window-splitting, retry, and budget-concurrency tests.
- Raw, normalized, scored, and feature records share immutable lineage.
- Historical snapshots are unchanged by later engagement, edits, or deletions.
- Model and tokenizer use immutable revisions.
- The advisor emits only `ResearchForecast`.
- Static and runtime tests prove no sentiment output can influence orders.

### Data quality

- At least 95% of scheduled windows have valid completeness metadata.
- At least 90% of otherwise valid posts score successfully.
- Feature nulling occurs whenever health is below threshold.
- Duplicate propagation and author concentration are monitored.
- Broad and curated populations remain separately inspectable.

### Research

Before promotion is even considered:

- at least 120 calendar days;
- at least 500 decisions;
- at least 100 effectively non-overlapping outcomes;
- overlap-adjusted uncertainty;
- results by asset, source class, volatility regime, and calendar period;
- a benchmark frozen before evaluation;
- comparison with feature-only and existing-desk baselines;
- explicit forecast coverage and abstention statistics;
- no thresholds selected on the final prospective evaluation period.

---

## 12. Recommended wording changes

### Replace

> fetch by query + time window, cursor pagination

### With

> Fetch by query and bounded event-time window through a provider-neutral continuation object. The TwitterAPI.io advanced-search adapter must use deterministic time-window splitting when required by provider documentation. Cursor pagination may be used only by endpoints that support it.

### Replace

> engagement-weighted

### With

> Equal-weighted or author-equal-weighted in v1. Engagement weighting is permitted only when engagement is observed at a pre-registered tweet age and remains point-in-time correct.

### Add

> Aggregate primarily at the author level. Preserve duplicate propagation separately from sentiment estimation.

### Add

> Persist negative, neutral, and positive probabilities. Define scalar sentiment as `p_positive - p_negative`. Pin the model, tokenizer, preprocessing, and library versions.

### Replace

```yaml
minimum_sample_size: 200_decisions
```

### With

```yaml
minimum_calendar_days: 120
minimum_decisions: 500
minimum_non_overlapping_outcomes: 100
overlap_adjustment: stationary_block_bootstrap_and_HAC
```

### Replace

> reliability weight zero until promoted

### With

> Emit a separate research-only schema that cannot be consumed by portfolio or execution interfaces. Reliability weight must also remain zero, but type-level and dependency-level isolation are mandatory.

---

## Final recommendation

Proceed after revision.

The addition has a strong safety architecture, but the current specification could create misleading “clean” features through incomplete query windows, mutable engagement, non-independent tweets, asset ambiguity, unstable baselines, and overlapping outcomes.

The most important implementation principle is:

> **Do not confuse a reproducible pipeline with a valid measurement.**

The system should first prove that it captures a complete, point-in-time, author-diverse and economically interpretable sentiment measure. Only then should the LLM advisor be evaluated for incremental predictive value.

## Sources checked

- [TwitterAPI.io Advanced Search](https://docs.twitterapi.io/api-reference/endpoint/tweet_advanced_search)
- [TwitterAPI.io Pricing](https://twitterapi.io/pricing)
- [CardiffNLP Twitter-RoBERTa sentiment model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [Official X search operators](https://docs.x.com/x-api/posts/search/integrate/operators)
