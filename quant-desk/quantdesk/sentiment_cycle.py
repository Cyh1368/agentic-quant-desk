"""One Twitter-sentiment research cycle (twitter-sentiment-plan-v2.md).

Run with ``python -m quantdesk.sentiment_cycle``. RESEARCH ONLY: this module
emits ResearchForecast records into the research store; nothing here can
reach the portfolio engine, risk engine, or executor (enforced by
tests/test_shadow_isolation.py).

Flow: reserve budget -> fetch windows -> normalize -> store -> dedup ->
score -> features -> baseline + LLM research advisors -> resolve outcomes.

Flags:
  --allow-unreviewed-terms   preflight/testing only; the terms gate (plan §4)
                             otherwise fail-closes the source.
  --fake-scorer              use the deterministic FakeScorer when the
                             [sentiment] extra (torch) is not installed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from quantdesk.advisors.sent_zscore_baseline import run_sent_zscore_contrarian_baseline_v1
from quantdesk.advisors.sentiment_llm import SentimentLlmAdvisorConfig, run_crypto_sentiment_llm_v1
from quantdesk.common.alerts import DiscordNotifier
from quantdesk.common.config import REPO_ROOT, load_config
from quantdesk.data.exclusion_policy import exclusion_flags
from quantdesk.data.raw_store import RawStore
from quantdesk.data.tweet_normalize import normalize_tweet
from quantdesk.data.tweet_store import TweetStore
from quantdesk.data.twitter import TwitterApiIoSource
from quantdesk.data.twitter_budget import BudgetExhausted, TwitterBudget
from quantdesk.features.sentiment import build_sentiment_snapshot_block, compute_sentiment_features
from quantdesk.ledger.store import LedgerStore
from quantdesk.ledger.research_store import ResearchStore

CYCLE_HORIZON = timedelta(hours=24)


def _load_sentiment_config() -> dict:
    import yaml
    with open(REPO_ROOT / "config" / "sentiment.yaml") as f:
        return yaml.safe_load(f)


def _get_scorer(use_fake: bool, cfg: dict):
    if use_fake:
        from quantdesk.scoring.tweet_scorer import FakeScorer
        return FakeScorer(), "fake"
    from pathlib import Path
    from quantdesk.scoring.tweet_scorer import TweetScorer
    local = Path(cfg["scorer"].get("model_local_dir", "")).expanduser()
    repo = str(local) if local.is_dir() else cfg["scorer"]["model_repo"]
    return TweetScorer(
        model_repo=repo,
        model_revision=None if local.is_dir() else cfg["scorer"].get("model_revision"),
        max_seq_len=cfg["scorer"]["max_sequence_length"],
    ), "real"


def _score_unscored(store: TweetStore, scorer) -> int:
    rows = store._conn.execute(
        "SELECT tweet_id, content_revision, model_input_text FROM tweets "
        "WHERE sentiment IS NULL AND is_deleted = 0 LIMIT 500"
    ).fetchall()
    if not rows:
        return 0
    cols = [r for r in rows]
    results = scorer.score_batch([r[2] or "" for r in cols])
    n = 0
    for (tweet_id, rev, _), res in zip(cols, results):
        if "quality_flags" in res and "unscored" in res.get("quality_flags", []):
            continue
        store.update_scores(
            tweet_id, rev,
            p_negative=res["p_negative"], p_neutral=res["p_neutral"],
            p_positive=res["p_positive"], scorer_confidence=res["scorer_confidence"],
            scorer_version=res["scorer_version"],
            preprocessor_version=res["preprocessor_version"],
        )
        n += 1
    return n


def _rows_with_exclusions(store: TweetStore, asset: str, start: datetime,
                          end: datetime, as_of: datetime, scfg: dict) -> list[dict]:
    rows = store.tweets_for_feature_window(asset, start, end, as_of)
    per_author = Counter(r["author_id"] for r in rows)
    out = []
    for r in rows:
        d = dict(r)
        for k in ("event_time", "available_to_strategy_time", "first_seen_at", "last_seen_at"):
            if isinstance(d.get(k), str):
                d[k] = datetime.fromisoformat(d[k])
        d["author_window_tweet_count"] = per_author[r["author_id"]]
        d["exclusions"] = exclusion_flags(d, scfg["spam"])
        out.append(d)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unreviewed-terms", action="store_true")
    ap.add_argument("--fake-scorer", action="store_true")
    args = ap.parse_args(argv)

    desk_cfg = load_config()
    scfg = _load_sentiment_config()
    now = datetime.now(timezone.utc)
    universe = list(scfg["assets"].keys())
    discord = DiscordNotifier()
    trail: list[str] = [f"sentiment cycle start {now:%Y-%m-%d %H:%M} UTC"]

    if not os.environ.get(scfg["provider"]["key_env"], "").strip():
        print(f"ERROR: {scfg['provider']['key_env']} not set in .env — cannot fetch.")
        return 2

    raw_store = RawStore(REPO_ROOT / desk_cfg["storage"]["raw_landing_dir"])
    tweet_store = TweetStore(REPO_ROOT / scfg["storage"]["tweet_store_path"])
    research_store = ResearchStore(REPO_ROOT / "data" / "research.sqlite")
    ledger = LedgerStore(REPO_ROOT / desk_cfg["storage"]["ledger_path"])
    budget = TwitterBudget(
        REPO_ROOT / "data" / "twitter_budget.sqlite",
        max_requests_per_day=scfg["budget"]["max_requests_per_day"],
        max_tweets_per_cycle=scfg["budget"]["max_tweets_per_cycle"],
        max_monthly_usd=scfg["budget"]["max_monthly_usd"],
    )
    source = TwitterApiIoSource(
        raw_store, allow_unreviewed_terms=args.allow_unreviewed_terms
    )

    # --- fetch one cadence window per asset -----------------------------
    window_start = now - timedelta(minutes=scfg["cadence_minutes"])
    cycle_id = f"cycle-{now:%Y%m%dT%H%M}"
    fetched_total = 0
    for asset, acfg in scfg["assets"].items():
        try:
            reservation = budget.reserve(
                "advanced_search", est_credits=20.0,
                est_usd=scfg["budget"]["max_monthly_usd"] / (30 * 96 * len(universe)),
                cycle_id=cycle_id, est_tweets=scfg["budget"]["max_tweets_per_cycle"] // len(universe),
                now=now,
            )
        except BudgetExhausted as e:
            trail.append(f"P2 BUDGET: {asset} fetch skipped ({e.cap_name} cap)")
            tweet_store.insert_query_window(
                asset, window_start, now, complete=False, tweet_count=0,
                meta={"source_class": "broad", "truncation_reason": "budget_cap"},
            )
            continue
        tweets, window = source.fetch_asset_window(
            asset, acfg["cashtag_query"], window_start, now, source_class="broad",
            max_tweets=scfg["budget"]["max_tweets_per_cycle"] // len(universe),
        )
        # Provider reports no per-call cost; carry the estimate as reported.
        budget.reconcile(reservation.reservation_id,
                         reported_credits=reservation.est_credits,
                         reported_usd=reservation.est_usd,
                         records_returned=len(tweets))
        tweet_store.insert_query_window(
            window.asset, window.requested_start, window.requested_end,
            complete=window.complete, tweet_count=window.fetched_count,
            meta={"source_class": window.source_class,
                  "truncation_reason": window.truncation_reason,
                  "coverage_start": window.coverage_start.isoformat(),
                  "coverage_end": window.coverage_end.isoformat()},
        )
        for raw in tweets:
            normalized = normalize_tweet(
                raw, ingested_time=now,
                raw_payload_hash=raw.get("_raw_payload_hash", ""),
                source_class="broad", asset_query_matches=[asset],
                assets_config=scfg["assets"],
            )
            tweet_store.upsert_tweet(normalized, now=now)
        fetched_total += len(tweets)
        trail.append(f"fetch {asset}: {len(tweets)} tweets, complete={window.complete}")

    tweet_store.assign_canonical(
        now - timedelta(hours=scfg["global_filters"]["dedupe_lookback_hours"]), now
    )

    # --- score ----------------------------------------------------------
    try:
        scorer, scorer_kind = _get_scorer(args.fake_scorer, scfg)
        n_scored = _score_unscored(tweet_store, scorer)
        trail.append(f"scored {n_scored} tweets ({scorer_kind} scorer)")
    except ImportError:
        trail.append("P2 SCORER: [sentiment] extra not installed and --fake-scorer "
                     "not passed; skipping scoring this cycle")

    # --- features + research advisors ------------------------------------
    snapshot_id = uuid4()
    all_features: dict = {}
    sample_rows: list[dict] = []
    windows_24h = [
        {"asset": w.asset, "window_start": w.start_time, "window_end": w.end_time,
         "complete": w.complete, "tweet_count": w.tweet_count, **(w.meta or {})}
        for a in universe for w in tweet_store.get_windows(a, now - CYCLE_HORIZON, now)
    ]
    for asset in universe:
        rows = _rows_with_exclusions(tweet_store, asset, now - CYCLE_HORIZON, now, now, scfg)
        feats = compute_sentiment_features(
            asset, rows,
            [w for w in windows_24h if w.get("asset") == asset],
            now, scfg["features"],
        )
        all_features.update({f"{asset}:{k}": v for k, v in feats.items()})
        sample_rows.extend(rows)
        a = asset.lower()
        m1 = feats.get(f"{a}_sent_mean_1h_broad@v1")
        m24 = feats.get(f"{a}_sent_mean_24h_broad@v1")
        disp = feats.get(f"{a}_sent_dispersion_1h@v1")
        health = feats.get(f"{a}_data_health@v1")
        trail.append(
            f"sentiment {asset}: 1h={'n/a' if m1 is None else f'{m1:+.2f}'} "
            f"24h={'n/a' if m24 is None else f'{m24:+.2f}'} "
            f"dispersion={'n/a' if disp is None else f'{disp:.2f}'} "
            f"health={health}"
        )
    block = build_sentiment_snapshot_block(all_features, sample_rows, scfg["snapshot_samples"])
    block["data_health"] = min(
        (v for k, v in all_features.items() if k.endswith("data_health@v1") and v is not None),
        default=0.0,
    )
    snap_path = REPO_ROOT / desk_cfg["storage"]["snapshot_dir"] / f"sentiment-{snapshot_id}.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps({"snapshot_id": str(snapshot_id), "block": block,
                                     "data_cutoff_at": now.isoformat()}, default=str))

    forecasts = list(run_sent_zscore_contrarian_baseline_v1(
        snapshot=block, instrument_ids=universe,
        generated_at=now, data_cutoff_at=now, snapshot_id=snapshot_id,
    ))
    llm_cfg = desk_cfg["llm"]
    forecasts += run_crypto_sentiment_llm_v1(
        snapshot=block, instrument_ids=universe,
        config=SentimentLlmAdvisorConfig(
            advisor_model=llm_cfg["advisor_model"],
            monthly_budget_usd=Decimal(str(llm_cfg["monthly_budget_usd"])),
            max_cost_per_decision_usd=Decimal(str(llm_cfg["max_cost_per_decision_usd"])),
            provider=llm_cfg["provider"],
        ),
        snapshot_id=snapshot_id, generated_at=now, data_cutoff_at=now,
        monthly_spend=lambda: ledger.monthly_llm_spend(now.year, now.month),
        persist=lambda prov: (
            ledger.insert_model_provenance(prov),
            ledger.log_llm_cost(prov.cost_usd, advisor_id="crypto_sentiment_llm_v1"),
        ),
    )
    for fc in forecasts:
        research_store.insert_forecast(fc)
        trail.append(f"research {fc.advisor_id} {fc.instrument_id}: "
                     f"{'abstain' if fc.abstain else fc.direction}")

    # --- resolve closed outcome windows (next_24h excess log return) -----
    pending = research_store.pending_outcomes(now)
    if pending:
        import asyncio
        from quantdesk.data.hyperliquid import HyperliquidClient

        async def _closes(coin, t0, t1):
            async with HyperliquidClient(raw_store) as hl:
                _, rows = await hl.candle_snapshot(
                    coin, "1h", int(t0.timestamp() * 1000), int(t1.timestamp() * 1000))
            return rows

        n_resolved = 0
        for p in pending[:50]:
            gen = datetime.fromisoformat(p["generated_at"])
            rows = asyncio.run(_closes(p["instrument_id"], gen - timedelta(hours=2),
                                       gen + CYCLE_HORIZON + timedelta(hours=2)))
            if len(rows) < 2:
                continue
            def px_at(t):
                best = min(rows, key=lambda r: abs(
                    datetime.fromisoformat(r["close_time"]) - t))
                return float(best["close"])
            p0, p1 = px_at(gen), px_at(gen + CYCLE_HORIZON)
            research_store.record_outcome(
                p["forecast_id"], realized_excess_log_return=math.log(p1 / p0),
                outcome_window_close_at=gen + CYCLE_HORIZON, now=now,
            )
            n_resolved += 1
        trail.append(f"resolved {n_resolved} outcome windows")

    trail.append(f"cycle complete: {fetched_total} tweets fetched, "
                 f"{len(forecasts)} research forecasts")
    print("\n".join(trail))
    if discord.enabled:
        discord.send("**sentiment research trail**\n" + "\n".join(f"- {t}" for t in trail))
    for s in (tweet_store, research_store, ledger):
        s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
