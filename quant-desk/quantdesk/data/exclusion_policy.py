"""Central, versioned exclusion policy for tweet-derived sentiment (plan §5).

The normalizer and store never delete or hide spam/ambiguous tweets — they
only flag. This module is the *single* place that decides which flags
exclude a tweet from which aggregate, so feature code never hard-codes a
scattered filter condition.

Also provides the author-first aggregation helpers required by plan §5:
five tweets from one author must never count as five independent
observations.
"""
from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

EXCLUSION_POLICY_VERSION = "exclusion_policy_v1"


def exclusion_flags(tweet_row: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Return the list of exclusion flags that apply to ``tweet_row``.

    ``config`` carries thresholds normally sourced from the ``spam:``
    section of ``config/sentiment.yaml``:
        min_author_followers, min_account_age_days,
        max_tweets_per_author_per_window.

    ``tweet_row`` is a dict-like row as returned by
    ``TweetStore.tweets_for_feature_window`` / ``get_latest_revision``,
    plus an optional ``author_tweet_count_in_window`` key the caller
    computes and injects (this module has no store access, so per-author
    counts across a window are the caller's responsibility to supply).

    Flags are additive — a tweet may carry several at once. ``multi_asset``
    is intentionally named ``multi_asset_ambiguous`` so it reads distinctly
    from the normalizer's ``target_ambiguity`` field: it excludes the tweet
    from *asset-specific* means only, not from population-wide aggregates.
    """
    flags: list[str] = []

    followers = tweet_row.get("author_followers")
    min_followers = config.get("min_author_followers")
    if min_followers is not None and (followers is None or followers < min_followers):
        flags.append("low_followers")

    age_days = tweet_row.get("author_age_days")
    min_age = config.get("min_account_age_days")
    if min_age is not None and (age_days is None or age_days < min_age):
        flags.append("young_account")

    max_per_author = config.get("max_tweets_per_author_per_window")
    author_count = tweet_row.get("author_tweet_count_in_window")
    if max_per_author is not None and author_count is not None and author_count > max_per_author:
        flags.append("author_over_window_cap")

    sentiment = tweet_row.get("sentiment")
    quality_flags = tweet_row.get("quality_flags") or []
    if sentiment is None or "unscored" in quality_flags:
        flags.append("unscored")

    if tweet_row.get("is_deleted"):
        flags.append("deleted")

    canonical_tweet_id = tweet_row.get("canonical_tweet_id")
    tweet_id = tweet_row.get("tweet_id")
    if canonical_tweet_id is not None and tweet_id is not None and canonical_tweet_id != tweet_id:
        flags.append("non_canonical_duplicate")

    if tweet_row.get("target_ambiguity") == "multi_asset":
        flags.append("multi_asset_ambiguous")

    return flags


def is_excluded_from_asset_mean(flags: Sequence[str]) -> bool:
    """True if any flag excludes the tweet from asset-specific sentiment means."""
    return len(flags) > 0


def is_excluded_from_population_mean(flags: Sequence[str]) -> bool:
    """True if any flag excludes the tweet from the broad (non-asset-specific)
    population mean. ``multi_asset_ambiguous`` alone does NOT exclude a
    tweet from the population mean (plan §6: it only routes out of
    asset-specific means)."""
    return any(f != "multi_asset_ambiguous" for f in flags)


def author_sentiment(tweet_sentiments: Sequence[float]) -> float | None:
    """Robust per-author sentiment: the median of one author's tweet sentiments
    in the window. Returns None if there are no scored tweets."""
    values = [s for s in tweet_sentiments if s is not None]
    if not values:
        return None
    return statistics.median(values)


def population_sentiment(author_sentiments: Sequence[float]) -> float | None:
    """Robust population sentiment: the median across per-author sentiments.
    Author-equal-weighted — one author's five tweets contribute exactly one
    value to this median, same as an author with one tweet."""
    values = [s for s in author_sentiments if s is not None]
    if not values:
        return None
    return statistics.median(values)
