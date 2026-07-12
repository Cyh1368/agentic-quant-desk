from quantdesk.data.exclusion_policy import (
    EXCLUSION_POLICY_VERSION,
    author_sentiment,
    exclusion_flags,
    is_excluded_from_asset_mean,
    is_excluded_from_population_mean,
    population_sentiment,
)

CONFIG = {
    "min_author_followers": 150,
    "min_account_age_days": 30,
    "max_tweets_per_author_per_window": 5,
}


def base_row(**overrides):
    row = {
        "author_followers": 5000,
        "author_age_days": 400,
        "author_tweet_count_in_window": 1,
        "sentiment": 0.3,
        "quality_flags": [],
        "is_deleted": False,
        "canonical_tweet_id": "t1",
        "tweet_id": "t1",
        "target_ambiguity": "single",
    }
    row.update(overrides)
    return row


def test_no_flags_for_clean_tweet():
    assert exclusion_flags(base_row(), CONFIG) == []


def test_low_followers_flag():
    row = base_row(author_followers=10)
    assert "low_followers" in exclusion_flags(row, CONFIG)


def test_young_account_flag():
    row = base_row(author_age_days=5)
    assert "young_account" in exclusion_flags(row, CONFIG)


def test_author_over_window_cap_flag():
    row = base_row(author_tweet_count_in_window=6)
    assert "author_over_window_cap" in exclusion_flags(row, CONFIG)


def test_unscored_flag_when_sentiment_none():
    row = base_row(sentiment=None)
    assert "unscored" in exclusion_flags(row, CONFIG)


def test_unscored_flag_from_quality_flags():
    row = base_row(sentiment=0.1, quality_flags=["unscored"])
    assert "unscored" in exclusion_flags(row, CONFIG)


def test_deleted_flag():
    row = base_row(is_deleted=True)
    assert "deleted" in exclusion_flags(row, CONFIG)


def test_non_canonical_duplicate_flag():
    row = base_row(canonical_tweet_id="other", tweet_id="t1")
    assert "non_canonical_duplicate" in exclusion_flags(row, CONFIG)


def test_multi_asset_ambiguous_flag():
    row = base_row(target_ambiguity="multi_asset")
    flags = exclusion_flags(row, CONFIG)
    assert "multi_asset_ambiguous" in flags


def test_multi_asset_excluded_from_asset_mean_but_not_population():
    row = base_row(target_ambiguity="multi_asset")
    flags = exclusion_flags(row, CONFIG)
    assert is_excluded_from_asset_mean(flags) is True
    assert is_excluded_from_population_mean(flags) is False


def test_clean_tweet_included_in_both_means():
    flags = exclusion_flags(base_row(), CONFIG)
    assert is_excluded_from_asset_mean(flags) is False
    assert is_excluded_from_population_mean(flags) is False


def test_author_sentiment_is_median_of_authors_own_tweets():
    assert author_sentiment([0.1, 0.5, 0.9]) == 0.5


def test_author_sentiment_none_when_empty():
    assert author_sentiment([]) is None
    assert author_sentiment([None, None]) is None


def test_population_sentiment_author_equal_weighted():
    # Author A posted 5 times (all sentiment 1.0), author B posted once (-1.0).
    # A naive tweet-level mean would be biased toward A; author-first
    # aggregation collapses A's five posts into a single median first.
    author_a = author_sentiment([1.0, 1.0, 1.0, 1.0, 1.0])
    author_b = author_sentiment([-1.0])
    pop = population_sentiment([author_a, author_b])
    assert pop == 0.0  # median of [1.0, -1.0], not skewed by volume


def test_version_constant_exists():
    assert isinstance(EXCLUSION_POLICY_VERSION, str) and EXCLUSION_POLICY_VERSION
