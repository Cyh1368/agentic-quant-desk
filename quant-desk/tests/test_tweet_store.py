from datetime import datetime, timedelta, timezone

import pytest

from quantdesk.data.tweet_normalize import normalize_tweet
from quantdesk.data.tweet_store import TweetStore

T0 = datetime(2026, 7, 12, 0, 0, 0, tzinfo=timezone.utc)


def mk(tweet_id, text, event_offset_min=0, followers=1000, author_id="a1", author="alice", **kw):
    raw = {
        "id": tweet_id,
        "text": text,
        "createdAt": (T0 + timedelta(minutes=event_offset_min)).isoformat(),
        "author": {
            "userName": author,
            "id": author_id,
            "followers": followers,
            "createdAt": "2020-01-01T00:00:00Z",
        },
        "retweetCount": 0,
        "replyCount": 0,
        "likeCount": 0,
        "viewCount": 0,
    }
    raw.update(kw)
    return normalize_tweet(raw, T0, "hash", "curated", ["BTC"])


@pytest.fixture
def store(tmp_path):
    s = TweetStore(tmp_path / "sentiment.sqlite")
    yield s
    s.close()


def test_upsert_creates_first_revision(store):
    norm = mk("1", "BTC to the moon")
    rev = store.upsert_tweet(norm, now=T0)
    assert rev == 0
    row = store.get_latest_revision("1")
    assert row["content_revision"] == 0
    assert row["raw_text_restricted"] == "BTC to the moon"


def test_upsert_same_content_bumps_last_seen_not_new_revision(store):
    norm = mk("1", "BTC to the moon")
    store.upsert_tweet(norm, now=T0)
    later = T0 + timedelta(minutes=10)
    rev = store.upsert_tweet(norm, now=later)
    assert rev == 0
    revisions = store.get_all_revisions("1")
    assert len(revisions) == 1
    assert revisions[0]["last_seen_at"] == later.astimezone(timezone.utc).isoformat()


def test_edit_creates_new_revision(store):
    norm_v0 = mk("1", "BTC to the moon")
    store.upsert_tweet(norm_v0, now=T0)
    norm_v1 = mk("1", "BTC to the moon, edited")
    t1 = T0 + timedelta(minutes=5)
    rev = store.upsert_tweet(norm_v1, now=t1)
    assert rev == 1
    revisions = store.get_all_revisions("1")
    assert len(revisions) == 2
    assert revisions[0]["raw_text_restricted"] == "BTC to the moon"
    assert revisions[1]["raw_text_restricted"] == "BTC to the moon, edited"


def test_historical_as_of_query_returns_old_revision(store):
    norm_v0 = mk("1", "BTC to the moon")
    store.upsert_tweet(norm_v0, now=T0)
    t1 = T0 + timedelta(minutes=30)
    norm_v1 = mk("1", "BTC to the moon, edited")
    store.upsert_tweet(norm_v1, now=t1)

    # as_of before the edit: must see the OLD revision.
    as_of_before = T0 + timedelta(minutes=10)
    rows = store.tweets_for_feature_window(
        "BTC", T0 - timedelta(minutes=1), T0 + timedelta(hours=1), as_of_before
    )
    assert len(rows) == 1
    assert rows[0]["raw_text_restricted"] == "BTC to the moon"
    assert rows[0]["content_revision"] == 0

    # as_of after the edit: must see the NEW revision.
    as_of_after = T0 + timedelta(hours=1)
    rows2 = store.tweets_for_feature_window(
        "BTC", T0 - timedelta(minutes=1), T0 + timedelta(hours=1), as_of_after
    )
    assert len(rows2) == 1
    assert rows2[0]["raw_text_restricted"] == "BTC to the moon, edited"
    assert rows2[0]["content_revision"] == 1


def test_available_to_strategy_time_gates_visibility(store):
    norm = mk("1", "BTC to the moon")
    store.upsert_tweet(norm, now=T0)
    too_early = T0 - timedelta(minutes=1)
    rows = store.tweets_for_feature_window(
        "BTC", T0 - timedelta(minutes=1), T0 + timedelta(hours=1), too_early
    )
    assert rows == []


def test_mark_deleted_flags_latest_revision(store):
    norm = mk("1", "BTC to the moon")
    store.upsert_tweet(norm, now=T0)
    store.mark_deleted("1", T0 + timedelta(minutes=1))
    row = store.get_latest_revision("1")
    assert row["is_deleted"] == 1
    assert row["deleted_observed_at"] is not None


def test_engagement_observations_append_only_and_never_mutate_historical_query(store):
    norm = mk("1", "BTC to the moon")
    store.upsert_tweet(norm, now=T0)

    store.add_engagement_observation("1", T0 + timedelta(minutes=5), 5, likes=10, views=100)
    rows_before = store.tweets_for_feature_window(
        "BTC", T0 - timedelta(minutes=1), T0 + timedelta(hours=1), T0 + timedelta(minutes=6)
    )
    assert len(rows_before) == 1
    # tweets_for_feature_window doesn't join engagement at all -- confirm the
    # tweet row itself is unaffected by observation content.
    snapshot_row = dict(rows_before[0])

    # A later, very different observation must not alter the historical row.
    store.add_engagement_observation("1", T0 + timedelta(hours=2), 125, likes=99999, views=999999)
    rows_after = store.tweets_for_feature_window(
        "BTC", T0 - timedelta(minutes=1), T0 + timedelta(hours=1), T0 + timedelta(minutes=6)
    )
    assert dict(rows_after[0]) == snapshot_row

    observations = store.get_engagement_observations("1")
    assert len(observations) == 2
    assert observations[0]["likes"] == 10
    assert observations[1]["likes"] == 99999


def test_author_independence_five_posts_one_author(store):
    for i in range(5):
        norm = mk(str(i), f"BTC post {i}", event_offset_min=i, author_id="a1", author="alice")
        store.upsert_tweet(norm, now=T0)

    rows = store.tweets_for_feature_window(
        "BTC", T0 - timedelta(minutes=1), T0 + timedelta(hours=1), T0 + timedelta(hours=1)
    )
    assert len(rows) == 5
    author_ids = {r["author_id"] for r in rows}
    assert author_ids == {"a1"}
    # Author-first aggregation (collapsing to one observation) is exercised
    # in test_exclusion_policy.py; here we just confirm the store keeps all
    # five raw rows available for that collapse to happen downstream.


def test_multi_asset_tweet_flagged_ambiguous_in_store(store):
    raw = {
        "id": "9",
        "text": "ETH strong, BTC weak",
        "createdAt": T0.isoformat(),
        "author": {"userName": "bob", "id": "b1", "followers": 1000, "createdAt": "2020-01-01T00:00:00Z"},
    }
    norm = normalize_tweet(raw, T0, "h", "curated", ["BTC", "ETH"])
    store.upsert_tweet(norm, now=T0)
    row = store.get_latest_revision("9")
    assert row["target_ambiguity"] == "multi_asset"
    assert set(row["asset_mentions"]) == {"BTC", "ETH"}


def test_dedup_canonicalization_collapses_near_duplicates(store):
    # Same content_for_dedup after stripping handles/URLs/case.
    norm_a = mk("1", "Big BTC news today!", event_offset_min=0, author_id="a1", author="alice")
    norm_b = mk("2", "big btc news today!", event_offset_min=1, author_id="a2", author="bob")
    norm_c = mk("3", "BIG BTC NEWS TODAY!", event_offset_min=2, author_id="a3", author="carol")
    store.upsert_tweet(norm_a, now=T0)
    store.upsert_tweet(norm_b, now=T0)
    store.upsert_tweet(norm_c, now=T0)

    n_groups = store.assign_canonical(T0 - timedelta(hours=1), T0 + timedelta(hours=1))
    assert n_groups >= 1

    row_a = store.get_latest_revision("1")
    row_b = store.get_latest_revision("2")
    row_c = store.get_latest_revision("3")

    # All three share the same canonical tweet (the earliest, tweet 1).
    assert row_a["canonical_tweet_id"] == "1"
    assert row_b["canonical_tweet_id"] == "1"
    assert row_c["canonical_tweet_id"] == "1"

    # Propagation counts are retained on every member of the group.
    assert row_a["duplicate_count"] == 3
    assert row_b["duplicate_count"] == 3
    assert row_c["duplicate_count"] == 3
    assert row_a["duplicate_unique_authors"] == 3


def test_dedup_distinct_content_not_collapsed(store):
    norm_a = mk("1", "BTC pump incoming", author_id="a1")
    norm_b = mk("2", "completely different topic here", author_id="a2")
    store.upsert_tweet(norm_a, now=T0)
    store.upsert_tweet(norm_b, now=T0)
    store.assign_canonical(T0 - timedelta(hours=1), T0 + timedelta(hours=1))
    row_a = store.get_latest_revision("1")
    row_b = store.get_latest_revision("2")
    assert row_a["canonical_tweet_id"] != row_b["canonical_tweet_id"]
    assert row_a["duplicate_count"] == 1


def test_query_windows_insert_and_get(store):
    store.insert_query_window(
        "BTC", T0, T0 + timedelta(hours=1), complete=True, tweet_count=42
    )
    windows = store.get_windows("BTC", T0, T0 + timedelta(hours=1))
    assert len(windows) == 1
    assert windows[0].complete is True
    assert windows[0].tweet_count == 42


def test_update_scores_fills_nullable_columns(store):
    norm = mk("1", "BTC to the moon")
    store.upsert_tweet(norm, now=T0)
    store.update_scores(
        "1",
        0,
        p_negative=0.1,
        p_neutral=0.2,
        p_positive=0.7,
        scorer_confidence=0.7,
        scorer_version="v1",
        preprocessor_version="v1",
    )
    row = store.get_latest_revision("1")
    assert row["sentiment"] == pytest.approx(0.6)
    assert row["scorer_confidence"] == 0.7
