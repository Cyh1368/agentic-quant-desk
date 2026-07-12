from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from quantdesk.data.twitter_budget import BudgetExhausted, TwitterBudget


def _now():
    return datetime(2026, 7, 12, 10, 0, tzinfo=timezone.utc)


def test_reserve_inserts_row_and_reconcile_updates_actuals(tmp_path):
    budget = TwitterBudget(
        tmp_path / "budget.sqlite",
        max_requests_per_day=10,
        max_tweets_per_cycle=100,
        max_monthly_usd=5.0,
    )
    res = budget.reserve("advanced_search", est_credits=1.0, est_usd=0.01, now=_now())
    assert res.reservation_id is not None

    budget.reconcile(res.reservation_id, reported_credits=1.0, reported_usd=0.008, records_returned=12)
    counters = budget.endpoint_counters("advanced_search")
    assert counters.request_count == 1
    assert counters.records_returned == 12
    assert counters.cost_reported_usd == pytest.approx(0.008)


def test_reserve_raises_budget_exhausted_on_daily_request_cap(tmp_path):
    budget = TwitterBudget(
        tmp_path / "budget.sqlite",
        max_requests_per_day=2,
        max_tweets_per_cycle=1000,
        max_monthly_usd=50.0,
    )
    budget.reserve("advanced_search", 1.0, 0.01, now=_now())
    budget.reserve("advanced_search", 1.0, 0.01, now=_now())
    with pytest.raises(BudgetExhausted) as exc_info:
        budget.reserve("advanced_search", 1.0, 0.01, now=_now())
    assert exc_info.value.cap_name == "max_requests_per_day"


def test_reserve_raises_budget_exhausted_on_cycle_tweet_cap(tmp_path):
    budget = TwitterBudget(
        tmp_path / "budget.sqlite",
        max_requests_per_day=1000,
        max_tweets_per_cycle=50,
        max_monthly_usd=50.0,
    )
    budget.reserve("advanced_search", 1.0, 0.01, cycle_id="cycle1", est_tweets=40, now=_now())
    with pytest.raises(BudgetExhausted) as exc_info:
        budget.reserve("advanced_search", 1.0, 0.01, cycle_id="cycle1", est_tweets=20, now=_now())
    assert exc_info.value.cap_name == "max_tweets_per_cycle"


def test_reserve_raises_budget_exhausted_on_monthly_usd_cap(tmp_path):
    budget = TwitterBudget(
        tmp_path / "budget.sqlite",
        max_requests_per_day=1000,
        max_tweets_per_cycle=10000,
        max_monthly_usd=1.0,
    )
    budget.reserve("advanced_search", 1.0, 0.6, now=_now())
    with pytest.raises(BudgetExhausted) as exc_info:
        budget.reserve("advanced_search", 1.0, 0.6, now=_now())
    assert exc_info.value.cap_name == "max_monthly_usd"


def test_reconcile_reflects_in_monthly_spend(tmp_path):
    budget = TwitterBudget(
        tmp_path / "budget.sqlite",
        max_requests_per_day=1000,
        max_tweets_per_cycle=10000,
        max_monthly_usd=1.0,
    )
    res = budget.reserve("advanced_search", 1.0, 0.9, now=_now())
    budget.reconcile(res.reservation_id, 1.0, 0.1, records_returned=5)
    # After reconciling down to 0.1 actual spend, there's room for more.
    budget.reserve("advanced_search", 1.0, 0.8, now=_now())


def test_budget_exhausted_message_has_no_secret(tmp_path):
    budget = TwitterBudget(
        tmp_path / "budget.sqlite",
        max_requests_per_day=1,
        max_tweets_per_cycle=100,
        max_monthly_usd=50.0,
    )
    budget.reserve("advanced_search", 1.0, 0.01, now=_now())
    with pytest.raises(BudgetExhausted) as exc_info:
        budget.reserve("advanced_search", 1.0, 0.01, now=_now())
    assert "fake-key-not-real" not in str(exc_info.value)


def test_concurrent_reservations_against_near_exhausted_cap_cannot_both_succeed(tmp_path):
    """Two threads racing to reserve the last unit of a nearly-exhausted cap
    must not both succeed — reservation is atomic (BEGIN IMMEDIATE)."""
    budget = TwitterBudget(
        tmp_path / "budget.sqlite",
        max_requests_per_day=3,
        max_tweets_per_cycle=100000,
        max_monthly_usd=50.0,
    )
    # Use up 2 of the 3 daily requests up front, leaving exactly 1 slot.
    budget.reserve("advanced_search", 1.0, 0.01, now=_now())
    budget.reserve("advanced_search", 1.0, 0.01, now=_now())

    results = []
    errors = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        try:
            budget.reserve("advanced_search", 1.0, 0.01, now=_now())
            results.append("ok")
        except BudgetExhausted:
            errors.append("exhausted")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 1, f"expected exactly one success, got {results}"
    assert len(errors) == 1, f"expected exactly one BudgetExhausted, got {errors}"
    counters = budget.endpoint_counters("advanced_search")
    assert counters.request_count == 3


def test_remaining_requests_and_monthly_usd_helpers(tmp_path):
    budget = TwitterBudget(
        tmp_path / "budget.sqlite",
        max_requests_per_day=5,
        max_tweets_per_cycle=100,
        max_monthly_usd=10.0,
    )
    assert budget.remaining_requests_today(now=_now()) == 5
    assert budget.remaining_monthly_usd(now=_now()) == pytest.approx(10.0)
    budget.reserve("advanced_search", 1.0, 2.0, now=_now())
    assert budget.remaining_requests_today(now=_now()) == 4
    assert budget.remaining_monthly_usd(now=_now()) == pytest.approx(8.0)


def test_reserve_rejects_naive_datetime(tmp_path):
    budget = TwitterBudget(
        tmp_path / "budget.sqlite",
        max_requests_per_day=5,
        max_tweets_per_cycle=100,
        max_monthly_usd=10.0,
    )
    with pytest.raises(ValueError):
        budget.reserve("advanced_search", 1.0, 0.01, now=datetime(2026, 7, 12, 10, 0))
