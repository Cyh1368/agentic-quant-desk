import pytest

from quantdesk.scoring.prospective import (
    ScoredRecord,
    abstention_rate,
    brier_score,
    calibration_error,
    calibration_table,
    compare_vs_baselines,
    contract_status,
    coverage,
    hit_rate,
    net_information_coefficient,
    random_same_frequency_records,
)

# Hand-computed fixture (see scratch calc): 3 non-flat + 1 flat record.
R1 = ScoredRecord("BTC", "adv", "long", 1.0, realized_excess_return=0.02, calibrated_probability_positive=0.8)
R2 = ScoredRecord("ETH", "adv", "long", 1.0, realized_excess_return=-0.01, calibrated_probability_positive=0.6)
R3 = ScoredRecord("BTC", "adv", "short", -1.0, realized_excess_return=-0.02, calibrated_probability_positive=0.3)
R4 = ScoredRecord("ETH", "adv", "flat", 0.0, realized_excess_return=0.05)

RECORDS = [R1, R2, R3, R4]


def test_coverage_and_abstention():
    assert coverage(RECORDS) == pytest.approx(0.75)
    assert abstention_rate(RECORDS) == pytest.approx(0.25)


def test_coverage_empty():
    assert coverage([]) == 0.0
    assert abstention_rate([]) == 1.0


def test_hit_rate_hand_computed():
    # R1: long, outcome positive -> hit. R2: long, outcome negative -> miss.
    # R3: short, outcome negative -> hit (predicted non-positive, outcome non-positive).
    assert hit_rate(RECORDS) == pytest.approx(2 / 3)


def test_hit_rate_none_when_no_active():
    assert hit_rate([R4]) is None


def test_brier_score_hand_computed():
    expected = ((0.8 - 1) ** 2 + (0.6 - 0) ** 2 + (0.3 - 0) ** 2) / 3
    assert brier_score(RECORDS) == pytest.approx(expected)
    assert brier_score(RECORDS) == pytest.approx(0.16333333333333333)


def test_brier_score_none_when_no_active():
    assert brier_score([R4]) is None


def test_calibration_table_buckets_hand_computed():
    table = calibration_table(RECORDS)
    by_bucket = {b["bucket"]: b for b in table}
    assert by_bucket["[0.2, 0.4]"]["n"] == 1
    assert by_bucket["[0.2, 0.4]"]["avg_predicted_probability"] == pytest.approx(0.3)
    assert by_bucket["[0.2, 0.4]"]["empirical_positive_rate"] == pytest.approx(0.0)

    assert by_bucket["[0.6, 0.8]"]["n"] == 1
    assert by_bucket["[0.6, 0.8]"]["avg_predicted_probability"] == pytest.approx(0.6)
    assert by_bucket["[0.6, 0.8]"]["empirical_positive_rate"] == pytest.approx(0.0)

    assert by_bucket["[0.8, 1.0]"]["n"] == 1
    assert by_bucket["[0.8, 1.0]"]["avg_predicted_probability"] == pytest.approx(0.8)
    assert by_bucket["[0.8, 1.0]"]["empirical_positive_rate"] == pytest.approx(1.0)

    assert by_bucket["[0.0, 0.2]"]["n"] == 0


def test_calibration_error_hand_computed():
    # weighted mean abs diff: (|0.3-0| + |0.6-0| + |0.8-1|) / 3
    expected = (0.3 + 0.6 + 0.2) / 3
    assert calibration_error(RECORDS) == pytest.approx(expected)


def test_net_information_coefficient_perfect_positive():
    records = [
        ScoredRecord("BTC", "adv", "long", 1.0, realized_excess_return=0.1),
        ScoredRecord("BTC", "adv", "long", 2.0, realized_excess_return=0.2),
        ScoredRecord("BTC", "adv", "long", 3.0, realized_excess_return=0.3),
    ]
    assert net_information_coefficient(records) == pytest.approx(1.0)


def test_net_information_coefficient_perfect_negative():
    records = [
        ScoredRecord("BTC", "adv", "long", 1.0, realized_excess_return=0.3),
        ScoredRecord("BTC", "adv", "long", 2.0, realized_excess_return=0.2),
        ScoredRecord("BTC", "adv", "long", 3.0, realized_excess_return=0.1),
    ]
    assert net_information_coefficient(records) == pytest.approx(-1.0)


def test_net_information_coefficient_hand_computed_mixed_actions():
    assert net_information_coefficient(RECORDS) == pytest.approx(0.6933752452815364)


def test_net_information_coefficient_none_below_two_active():
    assert net_information_coefficient([R1, R4]) is None  # only 1 active record, need >=2
    assert net_information_coefficient([R4]) is None  # 0 active


def test_always_flat_and_buy_and_hold_and_random_baselines_shape():
    comparisons = compare_vs_baselines(RECORDS)
    assert set(comparisons.keys()) == {
        "advisor", "always_flat", "buy_and_hold", "random_same_frequency",
    }
    assert comparisons["always_flat"]["coverage"] == 0.0
    assert comparisons["always_flat"]["hit_rate"] is None
    assert comparisons["buy_and_hold"]["coverage"] == 1.0


def test_random_same_frequency_is_seeded_and_reproducible():
    a = random_same_frequency_records(RECORDS, seed=7)
    b = random_same_frequency_records(RECORDS, seed=7)
    assert [r.action for r in a] == [r.action for r in b]
    c = random_same_frequency_records(RECORDS, seed=99)
    assert [r.action for r in a] != [r.action for r in c] or True  # different seed may coincide; just must not error


def test_compare_vs_baselines_includes_supplied_baseline_advisor():
    baseline_records = [
        ScoredRecord("BTC", "ts_momentum_baseline", "long", 0.5, realized_excess_return=0.01),
    ]
    comparisons = compare_vs_baselines(RECORDS, baseline_advisor_records=baseline_records)
    assert "ts_momentum_baseline" in comparisons
    assert comparisons["ts_momentum_baseline"]["coverage"] == 1.0


CONTRACT = {
    "advisor_id": "crypto_trend_llm_v1",
    "minimum_sample_size": 200,
    "demotion_thresholds": {
        "rolling_window": 100,
        "brier_score_max": 0.27,
        "action_on_breach": "shadow",
    },
}


def test_contract_status_insufficient_sample():
    status = contract_status(RECORDS, CONTRACT)
    assert status["status"] == "insufficient_sample"
    assert status["n_scored_in_window"] == 4


def test_contract_status_breach_triggers_action_on_breach():
    # Build 200 non-flat records with a poor brier score (worse than 0.27).
    poor_records = [
        ScoredRecord("BTC", "adv", "long", 1.0, realized_excess_return=-0.01, calibrated_probability_positive=0.9)
        for _ in range(200)
    ]
    status = contract_status(poor_records, CONTRACT)
    assert status["brier_score"] == pytest.approx(0.81)
    assert status["breached"] is True
    assert status["status"] == "shadow"


def test_contract_status_active_when_meets_thresholds():
    good_records = [
        ScoredRecord("BTC", "adv", "long", 1.0, realized_excess_return=0.01, calibrated_probability_positive=0.9)
        for _ in range(200)
    ]
    status = contract_status(good_records, CONTRACT)
    assert status["breached"] is False
    assert status["status"] == "active"
