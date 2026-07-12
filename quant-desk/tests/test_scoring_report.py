from decimal import Decimal

from quantdesk.scoring.prospective import ScoredRecord
from quantdesk.scoring.report import CostLogEntry, generate_weekly_report


def _records():
    return {
        "crypto_trend_llm_v1": [
            ScoredRecord("BTC", "crypto_trend_llm_v1", "long", 1.0, realized_excess_return=0.02, calibrated_probability_positive=0.8),
            ScoredRecord("ETH", "crypto_trend_llm_v1", "flat", 0.0, realized_excess_return=0.01),
        ],
        "ts_momentum_baseline": [
            ScoredRecord("BTC", "ts_momentum_baseline", "long", 0.5, realized_excess_return=0.02),
            ScoredRecord("ETH", "ts_momentum_baseline", "short", -0.4, realized_excess_return=-0.01),
        ],
    }


def test_report_is_deterministic_string():
    cost_log = [
        CostLogEntry("crypto_trend_llm_v1", "claude-haiku-4-5-20251001", Decimal("0.0123")),
        CostLogEntry("crypto_trend_llm_v1", "claude-haiku-4-5-20251001", Decimal("0.0098")),
    ]
    report1 = generate_weekly_report(
        period_label="2026-W28", advisor_records=_records(), cost_log=cost_log
    )
    report2 = generate_weekly_report(
        period_label="2026-W28", advisor_records=_records(), cost_log=cost_log
    )
    assert report1 == report2
    assert "crypto_trend_llm_v1" in report1
    assert "ts_momentum_baseline" in report1
    assert "coverage" in report1.lower()
    assert "abstention" in report1.lower()


def test_report_includes_total_spend():
    cost_log = [
        CostLogEntry("crypto_trend_llm_v1", "claude-haiku-4-5-20251001", Decimal("0.0123")),
    ]
    report = generate_weekly_report(
        period_label="2026-W28", advisor_records=_records(), cost_log=cost_log
    )
    assert "0.0123" in report
    assert "Total spend" in report


def test_report_handles_no_cost_log():
    report = generate_weekly_report(period_label="2026-W28", advisor_records=_records())
    assert "Total spend this period ($): 0" in report


def test_report_uses_baseline_comparison_when_present():
    report = generate_weekly_report(period_label="2026-W28", advisor_records=_records())
    # crypto_trend_llm_v1 section should include comparison line against ts_momentum_baseline
    assert "ts_momentum_baseline" in report
