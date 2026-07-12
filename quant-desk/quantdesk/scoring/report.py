"""Plain-text weekly report generator (deterministic, no LLM, plan §6/§11).

Consumes already-computed scoring results (see prospective.py) and a cost
log, and renders a plain-text summary. No LLM calls, no I/O beyond
formatting a string; callers own how the cost log and scored records are
sourced (this module never imports the ledger).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quantdesk.scoring.prospective import (
    ScoredRecord,
    abstention_rate,
    coverage,
    compare_vs_baselines,
)


@dataclass
class CostLogEntry:
    advisor_id: str
    model_id: str
    cost_usd: Decimal


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _fmt_num(x: float | None, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def generate_weekly_report(
    *,
    period_label: str,
    advisor_records: dict[str, list[ScoredRecord]],
    baseline_advisor_id: str = "ts_momentum_baseline",
    cost_log: list[CostLogEntry] | None = None,
) -> str:
    """Build a deterministic plain-text weekly report.

    advisor_records: mapping of advisor_id -> its ScoredRecord list for the
      period. If baseline_advisor_id is present as a key, its records are
      used as the baseline comparison for every other advisor.
    cost_log: flat list of per-call cost entries for the period; spend is
      summed per advisor_id.
    """
    cost_log = cost_log or []
    lines: list[str] = []
    lines.append(f"Quant Desk Weekly Advisor Report — {period_label}")
    lines.append("=" * 60)

    spend_by_advisor: dict[str, Decimal] = {}
    for entry in cost_log:
        spend_by_advisor[entry.advisor_id] = (
            spend_by_advisor.get(entry.advisor_id, Decimal("0")) + entry.cost_usd
        )
    total_spend = sum(spend_by_advisor.values(), Decimal("0"))

    baseline_records = advisor_records.get(baseline_advisor_id)

    for advisor_id, records in sorted(advisor_records.items()):
        lines.append("")
        lines.append(f"Advisor: {advisor_id}")
        lines.append("-" * 60)
        n = len(records)
        lines.append(f"  decisions scored:       {n}")
        lines.append(f"  coverage:                {_fmt_pct(coverage(records))}")
        lines.append(f"  abstention rate:         {_fmt_pct(abstention_rate(records))}")

        cmp_baseline = (
            baseline_records
            if (baseline_records is not None and advisor_id != baseline_advisor_id)
            else None
        )
        comparisons = compare_vs_baselines(records, baseline_advisor_records=cmp_baseline)

        advisor_summary = comparisons["advisor"]
        lines.append(f"  hit rate:                {_fmt_pct(advisor_summary['hit_rate'])}")
        lines.append(f"  brier score:             {_fmt_num(advisor_summary['brier_score'])}")
        lines.append(
            f"  net information coef.:   {_fmt_num(advisor_summary['net_information_coefficient'])}"
        )

        lines.append("  vs. baselines:")
        for name in ("always_flat", "buy_and_hold", "random_same_frequency", "ts_momentum_baseline"):
            if name not in comparisons:
                continue
            c = comparisons[name]
            lines.append(
                f"    {name:<24s} hit_rate={_fmt_pct(c['hit_rate']):>7s}  "
                f"brier={_fmt_num(c['brier_score']):>7s}  "
                f"IC={_fmt_num(c['net_information_coefficient']):>7s}"
            )

        spend = spend_by_advisor.get(advisor_id, Decimal("0"))
        lines.append(f"  spend this period ($):   {spend}")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Total spend this period ($): {total_spend}")

    return "\n".join(lines)
