"""Scorer validation harness (plan §6, Gate D blocker).

Loads a human-labeled CSV (as produced by
``scripts/sample_tweets_for_labeling.py`` after hand-labeling) and produces a
confusion matrix, per-class precision/recall, a calibration table + expected
calibration error (ECE), and a plain-text report. This module ships the
harness; the human labels themselves come later and are out of scope here.

Expected CSV columns: ``tweet_id, model_input_text, p_negative, p_neutral,
p_positive, human_label`` where ``human_label`` in
``{"negative", "neutral", "positive", ""}`` (blank rows are skipped -- not
yet labeled).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any, Sequence

LABELS = ("negative", "neutral", "positive")
DEFAULT_CALIBRATION_BINS = 10


@dataclass
class LabeledRow:
    tweet_id: str
    p_negative: float
    p_neutral: float
    p_positive: float
    human_label: str

    @property
    def predicted_label(self) -> str:
        probs = {"negative": self.p_negative, "neutral": self.p_neutral, "positive": self.p_positive}
        return max(probs, key=probs.get)

    @property
    def predicted_confidence(self) -> float:
        return max(self.p_negative, self.p_neutral, self.p_positive)


def load_labeled_csv(path: str) -> list[LabeledRow]:
    """Load a labeled CSV, skipping rows without a human_label."""
    rows: list[LabeledRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            label = (raw.get("human_label") or "").strip().lower()
            if label not in LABELS:
                continue
            rows.append(
                LabeledRow(
                    tweet_id=raw["tweet_id"],
                    p_negative=float(raw["p_negative"]),
                    p_neutral=float(raw["p_neutral"]),
                    p_positive=float(raw["p_positive"]),
                    human_label=label,
                )
            )
    return rows


def confusion_matrix(rows: Sequence[LabeledRow]) -> dict[str, dict[str, int]]:
    """confusion[true_label][predicted_label] = count."""
    matrix = {t: {p: 0 for p in LABELS} for t in LABELS}
    for r in rows:
        matrix[r.human_label][r.predicted_label] += 1
    return matrix


def precision_recall(rows: Sequence[LabeledRow]) -> dict[str, dict[str, float | None]]:
    """Per-class precision/recall/f1/support from the confusion matrix."""
    matrix = confusion_matrix(rows)
    out: dict[str, dict[str, float | None]] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fn = sum(matrix[label][p] for p in LABELS if p != label)
        fp = sum(matrix[t][label] for t in LABELS if t != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return out


def calibration_table(rows: Sequence[LabeledRow], n_bins: int = DEFAULT_CALIBRATION_BINS) -> list[dict[str, Any]]:
    """Bin predictions by predicted_confidence, compare to observed accuracy.

    Each bin: {bin_low, bin_high, n, avg_confidence, accuracy}.
    """
    bins: list[list[LabeledRow]] = [[] for _ in range(n_bins)]
    for r in rows:
        conf = r.predicted_confidence
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append(r)

    table = []
    for i, bucket in enumerate(bins):
        bin_low, bin_high = i / n_bins, (i + 1) / n_bins
        if not bucket:
            table.append(
                {"bin_low": bin_low, "bin_high": bin_high, "n": 0, "avg_confidence": None, "accuracy": None}
            )
            continue
        avg_conf = sum(r.predicted_confidence for r in bucket) / len(bucket)
        correct = sum(1 for r in bucket if r.predicted_label == r.human_label)
        accuracy = correct / len(bucket)
        table.append(
            {"bin_low": bin_low, "bin_high": bin_high, "n": len(bucket), "avg_confidence": avg_conf, "accuracy": accuracy}
        )
    return table


def expected_calibration_error(rows: Sequence[LabeledRow], n_bins: int = DEFAULT_CALIBRATION_BINS) -> float | None:
    """ECE = sum_bins (n_bin / n_total) * |accuracy - avg_confidence|."""
    table = calibration_table(rows, n_bins)
    total = sum(b["n"] for b in table)
    if total == 0:
        return None
    ece = 0.0
    for b in table:
        if b["n"] == 0:
            continue
        ece += (b["n"] / total) * abs(b["accuracy"] - b["avg_confidence"])
    return ece


def render_report(rows: Sequence[LabeledRow], n_bins: int = DEFAULT_CALIBRATION_BINS) -> str:
    """Plain-text scorer validation report."""
    lines: list[str] = []
    lines.append(f"Scorer validation report -- {len(rows)} labeled tweets")
    lines.append("")

    if not rows:
        lines.append("No labeled rows found. Nothing to report yet.")
        return "\n".join(lines)

    matrix = confusion_matrix(rows)
    lines.append("Confusion matrix (rows=true, cols=predicted):")
    header = "true\\pred".ljust(12) + "".join(label.ljust(12) for label in LABELS)
    lines.append(header)
    for t in LABELS:
        row_str = t.ljust(12) + "".join(str(matrix[t][p]).ljust(12) for p in LABELS)
        lines.append(row_str)
    lines.append("")

    pr = precision_recall(rows)
    lines.append("Per-class precision / recall / f1 / support:")
    for label in LABELS:
        m = pr[label]
        p_str = f"{m['precision']:.3f}" if m["precision"] is not None else "n/a"
        r_str = f"{m['recall']:.3f}" if m["recall"] is not None else "n/a"
        f1_str = f"{m['f1']:.3f}" if m["f1"] is not None else "n/a"
        lines.append(f"  {label:10s} precision={p_str} recall={r_str} f1={f1_str} support={m['support']}")
    lines.append("")

    ece = expected_calibration_error(rows, n_bins)
    lines.append(f"Expected calibration error (ECE, {n_bins} bins): {ece:.4f}" if ece is not None else "ECE: n/a")
    lines.append("")
    lines.append("Calibration table (bin -> n, avg_confidence, accuracy):")
    for b in calibration_table(rows, n_bins):
        if b["n"] == 0:
            continue
        lines.append(
            f"  [{b['bin_low']:.1f}, {b['bin_high']:.1f}) n={b['n']} "
            f"avg_confidence={b['avg_confidence']:.3f} accuracy={b['accuracy']:.3f}"
        )

    return "\n".join(lines)
