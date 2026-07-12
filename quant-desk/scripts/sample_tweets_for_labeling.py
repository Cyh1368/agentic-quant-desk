#!/usr/bin/env python
"""Stratified sampling script for scorer validation labeling (plan §6).

Draws ~300 tweets across strata that are known to stress a generic sentiment
model on crypto-financial language: sarcasm-suspect, emoji-heavy, price
targets, multi-asset, incident keywords, and a random-fill stratum. Writes a
CSV with ``tweet_id, model_input_text, p_negative, p_neutral, p_positive,
human_label`` (the last column left empty for hand-labeling).

This script accepts an in-memory or file-sourced list of already-scored tweet
rows (plain dicts) -- it does not fetch or score tweets itself (out of
scope: fetch adapter / tweet store). Usage:

    python scripts/sample_tweets_for_labeling.py --input rows.json --output sample.csv

``rows.json`` is a JSON list of dicts with at least: tweet_id,
model_input_text, p_negative, p_neutral, p_positive, and (optionally)
asset_mentions (list[str]) for the multi-asset stratum.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from typing import Any

TARGET_TOTAL = 300

STRATA_QUOTAS = {
    "sarcasm_suspect": 60,
    "emoji_heavy": 60,
    "price_targets": 60,
    "multi_asset": 40,
    "incident_keywords": 40,
    "random_fill": 40,
}

_SARCASM_MARKERS = ("sure", "totally", "\U0001f644", '"')
_EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
_PRICE_TARGET_PATTERN = re.compile(r"\$\s?\d|\d+\s?%")
_INCIDENT_KEYWORDS = (
    "hack",
    "exploit",
    "outage",
    "halt",
    "depeg",
    "rug",
    "liquidation",
    "insolvent",
    "delist",
    "breach",
)


def is_sarcasm_suspect(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SARCASM_MARKERS)


def is_emoji_heavy(text: str, min_emojis: int = 2) -> bool:
    return len(_EMOJI_PATTERN.findall(text)) >= min_emojis


def has_price_target(text: str) -> bool:
    return bool(_PRICE_TARGET_PATTERN.search(text))


def is_multi_asset(row: dict) -> bool:
    mentions = row.get("asset_mentions") or []
    return len(mentions) > 1


def has_incident_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in _INCIDENT_KEYWORDS)


def classify_strata(row: dict) -> list[str]:
    text = row.get("model_input_text") or ""
    strata = []
    if is_sarcasm_suspect(text):
        strata.append("sarcasm_suspect")
    if is_emoji_heavy(text):
        strata.append("emoji_heavy")
    if has_price_target(text):
        strata.append("price_targets")
    if is_multi_asset(row):
        strata.append("multi_asset")
    if has_incident_keyword(text):
        strata.append("incident_keywords")
    return strata


def stratified_sample(
    rows: list[dict],
    quotas: dict[str, int] = STRATA_QUOTAS,
    seed: int = 42,
) -> list[dict]:
    """Draw a stratified sample. A tweet may satisfy multiple strata; once
    selected it is removed from the remaining pool so counts stay disjoint
    and each output row appears once."""
    rng = random.Random(seed)
    remaining = list(rows)
    rng.shuffle(remaining)

    selected: list[dict] = []
    selected_ids: set[str] = set()

    for stratum, quota in quotas.items():
        if stratum == "random_fill":
            continue
        matches = [r for r in remaining if r["tweet_id"] not in selected_ids and stratum in classify_strata(r)]
        take = matches[:quota]
        for r in take:
            selected.append(r)
            selected_ids.add(r["tweet_id"])

    # Random fill from whatever's left.
    fill_quota = quotas.get("random_fill", 0)
    pool = [r for r in remaining if r["tweet_id"] not in selected_ids]
    rng.shuffle(pool)
    for r in pool[:fill_quota]:
        selected.append(r)
        selected_ids.add(r["tweet_id"])

    return selected


def write_labeling_csv(rows: list[dict], output_path: str) -> None:
    fieldnames = ["tweet_id", "model_input_text", "p_negative", "p_neutral", "p_positive", "human_label"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "tweet_id": r.get("tweet_id"),
                    "model_input_text": r.get("model_input_text", ""),
                    "p_negative": r.get("p_negative", ""),
                    "p_neutral": r.get("p_neutral", ""),
                    "p_positive": r.get("p_positive", ""),
                    "human_label": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a JSON file: list of scored tweet row dicts.")
    parser.add_argument("--output", required=True, help="Path to write the labeling CSV.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows: list[dict[str, Any]] = json.load(f)

    sample = stratified_sample(rows, seed=args.seed)
    write_labeling_csv(sample, args.output)
    print(f"Wrote {len(sample)} rows to {args.output} (target {TARGET_TOTAL})")


if __name__ == "__main__":
    main()
