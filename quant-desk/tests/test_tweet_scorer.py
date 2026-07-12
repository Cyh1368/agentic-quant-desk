from __future__ import annotations

import importlib.util

import pytest

from quantdesk.scoring.tweet_scorer import (
    PREPROCESSOR_VERSION,
    SCORER_VERSION,
    FakeScorer,
    TweetScorer,
    preprocess_text,
)

TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None and (
    importlib.util.find_spec("torch") is not None
)


def test_fake_scorer_deterministic_pipeline():
    scorer = FakeScorer(
        fixed_probs={
            "bullish af": (0.05, 0.15, 0.80),
            "this is terrible": (0.85, 0.10, 0.05),
        }
    )
    results = scorer.score_batch(["bullish af", "this is terrible", "unmapped text"])

    assert len(results) == 3
    bullish, terrible, unmapped = results

    assert bullish["p_positive"] == pytest.approx(0.80)
    assert bullish["sentiment"] == pytest.approx(0.80 - 0.05)
    assert bullish["scorer_confidence"] == pytest.approx(0.80)
    assert bullish["scorer_version"] == SCORER_VERSION
    assert bullish["preprocessor_version"] == PREPROCESSOR_VERSION
    assert bullish["quality_flags"] == []

    assert terrible["sentiment"] == pytest.approx(0.05 - 0.85)

    # Unmapped text falls back to default_probs deterministically.
    assert unmapped["p_negative"] == pytest.approx(0.2)
    assert unmapped["p_neutral"] == pytest.approx(0.6)
    assert unmapped["p_positive"] == pytest.approx(0.2)


def test_fake_scorer_never_raises_on_bad_text():
    scorer = FakeScorer()
    results = scorer.score_batch(["good text", "__RAISE__", "more good text"])

    assert len(results) == 3
    assert results[1]["quality_flags"] == ["unscored"]
    assert results[1]["sentiment"] is None
    assert results[0]["quality_flags"] == []
    assert results[2]["quality_flags"] == []


def test_fake_scorer_empty_batch():
    scorer = FakeScorer()
    assert scorer.score_batch([]) == []


def test_fake_scorer_pinning_contains_required_fields():
    scorer = FakeScorer()
    pinning = scorer.pinning()
    for key in (
        "model_repo",
        "model_revision",
        "tokenizer_revision",
        "transformers_version",
        "torch_version",
        "max_seq_len",
        "truncation_policy",
        "preprocessor_version",
        "scorer_version",
    ):
        assert key in pinning


def test_preprocess_text_handles_edge_cases():
    assert preprocess_text(None) == ""
    assert preprocess_text("  hi  ") == "hi"
    assert preprocess_text(123) == "123"
    long_text = "a" * 5000
    assert len(preprocess_text(long_text, max_chars=100)) == 100


def test_tweet_scorer_importable_without_transformers():
    # Constructing a TweetScorer must not require transformers/torch --
    # only calling load()/score_batch() should.
    scorer = TweetScorer(model_revision="abc123")
    assert scorer.model_repo == "cardiffnlp/twitter-roberta-base-sentiment-latest"
    assert scorer.max_seq_len == 512


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers/torch not installed")
def test_real_model_determinism_pinned_texts():
    scorer = TweetScorer()
    texts = [
        "Bitcoin just broke through resistance, feeling very bullish right now.",
        "This project is a complete disaster, I lost everything.",
        "BTC is trading sideways today, nothing much happening.",
    ]
    first = scorer.score_batch(texts)
    second = scorer.score_batch(texts)

    for a, b in zip(first, second):
        assert a["p_negative"] == pytest.approx(b["p_negative"], abs=1e-4)
        assert a["p_neutral"] == pytest.approx(b["p_neutral"], abs=1e-4)
        assert a["p_positive"] == pytest.approx(b["p_positive"], abs=1e-4)
        assert a["quality_flags"] == []
