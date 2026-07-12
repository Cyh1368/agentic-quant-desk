"""Deterministic, fully pinned tweet sentiment scorer (plan §6).

Loads ``cardiffnlp/twitter-roberta-base-sentiment-latest`` via ``transformers``
on CPU. ``transformers``/``torch`` are declared as the optional ``[sentiment]``
extra (see pyproject.toml) and are imported *lazily* inside methods so the
base install (and CI) never needs them. Tests exercise the pure scoring
contract via a ``FakeScorer`` fixture that implements the same interface with
deterministic probabilities.

Persisted per tweet: ``p_negative, p_neutral, p_positive, sentiment,
scorer_confidence, scorer_version, preprocessor_version``, with
``sentiment = p_positive - p_negative`` and
``scorer_confidence = max(p_negative, p_neutral, p_positive)``. Scoring
failures never raise for a single bad text -- they emit
``quality_flags: ["unscored"]`` and are excluded from aggregates by feature
code (see ``quantdesk/features/sentiment.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

SCORER_VERSION = "v1"
PREPROCESSOR_VERSION = "v1"


def preprocess_text(text: str, *, max_chars: int = 4000) -> str:
    """Versioned lightweight preprocessing applied before tokenization.

    This is deliberately conservative: the heavier, policy-laden transform
    (URL/mention substitution, cashtag/hashtag/emoji preservation) is the
    ``model_input_text`` pipeline built upstream of this module (plan §4);
    here we just guard against degenerate inputs (empty/non-str/oversized)
    before handing text to the tokenizer.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [x / total for x in exps]


def _score_dict_from_probs(probs: Sequence[float]) -> dict[str, Any]:
    p_negative, p_neutral, p_positive = probs[0], probs[1], probs[2]
    return {
        "p_negative": p_negative,
        "p_neutral": p_neutral,
        "p_positive": p_positive,
        "sentiment": p_positive - p_negative,
        "scorer_confidence": max(p_negative, p_neutral, p_positive),
        "scorer_version": SCORER_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "quality_flags": [],
    }


def _unscored_result(reason: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "p_negative": None,
        "p_neutral": None,
        "p_positive": None,
        "sentiment": None,
        "scorer_confidence": None,
        "scorer_version": SCORER_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "quality_flags": ["unscored"],
    }
    if reason:
        out["error"] = reason
    return out


@dataclass
class FakeScorer:
    """Deterministic scorer fixture for tests -- no torch/transformers needed.

    ``fixed_probs`` maps an exact input text (post-preprocessing) to a
    ``(p_negative, p_neutral, p_positive)`` tuple. Unmapped texts fall back to
    ``default_probs``. Texts equal to ``"__RAISE__"`` simulate an inference
    failure to exercise the never-raises-per-text contract.
    """

    fixed_probs: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    default_probs: tuple[float, float, float] = (0.2, 0.6, 0.2)

    def pinning(self) -> dict[str, Any]:
        return {
            "model_repo": "fake/fake-scorer",
            "model_revision": "fake-sha",
            "tokenizer_revision": "fake-sha",
            "transformers_version": "fake",
            "torch_version": "fake",
            "max_seq_len": 512,
            "truncation_policy": "head_truncate",
            "preprocessor_version": PREPROCESSOR_VERSION,
            "scorer_version": SCORER_VERSION,
        }

    def score_batch(self, texts: list[str]) -> list[dict]:
        results: list[dict] = []
        for raw_text in texts:
            text = preprocess_text(raw_text)
            try:
                if text == "__RAISE__":
                    raise RuntimeError("simulated scoring failure")
                probs = self.fixed_probs.get(text, self.default_probs)
                results.append(_score_dict_from_probs(probs))
            except Exception as exc:  # never propagate a single-text failure
                results.append(_unscored_result(reason=str(exc)))
        return results


class TweetScorer:
    """Loads and runs cardiffnlp/twitter-roberta-base-sentiment-latest on CPU.

    ``transformers``/``torch`` are imported lazily inside ``_load`` so this
    module can be imported (and TweetScorer instantiated) without the
    ``[sentiment]`` extra installed; only calling ``.load()`` or
    ``score_batch`` requires it.
    """

    def __init__(
        self,
        model_repo: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        model_revision: str | None = None,
        max_seq_len: int = 512,
        device: str = "cpu",
    ) -> None:
        self.model_repo = model_repo
        self._requested_revision = model_revision
        self.max_seq_len = max_seq_len
        self.device = device
        self._model = None
        self._tokenizer = None
        self._resolved_model_sha: str | None = None
        self._resolved_tokenizer_sha: str | None = None
        self._transformers_version: str | None = None
        self._torch_version: str | None = None
        self._loaded = False

    def load(self) -> None:
        """Load model + tokenizer, resolving and pinning the commit SHA."""
        if self._loaded:
            return
        import transformers  # lazy import -- optional [sentiment] extra
        import torch

        revision = self._requested_revision
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_repo, revision=revision
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_repo, revision=revision
        )
        model.to(self.device)
        model.eval()

        # Resolve the immutable commit SHA actually loaded, even when the
        # config requested a floating ref (None / branch name).
        resolved_sha = getattr(getattr(model, "config", None), "_commit_hash", None)
        if resolved_sha is None:
            resolved_sha = revision or "unknown"
        tok_sha = getattr(tokenizer, "_commit_hash", None) or resolved_sha

        self._model = model
        self._tokenizer = tokenizer
        self._resolved_model_sha = resolved_sha
        self._resolved_tokenizer_sha = tok_sha
        self._transformers_version = transformers.__version__
        self._torch_version = torch.__version__
        self._loaded = True

    def pinning(self) -> dict[str, Any]:
        """Everything persisted alongside each score, per plan §6."""
        if not self._loaded:
            self.load()
        return {
            "model_repo": self.model_repo,
            "model_revision": self._resolved_model_sha,
            "tokenizer_revision": self._resolved_tokenizer_sha,
            "transformers_version": self._transformers_version,
            "torch_version": self._torch_version,
            "max_seq_len": self.max_seq_len,
            "truncation_policy": "head_truncate",
            "preprocessor_version": PREPROCESSOR_VERSION,
            "scorer_version": SCORER_VERSION,
        }

    def score_batch(self, texts: list[str]) -> list[dict]:
        """Score a batch of texts; never raises for a single bad text."""
        if not texts:
            return []
        try:
            self.load()
        except Exception as exc:
            return [_unscored_result(reason=str(exc)) for _ in texts]

        import torch

        results: list[dict] = [None] * len(texts)  # type: ignore[list-item]
        good_indices: list[int] = []
        good_texts: list[str] = []
        for i, raw_text in enumerate(texts):
            text = preprocess_text(raw_text)
            if not text:
                results[i] = _unscored_result(reason="empty_text")
                continue
            good_indices.append(i)
            good_texts.append(text)

        if good_texts:
            try:
                encoded = self._tokenizer(
                    good_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_len,
                    padding=True,
                )
                with torch.no_grad():
                    output = self._model(**encoded)
                logits = output.logits.tolist()
                for idx, text_logits in zip(good_indices, logits):
                    probs = _softmax(text_logits)
                    results[idx] = _score_dict_from_probs(probs)
            except Exception as exc:
                for idx in good_indices:
                    results[idx] = _unscored_result(reason=str(exc))

        return results
