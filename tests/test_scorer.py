"""
Scorer tests — golden set + consistency + edge cases.

Run: pytest tests/test_scorer.py -v
"""

import pytest
from unittest.mock import patch

from evalloop.scorer import Score, score
from tests.golden_set import GOLDEN_SET


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and",
    "or", "it", "its", "for", "on", "at", "by", "as", "be", "has", "have",
    "that", "this", "with", "from", "not", "but", "so", "if", "than",
}

# Vocabulary built from all golden set outputs + baselines (deterministic)
_VOCAB = sorted({
    w for text in [
        "paris capital france city northern largest",
        "mitochondria powerhouse cell biology",
        "exercise memory older adults study",
        "def add return sum function",
        "word short long output baseline",
    ]
    for w in text.split()
    if w not in _STOP_WORDS
})


def fake_embed(texts: list[str]) -> list[list[float]]:
    """
    Deterministic fake embeddings using content-word frequency vectors.
    Excludes stop words so semantically unrelated texts score low.
    """
    def to_vec(text: str) -> list[float]:
        words = [w.strip(".,!?;:") for w in text.lower().split()]
        content_words = [w for w in words if w not in _STOP_WORDS]
        vec = [content_words.count(w) for w in _VOCAB]
        norm = (sum(x ** 2 for x in vec) ** 0.5) or 1.0
        return [x / norm for x in vec]

    return [to_vec(t) for t in texts]


# ---------------------------------------------------------------------------
# Golden set: parametrized
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("example", GOLDEN_SET, ids=[e.label for e in GOLDEN_SET])
def test_golden_set(example):
    with patch("evalloop.scorer._embed", side_effect=fake_embed):
        result = score(example.output, example.baseline_outputs)

    assert isinstance(result, Score), "score() must return a Score"
    assert example.min_score <= result.value <= example.max_score, (
        f"[{example.label}] expected score in [{example.min_score}, {example.max_score}], "
        f"got {result.value:.3f}"
    )
    for flag in example.must_have_flags:
        assert flag in result.flags, (
            f"[{example.label}] expected flag '{flag}' in {result.flags}"
        )


# ---------------------------------------------------------------------------
# Consistency guarantee — same input ALWAYS gets same score
# ---------------------------------------------------------------------------

def test_consistency_same_input_same_score():
    """Core bet: deterministic. Same input must produce identical score every time."""
    output = "Paris is the capital of France."
    baselines = ["The capital of France is Paris.", "France's capital is Paris."]

    with patch("evalloop.scorer._embed", side_effect=fake_embed):
        scores = [score(output, baselines).value for _ in range(5)]

    assert len(set(scores)) == 1, (
        f"score() is not deterministic: got {scores}"
    )


# ---------------------------------------------------------------------------
# Edge cases — must never raise
# ---------------------------------------------------------------------------

def test_none_output_does_not_raise():
    """None output must return score 0.0 with 'empty' flag, not raise."""
    with patch("evalloop.scorer._embed", side_effect=fake_embed):
        result = score(None, ["some baseline"])  # type: ignore[arg-type]
    assert result.value == 0.0
    assert "empty" in result.flags


def test_none_baseline_item_does_not_raise():
    """A None inside baseline_outputs must be skipped, not crash."""
    with patch("evalloop.scorer._embed", side_effect=fake_embed):
        result = score("some output", [None, "valid baseline"])  # type: ignore[list-item]
    assert isinstance(result, Score)


def test_single_char_output():
    with patch("evalloop.scorer._embed", side_effect=fake_embed):
        result = score("A", ["The capital of France is Paris."])
    assert "too_short" in result.flags


def test_unicode_output_does_not_raise():
    with patch("evalloop.scorer._embed", side_effect=fake_embed):
        result = score("巴黎是法国的首都。", ["The capital of France is Paris."])
    assert isinstance(result, Score)


def test_very_large_output_does_not_raise():
    big = "word " * 10_000
    with patch("evalloop.scorer._embed", side_effect=fake_embed):
        result = score(big, ["short baseline"])
    assert "too_long" in result.flags


# ---------------------------------------------------------------------------
# Score fields
# ---------------------------------------------------------------------------

def test_score_value_is_between_0_and_1():
    with patch("evalloop.scorer._embed", side_effect=fake_embed):
        result = score("Paris is the capital of France.", ["Paris is the capital."])
    assert 0.0 <= result.value <= 1.0


def test_score_has_confidence():
    with patch("evalloop.scorer._embed", side_effect=fake_embed):
        result = score("Paris is the capital of France.", ["Paris is the capital."])
    assert hasattr(result, "confidence")
    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Degraded mode — embed failure falls back to heuristics-only score
# ---------------------------------------------------------------------------

def test_degraded_mode_returns_partial_score_not_zero():
    """When embed fails, non-empty normal-length output gets 0.5, not 0.0."""
    def failing_embed(texts):
        raise ConnectionError("Voyage AI unavailable")

    result = score(
        "The capital of France is Paris.",
        ["The capital of France is Paris."],
        embed_fn=failing_embed,
    )
    assert "degraded_mode" in result.flags
    assert result.value == 0.5
    assert result.confidence < 0.5  # low confidence signalled


def test_degraded_mode_still_catches_empty():
    """Even in degraded mode, empty output returns 0.0 (heuristic fires first)."""
    def failing_embed(texts):
        raise ConnectionError("Voyage AI unavailable")

    result = score("", ["some baseline"], embed_fn=failing_embed)
    assert result.value == 0.0
    assert "empty" in result.flags


def test_degraded_mode_still_catches_too_short():
    """Even in degraded mode, too-short output returns 0.0."""
    def failing_embed(texts):
        raise ConnectionError("Voyage AI unavailable")

    result = score("Hi", ["The capital of France is Paris."], embed_fn=failing_embed)
    assert result.value == 0.0
    assert "too_short" in result.flags
