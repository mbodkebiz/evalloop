"""
Scorer — consistency-first LLM output scoring.

Design principles:
  - Deterministic: same input always produces the same score.
  - Heuristics first: length/format checks run before embeddings.
  - Embeddings second: cosine similarity to baseline examples.
  - LLM-as-judge: NOT used here. Reserved for ambiguous cases (future).
  - Never raises: all errors produce Score(value=0.0) with an error flag.

Score pipeline:
  output
    │
    ├─▶ heuristics (empty, too_short, too_long)
    │       │ if disqualifying → return early with score 0.0
    │
    ├─▶ baseline check (no_baseline → return score 0.0)
    │
    ├─▶ embedding cosine similarity → composite score
    │         │
    │         └──▶ Score(value, flags, confidence)
    │
    └─▶ [embed fails] → degraded_mode fallback
              │   heuristics-only score (0.5 for normal-length outputs)
              └──▶ Score(value=0.5, flags=["degraded_mode"], confidence=0.1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class Score:
    value: float          # 0.0–1.0
    flags: list[str] = field(default_factory=list)
    confidence: float = 1.0  # 0.0–1.0; lower when baseline is small


# ---------------------------------------------------------------------------
# Thresholds (tune against golden set, not guesswork)
# ---------------------------------------------------------------------------

# Length ratio vs median baseline length
_TOO_SHORT_RATIO = 0.15   # output is < 15% of median baseline length
_TOO_LONG_RATIO = 5.0     # output is > 5x median baseline length

# Minimum words to not be flagged too_short
_MIN_WORDS = 3


# ---------------------------------------------------------------------------
# Embedding hook (injectable for tests)
# ---------------------------------------------------------------------------

def _embed(texts: list[str]) -> list[list[float]]:
    """
    Default: calls Voyage AI voyage-3-lite (Anthropic's recommended embedding partner).
    Requires VOYAGE_API_KEY env var or voyageai package config.
    Injected in tests via unittest.mock.patch("evalloop.scorer._embed", ...).
    """
    import voyageai  # lazy import — not required at module load

    client = voyageai.Client()
    result = client.embed(texts, model="voyage-3-lite")
    return result.embeddings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


def _median_length(texts: list[str]) -> float:
    lengths = sorted(len(t.split()) for t in texts)
    n = len(lengths)
    mid = n // 2
    return (lengths[mid] if n % 2 else (lengths[mid - 1] + lengths[mid]) / 2)


def _centroid(embeddings: list[list[float]]) -> list[float]:
    n = len(embeddings)
    dim = len(embeddings[0])
    return [sum(e[i] for e in embeddings) / n for i in range(dim)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(
    output: str | None,
    baseline_outputs: list[str | None],
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> Score:
    """
    Score an LLM output against a set of baseline (known-good) outputs.

    Args:
        output:           The LLM output to score.
        baseline_outputs: Known-good outputs for this task. Empty → no_baseline.
        embed_fn:         Override the embedding function (used in tests).

    Returns:
        Score with value in [0.0, 1.0], flags, and confidence.
        Never raises — all error conditions return Score(value=0.0).
    """
    _embed_fn = embed_fn or _embed

    # --- Sanitize inputs -----------------------------------------------------
    if output is None:
        output = ""

    clean_baselines = [b for b in (baseline_outputs or []) if b is not None]

    # --- Heuristic: empty ----------------------------------------------------
    stripped = output.strip()
    if not stripped:
        return Score(value=0.0, flags=["empty"], confidence=1.0)

    # --- Heuristic: no baseline ----------------------------------------------
    if not clean_baselines:
        return Score(value=0.0, flags=["no_baseline"], confidence=0.0)

    # --- Heuristic: length checks --------------------------------------------
    flags: list[str] = []
    output_words = len(stripped.split())
    median_baseline_words = _median_length(clean_baselines)

    if output_words < _MIN_WORDS or (
        median_baseline_words > 0
        and output_words / median_baseline_words < _TOO_SHORT_RATIO
    ):
        # Too short: return early — length anomaly is disqualifying
        return Score(value=0.0, flags=["too_short"], confidence=1.0)

    if median_baseline_words > 0 and output_words / median_baseline_words > _TOO_LONG_RATIO:
        # Too long: flag but still compute similarity (partial score possible)
        flags.append("too_long")

    # --- Embedding similarity ------------------------------------------------
    try:
        all_texts = [stripped] + clean_baselines
        all_embeddings = _embed_fn(all_texts)
        output_embedding = all_embeddings[0]
        baseline_embeddings = all_embeddings[1:]
    except Exception:
        # Embedding unavailable — fall back to heuristics-only score.
        # A non-empty, normal-length output scores 0.5 (neutral, not failure).
        # Signals that scoring is degraded, not that the output is bad.
        return Score(value=0.5, flags=flags + ["degraded_mode"], confidence=0.1)

    baseline_centroid = _centroid(baseline_embeddings)
    similarity = _cosine(output_embedding, baseline_centroid)

    # Confidence scales with number of baseline examples (more = more reliable)
    confidence = min(1.0, len(clean_baselines) / 5)

    # If too_long was flagged, cap the score at 0.5
    final_value = similarity if "too_long" not in flags else min(similarity, 0.5)

    return Score(value=round(final_value, 4), flags=flags, confidence=round(confidence, 4))
