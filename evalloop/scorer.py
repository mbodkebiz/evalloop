"""
Scorer — consistency-first LLM output scoring.

Design principles:
  - Heuristics first: length/format checks run before any scoring.
  - Three scoring backends (auto-detected by capture.py):
      voyage      → cosine similarity via Voyage AI embeddings (most accurate)
      anthropic   → LLM-as-judge via Claude Haiku (uses existing API key)
      heuristics  → length/format checks only (no semantic scoring)
  - Never raises: all errors produce a Score with an error flag.

Score pipelines:

  Voyage (score()):
    output
      ├─▶ heuristics → early return if disqualifying
      ├─▶ embed via Voyage AI → cosine similarity to baseline centroid
      └─▶ [embed fails] → degraded_mode fallback (0.5, confidence=0.1)

  LLM-as-judge (llm_judge_score()):
    output
      ├─▶ heuristics → early return if disqualifying
      ├─▶ Claude Haiku rates output vs baseline examples → 0-10 normalised
      └─▶ [LLM call fails] → degraded_mode fallback (0.5, confidence=0.1)

  Heuristics-only:
    output
      └─▶ heuristics only → 0.0 (fail) or 0.5 (pass, degraded_mode flag)
"""

from __future__ import annotations

import math
import re
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


# ---------------------------------------------------------------------------
# LLM-as-judge scoring (Anthropic backend)
# ---------------------------------------------------------------------------


def _call_llm_judge(output: str, baseline_outputs: list[str]) -> float:
    """
    Ask Claude Haiku to rate output quality against baselines.
    Returns a float in [0.0, 1.0]. Raises on any failure.
    Lazy-imports anthropic — only used when ANTHROPIC_API_KEY is set.
    """
    import anthropic  # lazy — user already has this installed

    client = anthropic.Anthropic()

    if baseline_outputs:
        examples = "\n".join(f"- {b[:300]}" for b in baseline_outputs[:3])
        prompt = (
            f"Rate this AI response against the quality examples below.\n\n"
            f"Examples of good responses:\n{examples}\n\n"
            f"Response to rate:\n{output[:600]}\n\n"
            f"Reply with only a number 0-10 where:\n"
            f"10 = matches or exceeds example quality\n"
            f"5  = acceptable but noticeably worse\n"
            f"0  = wrong, off-topic, or useless"
        )
    else:
        prompt = (
            f"Rate this AI response for quality and helpfulness.\n\n"
            f"Response:\n{output[:600]}\n\n"
            f"Reply with only a number 0-10 where 10=excellent, 0=useless."
        )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip() if resp.content else ""
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return max(0.0, min(1.0, float(match.group()) / 10.0))
    return 0.5  # unparseable response — neutral fallback


def llm_judge_score(
    output: str | None,
    baseline_outputs: list[str | None],
) -> Score:
    """
    Score an LLM output using Claude Haiku as judge.

    Runs the same heuristics as score() first (empty, too_short).
    Falls back to degraded_mode (0.5) if the LLM call fails.
    Never raises.

    Args:
        output:           The LLM output to score.
        baseline_outputs: Known-good outputs for context (optional).

    Returns:
        Score with value in [0.0, 1.0], flags including "llm_judge",
        and confidence=0.7 (LLM judge is less consistent than embeddings).
    """
    if output is None:
        output = ""

    clean_baselines = [b for b in (baseline_outputs or []) if b is not None]
    stripped = output.strip()

    # Heuristics first — same gates as score()
    if not stripped:
        return Score(value=0.0, flags=["empty"], confidence=1.0)

    output_words = len(stripped.split())
    if output_words < _MIN_WORDS:
        return Score(value=0.0, flags=["too_short"], confidence=1.0)

    if clean_baselines:
        median_baseline_words = _median_length(clean_baselines)
        if median_baseline_words > 0 and output_words / median_baseline_words > _TOO_LONG_RATIO:
            # Too long — flag and cap, but still judge
            try:
                value = _call_llm_judge(stripped, clean_baselines)
                return Score(value=round(min(value, 0.5), 4), flags=["too_long", "llm_judge"], confidence=0.7)
            except Exception:
                return Score(value=0.5, flags=["too_long", "llm_judge", "degraded_mode"], confidence=0.1)

    try:
        value = _call_llm_judge(stripped, clean_baselines)
        return Score(value=round(value, 4), flags=["llm_judge"], confidence=0.7)
    except Exception:
        return Score(value=0.5, flags=["llm_judge", "degraded_mode"], confidence=0.1)
