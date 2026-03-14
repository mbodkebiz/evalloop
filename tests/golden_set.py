"""
Golden set: labeled (output, baseline_outputs, expected_score_range, expected_flags) examples.

These define what the scorer MUST do. Write tests against this set.
Add examples here when you find a new failure mode in production.

Score range: (min, max) — both inclusive.
Flags: subset of flags the score MUST contain (others are allowed).
"""

from dataclasses import dataclass


@dataclass
class GoldenExample:
    label: str
    output: str
    baseline_outputs: list[str]
    min_score: float
    max_score: float
    must_have_flags: list[str]


# ---------------------------------------------------------------------------
# Baselines used across examples
# ---------------------------------------------------------------------------

QA_BASELINES = [
    "The capital of France is Paris.",
    "Paris is the capital and largest city of France.",
    "France's capital city is Paris, located in northern France.",
]

SUMMARIZATION_BASELINES = [
    "The study found that exercise improves memory by 20% in older adults.",
    "Regular exercise was shown to enhance memory performance by 20% in elderly participants.",
]

CODE_BASELINES = [
    "def add(a, b):\n    return a + b",
    "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a + b",
]

# ---------------------------------------------------------------------------
# Golden examples
# ---------------------------------------------------------------------------

GOLDEN_SET: list[GoldenExample] = [
    # --- Empty / None-like outputs -------------------------------------------
    GoldenExample(
        label="empty_string",
        output="",
        baseline_outputs=QA_BASELINES,
        min_score=0.0,
        max_score=0.0,
        must_have_flags=["empty"],
    ),
    GoldenExample(
        label="whitespace_only",
        output="   \n\t  ",
        baseline_outputs=QA_BASELINES,
        min_score=0.0,
        max_score=0.0,
        must_have_flags=["empty"],
    ),
    # --- No baseline ---------------------------------------------------------
    GoldenExample(
        label="no_baseline",
        output="Paris is the capital of France.",
        baseline_outputs=[],
        min_score=0.0,
        max_score=0.0,
        must_have_flags=["no_baseline"],
    ),
    # --- Good outputs (high similarity to baseline) --------------------------
    GoldenExample(
        label="near_identical_to_baseline",
        output="The capital of France is Paris.",
        baseline_outputs=QA_BASELINES,
        min_score=0.85,
        max_score=1.0,
        must_have_flags=[],
    ),
    GoldenExample(
        label="good_paraphrase",
        output="Paris serves as the capital city of France.",
        baseline_outputs=QA_BASELINES,
        min_score=0.70,
        max_score=1.0,
        must_have_flags=[],
    ),
    # --- Length anomalies ----------------------------------------------------
    GoldenExample(
        label="rambler_10x_length",
        output=(
            "The capital of France is Paris. Paris is a beautiful city located in "
            "northern France along the Seine river. It has been the capital for many "
            "centuries and is known for the Eiffel Tower, the Louvre museum, Notre Dame "
            "cathedral, and many other famous landmarks. The city has a rich history "
            "dating back to Roman times when it was called Lutetia. Today it is home to "
            "over 2 million people in the city proper and over 12 million in the greater "
            "metropolitan area. It is a major center of art, fashion, gastronomy, and "
            "culture, and is consistently ranked as one of the world's top tourist "
            "destinations, attracting tens of millions of visitors each year."
        ),
        baseline_outputs=QA_BASELINES,
        min_score=0.0,
        max_score=0.5,
        must_have_flags=["too_long"],
    ),
    GoldenExample(
        label="too_short_single_word",
        output="Paris.",
        baseline_outputs=QA_BASELINES,
        min_score=0.0,
        max_score=0.5,
        must_have_flags=["too_short"],
    ),
    # --- Off-topic output ----------------------------------------------------
    GoldenExample(
        label="completely_off_topic",
        output="The mitochondria is the powerhouse of the cell.",
        baseline_outputs=QA_BASELINES,
        min_score=0.0,
        max_score=0.3,
        must_have_flags=[],
    ),
    # --- Different task types ------------------------------------------------
    GoldenExample(
        label="good_summarization",
        output="Exercise boosts memory by 20% in older adults.",
        baseline_outputs=SUMMARIZATION_BASELINES,
        min_score=0.70,
        max_score=1.0,
        must_have_flags=[],
    ),
    GoldenExample(
        label="good_code_output",
        output="def add(a, b):\n    return a + b",
        baseline_outputs=CODE_BASELINES,
        min_score=0.85,
        max_score=1.0,
        must_have_flags=[],
    ),
    # --- Consistency guarantee (same input must produce same score) ----------
    # Tested procedurally in test_scorer.py, not as a GoldenExample.
]
