"""
defaults.py — curated default baselines by task type.

Solves the cold-start problem: new users get meaningful scores immediately,
before they've accumulated their own history.

Each task type has 5 baseline examples — enough to form a reliable centroid
without being so opinionated that they mislead on niche use cases.

Users can always override with their own baselines:
    evalloop baseline add "my good output" --tag qa

Task types:
  qa              — question answering (factual, concise)
  summarization   — document/article summarization
  code            — code generation
  customer-service — support / helpdesk responses
  classification  — label / category output
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Default baseline examples per task type
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, list[str]] = {
    "qa": [
        "The capital of France is Paris.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "The mitochondria generates ATP through oxidative phosphorylation.",
        "The HTTP 404 status code means the requested resource was not found.",
        "To reverse a list in Python, use list.reverse() or list[::-1].",
    ],
    "summarization": [
        "The study found that regular exercise improves memory by 20% in older adults.",
        "The report concludes that renewable energy adoption has accelerated, with solar capacity doubling in three years.",
        "Researchers identified three key factors driving customer churn: price, support quality, and onboarding friction.",
        "The proposal outlines a two-phase rollout: a pilot with 500 users in Q1, followed by a full launch in Q2.",
        "In summary, the treatment group showed a statistically significant improvement over placebo at the 12-week mark.",
    ],
    "code": [
        "def add(a, b):\n    return a + b",
        "def reverse_string(s: str) -> str:\n    return s[::-1]",
        "def is_even(n: int) -> bool:\n    return n % 2 == 0",
        "const greet = (name: string): string => `Hello, ${name}!`;",
        "function sum(arr) {\n  return arr.reduce((a, b) => a + b, 0);\n}",
    ],
    "customer-service": [
        "Thank you for reaching out. I've looked into your account and can confirm the refund has been processed. You should see it within 3-5 business days.",
        "I completely understand your frustration and I'm sorry for the inconvenience. Let me escalate this to our technical team right away.",
        "Your order #12345 is currently in transit and is expected to arrive by Thursday. You can track it here: [link].",
        "I'd be happy to help with that! To reset your password, click 'Forgot password' on the login page and follow the instructions.",
        "We apologize for the delay. Your ticket has been prioritized and a specialist will follow up within 2 hours.",
    ],
    "classification": [
        "positive",
        "negative",
        "neutral",
        "spam",
        "not spam",
    ],
}

# ---------------------------------------------------------------------------
# Tag inference from system prompt keywords
# ---------------------------------------------------------------------------

_TAG_HINTS: list[tuple[list[str], str]] = [
    (["summarize", "summary", "tldr", "brief"], "summarization"),
    (["classify", "label", "category", "sentiment"], "classification"),
    (["code", "function", "implement", "write a", "script"], "code"),
    (["support", "customer", "help desk", "ticket", "refund", "order"], "customer-service"),
    (["answer", "question", "what is", "who is", "explain"], "qa"),
]


def infer_tag(system_prompt: str) -> str | None:
    """
    Infer a task tag from a system prompt.
    Returns the best-matching tag, or None if no confident match.
    """
    if not system_prompt:
        return None
    lower = system_prompt.lower()
    for keywords, tag in _TAG_HINTS:
        if any(kw in lower for kw in keywords):
            return tag
    return None


# ---------------------------------------------------------------------------
# Install defaults into baseline storage
# ---------------------------------------------------------------------------

def install(tag: str, overwrite: bool = False) -> int:
    """
    Install default baselines for a task tag.

    Args:
        tag:       Task tag (e.g. "qa", "summarization").
        overwrite: If True, clear existing baselines first.

    Returns:
        Number of examples installed. 0 if tag unknown or already exists.
    """
    from evalloop.baseline import add, clear, load

    if tag not in DEFAULTS:
        return 0

    existing = load(tag)
    if existing and not overwrite:
        return 0

    if overwrite:
        clear(tag)

    for example in DEFAULTS[tag]:
        add(example, task_tag=tag)

    return len(DEFAULTS[tag])


def install_all(overwrite: bool = False) -> dict[str, int]:
    """Install defaults for all task types. Returns {tag: count_installed}."""
    return {tag: install(tag, overwrite=overwrite) for tag in DEFAULTS}
