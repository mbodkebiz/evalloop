# evalloop

**Sentry for AI behavior.** Closed-loop eval monitoring for LLM-powered products.

Wrap your Anthropic or OpenAI client with one line. Every call is scored against your known-good baselines in the background — zero added latency. Watch for regressions before your users do.

```python
from evalloop import wrap
import anthropic

client = wrap(anthropic.Anthropic(), task_tag="qa")

# Use exactly like the original client
resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
```

Then check your dashboard:

```
$ evalloop status

🟢  [qa]
   Calls captured : 142
   Scored         : 142
   Avg (7d)       : 0.81
   Avg (24h)      : 0.79
   Trend (recent) : ▆▆▇▇▆▅▆▇▇▆▇▆▇▇▆▆▇▆▇▆
```

---

## Install

```bash
pip install evalloop
```

Requires Python 3.9+.

For **Anthropic** (primary):
```bash
pip install evalloop
export VOYAGE_API_KEY=your_voyage_key   # for embeddings
export ANTHROPIC_API_KEY=your_key
```

For **OpenAI**:
```bash
pip install "evalloop[openai]"
export VOYAGE_API_KEY=your_voyage_key
export OPENAI_API_KEY=your_key
```

Get a free Voyage AI key at [voyageai.com](https://www.voyageai.com) — used for semantic similarity scoring.

---

## How it works

```
your call
    │
    ▼
wrap(client).messages.create()
    │  returns immediately
    │
    └─▶ background thread
            │
            ├─▶ score(output, baselines)   ← cosine similarity via Voyage AI
            └─▶ sqlite insert              ← ~/.evalloop/calls.db
```

- **Zero latency** — scoring runs in a background thread, never on your call path
- **Silent on errors** — disk full, API down, DB locked → log to stderr, never crash your app
- **Self-hosted** — all data stays local in `~/.evalloop/`

---

## Scoring

evalloop uses **consistency-first scoring**: deterministic heuristics + semantic similarity to your baselines. No LLM-as-judge.

| Signal | What it catches |
|--------|----------------|
| Empty / whitespace output | Total failures |
| Length anomalies (too short / too long) | Truncation, rambling |
| Cosine similarity to baseline centroid | Semantic drift, off-topic responses |

Scores range 0.0–1.0. A score below your 7-day average by >5pp triggers a regression flag.

---

## Baselines

Baselines are your known-good outputs. evalloop ships with curated defaults for common task types so you get scores immediately on first install.

```bash
# See available built-in task types
evalloop defaults

# Install defaults for all task types
evalloop baseline install

# Add your own known-good output
evalloop baseline add "The capital of France is Paris." --tag qa

# List all tags with baselines
evalloop baseline list
```

Built-in task types: `qa`, `summarization`, `code`, `customer-service`, `classification`

---

## CLI

```bash
# Score trends for all task tags
evalloop status

# Filter to one tag
evalloop status --tag qa

# Manage baselines
evalloop baseline add "your good output" --tag my-task
evalloop baseline list
evalloop baseline install --tag qa
evalloop baseline install --overwrite   # replace all defaults
```

---

## OpenAI support

```python
from evalloop import wrap
import openai

client = wrap(openai.OpenAI(), task_tag="summarization")

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this article: ..."}],
)
```

---

## Tag inference

If you don't set `task_tag`, evalloop tries to infer it from your system prompt:

```python
client = wrap(anthropic.Anthropic())  # tag_inferred from system prompt

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    system="You are a helpful assistant that summarizes documents.",  # → "summarization"
    messages=[{"role": "user", "content": "Summarize: ..."}],
)
```

---

## Architecture

- **Storage**: SQLite at `~/.evalloop/calls.db`
- **Baselines**: JSONL files at `~/.evalloop/baselines/<tag>.jsonl`
- **Embeddings**: [Voyage AI](https://www.voyageai.com) `voyage-3-lite` (512-dim, fast, cheap)
- **No cloud dependency** — everything runs locally

---

## License

MIT
