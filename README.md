# evalloop

**Sentry for AI behavior.** Closed-loop eval monitoring for LLM-powered products.

You changed your prompt on Friday. Your bot broke. Your users noticed before you did.

evalloop wraps your LLM client with one line. Every call is scored against your known-good baselines in the background — zero added latency. Regressions surface in your terminal before they reach your users.

---

## Who this is for

**You're building an AI product** — a chatbot, support bot, summarizer, coding assistant. You're making prompt changes regularly. You have no eval system. You've shipped a regression at least once and found out from a user.

| If you're... | Your pain | evalloop gives you... |
|---|---|---|
| A solo founder shipping fast | You're the only engineer. When a prompt change breaks your bot over a weekend, users notice before you do. You have no eval infra — no time to build one. | Score trends after every call so regressions surface in 2 minutes, not Monday morning |
| An AI engineer at a startup | You improve prompts weekly but can't prove quality went up. Your PM asks "did the last change make it better?" and you have no answer. | A shareable eval report — timestamped scores by prompt version — you can show your PM exactly what changed and when |
| An ML engineer with internal LLM tools | Internal tools have no user feedback loop. If your tool degrades, users just quietly stop using it. You'd never know without monitoring. | Automated regression alerts so quality drops surface before users abandon the tool |

**Before evalloop:**
```
Prompt change pushed Friday 5pm
→ Users complain Monday morning
→ 4 hours debugging which change broke what
→ Average detection time: 3 days
```

**After evalloop:**
```
Prompt change pushed Friday 5pm
→ evalloop watch alerts Friday 5:02pm
→ Developer reverts before closing laptop
→ Average detection time: 2 minutes
```

---

## Install

```bash
pip install evalloop
```

Requires Python 3.9+. Needs a [Voyage AI](https://www.voyageai.com) key for semantic scoring (free tier available).

```bash
export VOYAGE_API_KEY=your_voyage_key
export ANTHROPIC_API_KEY=your_key   # or OPENAI_API_KEY
```

---

## Quickstart

```python
from evalloop import wrap
import anthropic

# One line change — use your client exactly as before
client = wrap(anthropic.Anthropic(), task_tag="qa")

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
# ^ scored against your baselines in the background. Zero added latency.
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

Or watch for regressions automatically:

```
$ evalloop watch --interval 60

evalloop watch — polling every 60s. Ctrl-C to stop.

🔴  [qa]
   Calls captured : 201
   Avg (7d)       : 0.81
   Avg (24h)      : 0.71
   ⚠  Regression   : score dropped 10.0pp in last 24h

^C
Watch stopped.
```

---

## How it works

```
your call
    │
    ▼
wrap(client).messages.create()
    │  returns immediately (zero latency added)
    │
    └─▶ background thread
            │
            ├─▶ infer task tag (from system prompt, if not set)
            ├─▶ score(output, baselines)
            │       ├─▶ heuristics (empty, too short, too long)
            │       └─▶ cosine similarity via Voyage AI
            └─▶ sqlite insert → ~/.evalloop/calls.db
```

- **Zero latency** — scoring runs in a background thread
- **Silent on errors** — disk full, API down, DB locked → log to stderr, never crash your app
- **Graceful exit** — on normal Python exit, flushes the queue so no captures are lost
- **Degraded mode** — if Voyage AI is unavailable, heuristic-only scoring continues (flags `degraded_mode`)
- **Self-hosted** — all data stays local in `~/.evalloop/`

---

## Auto tag inference

If you don't set `task_tag`, evalloop infers it from your system prompt:

```python
client = wrap(anthropic.Anthropic())  # no task_tag needed

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    system="You are a helpful assistant that summarizes documents.",
    # ↑ evalloop detects → task_tag="summarization"
    messages=[{"role": "user", "content": "Summarize this article..."}],
)
```

Supported task types for auto-inference: `qa`, `summarization`, `code`, `customer-service`, `classification`

---

## CLI commands

```bash
# Score trends for all tags
evalloop status

# Filter to one tag
evalloop status --tag qa

# Watch for regressions (polls every 60s by default)
evalloop watch
evalloop watch --tag qa --interval 30

# Export calls + scores for sharing or analysis
evalloop export                         # JSON to stdout
evalloop export --format csv -o out.csv # CSV to file
evalloop export --tag qa --limit 500    # filtered export

# Manage baselines
evalloop baseline add "your good output" --tag my-task
evalloop baseline list
evalloop baseline install               # install all built-in defaults
evalloop baseline install --tag qa     # install for one tag
evalloop baseline install --overwrite  # replace existing defaults

# List available built-in task types
evalloop defaults
```

---

## Scoring

evalloop uses **consistency-first scoring**: deterministic heuristics + semantic similarity to your baselines. No LLM-as-judge.

| Signal | What it catches |
|--------|----------------|
| Empty / whitespace output | Total failures |
| Too short (< 15% of baseline length) | Truncation |
| Too long (> 5x baseline length) | Rambling, prompt injection |
| Cosine similarity to baseline centroid | Semantic drift, off-topic responses |
| `degraded_mode` flag | Voyage AI unavailable — heuristics only |

Scores range 0.0–1.0. A score below your 7-day average by >5pp triggers a regression flag.

---

## Baselines

Baselines are your known-good outputs. evalloop ships with curated defaults for common task types — you get meaningful scores immediately on install.

```bash
# Install defaults for all task types
evalloop baseline install

# Add your own known-good output
evalloop baseline add "The capital of France is Paris." --tag qa

# List all tags with baselines
evalloop baseline list
```

Built-in task types: `qa`, `summarization`, `code`, `customer-service`, `classification`

---

## OpenAI support

```python
from evalloop import wrap
import openai

# pip install "evalloop[openai]"
client = wrap(openai.OpenAI(), task_tag="summarization")

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You summarize documents."},
        {"role": "user", "content": "Summarize: ..."},
    ],
)
```

---

## Data & Privacy

evalloop stores the following locally in `~/.evalloop/calls.db`:

| Field | Stored by default | With `store_inputs=False` |
|-------|:-----------------:|:------------------------:|
| Timestamp, model, latency | ✅ | ✅ |
| Output text | ✅ | ✅ |
| Score, flags, confidence | ✅ | ✅ |
| Input messages (may contain PII) | ✅ | ❌ |

For PII-sensitive environments (HIPAA, GDPR), opt out of input storage:

```python
client = wrap(anthropic.Anthropic(), task_tag="support", store_inputs=False)
# Input messages are never written to disk. Output + scores are still captured.
```

All data stays on your machine. evalloop has no cloud component.

---

## Export & share

Share your eval data with your team:

```bash
# Export last 1000 calls as JSON
evalloop export > eval-report.json

# Export as CSV for Excel/Sheets
evalloop export --format csv -o eval-report.csv

# Export a specific tag
evalloop export --tag qa --limit 500
```

---

## Architecture

- **Storage**: SQLite at `~/.evalloop/calls.db` (WAL mode — concurrent reads while writing)
- **Baselines**: JSONL files at `~/.evalloop/baselines/<tag>.jsonl`
- **Embeddings**: [Voyage AI](https://www.voyageai.com) `voyage-3-lite` (512-dim, fast, cheap)
- **Embed model provenance**: stored per-row so score history remains interpretable if model changes
- **No cloud dependency** — everything runs locally

---

## License

MIT
