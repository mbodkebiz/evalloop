# evalloop — Critique & Action Items

Brutal honest feedback from four seats. Each action item is discrete and buildable.

---

## Prompt Engineer

**Core problem: the scoring signal is too weak to trust.**

| # | Action Item | Why |
|---|-------------|-----|
| PE-1 | **Add LLM-as-judge scoring** (optional, uses user's own API key) | Cosine similarity misses factually wrong answers that sound right. "Paris is the capital of Germany" scores high against "Paris is the capital of France." That's the regression that matters. |
| PE-2 | **Remove or gate the `too_short` / `too_long` heuristics behind a configurable flag** | Length is not quality. A perfect one-sentence answer fails `too_short` against a 5-sentence baseline. Currently penalizing correct concise answers. |
| PE-3 | **Calibrate the 15% and 5x length thresholds against real regression data** | These are magic numbers. Run them against 3-5 real prompt regression examples and adjust, or expose them as config options. |
| PE-4 | **Fix tag inference to handle ambiguous system prompts** | Keyword matching breaks when a prompt mentions multiple task types ("summarize problems to write code for customer service"). Needs priority order or LLM-based classification. |
| PE-5 | **Document exactly what evalloop catches and what it doesn't** | "No LLM-as-judge" is a real limitation. The README should say explicitly: "evalloop catches semantic drift and structural failures. It does not catch factual errors or subtle quality regressions." |

---

## PM

**Core problem: the product alerts you but doesn't help you fix anything.**

| # | Action Item | Why |
|---|-------------|-----|
| PM-1 | **Add push alerts — Slack webhook and/or email** | `evalloop watch` requires the developer to babysit a terminal. The before/after story ("alerts Friday 5:02pm") is only true if watch is running. Push makes the core promise unconditionally true. |
| PM-2 | **Add prompt version tracking** | Score drops without knowing which prompt change caused them is an alert with no remediation path. Need to correlate score drops to specific prompt versions. Could be as simple as `wrap(client, prompt_version="v1.3")`. |
| PM-3 | **Build a human-readable report** (HTML or Markdown, not raw JSON/CSV) | "Shareable eval report" is the AI engineer's value prop. A PM cannot read a JSON file. The export should produce something you can paste into a Notion doc or email. |
| PM-4 | **Make the regression threshold configurable per tag** | `threshold=0.05` is hardcoded. For a medical triage bot, 1pp is a patient safety issue. For a creative writing bot, 10pp is noise. Expose `regression_threshold` as a `wrap()` param. |
| PM-5 | **Add a "what to do next" section to regression alerts** | When `evalloop watch` fires, it should tell you: "Score dropped 10pp. Last 5 flagged outputs: [list]. Run `evalloop export --tag qa --limit 5` to review them." Give the developer an immediate next action. |

---

## Engineering Manager

**Core problem: this is a local dev tool, not a production monitoring system.**

| # | Action Item | Why |
|---|-------------|-----|
| EM-1 | **Support a shared DB path via env var or config** (`EVALLOOP_DB_PATH`) | The `~/.evalloop/` default means production workers each write to their own local file. Nobody is reading those. A shared path (NFS mount, or a future cloud option) makes this usable in real deployments. |
| EM-2 | **Handle SIGTERM gracefully in long-running servers** | `atexit` doesn't fire on SIGKILL or uncaught SIGTERM. Add a `signal.signal(SIGTERM, ...)` handler in `_CaptureWorker` so Gunicorn/K8s worker recycling doesn't silently drop captures. |
| EM-3 | **Add at least one real integration test** (not mocked) | The entire test suite mocks Voyage and mocks the DB. There's no proof the system works end-to-end. Add one test that makes a real Voyage AI call (skipped if `VOYAGE_API_KEY` not set). |
| EM-4 | **Add a `evalloop check` command for CI** | There's no way to use evalloop in a PR workflow today. `evalloop check --tag qa --fail-below 0.7` should exit non-zero so CI can block a deploy on quality regression. |
| EM-5 | **Document the production deployment story** | README shows local dev usage only. Add a section: "Running in production (web servers, workers, containers)" with Gunicorn, Celery, and Docker examples. |
| EM-6 | **Replace the column migration hack with proper versioning** | `ALTER TABLE ... ADD COLUMN` guarded by `PRAGMA table_info` is fragile. Use a schema version table and a proper migration list. Will break otherwise when a column type needs to change. |

---

## CEO

**Core problem: the product has no expansion motion, no monetization path, and a self-imposed distribution ceiling.**

| # | Action Item | Why |
|---|-------------|-----|
| CEO-1 | **Define the paid tier before the free tier gets too established** | "All data stays on your machine" kills every cloud monetization path. Decide now: is the business OSS + consulting, OSS + cloud sync (with explicit privacy tradeoff), or something else. Not deciding is deciding wrong. |
| CEO-2 | **Build a team/cloud sync story** (even if v2) | The current architecture has no virality, no network effect, no collaboration. One engineer finds a regression → shares the eval report → their PM wants a dashboard → signs up for cloud. That loop doesn't exist today. |
| CEO-3 | **Target developers who already feel the pain, not those who don't yet know they have it** | "No eval system" customers have no budget and no urgency. The wedge is developers at companies already using LangSmith or Braintrust who find it too heavyweight. Position as the lightweight alternative for teams that want monitoring without the platform overhead. |
| CEO-4 | **Remove the Voyage AI free-tier dependency from the core value prop** | Voyage is a third-party free tier. It can change pricing, rate-limit you, or get acquired. The core scoring signal shouldn't depend on a vendor you don't control. Either (a) support multiple embed providers, (b) offer local embeddings via sentence-transformers, or (c) make Voyage optional with a fully-capable heuristic fallback. |
| CEO-5 | **Write down the 12-month product thesis** | What does evalloop look like if it works? 1000 developers using it? What do they pay for? What do they tell their friends? What does the company look like? Without this, every feature decision is arbitrary. |

---

## Priority stack rank (cross-seat view)

| Priority | Item | Seat | Effort |
|----------|------|------|--------|
| P0 | PM-1: Push alerts (Slack webhook) | PM | S |
| P0 | PM-2: Prompt version tracking | PM | S |
| P1 | PE-1: LLM-as-judge scoring (opt-in) | PE | M |
| P1 | EM-4: `evalloop check` for CI | EM | S |
| P1 | CEO-4: Multi-provider embeddings / local fallback | CEO | M |
| P2 | PM-3: Human-readable HTML report | PM | M |
| P2 | EM-1: Shared DB path via env var | EM | S |
| P2 | EM-2: SIGTERM handling | EM | S |
| P2 | PE-2: Configurable length thresholds | PE | S |
| P2 | PM-4: Configurable regression threshold per tag | PM | S |
| P3 | CEO-1: Define paid tier | CEO | — |
| P3 | CEO-3: Repositioning against LangSmith | CEO | — |
| P3 | EM-3: Real integration test | EM | S |
| P3 | EM-6: Proper schema versioning | EM | M |
| P4 | CEO-2: Team/cloud sync story | CEO | XL |
| P4 | CEO-5: 12-month product thesis | CEO | — |
