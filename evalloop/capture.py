"""
capture.py — intercept LLM calls and queue them for scoring.

Usage:
    from evalloop import wrap
    import anthropic

    client = wrap(anthropic.Anthropic(), task_tag="my-bot")
    # All calls through client are captured automatically.

    # OpenAI also supported:
    client = wrap(openai.OpenAI(), task_tag="my-bot")

    # Opt out of storing input messages (for PII-sensitive environments):
    client = wrap(anthropic.Anthropic(), task_tag="my-bot", store_inputs=False)

Design:
  - Client wrapper pattern (not monkey-patch): explicit, async-safe, testable.
  - Fire-and-forget: background thread drains a queue so the user's call path
    has zero added latency.
  - Silent on all errors: disk full, DB locked, embed failure — log to stderr,
    never propagate into user code.
  - atexit flush: on normal Python exit, waits up to 5s for queue to drain
    so script users don't lose captures.

Architecture:
  user call
      │
      ▼
  wrap(client).messages.create()   ← Anthropic
  wrap(client).chat.completions.create()  ← OpenAI
      │  returns immediately (response handed back to user)
      │
      └─▶ _queue.put_nowait(CapturedCall(...))   ← non-blocking
                │
                ▼ (background daemon thread)
          _worker() drains queue
                │
                ├─▶ infer_tag() if tag="default"
                ├─▶ scorer.score(output, baselines)
                │       └─▶ [embed fails] degraded_mode score (not 0.0)
                └─▶ db.insert(call + score + embed_model)  ← silent on failure

Response shape differences:
  Anthropic: response.content[0].text  (TextBlock)
  OpenAI:    response.choices[0].message.content
"""

from __future__ import annotations

import atexit
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from evalloop._utils import _warn
from evalloop.db import DB
from evalloop.scorer import heuristics_score as _heuristics_score
from evalloop.scorer import llm_judge_score as _llm_judge_score
from evalloop.scorer import score as _score

# Scoring backend model names stored per-row for provenance
_VOYAGE_MODEL = "voyage-3-lite"
_LLM_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def _detect_scorer() -> str:
    """
    Auto-detect the best available scoring backend from env vars.
    Priority: voyage > anthropic > heuristics (silent — no warning).

    Returns: "voyage" | "anthropic" | "heuristics"
    """
    if os.environ.get("VOYAGE_API_KEY"):
        return "voyage"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "heuristics"


# ---------------------------------------------------------------------------
# Internal captured call record
# ---------------------------------------------------------------------------


@dataclass
class CapturedCall:
    ts: float
    model: str
    input_messages: list[dict] | None  # None when store_inputs=False
    output_text: str
    latency_ms: float
    task_tag: str = "default"


# ---------------------------------------------------------------------------
# Response text extraction (handles both Anthropic and OpenAI shapes)
# ---------------------------------------------------------------------------


def _extract_output(response: Any) -> str:
    """
    Extract text output from an LLM response object.
    Anthropic: response.content is a list of blocks; grab text blocks.
    OpenAI:    response.choices[0].message.content
    Returns "" on any unexpected shape — never raises.
    """
    # Anthropic shape: response.content = [TextBlock(text=..., type="text"), ...]
    content = getattr(response, "content", None)
    if isinstance(content, list) and content:
        parts = []
        for block in content:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        if parts:
            return "".join(parts)

    # OpenAI shape: response.choices[0].message.content
    choices = getattr(response, "choices", None)
    if choices:
        try:
            return choices[0].message.content or ""
        except (AttributeError, IndexError):
            pass

    return ""


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


class _CaptureWorker:
    """
    Drains the capture queue in a daemon thread.
    All errors are caught and logged to stderr — never propagated.

    Throughput note: each item requires one Voyage AI embed call (~200-500ms).
    The single thread can process ~2-5 calls/sec. For high-volume batch use,
    queue items will back up. Queue maxsize=1000 provides ~3-8 min of buffer
    at typical LLM call rates. Batched embedding is a future optimization.
    """

    def __init__(self, db_path: str | None = None):
        self._queue: queue.Queue[CapturedCall | None] = queue.Queue(maxsize=1000)
        # DB init failure must not propagate into user code
        try:
            self._db: DB | None = DB(db_path)
        except Exception as exc:  # noqa: BLE001
            _warn(f"evalloop: DB init failed — scoring disabled ({exc})")
            self._db = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="evalloop-capture")
        self._thread.start()
        # Flush on normal Python exit so script users don't lose captures
        atexit.register(self.flush)

    def put(self, call: CapturedCall) -> None:
        """Non-blocking enqueue. Drops silently if queue is full."""
        try:
            self._queue.put_nowait(call)
        except queue.Full:
            _warn("evalloop: capture queue full — dropping call")

    def _run(self) -> None:
        while True:
            try:
                call = self._queue.get(timeout=1.0)
                if call is None:
                    self._queue.task_done()
                    break
                try:
                    self._process(call)
                except Exception as exc:  # noqa: BLE001
                    _warn(f"evalloop: worker error — {exc}")
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

    def _process(self, call: CapturedCall) -> None:
        result = None
        scorer_backend = None
        try:
            from evalloop.baseline import load as _load_baselines
            from evalloop.defaults import install as _install_defaults
            baselines = _load_baselines(call.task_tag)
            # Auto-install defaults on first use if no baselines exist yet
            if not baselines:
                _install_defaults(call.task_tag)
                baselines = _load_baselines(call.task_tag)

            scorer_backend = _detect_scorer()
            if scorer_backend == "anthropic":
                result = _llm_judge_score(call.output_text, baselines)
            elif scorer_backend == "voyage":
                result = _score(call.output_text, baselines)
            else:
                # heuristics-only — skip embedding entirely
                result = _heuristics_score(call.output_text, baselines)
        except Exception as exc:  # noqa: BLE001
            _warn(f"evalloop: scoring error — {exc}")

        if self._db is None:
            return

        try:
            if result and "degraded_mode" not in result.flags:
                if scorer_backend == "anthropic":
                    embed_model = _LLM_JUDGE_MODEL
                elif scorer_backend == "voyage":
                    embed_model = _VOYAGE_MODEL
                else:
                    embed_model = None
            else:
                embed_model = None
            self._db.insert(call, result, embed_model=embed_model)
        except Exception as exc:  # noqa: BLE001
            _warn(f"evalloop: db insert error — {exc}")

    def flush(self, timeout: float = 5.0) -> None:
        """Block until all queued calls are fully processed or timeout."""
        done = threading.Event()

        def _join() -> None:
            self._queue.join()
            done.set()

        t = threading.Thread(target=_join, daemon=True)
        t.start()
        done.wait(timeout=timeout)


# Module-level singleton worker (lazy init on first wrap() call)
_worker: _CaptureWorker | None = None
_worker_lock = threading.Lock()


def _get_worker() -> _CaptureWorker:
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = _CaptureWorker()
    return _worker


# ---------------------------------------------------------------------------
# Proxy classes — Anthropic
# ---------------------------------------------------------------------------


class _WrappedMessages:
    """Wraps anthropic.Anthropic().messages"""

    def __init__(self, original: Any, task_tag: str, store_inputs: bool):
        self._original = original
        self._task_tag = task_tag
        self._store_inputs = store_inputs

    def create(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        response = self._original.create(*args, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        try:
            # Infer tag from system prompt if user didn't specify one
            tag = self._task_tag
            if tag == "default":
                try:
                    from evalloop.defaults import infer_tag
                    system = kwargs.get("system", "")
                    if not system:
                        # Also check messages list for a system-role message
                        for msg in kwargs.get("messages", []):
                            if isinstance(msg, dict) and msg.get("role") == "system":
                                system = msg.get("content", "")
                                break
                    tag = infer_tag(system) or "default"
                except Exception:  # noqa: BLE001
                    tag = "default"

            _get_worker().put(CapturedCall(
                ts=time.time(),
                model=kwargs.get("model", getattr(response, "model", "unknown")),
                input_messages=list(kwargs.get("messages", [])) if self._store_inputs else None,
                output_text=_extract_output(response),
                latency_ms=latency_ms,
                task_tag=tag,
            ))
        except Exception as exc:  # noqa: BLE001
            _warn(f"evalloop: capture error — {exc}")

        return response


class _WrappedAnthropicClient:
    """Proxy for anthropic.Anthropic()"""

    def __init__(self, client: Any, task_tag: str, store_inputs: bool):
        self._client = client
        self.messages = _WrappedMessages(client.messages, task_tag, store_inputs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Proxy classes — OpenAI (secondary)
# ---------------------------------------------------------------------------


class _WrappedCompletions:
    """Wraps openai.OpenAI().chat.completions"""

    def __init__(self, original: Any, task_tag: str, store_inputs: bool):
        self._original = original
        self._task_tag = task_tag
        self._store_inputs = store_inputs

    def create(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        response = self._original.create(*args, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        try:
            # Infer tag from system-role message if user didn't specify one
            tag = self._task_tag
            if tag == "default":
                try:
                    from evalloop.defaults import infer_tag
                    system = ""
                    for msg in kwargs.get("messages", []):
                        if isinstance(msg, dict) and msg.get("role") == "system":
                            system = msg.get("content", "")
                            break
                    tag = infer_tag(system) or "default"
                except Exception:  # noqa: BLE001
                    tag = "default"

            _get_worker().put(CapturedCall(
                ts=time.time(),
                model=kwargs.get("model", "unknown"),
                input_messages=list(kwargs.get("messages", [])) if self._store_inputs else None,
                output_text=_extract_output(response),
                latency_ms=latency_ms,
                task_tag=tag,
            ))
        except Exception as exc:  # noqa: BLE001
            _warn(f"evalloop: capture error — {exc}")

        return response


class _WrappedOpenAIClient:
    """Proxy for openai.OpenAI()"""

    def __init__(self, client: Any, task_tag: str, store_inputs: bool):
        self._client = client
        self._task_tag = task_tag
        self.chat = _WrappedOpenAIChat(client.chat, task_tag, store_inputs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _WrappedOpenAIChat:
    def __init__(self, original: Any, task_tag: str, store_inputs: bool):
        self.completions = _WrappedCompletions(original.completions, task_tag, store_inputs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def wrap(client: Any, task_tag: str = "default", store_inputs: bool = True) -> Any:
    """
    Wrap an Anthropic or OpenAI client to capture all LLM calls.

    Anthropic (primary):
        import anthropic
        from evalloop import wrap

        client = wrap(anthropic.Anthropic(), task_tag="my-bot")
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

    OpenAI (also supported):
        import openai
        from evalloop import wrap

        client = wrap(openai.OpenAI(), task_tag="my-bot")
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

    PII-sensitive environments:
        client = wrap(anthropic.Anthropic(), task_tag="my-bot", store_inputs=False)
        # Captures output + score only. Input messages are not stored.

    Args:
        client:       anthropic.Anthropic() or openai.OpenAI() instance.
        task_tag:     Label used to look up the right baselines. If not set,
                      evalloop will try to infer it from your system prompt.
        store_inputs: If False, input messages are not stored in the DB.
                      Useful for PII-sensitive environments. Default: True.

    Returns:
        A proxy client. Use it exactly like the original.
        Zero latency added — capture happens in a background thread.
    """
    # Detect client type by module name — avoids hard import deps
    module = type(client).__module__ or ""
    if module.startswith("anthropic"):
        return _WrappedAnthropicClient(client, task_tag, store_inputs)
    return _WrappedOpenAIClient(client, task_tag, store_inputs)
