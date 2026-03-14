"""
capture.py — intercept LLM calls and queue them for scoring.

Usage:
    from evalloop import wrap
    import anthropic

    client = wrap(anthropic.Anthropic(), task_tag="my-bot")
    # All calls through client are captured automatically.

    # OpenAI also supported:
    client = wrap(openai.OpenAI(), task_tag="my-bot")

Design:
  - Client wrapper pattern (not monkey-patch): explicit, async-safe, testable.
  - Fire-and-forget: background thread drains a queue so the user's call path
    has zero added latency.
  - Silent on all errors: disk full, DB locked, embed failure — log to stderr,
    never propagate into user code.

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
                ▼ (background thread)
          _worker() drains queue
                │
                ├─▶ scorer.score(output, baselines)
                └─▶ db.insert(call + score)        ← silent on failure

Response shape differences:
  Anthropic: response.content[0].text  (TextBlock)
  OpenAI:    response.choices[0].message.content
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from evalloop.db import DB
from evalloop.scorer import score as _score


# ---------------------------------------------------------------------------
# Internal captured call record
# ---------------------------------------------------------------------------


@dataclass
class CapturedCall:
    ts: float
    model: str
    input_messages: list[dict]
    output_text: str
    latency_ms: float
    task_tag: str = "default"
    extra: dict = field(default_factory=dict)


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
    """

    def __init__(self, db_path: str | None = None):
        self._queue: queue.Queue[CapturedCall | None] = queue.Queue(maxsize=1000)
        self._db = DB(db_path)
        self._thread = threading.Thread(target=self._run, daemon=True, name="evalloop-capture")
        self._thread.start()

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
        try:
            from evalloop.baseline import load as _load_baselines
            from evalloop.defaults import install as _install_defaults
            baselines = _load_baselines(call.task_tag)
            # Auto-install defaults on first use if no baselines exist yet
            if not baselines:
                _install_defaults(call.task_tag)
                baselines = _load_baselines(call.task_tag)
            result = _score(call.output_text, baselines)
        except Exception as exc:  # noqa: BLE001
            _warn(f"evalloop: scoring error — {exc}")
            result = None

        try:
            self._db.insert(call, result)
        except Exception as exc:  # noqa: BLE001
            _warn(f"evalloop: db insert error — {exc}")

    def flush(self, timeout: float = 5.0) -> None:
        """Block until all queued calls are fully processed or timeout."""
        # queue.join() blocks until every put()'d item has called task_done()
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

    def __init__(self, original: Any, task_tag: str):
        self._original = original
        self._task_tag = task_tag

    def create(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        response = self._original.create(*args, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        try:
            _get_worker().put(CapturedCall(
                ts=time.time(),
                model=kwargs.get("model", getattr(response, "model", "unknown")),
                input_messages=list(kwargs.get("messages", [])),
                output_text=_extract_output(response),
                latency_ms=latency_ms,
                task_tag=self._task_tag,
            ))
        except Exception as exc:  # noqa: BLE001
            _warn(f"evalloop: capture error — {exc}")

        return response


class _WrappedAnthropicClient:
    """Proxy for anthropic.Anthropic()"""

    def __init__(self, client: Any, task_tag: str):
        self._client = client
        self.messages = _WrappedMessages(client.messages, task_tag)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Proxy classes — OpenAI (secondary)
# ---------------------------------------------------------------------------


class _WrappedCompletions:
    """Wraps openai.OpenAI().chat.completions"""

    def __init__(self, original: Any, task_tag: str):
        self._original = original
        self._task_tag = task_tag

    def create(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        response = self._original.create(*args, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        try:
            _get_worker().put(CapturedCall(
                ts=time.time(),
                model=kwargs.get("model", "unknown"),
                input_messages=list(kwargs.get("messages", [])),
                output_text=_extract_output(response),
                latency_ms=latency_ms,
                task_tag=self._task_tag,
            ))
        except Exception as exc:  # noqa: BLE001
            _warn(f"evalloop: capture error — {exc}")

        return response


class _WrappedOpenAIClient:
    """Proxy for openai.OpenAI()"""

    def __init__(self, client: Any, task_tag: str):
        self._client = client
        self._task_tag = task_tag
        self.chat = _WrappedOpenAIChat(client.chat, task_tag)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _WrappedOpenAIChat:
    def __init__(self, original: Any, task_tag: str):
        self.completions = _WrappedCompletions(original.completions, task_tag)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def wrap(client: Any, task_tag: str = "default") -> Any:
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

    Args:
        client:   anthropic.Anthropic() or openai.OpenAI() instance.
        task_tag: Label used to look up the right baselines.

    Returns:
        A proxy client. Use it exactly like the original.
        Zero latency added — capture happens in a background thread.
    """
    # Detect client type by module name — avoids hard import deps
    module = type(client).__module__ or ""
    if module.startswith("anthropic"):
        return _WrappedAnthropicClient(client, task_tag)
    return _WrappedOpenAIClient(client, task_tag)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _warn(msg: str) -> None:
    """Write a warning to stderr. Never raises."""
    try:
        print(msg, file=sys.stderr)
    except Exception:  # noqa: BLE001
        pass
