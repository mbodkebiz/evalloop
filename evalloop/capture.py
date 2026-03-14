"""
capture.py — intercept LLM calls and queue them for scoring.

Usage:
    from evalloop import wrap
    client = wrap(openai.OpenAI())
    # All calls through client are captured automatically.

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
  WrappedCompletions.create()
      │  returns immediately
      │
      ├─▶ original openai call (unchanged, full response returned to user)
      │
      └─▶ _queue.put_nowait(CapturedCall(...))   ← non-blocking
                │
                ▼ (background thread)
          _worker() drains queue
                │
                ├─▶ scorer.score(output, baselines)
                └─▶ db.insert(call + score)        ← silent on failure
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
                    break
                self._process(call)
            except queue.Empty:
                continue
            except Exception as exc:  # noqa: BLE001
                _warn(f"evalloop: worker error — {exc}")

    def _process(self, call: CapturedCall) -> None:
        try:
            from evalloop.baseline import load as _load_baselines
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
        """Block until queue is empty or timeout. Useful in tests."""
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.05)


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
# Proxy classes
# ---------------------------------------------------------------------------


class _WrappedCompletions:
    def __init__(self, original: Any, task_tag: str):
        self._original = original
        self._task_tag = task_tag

    def create(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        response = self._original.create(*args, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        try:
            messages = kwargs.get("messages", [])
            model = kwargs.get("model", "unknown")
            output_text = response.choices[0].message.content or ""

            _get_worker().put(CapturedCall(
                ts=time.time(),
                model=model,
                input_messages=list(messages),
                output_text=output_text,
                latency_ms=latency_ms,
                task_tag=self._task_tag,
            ))
        except Exception as exc:  # noqa: BLE001
            _warn(f"evalloop: capture error — {exc}")

        return response


class _WrappedChat:
    def __init__(self, original: Any, task_tag: str):
        self.completions = _WrappedCompletions(original.completions, task_tag)


class _WrappedClient:
    """Thin proxy that wraps only the surfaces evalloop intercepts."""

    def __init__(self, client: Any, task_tag: str):
        self._client = client
        self.chat = _WrappedChat(client.chat, task_tag)

    def __getattr__(self, name: str) -> Any:
        # Pass through everything else unchanged (embeddings, files, etc.)
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def wrap(client: Any, task_tag: str = "default") -> _WrappedClient:
    """
    Wrap an OpenAI client to capture all chat.completions.create calls.

    Args:
        client:   An openai.OpenAI() (or compatible) client instance.
        task_tag: Label used to look up the right baselines (e.g. "customer-service").

    Returns:
        A proxy client. Use it exactly like the original.

    Example:
        import openai
        from evalloop import wrap

        client = wrap(openai.OpenAI(), task_tag="qa-bot")
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        # Capture queued in background. Zero latency added.
    """
    return _WrappedClient(client, task_tag)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _warn(msg: str) -> None:
    """Write a warning to stderr. Never raises."""
    try:
        print(msg, file=sys.stderr)
    except Exception:  # noqa: BLE001
        pass
