"""
capture.py tests — Anthropic-first wrapping, OpenAI secondary, non-blocking,
silent error handling, response shape extraction, tag inference, store_inputs.
"""

import os
import time
from unittest.mock import MagicMock, patch

from evalloop.capture import CapturedCall, _CaptureWorker, _extract_output, wrap


# ---------------------------------------------------------------------------
# Helpers — fake Anthropic and OpenAI response shapes
# ---------------------------------------------------------------------------


def _anthropic_response(text: str, model: str = "claude-haiku-4-5-20251001") -> MagicMock:
    """Mimics anthropic.types.Message response shape."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.model = model
    # Ensure .choices is absent so OpenAI path is not hit
    del resp.choices
    type(resp).__module__ = "anthropic.types"
    return resp


def _openai_response(text: str) -> MagicMock:
    """Mimics openai.types.chat.ChatCompletion response shape."""
    resp = MagicMock()
    resp.choices[0].message.content = text
    # Ensure .content is absent so Anthropic path is not hit
    del resp.content
    return resp


def _anthropic_client() -> MagicMock:
    client = MagicMock()
    type(client).__module__ = "anthropic"
    return client


def _openai_client() -> MagicMock:
    client = MagicMock()
    type(client).__module__ = "openai"
    return client


# ---------------------------------------------------------------------------
# _extract_output — response shape extraction
# ---------------------------------------------------------------------------


def test_extract_output_anthropic_shape():
    resp = _anthropic_response("Paris is the capital of France.")
    assert _extract_output(resp) == "Paris is the capital of France."


def test_extract_output_openai_shape():
    resp = _openai_response("Paris is the capital of France.")
    assert _extract_output(resp) == "Paris is the capital of France."


def test_extract_output_anthropic_multiple_blocks():
    b1, b2 = MagicMock(), MagicMock()
    b1.type, b1.text = "text", "Hello "
    b2.type, b2.text = "text", "world"
    resp = MagicMock()
    resp.content = [b1, b2]
    del resp.choices
    assert _extract_output(resp) == "Hello world"


def test_extract_output_anthropic_skips_non_text_blocks():
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "answer"
    resp = MagicMock()
    resp.content = [tool_block, text_block]
    del resp.choices
    assert _extract_output(resp) == "answer"


def test_extract_output_unknown_shape_returns_empty():
    resp = MagicMock(spec=[])  # no .content, no .choices
    assert _extract_output(resp) == ""


# ---------------------------------------------------------------------------
# wrap() — client detection
# ---------------------------------------------------------------------------


def test_wrap_anthropic_client_returns_proxy_with_messages():
    client = _anthropic_client()
    wrapped = wrap(client)
    assert hasattr(wrapped, "messages")


def test_wrap_openai_client_returns_proxy_with_chat():
    client = _openai_client()
    wrapped = wrap(client)
    assert hasattr(wrapped, "chat")


def test_wrap_passes_through_non_intercepted_attrs():
    client = _anthropic_client()
    client.beta = "beta-attr"
    wrapped = wrap(client)
    assert wrapped.beta == "beta-attr"


# ---------------------------------------------------------------------------
# Anthropic: messages.create captures correctly
# ---------------------------------------------------------------------------


def test_anthropic_create_returns_original_response():
    resp = _anthropic_response("4")
    client = _anthropic_client()
    client.messages.create.return_value = resp

    wrapped = wrap(client, task_tag="math")
    result = wrapped.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": "2+2?"}],
    )
    assert result is resp


def test_anthropic_create_queues_captured_call():
    resp = _anthropic_response("Paris is the capital of France.", model="claude-haiku-4-5-20251001")
    client = _anthropic_client()
    client.messages.create.return_value = resp

    worker = MagicMock()
    with patch("evalloop.capture._get_worker", return_value=worker):
        wrapped = wrap(client, task_tag="qa")
        wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": "Capital of France?"}],
        )

    worker.put.assert_called_once()
    call: CapturedCall = worker.put.call_args[0][0]
    assert call.output_text == "Paris is the capital of France."
    assert call.task_tag == "qa"
    assert call.model == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# OpenAI: chat.completions.create captures correctly
# ---------------------------------------------------------------------------


def test_openai_create_returns_original_response():
    resp = _openai_response("4")
    client = _openai_client()
    client.chat.completions.create.return_value = resp

    wrapped = wrap(client)
    result = wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "2+2?"}],
    )
    assert result is resp


def test_openai_create_queues_captured_call():
    resp = _openai_response("Paris is the capital of France.")
    client = _openai_client()
    client.chat.completions.create.return_value = resp

    worker = MagicMock()
    with patch("evalloop.capture._get_worker", return_value=worker):
        wrapped = wrap(client, task_tag="qa")
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Capital of France?"}],
        )

    worker.put.assert_called_once()
    call: CapturedCall = worker.put.call_args[0][0]
    assert call.output_text == "Paris is the capital of France."
    assert call.task_tag == "qa"


# ---------------------------------------------------------------------------
# Non-blocking: latency impact must be near-zero
# ---------------------------------------------------------------------------


def test_capture_does_not_block_caller():
    resp = _anthropic_response("output")
    client = _anthropic_client()
    client.messages.create.return_value = resp

    worker = MagicMock()
    with patch("evalloop.capture._get_worker", return_value=worker):
        wrapped = wrap(client)
        t0 = time.monotonic()
        wrapped.messages.create(model="claude-haiku-4-5-20251001", max_tokens=10, messages=[])
        elapsed = time.monotonic() - t0

    assert elapsed < 0.05, f"capture blocked for {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Silent error handling — never raise into user code
# ---------------------------------------------------------------------------


def test_broken_response_does_not_raise():
    broken = MagicMock(spec=[])  # no .content, no .choices, no .model
    client = _anthropic_client()
    client.messages.create.return_value = broken

    wrapped = wrap(client)
    result = wrapped.messages.create(model="claude-haiku-4-5-20251001", max_tokens=10, messages=[])
    assert result is broken  # still returned to caller


def test_full_queue_does_not_raise():
    import queue as q
    worker = _CaptureWorker.__new__(_CaptureWorker)
    worker._queue = q.Queue(maxsize=1)
    worker._queue.put_nowait(MagicMock())  # fill it

    worker.put(MagicMock())  # must not raise


# ---------------------------------------------------------------------------
# Tag inference — system prompt → task_tag
# ---------------------------------------------------------------------------


def test_tag_inferred_from_system_prompt():
    """wrap() without task_tag should infer tag from system prompt."""
    resp = _anthropic_response("Short summary here.")
    client = _anthropic_client()
    client.messages.create.return_value = resp

    worker = MagicMock()
    with patch("evalloop.capture._get_worker", return_value=worker):
        wrapped = wrap(client)  # no task_tag
        wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            system="You are a helpful assistant that summarizes documents.",
            messages=[{"role": "user", "content": "Summarize this."}],
        )

    call: CapturedCall = worker.put.call_args[0][0]
    assert call.task_tag == "summarization"


def test_tag_defaults_when_no_match():
    """Unrecognized system prompt falls back to 'default'."""
    resp = _anthropic_response("Arr, I be a pirate!")
    client = _anthropic_client()
    client.messages.create.return_value = resp

    worker = MagicMock()
    with patch("evalloop.capture._get_worker", return_value=worker):
        wrapped = wrap(client)
        wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            system="You are a pirate.",
            messages=[{"role": "user", "content": "Say something."}],
        )

    call: CapturedCall = worker.put.call_args[0][0]
    assert call.task_tag == "default"


# ---------------------------------------------------------------------------
# store_inputs=False — PII opt-out
# ---------------------------------------------------------------------------


def test_store_inputs_false_sets_input_messages_to_none():
    resp = _anthropic_response("Paris is the capital.")
    client = _anthropic_client()
    client.messages.create.return_value = resp

    worker = MagicMock()
    with patch("evalloop.capture._get_worker", return_value=worker):
        wrapped = wrap(client, task_tag="qa", store_inputs=False)
        wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": "Capital of France?"}],
        )

    call: CapturedCall = worker.put.call_args[0][0]
    assert call.input_messages is None


def test_store_inputs_true_stores_messages():
    resp = _anthropic_response("Paris.")
    client = _anthropic_client()
    client.messages.create.return_value = resp

    worker = MagicMock()
    with patch("evalloop.capture._get_worker", return_value=worker):
        wrapped = wrap(client, task_tag="qa", store_inputs=True)
        wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": "Capital of France?"}],
        )

    call: CapturedCall = worker.put.call_args[0][0]
    assert call.input_messages is not None
    assert len(call.input_messages) == 1


# ---------------------------------------------------------------------------
# VOYAGE_API_KEY warning
# ---------------------------------------------------------------------------


def test_wrap_silent_when_no_scoring_keys(capsys, monkeypatch):
    """wrap() is silent when no scoring keys are set — heuristics mode, no wall."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = _anthropic_client()
    wrap(client)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_detect_scorer_voyage(monkeypatch):
    from evalloop.capture import _detect_scorer
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-testkey")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _detect_scorer() == "voyage"


def test_detect_scorer_anthropic(monkeypatch):
    from evalloop.capture import _detect_scorer
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-testkey")
    assert _detect_scorer() == "anthropic"


def test_detect_scorer_heuristics(monkeypatch):
    from evalloop.capture import _detect_scorer
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _detect_scorer() == "heuristics"


def test_detect_scorer_prefers_voyage_over_anthropic(monkeypatch):
    from evalloop.capture import _detect_scorer
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-testkey")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-testkey")
    assert _detect_scorer() == "voyage"


# ---------------------------------------------------------------------------
# DB init failure — worker continues without crashing user code
# ---------------------------------------------------------------------------


def test_worker_continues_when_db_init_fails():
    """If DB can't be created, worker still processes queue without raising."""
    with patch("evalloop.capture.DB", side_effect=Exception("disk full")):
        worker = _CaptureWorker(db_path="/nonexistent/🚫/path")

    # Worker should be running and accepting items without raising
    call = CapturedCall(
        ts=0.0,
        model="test",
        input_messages=None,
        output_text="test",
        latency_ms=0.0,
        task_tag="default",
    )
    worker.put(call)  # must not raise
    assert worker._db is None
