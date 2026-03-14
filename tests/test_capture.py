"""
capture.py tests — wrapping, non-blocking, silent error handling.
"""

import time
from unittest.mock import MagicMock, patch

from evalloop.capture import CapturedCall, _CaptureWorker, wrap


# ---------------------------------------------------------------------------
# wrap() returns a proxy
# ---------------------------------------------------------------------------


def test_wrap_returns_proxy():
    client = MagicMock()
    wrapped = wrap(client)
    assert wrapped is not client
    assert hasattr(wrapped, "chat")


def test_wrap_passes_through_non_chat_attrs():
    client = MagicMock()
    client.files = "files-attr"
    wrapped = wrap(client)
    assert wrapped.files == "files-attr"


# ---------------------------------------------------------------------------
# Proxy captures calls and returns original response
# ---------------------------------------------------------------------------


def test_create_returns_original_response():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "test output"

    client = MagicMock()
    client.chat.completions.create.return_value = mock_response

    wrapped = wrap(client)
    result = wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert result is mock_response


def test_create_queues_captured_call():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Paris is the capital of France."

    client = MagicMock()
    client.chat.completions.create.return_value = mock_response

    worker = MagicMock()
    with patch("evalloop.capture._get_worker", return_value=worker):
        wrapped = wrap(client, task_tag="qa")
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

    worker.put.assert_called_once()
    call: CapturedCall = worker.put.call_args[0][0]
    assert call.output_text == "Paris is the capital of France."
    assert call.task_tag == "qa"
    assert call.model == "gpt-4o"


# ---------------------------------------------------------------------------
# Non-blocking: latency impact must be near-zero
# ---------------------------------------------------------------------------


def test_capture_does_not_block_caller():
    """The caller must get the response before scoring completes."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "output"

    client = MagicMock()
    client.chat.completions.create.return_value = mock_response

    slow_worker = MagicMock()
    slow_worker.put = MagicMock()  # put is instant (queues to background)

    with patch("evalloop.capture._get_worker", return_value=slow_worker):
        wrapped = wrap(client)
        t0 = time.monotonic()
        wrapped.chat.completions.create(model="gpt-4o", messages=[])
        elapsed = time.monotonic() - t0

    # put() must return in < 50ms (it's a queue.put_nowait)
    assert elapsed < 0.05, f"capture blocked for {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Silent error handling — never raise into user code
# ---------------------------------------------------------------------------


def test_broken_response_does_not_raise():
    """If the response has an unexpected shape, capture fails silently."""
    broken_response = MagicMock()
    del broken_response.choices  # simulate unexpected response shape

    client = MagicMock()
    client.chat.completions.create.return_value = broken_response

    wrapped = wrap(client)
    result = wrapped.chat.completions.create(model="gpt-4o", messages=[])
    assert result is broken_response  # still returned to caller


def test_full_queue_does_not_raise():
    """Queue full must be silently dropped, not raise."""
    import queue as q
    worker = _CaptureWorker.__new__(_CaptureWorker)
    worker._queue = q.Queue(maxsize=1)
    worker._queue.put_nowait(MagicMock())  # fill it up

    # Should not raise even though queue is full
    worker.put(MagicMock())
