"""
baseline.py tests — add/load/clear, graceful empty handling, never raises.
"""

import pytest

from evalloop import baseline as bl


@pytest.fixture(autouse=True)
def tmp_baseline_dir(tmp_path, monkeypatch):
    """Redirect baseline storage to a temp dir for each test."""
    monkeypatch.setattr(bl, "_BASELINE_DIR", str(tmp_path))
    # Also patch _path to use the new dir
    monkeypatch.setattr(
        "evalloop.baseline._BASELINE_DIR", str(tmp_path)
    )
    yield


def test_load_empty_when_no_file():
    result = bl.load("nonexistent")
    assert result == []


def test_add_and_load():
    bl.add("Paris is the capital of France.", task_tag="qa")
    bl.add("France's capital is Paris.", task_tag="qa")
    outputs = bl.load("qa")
    assert len(outputs) == 2
    assert "Paris is the capital of France." in outputs


def test_load_ignores_blank_lines():
    import os
    from pathlib import Path
    p = Path(bl._BASELINE_DIR) / "test.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"output": "good output"}\n\n{"output": "another"}\n')
    outputs = bl.load("test")
    assert len(outputs) == 2


def test_load_skips_malformed_lines():
    from pathlib import Path
    p = Path(bl._BASELINE_DIR) / "test.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"output": "good"}\nnot-json\n{"output": "also good"}\n')
    outputs = bl.load("test")
    assert len(outputs) == 2


def test_clear_removes_file():
    bl.add("something", task_tag="todelete")
    assert bl.load("todelete") != []
    bl.clear("todelete")
    assert bl.load("todelete") == []


def test_clear_nonexistent_does_not_raise():
    bl.clear("does-not-exist")  # must not raise


def test_list_tags():
    bl.add("output a", task_tag="tag-a")
    bl.add("output b", task_tag="tag-b")
    tags = bl.list_tags()
    assert "tag-a" in tags
    assert "tag-b" in tags


def test_add_never_raises_on_bad_path(monkeypatch):
    monkeypatch.setattr("evalloop.baseline._BASELINE_DIR", "/nonexistent/🚫/path")
    bl.add("some output", task_tag="x")  # must not raise


def test_load_never_raises_on_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr("evalloop.baseline._BASELINE_DIR", str(tmp_path))
    p = tmp_path / "corrupt.jsonl"
    p.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")
    result = bl.load("corrupt")
    assert isinstance(result, list)  # empty list, no exception


# ---------------------------------------------------------------------------
# task_tag sanitization — path traversal prevention
# ---------------------------------------------------------------------------

def test_sanitize_tag_strips_path_traversal():
    from evalloop.baseline import _sanitize_tag
    assert _sanitize_tag("../../etc/passwd") == "etcpasswd"


def test_sanitize_tag_allows_valid_chars():
    from evalloop.baseline import _sanitize_tag
    assert _sanitize_tag("qa-bot_v2") == "qa-bot_v2"


def test_sanitize_tag_empty_result_becomes_default():
    from evalloop.baseline import _sanitize_tag
    assert _sanitize_tag("../..") == "default"
