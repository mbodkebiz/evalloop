"""
baseline.py — manage known-good output examples per task tag.

Baselines are stored as JSONL files in ~/.evalloop/baselines/<task_tag>.jsonl
Each line: {"output": "..."}

Returns empty list (never raises) when no baseline exists — caller must handle
the no_baseline case (scorer returns Score(0.0, ["no_baseline"])).

task_tag is sanitized to [a-zA-Z0-9_-] before use as a filename to prevent
path traversal and filesystem surprises.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from evalloop._utils import _warn


_BASELINE_DIR = os.path.expanduser("~/.evalloop/baselines")

# Only alphanumeric, dash, underscore allowed in task tags used as filenames
_TAG_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_tag(task_tag: str) -> str:
    """Strip characters that are unsafe in filenames. 'default' if empty result."""
    sanitized = _TAG_RE.sub("", task_tag)
    return sanitized or "default"


def _path(task_tag: str) -> Path:
    return Path(_BASELINE_DIR) / f"{_sanitize_tag(task_tag)}.jsonl"


def add(output: str, task_tag: str = "default") -> None:
    """
    Add a known-good output to the baseline for task_tag.
    Creates the baseline file if it doesn't exist.
    Never raises.
    """
    try:
        p = _path(task_tag)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"output": output}) + "\n")
    except Exception as exc:  # noqa: BLE001
        _warn(f"evalloop: baseline.add error — {exc}")


def load(task_tag: str = "default") -> list[str]:
    """
    Load baseline outputs for task_tag.

    Returns:
        List of output strings. Empty list if no baseline file exists.
        Never raises.
    """
    p = _path(task_tag)
    if not p.exists():
        return []
    outputs = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj.get("output"), str) and obj["output"].strip():
                        outputs.append(obj["output"])
                except json.JSONDecodeError:
                    continue
    except Exception as exc:  # noqa: BLE001
        _warn(f"evalloop: baseline.load error — {exc}")
    return outputs


def clear(task_tag: str = "default") -> None:
    """Remove all baselines for task_tag. Never raises."""
    try:
        p = _path(task_tag)
        if p.exists():
            p.unlink()
    except Exception as exc:  # noqa: BLE001
        _warn(f"evalloop: baseline.clear error — {exc}")


def list_tags() -> list[str]:
    """Return all task tags that have baseline files. Never raises."""
    try:
        return [p.stem for p in Path(_BASELINE_DIR).glob("*.jsonl")]
    except Exception:  # noqa: BLE001
        return []
