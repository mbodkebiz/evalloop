"""
_utils.py — shared internal helpers.
"""

from __future__ import annotations

import sys


def _warn(msg: str) -> None:
    """Write a warning to stderr. Never raises."""
    try:
        print(msg, file=sys.stderr)
    except Exception:  # noqa: BLE001
        pass
