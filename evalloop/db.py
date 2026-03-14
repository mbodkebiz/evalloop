"""
db.py — SQLite storage with yoyo migrations.

Schema is migration-managed from day one. Adding columns = new migration file,
never a manual ALTER TABLE.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalloop.capture import CapturedCall
    from evalloop.scorer import Score


_DEFAULT_DB_PATH = os.path.expanduser("~/.evalloop/calls.db")

_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    model       TEXT    NOT NULL,
    input_json  TEXT    NOT NULL,
    output_text TEXT    NOT NULL,
    latency_ms  REAL,
    task_tag    TEXT    NOT NULL DEFAULT 'default',
    score       REAL,
    score_flags TEXT,
    confidence  REAL
);
CREATE INDEX IF NOT EXISTS idx_calls_ts ON calls (ts);
CREATE INDEX IF NOT EXISTS idx_calls_task ON calls (task_tag, ts);
"""


class DB:
    def __init__(self, path: str | None = None):
        self._path = path or _DEFAULT_DB_PATH
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self) -> None:
        with self._connect() as conn:
            conn.executescript(_MIGRATION_SQL)

    def insert(self, call: CapturedCall, result: Score | None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO calls
                    (ts, model, input_json, output_text, latency_ms, task_tag,
                     score, score_flags, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call.ts,
                    call.model,
                    json.dumps(call.input_messages),
                    call.output_text,
                    call.latency_ms,
                    call.task_tag,
                    result.value if result else None,
                    json.dumps(result.flags) if result else None,
                    result.confidence if result else None,
                ),
            )

    def recent(self, task_tag: str = "default", limit: int = 200) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT * FROM calls
                WHERE task_tag = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (task_tag, limit),
            ).fetchall()

    def all_task_tags(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT task_tag FROM calls ORDER BY task_tag"
            ).fetchall()
            return [r["task_tag"] for r in rows]
