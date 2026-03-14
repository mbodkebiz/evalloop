"""
db.py — SQLite storage for captured LLM calls and scores.

Schema is managed via a single CREATE TABLE IF NOT EXISTS script.
To add columns in future versions: add an entry to _COLUMN_MIGRATIONS —
each entry is applied once via ALTER TABLE, guarded by a column-existence
check. Never drop or rename columns (backward-compat).

WAL mode is enabled so the CLI can read while the background capture
thread writes without either side blocking the other.
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
    input_json  TEXT,
    output_text TEXT    NOT NULL,
    latency_ms  REAL,
    task_tag    TEXT    NOT NULL DEFAULT 'default',
    score       REAL,
    score_flags TEXT,
    confidence  REAL,
    embed_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_ts ON calls (ts);
CREATE INDEX IF NOT EXISTS idx_calls_task ON calls (task_tag, ts);
"""

# Safe column additions for users upgrading from older schema versions.
# Each tuple: (column_name, ALTER TABLE sql). Applied once, idempotent.
_COLUMN_MIGRATIONS = [
    ("embed_model", "ALTER TABLE calls ADD COLUMN embed_model TEXT"),
]


class DB:
    def __init__(self, path: str | None = None):
        self._path = path or _DEFAULT_DB_PATH
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        # WAL: readers never block writers, writers never block readers
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _migrate(self) -> None:
        with self._connect() as conn:
            conn.executescript(_MIGRATION_SQL)
            # Apply column additions for existing DBs
            existing = {row[1] for row in conn.execute("PRAGMA table_info(calls)")}
            for col_name, alter_sql in _COLUMN_MIGRATIONS:
                if col_name not in existing:
                    conn.execute(alter_sql)

    def insert(
        self,
        call: CapturedCall,
        result: Score | None,
        embed_model: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO calls
                    (ts, model, input_json, output_text, latency_ms, task_tag,
                     score, score_flags, confidence, embed_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call.ts,
                    call.model,
                    json.dumps(call.input_messages) if call.input_messages is not None else None,
                    call.output_text,
                    call.latency_ms,
                    call.task_tag,
                    result.value if result else None,
                    json.dumps(result.flags) if result else None,
                    result.confidence if result else None,
                    embed_model,
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

    def update_score(
        self,
        call_id: int,
        result: Score,
        embed_model: str | None = None,
    ) -> None:
        """Overwrite score fields for a specific call row (used by rescore)."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE calls
                SET score = ?, score_flags = ?, confidence = ?, embed_model = ?
                WHERE id = ?
                """,
                (
                    result.value,
                    json.dumps(result.flags),
                    result.confidence,
                    embed_model,
                    call_id,
                ),
            )

    def export(
        self,
        task_tag: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Return calls as plain dicts for JSON/CSV export."""
        with self._connect() as conn:
            if task_tag:
                rows = conn.execute(
                    "SELECT * FROM calls WHERE task_tag = ? ORDER BY ts DESC LIMIT ?",
                    (task_tag, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM calls ORDER BY ts DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
