"""
Simple evaluation queue backed by SQLite for offline relevance scoring.

This module defines a small, stdlib-only helper to persist minimal artifacts of
RAG responses for offline evaluation. We capture the `interaction_id`, user
`question`, model `answer`, and a compact JSON-serialized list of top context
`chunks` so that a later background process can compute LLM-as-judge relevance
scores without adding latency to the request path. The storage is SQLite, which
is an embedded, serverless database that ships with Python via the `sqlite3`
module. The queue table uses a UNIQUE constraint (via a UNIQUE interaction_id
choice) to ensure idempotency; repeated enqueues for the same interaction are
ignored safely. Timestamps are stored in ISO 8601 format.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List
from datetime import datetime, timezone
import os
import logging


logger = logging.getLogger(__name__)


def init_eval_queue(db_path: str) -> None:
    """
    Initialize the evaluation queue table if it does not exist.

    This function ensures the parent directory of the database file exists, opens
    a connection to the SQLite database at `db_path`, and creates the `eval_queue`
    table with the schema required for offline evaluation. We define a UNIQUE row
    keyed by `interaction_id` (via a UNIQUE constraint on the column) so inserts
    are idempotent. The function commits and closes the connection; errors are
    allowed to propagate to the caller for visibility during startup.

    Args:
        db_path (str): Filesystem path to the SQLite database file.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT UNIQUE,
                question TEXT,
                answer TEXT,
                context_json TEXT,
                created_at TEXT,
                processed_at TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def enqueue_eval_item(
    db_path: str,
    interaction_id: str,
    question: str,
    answer: str,
    context_chunks: List[str],
) -> bool:
    """
    Insert one evaluation item into the queue; return True if inserted, False if duplicate.

    The function serializes `context_chunks` to JSON and performs an INSERT OR IGNORE
    on the `eval_queue` table using `interaction_id` as the uniqueness key. On the
    first insertion, `created_at` is set to the current UTC time in ISO format and
    `processed_at` remains NULL to signal pending evaluation. If a row with the same
    interaction_id already exists, the operation is ignored and the function returns
    False. Exceptions are allowed to propagate to the caller so errors can be logged
    and handled at the API layer without crashing the server.

    Args:
        db_path (str): Filesystem path to the SQLite database file.
        interaction_id (str): Unique identifier joining the request to a trace/log.
        question (str): The user's question.
        answer (str): The assistant's final answer text.
        context_chunks (List[str]): Up to a few context strings used to ground the answer.

    Returns:
        bool: True if a new row was inserted; False if it already existed (duplicate).
    """
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        now_iso = datetime.now(timezone.utc).isoformat()
        ctx_json = json.dumps(list(context_chunks), ensure_ascii=False)
        cur.execute(
            """
            INSERT OR IGNORE INTO eval_queue (
                interaction_id, question, answer, context_json, created_at, processed_at
            ) VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (interaction_id, question, answer, ctx_json, now_iso),
        )
        con.commit()
        return bool(cur.rowcount and cur.rowcount > 0)
    finally:
        con.close()


