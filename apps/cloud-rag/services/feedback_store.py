"""
SQLite-backed feedback store with deduplication by feedback_id.

This module provides a tiny persistence layer for feedback ingestion using SQLite,
an embedded, serverless database engine. "SQL" stands for Structured Query Language,
and "lite" emphasizes the lightweight, self-contained nature of SQLite. We store
feedback entries in a single table and enforce idempotency by declaring the
`feedback_id` column as PRIMARY KEY. Idempotent means that repeated submissions of
the same inputs have the same effect as a single submission, allowing safe retries
without creating duplicates. The API offered here is intentionally small: a database
initializer and a batch upsert function. Both rely solely on Python's standard library
(`sqlite3`, `os`, `datetime`, `typing`, and `pathlib`) to keep deployment simple.
"""

from __future__ import annotations

import datetime as _dt
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


def init_db(db_path: str) -> None:
    """
    Initialize the feedback SQLite database and ensure the feedback table exists.

    This function creates parent directories as needed and opens a connection to the
    SQLite database file at `db_path`. It then issues a `CREATE TABLE IF NOT EXISTS`
    statement to define the `feedback` table with a PRIMARY KEY on `feedback_id` to
    guarantee deduplication and idempotency for repeated inserts. The table captures
    basic fields needed for audit and analytics. The connection is committed and
    closed before returning. Errors propagate to the caller for visibility.

    Args:
        db_path (str): Filesystem path to the SQLite database file.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id   TEXT PRIMARY KEY,
                interaction_id TEXT,
                score          INTEGER,
                label          TEXT,
                comment        TEXT,
                created_at     TEXT,
                inserted_at    TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def upsert_feedback_batch(db_path: str, items: List[Dict]) -> Tuple[int, int]:
    """
    Insert a batch of feedback items with ON CONFLICT DO NOTHING semantics.

    Opens a single SQLite connection and transaction to write the provided items to
    the `feedback` table. For each item, attempts an INSERT and counts the row as
    accepted if it was inserted; if the `feedback_id` already exists, the operation
    is ignored and counted as a duplicate. The function returns a pair (accepted,
    duplicates). A best-effort `inserted_at` timestamp (UTC ISO 8601) is recorded.

    Args:
        db_path (str): Path to the SQLite database file.
        items (List[Dict]): Feedback items to upsert. Each should include:
            - feedback_id (str)
            - interactionId (str)
            - score (int)
            - label (str)
            - comment (str or None)
            - created_at (str)

    Returns:
        Tuple[int, int]: (accepted_count, duplicate_count)
    """
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        accepted = 0
        duplicates = 0
        now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
        for it in items:
            try:
                cur.execute(
                    """
                    INSERT INTO feedback (
                        feedback_id, interaction_id, score, label, comment, created_at, inserted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(it.get("feedback_id")),
                        str(it.get("interactionId")),
                        int(it.get("score")),
                        str(it.get("label")),
                        ("" if it.get("comment") is None else str(it.get("comment"))),
                        str(it.get("created_at")),
                        now_iso,
                    ),
                )
                accepted += 1
            except sqlite3.IntegrityError:
                duplicates += 1
        con.commit()
        return accepted, duplicates
    finally:
        con.close()


