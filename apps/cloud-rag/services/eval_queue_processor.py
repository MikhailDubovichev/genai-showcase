"""
Offline evaluation queue processor for Cloud RAG.

This module reads pending items from the SQLite-backed evaluation queue, computes
an LLM-as-judge relevance score for each item, and marks them as processed. The
intent is to remove any evaluation latency from the request path: the /api/rag/answer
endpoint simply enqueues artifacts (interaction id, question, answer, and top
context chunks), and this processor is run separately to attach scores later.
We use stdlib-only components: sqlite3 for persistence, json for serialization,
and datetime for ISO timestamps. When LangFuse is configured, the processor will
attempt to record a score on the corresponding trace and add a small metadata
payload; failures are swallowed so ingestion is robust. This file is safe to run
manually or as part of a background job.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from config import CONFIG
from eval.relevance_evaluator import evaluate_relevance
# Reuse existing helper; alias to match generic naming used here
from providers.langfuse import (
    add_user_feedback_score as add_trace_score,  # type: ignore
    update_trace_metadata,
)


logger = logging.getLogger(__name__)


def now_iso_utc() -> str:
    """
    Return the current UTC time in ISO 8601 format.

    This small utility centralizes timestamp generation to ensure consistent,
    timezone-aware values across queue operations. Using UTC prevents local
    clock/timezone issues from affecting ordering or comparisons.
    """
    return datetime.now(timezone.utc).isoformat()


def open_conn(db_path: str):
    """
    Open a sqlite3 connection to the given database path.

    The caller is responsible for closing the returned connection. This thin
    wrapper exists for readability and to keep connection creation consistent.
    """
    return sqlite3.connect(db_path)


def fetch_pending(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch pending evaluation queue rows up to the provided limit.

    Rows are selected in ascending id order where processed_at IS NULL. The function
    parses `context_json` into a Python list of strings best-effort (invalid JSON
    yields an empty list) and returns a list of dictionaries with keys: id,
    interaction_id, question, answer, and context_chunks.
    """
    con = open_conn(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, interaction_id, question, answer, context_json
            FROM eval_queue
            WHERE processed_at IS NULL
            ORDER BY id ASC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = []
        for rid, tid, q, a, ctx in cur.fetchall():
            chunks: List[str] = []
            try:
                raw = json.loads(ctx) if isinstance(ctx, str) else []
                if isinstance(raw, list):
                    chunks = [str(s) for s in raw]
            except Exception:
                chunks = []
            rows.append(
                {
                    "id": int(rid),
                    "interaction_id": str(tid),
                    "question": str(q or ""),
                    "answer": str(a or ""),
                    "context_chunks": chunks,
                }
            )
        return rows
    finally:
        con.close()


def mark_processed(db_path: str, row_ids: List[int], processed_at: str) -> None:
    """
    Mark the given queue row ids as processed with the provided timestamp.

    Uses a single UPDATE statement with an IN clause. If `row_ids` is empty,
    the function returns immediately. Any sqlite3 errors are allowed to bubble
    to the caller so they can be logged appropriately by the orchestrator.
    """
    if not row_ids:
        return
    con = open_conn(db_path)
    try:
        cur = con.cursor()
        placeholders = ",".join("?" for _ in row_ids)
        sql = f"UPDATE eval_queue SET processed_at=? WHERE id IN ({placeholders})"
        cur.execute(sql, (processed_at, *row_ids))
        con.commit()
    finally:
        con.close()


def process_pending_eval_items(limit: int = 50) -> Dict[str, int]:
    """
    Process up to `limit` pending eval queue items and return a small summary.

    The processor reads pending rows from the eval queue, computes a relevance score
    for each item via `evaluate_relevance(...)`, and best-effort records that score
    in LangFuse using the interaction id as trace id. To aid analysis, the processor
    also upserts a small metadata payload with the numeric `relevance` and a flag
    indicating that the evaluation was performed offline. Any failures in scoring or
    metadata updates are swallowed to keep the job robust. Successfully handled row
    ids are then marked with a `processed_at` timestamp so they are not re-processed.

    Args:
        limit (int): Maximum number of items to process in one run.

    Returns:
        Dict[str, int]: A summary with keys: `fetched` and `processed`.
    """
    db_path = str((CONFIG.get("paths", {}) or {}).get("db_path", "data/db.sqlite"))
    rows = fetch_pending(db_path, limit)
    processed_ids: List[int] = []

    for r in rows:
        try:
            score = float(
                evaluate_relevance(
                    r.get("question", ""),
                    list(r.get("context_chunks", []))[:3],
                    r.get("answer", ""),
                )
            )
        except Exception:
            score = 0.0
        # Clamp to [0, 1]
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0

        # Best-effort score + metadata to LangFuse
        try:
            add_trace_score(trace_id=str(r.get("interaction_id", "")), name="relevance", score_value=score, comment="offline-queue")
        except Exception:
            pass
        try:
            update_trace_metadata(str(r.get("interaction_id", "")), {"relevance": score, "eval_offline": True})
        except Exception:
            pass

        processed_ids.append(int(r["id"]))

    if processed_ids:
        try:
            mark_processed(db_path, processed_ids, now_iso_utc())
        except Exception:
            # If marking fails, we’ll try again next run; keep job robust
            logger.warning("failed to mark processed for ids=%s", processed_ids)

    return {"fetched": len(rows), "processed": len(processed_ids)}


if __name__ == "__main__":
    print(json.dumps(process_pending_eval_items(), indent=2))


