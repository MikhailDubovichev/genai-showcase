"""
Cloud feedback ingestion API with deduplication and SQLite persistence.

This module adds a batched POST endpoint that accepts edge-collected feedback
and persists it to a local SQLite database for downstream processing. The
endpoint is intentionally simple and idempotent: a PRIMARY KEY on `feedback_id`
ensures that resubmitted items are treated as duplicates rather than inserted
again, making the call safe to retry. We use FastAPI for routing and Pydantic
for request validation, which will return 422 for invalid payloads. Logging is
kept minimal: an INFO line for accepted/duplicate counts, and an ERROR line for
unexpected failures. No external dependencies are required beyond the standard
library `sqlite3` for database access and the existing application configuration.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import CONFIG
from services.feedback_store import init_db, upsert_feedback_batch
from providers.langfuse import add_user_feedback_score


logger = logging.getLogger(__name__)
router = APIRouter()


class FeedbackItem(BaseModel):
    """
    A single feedback record submitted by the edge client.

    Fields are intentionally permissive for the MVP; Pydantic validation ensures
    basic presence and typing, while the database layer enforces deduplication
    via a PRIMARY KEY on `feedback_id` to guarantee idempotent ingestion.
    """

    feedback_id: str = Field(..., description="Deterministic 32-hex identifier")
    interactionId: str = Field(..., description="Associated interaction identifier")
    score: int = Field(..., description="+1 for positive, -1 for negative")
    label: str = Field(..., description="Human label e.g., positive/negative")
    comment: str | None = Field("", description="Optional free-form comment")
    created_at: str = Field(..., description="ISO 8601 timestamp from edge")


class FeedbackBatch(BaseModel):
    """
    Batched feedback submission format.
    """

    items: List[FeedbackItem]


# Initialize database on import
db_path = str((CONFIG.get("paths", {}) or {}).get("db_path", "data/db.sqlite"))
init_db(db_path)


@router.post("/feedback/sync")
def sync_feedback(batch: FeedbackBatch) -> JSONResponse:
    """
    Persist a batch of feedback items with deduplication by feedback_id.

    If the batch is empty, returns zeros for both accepted and duplicates. Otherwise,
    inserts items within a single transaction using PRIMARY KEY conflict handling to
    count duplicates. Any unexpected error results in a 500 with a concise message,
    and the exception is logged for observability.
    """
    try:
        if not batch.items:
            return JSONResponse({"accepted": 0, "duplicates": 0})
        items_dicts = [item.model_dump() for item in batch.items]
        accepted, duplicates, accepted_ids = upsert_feedback_batch(db_path, items_dicts)

        # Attempt to add user_feedback score for newly accepted items only
        id_to_item = {it["feedback_id"]: it for it in items_dicts}
        scored_ok = 0
        for fid in accepted_ids:
            it = id_to_item.get(fid)
            if not it:
                continue
            try:
                add_user_feedback_score(
                    trace_id=str(it.get("interactionId", "")),
                    score_value=float(int(it.get("score", 0))),
                    comment=(it.get("comment") or ""),
                    name="user_feedback",
                )
                scored_ok += 1
            except Exception:
                # Best-effort: never fail ingestion due to scoring issues
                pass

        logger.info(
            "feedback_sync accepted=%s duplicates=%s scored=%s",
            accepted,
            duplicates,
            scored_ok,
        )
        return JSONResponse({"accepted": accepted, "duplicates": duplicates})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("feedback_sync failed: %s", exc)
        return JSONResponse(
            {"message": "feedback sync error", "type": "error", "detail": str(exc)},
            status_code=500,
        )


