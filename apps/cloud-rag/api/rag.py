"""
Cloud RAG API endpoints for retrieval-augmented answering.

This module defines the HTTP interface for the cloud-side RAG flow. It mirrors
the edge server's style by exposing a FastAPI router with verbose docstrings,
lightweight module-level logging, and a clear separation between transport
logic and application orchestration. The main endpoint, POST /rag/answer,
accepts a structured request with the user's question, an interaction id for
tracing, and an optional topK value that controls retrieval depth. The handler
invokes the composable RAG chain which retrieves context, renders the strict
JSON prompt, calls the LLM, and validates the result against the agreed schema
so that clients can always rely on a stable response shape.

Implementation details are intentionally kept minimal at this stage. Provider
construction is delegated to small factory helpers under providers/, while the
RAG logic is encapsulated in rag/chain.py. This modular design enables us to
iterate on each layer independently—embedding model selection, prompt tuning,
retriever configuration—without coupling HTTP concerns to RAG internals.
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import CONFIG
from providers.nebius_embeddings import get_embeddings
from providers.nebius_llm import get_llm
from providers.langfuse import create_trace, update_trace_metadata
from rag.chain import run_chain
from services.eval_queue import enqueue_eval_item


logger = logging.getLogger(__name__)
router = APIRouter()


class RAGRequest(BaseModel):
    """
    Request payload for the retrieval-augmented answer endpoint.

    Fields:
        question (str): The user's question to be grounded with retrieved context.
        interactionId (str): Unique identifier for this interaction, used for tracing and
            downstream analytics or A/B testing.
        topK (int): Number of context chunks to retrieve. Defaults to 3. Larger values can
            improve recall at the cost of latency and token usage.
    """

    question: str = Field(..., description="User question to answer with RAG")
    interactionId: str = Field(..., description="Unique interaction identifier")
    topK: int = Field(3, description="Number of chunks to retrieve")


@router.post("/rag/answer")
def answer_rag(req: RAGRequest) -> JSONResponse:
    """
    Answer a question using the Cloud RAG pipeline, create/update a LangFuse trace, and return
    validated JSON.

    This endpoint orchestrates the full retrieve‑then‑read flow and adds minimal observability.
    It first ensures a LangFuse trace exists whose id equals the request's `interactionId`.
    The handler then resolves configuration (FAISS index path, model), constructs the embeddings
    and LLM providers, and executes the RAG chain which retrieves context, renders the strict
    JSON prompt, calls the model, and validates the output against the `EnergyEfficiencyResponse`
    schema. On success, we compute lightweight telemetry (latency_ms, model name, retrieved_k,
    json_valid=True, placeholder token counts = None) and upsert these fields onto the same trace.
    On failure, we capture latency, set json_valid=False, include the exception type, and mark
    http_status=500. All LangFuse interactions are best‑effort and become no‑ops when the client
    is not configured or the library is missing; the API response contract remains unchanged.

    Returns:
        JSONResponse: A JSON payload conforming to `EnergyEfficiencyResponse` on success.
        On errors, returns a concise error JSON with HTTP 500 while attempting to update the
        trace with diagnostic metadata.
    """
    start_ts = time.monotonic()
    try:
        # Create a LangFuse trace for this request (no-op if disabled/missing)
        create_trace(
            trace_id=req.interactionId,
            name="rag.answer",
            metadata={"endpoint": "/api/rag/answer"},
        )

        faiss_dir = str((CONFIG.get("paths", {}) or {}).get("faiss_index_dir", "apps/cloud-rag/faiss_index"))
        embeddings = get_embeddings()
        llm = get_llm()
        result_json = run_chain(
            question=req.question,
            interaction_id=req.interactionId,
            top_k=req.topK,
            faiss_dir=faiss_dir,
            embeddings=embeddings,
            llm=llm,
        )
        obj = json.loads(result_json)
        latency_ms = int((time.monotonic() - start_ts) * 1000)
        model = (CONFIG.get("llm", {}) or {}).get("model", "unknown")
        retrieved_k = len(obj.get("content", [])) if isinstance(obj, dict) else 0
        update_trace_metadata(
            req.interactionId,
            {
                "latency_ms": latency_ms,
                "model": model,
                "tokens_prompt": None,
                "tokens_completion": None,
                "json_valid": True,
                "retrieved_k": retrieved_k,
                "http_status": 200,
            },
        )

        # Minimal, best-effort enqueue for offline evaluation (do not affect response)
        try:
            question = req.question
            answer_text = str(obj.get("message", "")) if isinstance(obj, dict) else ""
            context_chunks = []
            if isinstance(obj, dict):
                raw_content = obj.get("content", [])
                if isinstance(raw_content, list):
                    for it in raw_content:
                        if isinstance(it, dict) and "chunk" in it:
                            context_chunks.append(str(it.get("chunk", "")))
                            if len(context_chunks) >= 3:
                                break
            db_path = str((CONFIG.get("paths", {}) or {}).get("db_path", "data/db.sqlite"))
            enqueue_eval_item(
                db_path=db_path,
                interaction_id=req.interactionId,
                question=question,
                answer=answer_text,
                context_chunks=context_chunks,
            )
        except Exception:
            pass

        return JSONResponse(obj)
    except Exception as e:  # pragma: no cover - broad surface area during MVP
        logger.error("RAG pipeline error: %s", e)
        latency_ms = int((time.monotonic() - start_ts) * 1000)
        update_trace_metadata(
            req.interactionId,
            {
                "latency_ms": latency_ms,
                "model": (CONFIG.get("llm", {}) or {}).get("model", "unknown"),
                "tokens_prompt": None,
                "tokens_completion": None,
                "json_valid": False,
                "retrieved_k": req.topK,
                "error_type": type(e).__name__,
                "http_status": 500,
            },
        )
        return JSONResponse(
            {"message": "RAG pipeline error", "type": "error", "detail": str(e)},
            status_code=500,
        )


