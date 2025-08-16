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
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import CONFIG
from providers.embeddings import get_embeddings
from providers.nebius_llm import get_llm
from rag.chain import run_chain


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
    Answer a question using the Cloud RAG pipeline and return validated JSON.

    The handler builds the embeddings and LLM providers using the Cloud configuration,
    resolves the FAISS index directory from CONFIG paths, and executes the end-to-end
    RAG chain. The chain guarantees that the output conforms to the
    EnergyEfficiencyResponse schema; upon success we return the validated JSON object.
    If any error occurs (missing credentials, index missing, validation failure),
    a concise error JSON with HTTP 500 is returned to the client.

    Returns:
        JSONResponse: A JSON payload conforming to EnergyEfficiencyResponse on success, or an
        error object on failure.
    """
    try:
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
        return JSONResponse(json.loads(result_json))
    except Exception as e:  # pragma: no cover - broad surface area during MVP
        logger.error("RAG pipeline error: %s", e)
        return JSONResponse(
            {"message": "RAG pipeline error", "type": "error", "detail": str(e)},
            status_code=500,
        )


