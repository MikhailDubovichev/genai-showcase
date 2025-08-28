"""
OpenAI embeddings provider for the Cloud RAG service (factory wrapper).

This module exposes a single factory function that constructs a LangChain
`OpenAIEmbeddings` instance using configuration read from `CONFIG["embeddings"]`
and credentials read from the process environment. The function mirrors the
Nebius embeddings helper, keeping provider-specific details isolated so the rest
of the application can remain provider-agnostic.

Key behaviors:
- Read model name from CONFIG["embeddings"]["name"].
- Require `OPENAI_API_KEY` via environment variable (env comes from environment).
- Guard imports to avoid hard dependency unless OpenAI provider is selected.
- Avoid logging secrets; only log the chosen model at INFO level.

On import failures or missing credentials, the function raises a RuntimeError
with clear guidance on how to proceed, enabling fail-fast behavior at startup.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict


logger = logging.getLogger(__name__)


def build_openai_embeddings(cfg: Dict[str, Any]) -> Any:
    """
    Build and return an OpenAIEmbeddings instance using the provided config.

    This factory reads the embedding model name from `cfg` (expected to be
    `CONFIG["embeddings"]`) and constructs a LangChain `OpenAIEmbeddings`
    instance. The `OPENAI_API_KEY` must be present in the process environment.
    If the required packages are not installed, a RuntimeError is raised with a
    clear installation hint.

    Args:
        cfg (Dict[str, Any]): The embeddings configuration section from CONFIG.
            Expected key: "name" (str), the embedding model identifier.

    Returns:
        Any: An OpenAIEmbeddings instance suitable for FAISS seeding and retrieval.

    Raises:
        RuntimeError: If `OPENAI_API_KEY` is missing or if `langchain-openai`
        (and its dependency `openai`) is not installed. The error message will
        include a concise installation suggestion.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before seeding or running the service."
        )

    try:
        from langchain_openai import OpenAIEmbeddings  # type: ignore
    except ImportError as exc:  # pragma: no cover - import-time failure path
        raise RuntimeError(
            "Install OpenAI provider packages: pip install langchain-openai openai"
        ) from exc

    model = str(cfg.get("name", "text-embedding-3-small"))
    logger.info("Initializing OpenAIEmbeddings with model: %s", model)
    return OpenAIEmbeddings(model=model)


