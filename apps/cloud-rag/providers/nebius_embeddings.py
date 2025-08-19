"""
Nebius embeddings provider for Cloud RAG seeding and retrieval.

This module exposes a single helper, `get_embeddings()`, which constructs the
LangChain NebiusEmbeddings instance used to embed documents during index
seeding and for retrieval. We read the model name and credentials from the
app's configuration and environment so that deployments can switch models or
rotate credentials without changing code. The function validates that the
NEBIUS_API_KEY is present and raises a clear error if not, guiding the user to
export the key before running the seeding script or the application code that
depends on embeddings.
"""

from __future__ import annotations

import logging
from typing import Any

from config import CONFIG, ENV

logger = logging.getLogger(__name__)


def get_embeddings() -> Any:
    """
    Construct and return a NebiusEmbeddings instance using configured model and env.

    Returns:
        Any: A NebiusEmbeddings instance suitable for passing to LangChain FAISS
        loaders and vector store constructors.

    Raises:
        RuntimeError: If NEBIUS_API_KEY is missing or if the integration package
        is not installed, with a clear instruction on how to install.
    """
    model = (CONFIG.get("embeddings", {}) or {}).get("name", "BAAI/bge-en-icl")  # type: ignore[assignment]
    api_key = ENV.get("NEBIUS_API_KEY", "")  # type: ignore[assignment]

    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Please export NEBIUS_API_KEY before seeding the index."
        )

    try:
        from langchain_nebius import NebiusEmbeddings  # type: ignore
    except ImportError as exc:  # pragma: no cover - import-time failure path
        raise RuntimeError(
            "Install Nebius provider: pip install langchain-nebius"
        ) from exc

    # Ensure runtime environment is visible to the provider implementation
    import os as _os
    if _os.environ.get("NEBIUS_API_KEY") != api_key and api_key:
        _os.environ["NEBIUS_API_KEY"] = api_key

    logger.info("Initializing NebiusEmbeddings with model: %s", model)
    return NebiusEmbeddings(model=model)


