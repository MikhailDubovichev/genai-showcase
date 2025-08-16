"""
Nebius chat LLM provider for the Cloud RAG service.

This module provides a small factory function, `get_llm()`, that returns a chat
model instance backed by the official LangChain Nebius integration. The goal is
to keep provider wiring centralized and minimal: we read necessary settings from
the Cloud configuration and environment variables, validate that credentials are
present, and construct a model with sensible defaults for RAG usage. By keeping
this logic in a dedicated providers package, we avoid scattering environment
lookups or integration details across the application code, which improves
testability and separation of concerns.

The function enforces that a `NEBIUS_API_KEY` is available at runtime. If the
key is missing, a clear RuntimeError is raised with instructions to export the
variable. On success, the integration class `ChatNebius` is imported from the
`langchain-nebius` package. If the package is not installed in the current
environment, we raise a concise RuntimeError with an installation hint so that
developers can resolve the dependency explicitly. The returned model is
configured with conservative generation parameters (temperature and top_p) that
are appropriate for JSON-constrained RAG answers.
"""

from __future__ import annotations

import logging
from typing import Any

from config import CONFIG, ENV

logger = logging.getLogger(__name__)


def get_llm() -> Any:
    """
    Construct and return a Nebius chat model for use in the RAG chain.

    This function reads the target chat model name from the Cloud configuration
    under `llm.model`, defaulting to "Qwen/Qwen3-30B-A3B-fast" when not set.
    It also verifies that `NEBIUS_API_KEY` is present in the environment mapping
    (ENV). If the key is missing, a RuntimeError is raised with a clear message
    directing the user to export the variable. On success, the LangChain Nebius
    provider class `ChatNebius` is imported and instantiated. We ensure the
    process environment contains `NEBIUS_API_KEY` so the provider can read it.

    Returns:
        Any: A ChatNebius instance configured with the selected model and
        conservative generation parameters suitable for structured JSON output.

    Raises:
        RuntimeError: If `NEBIUS_API_KEY` is missing or if the Nebius provider
        package is not installed in the environment.
    """
    nebius_api_key = ENV.get("NEBIUS_API_KEY", "")  # type: ignore[assignment]
    model = (CONFIG.get("llm", {}) or {}).get("model", "Qwen/Qwen3-30B-A3B-fast")  # type: ignore[assignment]

    if not nebius_api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Please export NEBIUS_API_KEY before running the RAG endpoint."
        )

    try:
        from langchain_nebius import ChatNebius  # type: ignore
    except ImportError as exc:  # pragma: no cover - import-time failure path
        raise RuntimeError(
            "Install Nebius provider: pip install langchain-nebius"
        ) from exc

    # Ensure the provider can read credentials from the process environment
    import os as _os
    if _os.environ.get("NEBIUS_API_KEY") != nebius_api_key and nebius_api_key:
        _os.environ["NEBIUS_API_KEY"] = nebius_api_key

    logger.info("Initializing Nebius Chat model: %s", model)
    return ChatNebius(model=model, temperature=0.2, top_p=0.95)


