"""
Provider factory for chat LLM and embeddings (Nebius/OpenAI switch).

This module centralizes provider selection for the Cloud RAG service. It exposes
two functions—`get_chat_llm` and `get_embeddings`—that decide which concrete
provider implementation to instantiate based on the runtime configuration
(`CONFIG`). The design keeps provider imports guarded so that optional packages
are only required when their provider is selected.

Contract and behavior:
- `get_chat_llm(config)` reads `config["llm"]["provider"]` in {"nebius", "openai"}.
- `get_embeddings(config)` reads `config["embeddings"]["provider"]` similarly.
- On unknown provider values, the functions raise a `ValueError` with a clear
  message to aid debugging and configuration hygiene.
- This module does not validate secrets; that is handled centrally in the config
  loader (M11 Step 1). We only construct the provider objects assuming env is
  already validated.

The factory maintains separation of concerns and supports adding more providers
without changing business logic or API contracts in other parts of the app.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_chat_llm(config: Dict[str, Any]) -> Any:
    """
    Build a chat LLM instance according to the configured provider.

    This function inspects `config["llm"]["provider"]` to decide which vendor
    implementation to construct. Supported values are "nebius" and "openai"
    (case-insensitive). Imports for optional providers are guarded to avoid
    requiring packages that are not needed for the current selection.

    Args:
        config (Dict[str, Any]): The global CONFIG mapping (not just llm section).

    Returns:
        Any: A chat model instance compatible with the RAG chain.

    Raises:
        ValueError: If the provider value is unsupported.
        RuntimeError: If required packages for the selected provider are missing.
    """
    llm_cfg = (config.get("llm", {}) or {})
    provider = str(llm_cfg.get("provider", "nebius")).strip().lower()
    logger.info("Provider selection (LLM): %s", provider)

    if provider == "nebius":
        from .nebius_llm import get_llm as build_nebius_llm  # local import to avoid hard dep

        return build_nebius_llm()
    if provider == "openai":
        from .openai_llm import build_openai_chat_llm  # local import to avoid hard dep

        return build_openai_chat_llm(llm_cfg)

    raise ValueError(f"Unsupported llm provider: {provider}")


def get_embeddings(config: Dict[str, Any]) -> Any:
    """
    Build an embeddings instance according to the configured provider.

    This function inspects `config["embeddings"]["provider"]` to decide which
    vendor implementation to construct. Supported values are "nebius" and
    "openai" (case-insensitive). Imports are guarded so that optional packages
    need not be installed unless their provider is selected.

    Args:
        config (Dict[str, Any]): The global CONFIG mapping (not just embeddings section).

    Returns:
        Any: An embeddings instance suitable for FAISS seeding and retrieval.

    Raises:
        ValueError: If the provider value is unsupported.
        RuntimeError: If required packages for the selected provider are missing.
    """
    emb_cfg = (config.get("embeddings", {}) or {})
    provider = str(emb_cfg.get("provider", "nebius")).strip().lower()
    logger.info("Provider selection (Embeddings): %s", provider)

    if provider == "nebius":
        from .nebius_embeddings import get_embeddings as build_nebius_embeddings  # local import

        return build_nebius_embeddings()
    if provider == "openai":
        from .openai_embeddings import build_openai_embeddings  # local import

        return build_openai_embeddings(emb_cfg)

    raise ValueError(f"Unsupported embeddings provider: {provider}")


