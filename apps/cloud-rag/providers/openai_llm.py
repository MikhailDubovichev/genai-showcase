"""
OpenAI chat LLM provider for the Cloud RAG service (factory wrapper).

This module provides a small factory function to construct a LangChain-compatible
chat model backed by OpenAI. The goal is to mirror the Nebius provider wiring
while keeping provider-specific details encapsulated in the providers package.

Behavior and responsibilities:
- Read generation parameters from CONFIG["llm"] (model, temperature, top_p).
- Read `OPENAI_API_KEY` from the environment (env comes from environment).
- Guard imports so that OpenAI packages are only required when this provider is
  actually selected (avoids hard dependency when using Nebius).
- Never log secrets; log only provider selection and model identity at INFO.

The function returns a chat model instance that satisfies the same interface the
rest of the application expects. If the required packages are missing, a
RuntimeError is raised with a clear instruction to install `langchain-openai`
and `openai`. If the API key is missing, a RuntimeError is raised with guidance
to set `OPENAI_API_KEY`.

This design follows the separation-of-concerns principle: configuration and
credential handling remain centralized, and the rest of the codebase depends on
provider-agnostic factories rather than concrete vendor classes. This makes it
easy to switch providers without touching business logic or API layers.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict


logger = logging.getLogger(__name__)


def build_openai_chat_llm(cfg: Dict[str, Any]) -> Any:
    """
    Build and return an OpenAI chat model compatible with the RAG chain.

    This factory reads required generation parameters from the provided `cfg`
    mapping (which should be `CONFIG["llm"]`) and constructs a LangChain
    `ChatOpenAI` instance. The function requires `OPENAI_API_KEY` to be present
    in the process environment and does not cache or expose the secret. Imports
    are guarded to avoid hard dependency when OpenAI is not selected.

    Args:
        cfg (Dict[str, Any]): The LLM configuration section from CONFIG. Expected
            keys include "model" (str), "temperature" (float), and "top_p" (float).

    Returns:
        Any: A ChatOpenAI instance ready for use by the RAG chain.

    Raises:
        RuntimeError: If `OPENAI_API_KEY` is missing or if the `langchain-openai`
        (and its `openai` dependency) package is not installed, with installation
        hints that follow best practice.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before running the RAG service."
        )

    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - import-time failure path
        raise RuntimeError(
            "Install OpenAI provider packages: pip install langchain-openai openai"
        ) from exc

    model = str(cfg.get("model", "gpt-4o-mini"))
    temperature = float(cfg.get("temperature", 0.0))
    top_p = float(cfg.get("top_p", 0.95))

    logger.info("Initializing OpenAI Chat model: %s", model)
    # ChatOpenAI reads OPENAI_API_KEY from the environment.
    # Keep arguments minimal and aligned with our Nebius usage.
    return ChatOpenAI(model=model, temperature=temperature, top_p=top_p)


