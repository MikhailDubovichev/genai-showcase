"""
Configuration module for the Cloud RAG service (environment loader and runtime config).

This module centralizes how the Cloud RAG app reads configuration from the process environment
and exposes a simple, dictionary‑like structure that the rest of the application can depend on.
The goal for this milestone is intentionally narrow: detect optional Langfuse credentials for
observability and define server host/port with a small, explicit schema. We avoid any third‑party
libraries such as python‑dotenv to keep the footprint minimal, assuming that environments like
Docker Compose or deployment platforms will inject variables directly.

Two top‑level objects are exported:
- ENV: a mapping of raw environment variables relevant to this app. The values are read once at
  import time to make behavior deterministic and easy to reason about within a single process.
- CONFIG: a higher‑level mapping that expresses structured configuration for the server and
  integrations. The CONFIG object is what application code should prefer to read for runtime
  parameters, such as which TCP port to bind. Defaults are chosen to support local development
  without any extra setup.

In addition, we provide a small helper function, langfuse_present(), which indicates whether both
required Langfuse credentials are available. This allows later code paths to conditionally enable
Langfuse without scattering environment checks throughout the codebase. The philosophy mirrors the
edge server pattern: a single import site provides a consistent configuration surface that callers
can rely upon, improving separation of concerns and testability.
"""

# TODO: I want to refactor this file and keep all the variables but API keys in config.json. So config variables (besides
# API keys) should be read from config.json.

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Mapping
import logging


def _read_env() -> Dict[str, str]:
    """
    Read process environment variables relevant to the Cloud RAG app.

    This function extracts only the keys we care about for this milestone and provides
    well‑defined defaults. Reading once during module import ensures that configuration
    remains stable during the lifetime of the process, which makes it easier to reason
    about behavior and aligns with common Twelve‑Factor App practices.

    Returns:
        Dict[str, str]: A dictionary containing keys for Langfuse integration and any
        other environment‑scoped values needed by the application. Missing variables
        are replaced with safe defaults that enable local development out of the box.
    """
    return {
        "LANGFUSE_PUBLIC_KEY": os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        "LANGFUSE_SECRET_KEY": os.environ.get("LANGFUSE_SECRET_KEY", ""),
        "LANGFUSE_HOST": os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        # Server port is read separately in CONFIG to ensure integer conversion and defaults
        "CLOUD_RAG_PORT": os.environ.get("CLOUD_RAG_PORT", "8000"),
        # Nebius embeddings/API credentials
        "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
    }


ENV: Mapping[str, str] = _read_env()


def _load_json_config() -> Dict[str, object]:
    """
    Load optional JSON configuration from the app's config directory.

    This function attempts to read `config.json` that lives alongside this module. The JSON file
    provides overrides and additional structured settings such as LLM base URL, embeddings model
    name, and well-known paths used by scripts and services. Failure to read the file (missing,
    malformed, or permission-related issues) results in an empty mapping so that sane defaults
    remain in effect for local development.

    Returns:
        Dict[str, object]: The parsed JSON mapping or an empty dictionary if the file is absent or
        unreadable.
    """
    cfg_path = Path(__file__).parent / "config.json"
    try:
        if cfg_path.exists():
            return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        # Intentionally ignore and fall back to defaults
        pass
    return {}


def _build_config(env: Mapping[str, str]) -> Dict[str, object]:
    """
    Construct the high‑level CONFIG object consumed by the application.

    This function translates raw environment values into a structured configuration mapping
    that the service can use at runtime. It enforces simple type conversions (such as turning
    the port into an integer) and organizes related settings into nested dictionaries. The
    shape mirrors the edge server style and is intentionally conservative so it can be evolved
    safely as the Cloud service grows more features in later milestones.

    Args:
        env (Mapping[str, str]): The raw environment variables previously captured in ENV.

    Returns:
        Dict[str, object]: A dictionary with at least two top‑level keys: "server" for host/port
        and "langfuse" for integration credentials and endpoint. Callers should prefer this object
        over reading os.environ directly to keep configuration logic in one place.
    """
    port_str = env.get("CLOUD_RAG_PORT", "8000")
    try:
        port_val = int(port_str)
    except ValueError:
        port_val = 8000

    # Base defaults
    cfg: Dict[str, object] = {
        "server": {
            "host": "0.0.0.0",
            "port": port_val,
        },
        "langfuse": {
            "public_key": env.get("LANGFUSE_PUBLIC_KEY", ""),
            "secret_key": env.get("LANGFUSE_SECRET_KEY", ""),
            "host": env.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        },
        "llm": {
            "base_url": "https://api.studio.nebius.ai/v1",
            "timeout": 30,
        },
        "embeddings": {
            "name": "BAAI/bge-en-icl",
        },
        "paths": {
            "faiss_index_dir": "apps/cloud-rag/faiss_index",
            "seed_data_dir": "apps/cloud-rag/rag/data/seed",
        },
    }

    # Overlay JSON config if present
    json_cfg = _load_json_config()
    if isinstance(json_cfg, dict):
        llm_cfg = json_cfg.get("llm", {})
        if isinstance(llm_cfg, dict):
            cfg_llm = cfg.get("llm", {})  # type: ignore[assignment]
            if isinstance(cfg_llm, dict):
                if "base_url" in llm_cfg:
                    cfg_llm["base_url"] = str(llm_cfg.get("base_url"))
                if "timeout" in llm_cfg:
                    try:
                        cfg_llm["timeout"] = int(llm_cfg.get("timeout"))
                    except Exception:
                        pass
                # Provider toggle (added in M11 Step 1)
                if "provider" in llm_cfg:
                    cfg_llm["provider"] = str(llm_cfg.get("provider"))
                cfg["llm"] = cfg_llm

        emb_cfg = json_cfg.get("embeddings", {})
        if isinstance(emb_cfg, dict):
            cfg_emb = cfg.get("embeddings", {})  # type: ignore[assignment]
            if isinstance(cfg_emb, dict) and "name" in emb_cfg:
                cfg_emb["name"] = str(emb_cfg.get("name"))
                # Provider toggle (added in M11 Step 1)
                if "provider" in emb_cfg:
                    cfg_emb["provider"] = str(emb_cfg.get("provider"))
                cfg["embeddings"] = cfg_emb

        paths_cfg = json_cfg.get("paths", {})
        if isinstance(paths_cfg, dict):
            cfg_paths = cfg.get("paths", {})  # type: ignore[assignment]
            if isinstance(cfg_paths, dict):
                if "faiss_index_dir" in paths_cfg:
                    cfg_paths["faiss_index_dir"] = str(paths_cfg.get("faiss_index_dir"))
                if "seed_data_dir" in paths_cfg:
                    cfg_paths["seed_data_dir"] = str(paths_cfg.get("seed_data_dir"))
                cfg["paths"] = cfg_paths

        # NEW: retrieval section overlay (mode, k's, fusion alpha, etc.)
        retrieval_cfg = json_cfg.get("retrieval", {})
        if isinstance(retrieval_cfg, dict):
            cfg_retr: Dict[str, object] = {}
            if "mode" in retrieval_cfg:
                cfg_retr["mode"] = str(retrieval_cfg.get("mode"))
            if "semantic_k" in retrieval_cfg:
                try:
                    cfg_retr["semantic_k"] = int(retrieval_cfg.get("semantic_k"))
                except Exception:
                    pass
            if "keyword_k" in retrieval_cfg:
                try:
                    cfg_retr["keyword_k"] = int(retrieval_cfg.get("keyword_k"))
                except Exception:
                    pass
            if "default_top_k" in retrieval_cfg:
                try:
                    cfg_retr["default_top_k"] = int(retrieval_cfg.get("default_top_k"))
                except Exception:
                    pass
            if "allow_general_knowledge" in retrieval_cfg:
                try:
                    cfg_retr["allow_general_knowledge"] = bool(retrieval_cfg.get("allow_general_knowledge"))
                except Exception:
                    pass
            fusion_cfg = retrieval_cfg.get("fusion", {})
            if isinstance(fusion_cfg, dict) and "alpha" in fusion_cfg:
                try:
                    cfg_retr["fusion"] = {"alpha": float(fusion_cfg.get("alpha"))}
                except Exception:
                    pass
            if cfg_retr:
                cfg["retrieval"] = cfg_retr

        # NEW: rerank section overlay
        rerank_cfg_json = json_cfg.get("rerank", {})
        if isinstance(rerank_cfg_json, dict):
            cfg_rerank: Dict[str, object] = {}
            if "enabled" in rerank_cfg_json:
                try:
                    cfg_rerank["enabled"] = bool(rerank_cfg_json.get("enabled"))
                except Exception:
                    pass
            if "top_n" in rerank_cfg_json:
                try:
                    cfg_rerank["top_n"] = int(rerank_cfg_json.get("top_n"))
                except Exception:
                    pass
            if "timeout_ms" in rerank_cfg_json:
                try:
                    cfg_rerank["timeout_ms"] = int(rerank_cfg_json.get("timeout_ms"))
                except Exception:
                    pass
            if "preview_chars" in rerank_cfg_json:
                try:
                    cfg_rerank["preview_chars"] = int(rerank_cfg_json.get("preview_chars"))
                except Exception:
                    pass
            if "batch_size" in rerank_cfg_json:
                try:
                    cfg_rerank["batch_size"] = int(rerank_cfg_json.get("batch_size"))
                except Exception:
                    pass
            if "model" in rerank_cfg_json:
                cfg_rerank["model"] = str(rerank_cfg_json.get("model"))
            if cfg_rerank:
                cfg["rerank"] = cfg_rerank

    return cfg


CONFIG: Dict[str, object] = _build_config(ENV)


def langfuse_present() -> bool:
    """
    Determine whether Langfuse credentials are available in the environment.

    This helper centralizes the presence check for Langfuse configuration so that the rest of the
    application can make a simple boolean decision about enabling Langfuse. It requires both the
    public and secret keys to be non‑empty, which prevents partially configured states from
    attempting to initialize integrations that would fail at runtime. While unused in this milestone,
    it provides a clean extension point for subsequent work where RAG traces and metrics may be
    recorded via Langfuse when credentials are supplied.

    Returns:
        bool: True if both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are present and non‑empty;
        otherwise False.
    """
    return bool(ENV.get("LANGFUSE_PUBLIC_KEY") and ENV.get("LANGFUSE_SECRET_KEY"))


def get_required_env(var_name: str) -> str:
    """
    Read a required environment variable and fail fast if it is missing.

    This helper centralizes validation of sensitive configuration values that must be
    provided via the operating system environment ("env" comes from "environment").
    It is used to enforce that API credentials for external providers are present at
    process startup, which prevents deferred runtime errors deeper in the request
    handling path. The function does not cache, normalize, or log the secret value,
    and it never returns defaults – absence is treated as a configuration error.

    Args:
        var_name (str): The required environment variable name to fetch, e.g.,
            "NEBIUS_API_KEY" (Nebius) or "OPENAI_API_KEY" (OpenAI – OpenAI stands for
            Open Artificial Intelligence).

    Returns:
        str: The non‑empty value of the requested environment variable.

    Raises:
        RuntimeError: If the variable is not set or resolves to an empty string, with a
            clear instruction to set it in the shell or .env file.
    """
    value = os.getenv(var_name, "")
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {var_name}. Set it in your shell or .env"
        )
    return value


def _validate_providers(config: Dict[str, object]) -> None:
    """
    Validate provider selections and required credentials based on configuration.

    This function inspects the provider toggles under the `llm` and `embeddings` sections
    of CONFIG and ensures the corresponding environment variables are present. Supported
    providers are "nebius" and "openai" (case‑insensitive). For both LLM and embeddings in
    this step, we enforce presence of the same key per provider:
    - nebius → NEBIUS_API_KEY must be set
    - openai → OPENAI_API_KEY must be set

    The function emits concise INFO logs summarizing the chosen providers (without logging any
    secrets) and raises clear exceptions on misconfiguration to fail fast at startup.

    Args:
        config (Dict[str, object]): The resolved CONFIG mapping.

    Raises:
        ValueError: If an unknown provider is configured for either section.
        RuntimeError: If the required environment variable for a known provider is missing.
    """
    logger = logging.getLogger(__name__)

    llm_cfg = config.get("llm", {}) if isinstance(config, dict) else {}
    emb_cfg = config.get("embeddings", {}) if isinstance(config, dict) else {}

    llm_provider = str(getattr(llm_cfg, "get", lambda *_: "nebius")("provider", "nebius")).strip().lower()
    emb_provider = str(getattr(emb_cfg, "get", lambda *_: "nebius")("provider", "nebius")).strip().lower()

    logger.info("LLM provider selected: %s", llm_provider)
    logger.info("Embeddings provider selected: %s", emb_provider)

    # Validate LLM provider
    if llm_provider == "nebius":
        get_required_env("NEBIUS_API_KEY")
    elif llm_provider == "openai":
        get_required_env("OPENAI_API_KEY")
    else:
        raise ValueError(f"Unsupported llm provider: {llm_provider}")

    # Validate embeddings provider (same rule in this step)
    if emb_provider == "nebius":
        get_required_env("NEBIUS_API_KEY")
    elif emb_provider == "openai":
        get_required_env("OPENAI_API_KEY")
    else:
        raise ValueError(f"Unsupported embeddings provider: {emb_provider}")


# Perform provider validation at import time so the app fails fast on misconfiguration.
_validate_providers(CONFIG)

