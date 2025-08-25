"""
Gradio configuration loader and helpers for UI-only settings.

This module centralizes how the Gradio applications (like the chat UI) read
and validate their runtime configuration. The configuration is stored as a
consumer-facing JSON file under `apps/gradio/config/config.json` and should
contain only non-sensitive values such as base URLs and timeouts. This keeps
secrets out of the repository, while allowing quick edits for local or staging
targets. The term "env" comes from environment and refers to the set of
variables and settings provided by the operating system or shell that a program
can read at runtime. The term "URL" comes from Uniform Resource Locator and is
the standard way to specify the location of a resource on the network,
including the scheme (http/https), host, optional port, and path. The term
"JSON" comes from JavaScript Object Notation, a lightweight, text-based data
interchange format, which we use to define these configuration values.

The helpers below load the JSON once, normalize and validate the fields
required by our UIs, and provide convenience utilities for building request
URLs and converting timeouts into seconds for Python APIs. Keeping this logic
in one place avoids duplication across multiple Gradio apps and ensures
consistent behavior (e.g., URL normalization and minimum timeout enforcement)
without introducing external dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_gradio_config() -> Dict[str, Any]:
    """
    Load and parse the Gradio configuration JSON from the repository.

    This function reads `apps/gradio/config/config.json` from the repository
    tree and returns the parsed dictionary. It raises a ValueError if the file
    does not exist, cannot be read, or does not contain valid JSON. The intent
    is to fail fast with a clear message when configuration is missing or
    malformed. Callers should store the returned dictionary and pass it to
    helper functions to derive specific values.

    Returns:
        Dict[str, Any]: The parsed configuration object.

    Raises:
        ValueError: If the config file is missing or invalid JSON.
    """
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.json"
    if not config_path.exists():
        raise ValueError(f"Gradio config not found at {config_path}")
    try:
        text = config_path.read_text(encoding="utf-8")
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid Gradio config JSON at {config_path}: {exc}") from exc


def get_edge_base_url(config: Dict[str, Any]) -> str:
    """
    Extract and normalize the Edge API base URL from the configuration.

    This helper ensures that the returned string includes an explicit scheme
    (http or https) and that it does not end with a trailing slash. Normalizing
    the base URL prevents mistakes when joining paths for requests from the
    Gradio UI and keeps the behavior consistent across environments.

    Args:
        config (Dict[str, Any]): Parsed configuration dictionary.

    Returns:
        str: A normalized base URL such as "http://localhost:8080".

    Raises:
        ValueError: If the URL is missing or does not include http/https scheme.
    """
    raw = str((config.get("edge_api_base_url") or "")).strip()
    if not raw:
        raise ValueError("edge_api_base_url is missing in Gradio config")
    if not (raw.startswith("http://") or raw.startswith("https://")):
        raise ValueError("edge_api_base_url must start with http:// or https://")
    while raw.endswith("/"):
        raw = raw[:-1]
    return raw


def get_cloud_base_url(config: Dict[str, Any]) -> str:
    """
    Extract and normalize the Cloud RAG API base URL from the configuration.

    This helper ensures that the returned string includes an explicit scheme
    (http or https) and that it does not end with a trailing slash. Normalizing
    the base URL prevents mistakes when joining paths for requests from the
    Gradio UI and keeps the behavior consistent across environments. The
    cloud RAG service provides Retrieval Augmented Generation (RAG) capabilities
    using vector search and LLM responses.

    Args:
        config (Dict[str, Any]): Parsed configuration dictionary.

    Returns:
        str: A normalized base URL such as "http://localhost:8000".

    Raises:
        ValueError: If the URL is missing or does not include http/https scheme.
    """
    raw = str((config.get("cloud_rag_base_url") or "")).strip()
    if not raw:
        raise ValueError("cloud_rag_base_url is missing in Gradio config")
    if not (raw.startswith("http://") or raw.startswith("https://")):
        raise ValueError("cloud_rag_base_url must start with http:// or https://")
    while raw.endswith("/"):
        raw = raw[:-1]
    return raw


def get_timeout_seconds(config: Dict[str, Any]) -> float:
    """
    Convert the HTTP timeout from milliseconds to seconds for Python APIs.

    This helper reads the integer `http_timeout_ms` from the configuration,
    applies a minimum of 1 millisecond if needed, and returns the value in
    seconds as a float. Using a normalized seconds value allows us to pass the
    timeout directly to `urllib.request.urlopen` or similar functions without
    duplicating conversion logic.

    Args:
        config (Dict[str, Any]): Parsed configuration dictionary.

    Returns:
        float: Timeout in seconds suitable for Python networking APIs.
    """
    try:
        timeout_ms = int(config.get("http_timeout_ms", 5000))
    except Exception:
        timeout_ms = 5000
    if timeout_ms < 1:
        timeout_ms = 1
    return float(timeout_ms) / 1000.0


def build_url(base: str, path: str) -> str:
    """
    Build a request URL by joining a normalized base and a path.

    This function joins the base URL (which should not end with a trailing
    slash) with a path (which may or may not begin with a slash) to produce a
    single URL that contains exactly one slash between the segments.
    Centralizing this behavior prevents accidental double slashes or missing
    separators in the Gradio apps' HTTP requests.

    Args:
        base (str): Normalized base URL such as "http://localhost:8080".
        path (str): Request path such as "/api/prompt" or "api/prompt".

    Returns:
        str: The combined URL ready for requests.
    """
    normalized_base = base[:-1] if base.endswith("/") else base
    normalized_path = path[1:] if path.startswith("/") else path
    return f"{normalized_base}/{normalized_path}"