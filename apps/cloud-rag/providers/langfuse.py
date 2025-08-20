"""
LangFuse client initializer for the Cloud RAG service (minimal, lazy, and optional).

This module provides a tiny integration layer for LangFuse that matches the MVP
requirements: initialize a client once at application startup if credentials are
present, and shut it down gracefully during application shutdown. The goal is to
centralize all LangFuse wiring in a single, well-documented place so the rest of
the codebase remains agnostic to the presence or absence of the observability
backend. If credentials are not provided or the library is not installed, the
functions return None and log a concise hint instead of raising, ensuring that
the server remains fully functional in a no-op mode.

Behavior summary:
- Reads public/secret keys from the environment mapping exposed by `config.ENV`.
- Reads the host from `config.CONFIG["langfuse"]["host"]`, defaulting to the
  public cloud endpoint if not present in config.json.
- On first call to `get_langfuse()`, returns a memoized client or creates one
  if keys and library are available; otherwise logs a one-time warning and
  returns None.
- `close_langfuse()` safely shuts down the client if it was created; otherwise
  it performs a no-op. Errors during shutdown are swallowed with a light log.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config import CONFIG, ENV

logger = logging.getLogger(__name__)

_client: Optional[Any] = None
_warned_disabled: bool = False


def get_langfuse() -> Optional[Any]:
    """
    Lazily construct and return a Langfuse client if credentials are present.

    This function implements a minimal singleton pattern for the Langfuse client.
    It first checks whether a client has already been created in this process and
    immediately returns it if so. If no client exists, the function reads the
    `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` from the `ENV` mapping and the
    `host` value from `CONFIG["langfuse"]["host"]`. If either key is missing or
    empty, the function logs a one-time warning and returns None to keep the app
    operational in environments where Langfuse is not configured.

    If the `langfuse` package is not installed, a concise hint is logged with an
    installation command that developers can run locally. When all requirements are
    satisfied, the function constructs `Langfuse(public_key=..., secret_key=...,
    host=...)`, memoizes it, and returns the instance.

    Returns:
        Optional[Any]: A Langfuse client instance if configured and available; otherwise None.
    """
    global _client, _warned_disabled

    if _client is not None:
        return _client

    public_key = ENV.get("LANGFUSE_PUBLIC_KEY", "")  # type: ignore[assignment]
    secret_key = ENV.get("LANGFUSE_SECRET_KEY", "")  # type: ignore[assignment]
    host = (CONFIG.get("langfuse", {}) or {}).get("host", "https://cloud.langfuse.com")  # type: ignore[assignment]

    if not public_key or not secret_key:
        if not _warned_disabled:
            logger.warning(
                "LangFuse disabled: missing LANGFUSE_PUBLIC_KEY/SECRET_KEY."
            )
            _warned_disabled = True
        return None

    try:
        from langfuse import Langfuse  # type: ignore
    except ImportError:
        logger.warning("LangFuse library not installed. Install with: poetry add langfuse")
        return None

    try:
        _client = Langfuse(public_key=public_key, secret_key=secret_key, host=str(host))
        logger.info("LangFuse client initialized for host: %s", host)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("LangFuse initialization failed: %s", exc)
        _client = None
    return _client


def close_langfuse() -> None:
    """
    Close and flush the Langfuse client if it was previously initialized.

    This helper ensures that any buffered data is flushed and that resources are
    released when the application shuts down. If no client exists or if the
    underlying SDK does not expose a shutdown/flush method, the function returns
    quietly without raising. This behavior keeps application shutdown paths
    robust and avoids spurious errors in environments where Langfuse is not in use.
    """
    global _client
    if _client is None:
        return
    try:
        # Try common termination methods; ignore if not present
        if hasattr(_client, "shutdown"):
            _client.shutdown()  # type: ignore[attr-defined]
        elif hasattr(_client, "flush"):
            _client.flush()  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("LangFuse shutdown encountered an issue: %s", exc)
    finally:
        _client = None


def create_trace(trace_id: str, name: Optional[str] = None, metadata: Optional[dict] = None) -> Optional[object]:
    """
    Create a LangFuse trace for the given identifier, if the client is available.

    This helper encapsulates the minimal trace creation required for the MVP. It fetches the
    memoized LangFuse client via `get_langfuse()`. If the client is not initialized (because keys
    are missing or the library is not present), the function returns None and performs no work so
    that application behavior remains unchanged. When a client is present, it attempts to create or
    upsert a trace with the specified `trace_id`. Any exceptions thrown by the provider are caught
    and logged at warning level, and the function returns None in that case to keep the request
    path resilient.

    Args:
        trace_id (str): Stable identifier for the trace; we use the API request's interactionId.
        name (Optional[str]): Optional human-readable name for the trace. Defaults to "rag.answer".
        metadata (Optional[dict]): Optional metadata payload to attach at trace creation.

    Returns:
        Optional[object]: A trace object if successfully created; None if disabled or on failure.
    """
    client = get_langfuse()
    if client is None:
        return None
    try:
        # Establish an active trace context and upsert name/metadata.
        try:
            from langfuse.types import TraceContext  # type: ignore
            tc = TraceContext(trace_id=trace_id)
        except Exception:
            tc = None

        if tc is not None:
            with client.start_as_current_span(name=name or "rag.answer", trace_context=tc):
                client.update_current_trace(name=name or "rag.answer", metadata=metadata or {})
        else:
            # Fallback: set id seed and update without an active span
            client.create_trace_id(seed=trace_id)
            client.update_current_trace(name=name or "rag.answer", metadata=metadata or {})
        return {"id": trace_id}
    except Exception as exc:  # pragma: no cover - provider-level failure
        logger.warning("LangFuse trace creation failed for %s: %s", trace_id, exc)
        return None


def update_trace_metadata(trace_id: str, metadata: dict) -> None:
    """
    Upsert metadata on an existing LangFuse trace identified by trace_id.

    This function provides a minimal mechanism to enrich a trace with additional fields after
    it has been created. The implementation is deliberately defensive: it silently returns if
    LangFuse is not configured or the SDK is not available, and it catches any exceptions raised
    by the provider during the upsert operation to avoid impacting the request flow. Metadata is
    passed as-is to the provider. The trace name is fixed to "rag.answer" to align with our
    endpoint naming, while the trace id comes from the API request's interactionId for stable
    correlation across systems.

    Args:
        trace_id (str): The identifier of the trace to update (the API interactionId).
        metadata (dict): A flat dictionary of fields to be upserted onto the trace object.

    Returns:
        None: The function operates in a best-effort manner and never raises.
    """
    client = get_langfuse()
    if client is None:
        return
    try:
        # Ensure we are updating the intended trace id using an active span context when possible.
        try:
            from langfuse.types import TraceContext  # type: ignore
            tc = TraceContext(trace_id=trace_id)
        except Exception:
            tc = None

        if tc is not None:
            with client.start_as_current_span(name="rag.answer.meta", trace_context=tc):
                client.update_current_trace(metadata=metadata or {})
        else:
            client.create_trace_id(seed=trace_id)
            client.update_current_trace(metadata=metadata or {})
    except Exception as exc:  # pragma: no cover - provider-level failure
        logger.warning("LangFuse trace metadata update failed for %s: %s", trace_id, exc)
        return


