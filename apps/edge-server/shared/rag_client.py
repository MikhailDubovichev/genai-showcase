"""
Edge-to-Cloud RAG HTTP client (minimal, dependency-free).

This module provides a tiny helper to call the Cloud RAG service from the edge server using
only Python's standard library. It issues an HTTP (Hypertext Transfer Protocol) POST request
to the Cloud endpoint and returns the parsed JSON (JavaScript Object Notation) object on
success. Configuration is read from the edge app's CONFIG, which itself sources values from
the project's configuration file and the process env (environment variables; “env” comes
from environment). The client enforces a short timeout (time-out originally referred to a
protective cut-off after which a system gives up waiting), which is critical on the edge to
avoid blocking device control flows when the Cloud is unreachable or slow. The API is kept
small, explicit, and focused on clear error taxonomy so callers can distinguish network timeouts
from non-200 HTTP responses and JSON parsing problems. No external dependencies are required.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict
from urllib import error as urlerror
from urllib import request as urlrequest

from config import CONFIG


class RAGClientError(Exception):
    """
    Base exception for RAG client errors.

    Raised for non-timeout failures such as non-200 HTTP responses or invalid JSON payloads
    returned by the Cloud service. Callers can catch this type to handle generic client issues
    that are not specifically timeouts.
    """


class RAGClientTimeoutError(RAGClientError):
    """
    Timeout-specific client error.

    Raised when the underlying network operation exceeds the configured timeout. This is a
    distinct error class so edge callers can apply short-circuit fallbacks without conflating
    timeouts and other classes of failures (e.g., 500 errors or malformed JSON responses).
    """


def post_answer(
    base_url: str,
    question: str,
    interaction_id: str,
    top_k: int = 3,
    timeout_s: float = 1.5,
) -> Dict[str, Any]:
    """
    POST a question to the Cloud RAG endpoint and return the parsed JSON response.

    This function constructs the Cloud API URL by joining the given base_url with the
    "/api/rag/answer" path, sends a JSON body containing the question, the interactionId
    (aligned with edge-generated identifiers), and a retrieval depth topK. It enforces a
    short network timeout to prevent the edge server from stalling when the Cloud is slow
    or unavailable. On HTTP 200, the response body is parsed as JSON and returned as a
    dictionary. On non-200 responses, a RAGClientError is raised including the status code
    and response text. Timeouts raise RAGClientTimeoutError. Invalid JSON raises
    RAGClientError with a helpful message.

    Args:
        base_url (str): Cloud service base URL (e.g., "http://localhost:8000").
        question (str): User question to submit to the RAG service.
        interaction_id (str): Interaction identifier to correlate across systems.
        top_k (int): Number of context chunks to retrieve; defaults to 3.
        timeout_s (float): Socket timeout in seconds; defaults to 1.5 for edge responsiveness.

    Returns:
        Dict[str, Any]: Parsed JSON payload returned by the Cloud on HTTP 200.

    Raises:
        RAGClientTimeoutError: When the request exceeds the given timeout.
        RAGClientError: For non-200 HTTP responses or invalid JSON bodies.
    """
    url = f"{base_url.rstrip('/')}" \
          f"/api/rag/answer"

    payload = {
        "question": question,
        "interactionId": interaction_id,
        "topK": int(top_k),
    }
    data = json.dumps(payload).encode("utf-8")

    req = urlrequest.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read().decode("utf-8", errors="replace")
            if status != 200:
                raise RAGClientError(f"Cloud RAG HTTP {status}: {body}")
            try:
                return json.loads(body)
            except json.JSONDecodeError as exc:
                raise RAGClientError(
                    f"Invalid JSON from Cloud RAG: {exc}: body={body[:200]}"
                )
    except socket.timeout as exc:
        raise RAGClientTimeoutError(f"Request timed out after {timeout_s}s") from exc
    except urlerror.URLError as exc:
        # URLError may wrap socket.timeout or other transient network errors
        if isinstance(exc.reason, socket.timeout):
            raise RAGClientTimeoutError(f"Request timed out after {timeout_s}s") from exc
        raise RAGClientError(f"Network error calling Cloud RAG: {exc}") from exc


def post_answer_from_config(
    question: str,
    interaction_id: str,
    top_k: int = 3,
    timeout_s: float = 1.5,
) -> Dict[str, Any]:
    """
    Convenience wrapper that reads the Cloud base URL from CONFIG and delegates to post_answer.

    This function extracts the Cloud RAG base URL from the edge app's CONFIG mapping under
    `CONFIG["cloud_rag"]["base_url"]`. It raises RAGClientError if the base URL is missing,
    empty, or otherwise not a usable string. On success, it calls post_answer with the supplied
    question, interaction identifier, and optional retrieval and timeout arguments. Short timeouts
    are important on the edge to keep local device control responsive when the network path to the
    Cloud is slow or down.

    Args:
        question (str): User question to submit to the RAG service.
        interaction_id (str): Interaction identifier to correlate across systems.
        top_k (int): Number of context chunks to retrieve; defaults to 3.
        timeout_s (float): Socket timeout in seconds; defaults to 1.5 for edge responsiveness.

    Returns:
        Dict[str, Any]: Parsed JSON payload returned by the Cloud on HTTP 200.

    Raises:
        RAGClientError: If the Cloud base URL is missing or invalid; non-200/JSON issues
            from the underlying request are also surfaced as RAGClientError.
        RAGClientTimeoutError: When the request exceeds the given timeout.
    """
    cloud_cfg = CONFIG.get("cloud_rag", {}) or {}
    base_url = cloud_cfg.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise RAGClientError("Missing CONFIG['cloud_rag']['base_url'] for Cloud RAG client")

    return post_answer(
        base_url=base_url,
        question=question,
        interaction_id=interaction_id,
        top_k=top_k,
        timeout_s=timeout_s,
    )


