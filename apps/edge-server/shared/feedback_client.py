"""
Feedback HTTP client for syncing edge feedback to the Cloud (stdlib-only).

This provider encapsulates the network call used to synchronize locally
captured feedback with the Cloud service. It uses only Python's standard
library to send an HTTP (Hypertext Transfer Protocol) POST request with a
JSON (JavaScript Object Notation) body to the Cloud endpoint and parses the
response on success. A short timeout (time-out: a protective cut-off after
which we stop waiting) is enforced so edge workflows remain responsive even
if the network or Cloud service is slow. The goal is to separate networking
concerns from business logic, keeping callers focused on reading, filtering,
and checkpointing feedback while this module handles request/response details
and error taxonomy.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, List
from urllib import error as urlerror
from urllib import request as urlrequest


class FeedbackClientError(Exception):
    """
    Base exception for feedback client failures.

    Raised for non-timeout errors such as non-200 HTTP responses, invalid JSON
    responses, or generic network issues that are not specifically timeouts.
    Callers can catch this to handle permanent or application-level failures.
    """


class FeedbackClientTimeoutError(FeedbackClientError):
    """
    Timeout-specific feedback client failure.

    Raised when the request exceeds the configured timeout, either via
    socket.timeout or a URLError whose reason is a timeout-type error. This
    allows callers to distinguish transient slowness from other failure modes.
    """


def post_feedback_batch(
    base_url: str,
    items: List[Dict[str, Any]],
    timeout_s: float = 5.0,
) -> Dict[str, Any]:
    """
    POST a batch of feedback items to the Cloud feedback sync endpoint.

    Sends the payload to `{base_url}/api/feedback/sync` with Content-Type set to
    `application/json`. On HTTP 200, parses and returns the JSON body as a dict.
    On non-200, raises FeedbackClientError including status and body. On network
    timeouts, raises FeedbackClientTimeoutError. Invalid JSON bodies raise
    FeedbackClientError with a truncated body for safety.

    Args:
        base_url (str): Cloud base URL, e.g., "http://localhost:8000".
        items (List[Dict[str, Any]]): Normalized feedback objects to upload.
        timeout_s (float): Request timeout in seconds (protective cut-off).

    Returns:
        Dict[str, Any]: Parsed JSON dictionary from the Cloud on success.

    Raises:
        FeedbackClientTimeoutError: If the request exceeds the timeout.
        FeedbackClientError: For non-200 responses or invalid/malformed JSON.
    """
    # TODO: Move hardcoded URL construction details and default timeouts into
    # CONFIG["cloud_rag"] (e.g., base_url, timeout_s) to avoid literals here and
    # centralize configuration alongside other Cloud settings.
    url = f"{base_url.rstrip('/')}" \
          f"/api/feedback/sync"

    data = json.dumps({"items": items}).encode("utf-8")
    req = urlrequest.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read().decode("utf-8", errors="replace")
            if status != 200:
                raise FeedbackClientError(f"HTTP {status}: {body}")
            try:
                return json.loads(body)
            except json.JSONDecodeError as exc:
                snippet = body[:200]
                raise FeedbackClientError(
                    f"Invalid JSON from Cloud: {exc}: body={snippet}"
                ) from exc
    except socket.timeout as exc:
        raise FeedbackClientTimeoutError(f"Request timed out after {timeout_s}s") from exc
    except urlerror.URLError as exc:
        if isinstance(exc.reason, socket.timeout):
            raise FeedbackClientTimeoutError(f"Request timed out after {timeout_s}s") from exc
        raise FeedbackClientError(f"Network error: {exc}") from exc


