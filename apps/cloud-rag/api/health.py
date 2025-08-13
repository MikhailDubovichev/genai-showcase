"""
Health endpoint for the Cloud RAG service.

This module defines a minimal FastAPI router that exposes a liveness/readiness
check at the path "/health". The purpose of this endpoint is to provide a
lightweight, dependency‑free way to observe whether the HTTP process is up and
capable of serving requests. We return a small JSON object containing a static
status field set to "ok" and an ISO‑8601 timestamp captured in UTC. This shape
is intentionally simple and mirrors common health check conventions across web
services, making it convenient for local curl checks, CI smoke tests, container
orchestrators, and monitoring dashboards. Keeping this code isolated from other
application concerns helps ensure the health path remains reliable as we evolve
the rest of the service in later milestones.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    """
    Return a simple health status payload for the Cloud RAG service.

    This endpoint intentionally avoids external dependencies or complex logic to
    minimize the risk of false negatives during health checking. It reports the
    service as healthy by returning a constant status value of "ok" and includes
    a UTC timestamp in ISO‑8601 format to provide basic observability for when
    the check was served. The timestamp can assist in debugging issues where
    proxies or caches might have served stale responses, and it acts as a quick
    sanity check that the process's clock is advancing.

    Returns:
        Dict[str, str]: A JSON‑serializable dictionary with keys "status" and
        "timestamp". The timestamp is generated via
        datetime.now(timezone.utc).isoformat().
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


