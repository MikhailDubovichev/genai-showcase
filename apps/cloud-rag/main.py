"""
Cloud RAG service entrypoint and application factory.

This module bootstraps a minimal FastAPI application that will serve as the
Cloud Retrieval-Augmented Generation (RAG) service in this monorepo. The
responsibility here is intentionally narrow for the MVP milestone: we provide
an HTTP server with consistent logging, permissive CORS configuration for easy
local development, and a basic health endpoint to support readiness checks
from developers, orchestrators, and container platforms. The design mirrors the
style of the existing edge server while remaining independent so each app can
evolve at its own pace without tight coupling.

Key aspects implemented in this file include:
- Construction of the FastAPI application with a clear, documented purpose.
- Lightweight, structured logging using Python's standard logging facilities
  and a module-level logger obtained via logging.getLogger(__name__). This
  keeps logs consistent with the approach used on the edge server while avoiding
  cross‑app imports during the MVP stage.
- CORS middleware configured to allow all origins, headers, and methods to
  simplify iterative development and testing across multiple local ports.
- Mounting of a dedicated health router that responds on GET /health with a
  simple JSON payload, enabling health checks from curl, load balancers, or
  container orchestration systems. Later milestones will mount versioned APIs
  under the /api prefix for RAG answer and feedback endpoints.

This module is designed to be executed directly via "python apps/cloud-rag/main.py".
It includes a standard __main__ block that starts Uvicorn with sensible defaults
for local development. The wider application components such as RAG pipelines,
vector stores, and feedback persistence will be added in subsequent milestones,
leaving this file focused on application wiring and HTTP concerns.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure local package imports resolve when running this file directly via path.
# This makes "from api.health import router" refer to this app's API package.
CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from api.health import router as health_router  # noqa: E402  (import after sys.path adjustment)
from api import rag as rag_router  # noqa: E402
from providers.langfuse import get_langfuse, close_langfuse  # noqa: E402
from config import CONFIG  # noqa: E402


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application for the Cloud RAG service.

    This factory encapsulates app construction so tests and alternative entry
    points can reuse a consistent setup. The function applies permissive CORS
    middleware so requests from any local development origin will succeed, and
    it wires routing in two layers: a root‑level health route mounted directly
    on the app (exposed at /health) and a top‑level API router with the prefix
    "/api" reserved for future RAG and feedback endpoints. While the API router
    does not expose concrete routes in this milestone, pre‑registering it keeps
    the URL structure predictable and ready for incremental additions.

    Returns:
        FastAPI: A configured FastAPI instance with CORS enabled, a health route
        mounted, and an empty "/api" router reserved for future endpoints.
    """
    app = FastAPI(title="Cloud RAG Service", docs_url="/docs", redoc_url="/redoc")

    # Enable permissive CORS for local development; tighten for production later.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount health at the root (GET /health).
    app.include_router(health_router)

    # Prepare and mount API routers
    api_router = APIRouter(prefix="/api")
    app.include_router(api_router)
    app.include_router(rag_router.router, prefix="/api", tags=["RAG"])

    logger.info("Cloud RAG FastAPI app created with CORS and /health endpoint")
    return app


app = create_app()


@app.on_event("startup")
def _on_startup() -> None:
    """
    Initialize optional providers at application startup.

    We attempt to initialize LangFuse based on environment/configuration. If credentials are
    absent or the library is not installed, the provider returns None and the app continues
    to run in no-op mode.
    """
    client = get_langfuse()
    if client is not None:
        app.state.langfuse = client


@app.on_event("shutdown")
def _on_shutdown() -> None:
    """
    Cleanly shut down optional providers when the application stops.
    """
    close_langfuse()


if __name__ == "__main__":
    """
    Run the Cloud RAG development server using Uvicorn.

    This block allows launching the service directly with the standard Python
    interpreter. It uses host 0.0.0.0 and port 8000 by default so that the
    service is reachable from other local processes and, in containerized
    environments, from outside the container when the port is mapped. Logging
    is configured to the INFO level to provide a succinct operational signal
    while avoiding overly verbose output. For production, consider setting up a
    dedicated logging configuration and structured log export.

    Example:
        python apps/cloud-rag/main.py
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        import uvicorn

        host = str(CONFIG["server"]["host"])  # type: ignore[index]
        port = int(CONFIG["server"]["port"])  # type: ignore[index]
        logger.info("Starting Cloud RAG service on http://%s:%s …", host, port)
        uvicorn.run(app, host=host, port=port)
    except Exception as exc:  # pragma: no cover - defensive entrypoint guard
        logger.error("Failed to start Cloud RAG service: %s", exc)
        raise


