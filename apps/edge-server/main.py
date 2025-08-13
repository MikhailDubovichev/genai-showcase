""" main.py: FastAPI application entry point and runtime configuration.

This module builds the ASGI app, mounts API routers, configures CORS (Cross‑Origin Resource Sharing; CORS comes
from "cross origin resource sharing"), and exposes a Prometheus metrics endpoint. It centralizes web‑layer wiring
so the rest of the codebase can focus on business logic. When executed directly, it starts a Uvicorn server using
host/port values from configuration, making local development on macOS straightforward. The module purposely keeps
state in process‑local variables rather than globals to make testing simpler and to avoid surprises during import.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
import logging
from config import CONFIG

# --- Router Imports ---
from api import context as context_router
from api import prompt as prompt_router
from api import feedback as feedback_router

# Get a logger instance for this module
logger = logging.getLogger(__name__)

app = FastAPI()

# Include routers
app.include_router(context_router.router, prefix="/api", tags=["Context"])
app.include_router(prompt_router.router, prefix="/api", tags=["Prompt"])
app.include_router(feedback_router.router, prefix="/api", tags=["Feedback"])

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Configure CORS
allow_origins = CONFIG.get('cors', {}).get('allow_origins', ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The uvicorn server is used to run the FastAPI application. 
if __name__ == '__main__':
    import uvicorn
    logger.info("[__main__] Starting Uvicorn server for main.py\n")
    uvicorn.run(
        app,
        host=CONFIG.get('server', {}).get('host', '0.0.0.0'),
        port=CONFIG.get('server', {}).get('port', 8080)
    )

# to connect to the API endpoints via Wi-Fi, one needs to know the IP address of the machine that is running the web server:
# this can be found by running in the terminal (on macOS):
# ipconfig getifaddr en0
# This command will show your actual IP address for your Wi-Fi connection (en0 is typically your WiFi interface on MacOS).
# then one can connect to the API endpoints by using the following URL:
# http://<ip_address>:8080/api/prompt
# http://<ip_address>:8080/api/reset
# http://<ip_address>:8080/api/context



