"""
Initializes the tools package, sets up ToolManager, provider client, ToolExecutor,
and registers all tools from handlers.py.

This module wires the tool system to a Smart Home Integrator client instance. By
using a small factory that defaults to the mock integrator (selected via the
ENERGY_PROVIDER environment variable, where env stands for environment), the
application can run end‑to‑end without secrets while remaining pluggable.
"""

import logging
import os

# Import provider factory defaults
from provider_api import MockProviderClient

# Import core components from .core
from .core import ToolManager, ToolExecutor, Tool

# Import the registration function from .handlers
from .handlers import register_all_tools

# ---------------------------------------------------------------------------
# Global instances – explicit dependencies
# ---------------------------------------------------------------------------

def _make_provider_client():
    """
    Construct and return the active Smart Home Integrator client.

    Defaults to the in‑repo mock for local runs, demos, and CI. In the future, this
    function can resolve additional concrete implementations via ENERGY_PROVIDER.
    """
    provider = os.getenv("ENERGY_PROVIDER", "mock").lower()
    if provider == "mock":
        return MockProviderClient()
    # Fallback to mock if unknown selector is provided
    logging.warning("Unknown ENERGY_PROVIDER '%s'; falling back to mock.", provider)
    return MockProviderClient()

# 1. Instantiate provider client
provider_client_instance = _make_provider_client()
logging.info("Provider client instantiated in tools package (%s).", type(provider_client_instance).__name__)

# 2. Instantiate ToolManager
tool_manager = ToolManager()
logging.info("ToolManager instantiated in tools package.")

# 3. Register all tools using the function from handlers.py
register_all_tools(tool_manager)

# 4. Instantiate ToolExecutor with the manager and client
tool_executor = ToolExecutor(tool_manager, provider_client_instance)
logging.info("ToolExecutor instantiated in tools package.")

# --- Define what's available when importing from llm_cloud.tools ---
__all__ = [
    'Tool',
    'ToolManager',
    'ToolExecutor',
    'tool_manager',
    'tool_executor',
    'provider_client_instance',
    'get_tool_definitions'
]

def get_tool_definitions() -> list:
    """
    Convenience function to get tool definitions from the global tool_manager.
    """
    if tool_manager:
        return tool_manager.get_definitions()
    return []

logging.info("llm_cloud.tools package fully initialized.")
