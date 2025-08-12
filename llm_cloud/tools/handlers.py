# llm_cloud/tools/handlers.py
"""
Handlers for Smart Home Integrator tools and tool registration logic.

This module centralizes tool handler implementations that interact with the active
Smart Home Integrator client so the LLM has a coherent toolbox. Each handler is a
thin adapter that validates arguments, delegates to the provider client instance
(for example, the injected `api_client.get_devices`), and returns results in a form
suitable for a follow‑up completion. The `register_all_tools` function wires
handlers into the shared `ToolManager`, keeping discovery and exposure of tools
consistent across the application.
"""

import logging
import json
from typing import Callable, Any, Dict, List
import datetime

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Import Tool and ToolManager from the .core module for registration
from .core import Tool, ToolManager

# ---------------------------------------------------------------------------
# Device actions ------------------------------------------------------------
# ---------------------------------------------------------------------------

def _control_device_handler(args: Dict[str, Any], token: str, location_id: str, api_client: Any) -> str:
    """Turn a smart device on or off via the active Smart Home Integrator client.

    Args:
        args (Dict[str, Any]): Expected keys: "device_id" (str), "action" (str: "on"|"off").
        token (str): Authentication token for the integrator.
        location_id (str): Logical location identifier.
        api_client (Any): The injected provider client instance.

    Returns:
        str: A human‑readable confirmation or error message.
    """
    logger.info(
        f"[_control_device_handler] device_id={args.get('device_id')}, action={args.get('action')}, location_id={location_id}\n"
    )
    result = api_client.control_device(
        token=token,
        device_id=args["device_id"],
        action=args["action"],
    )
    if isinstance(result, dict) and result.get("ok"):
        device = result.get("device", {})
        return json.dumps({"status": "success", "device": device})
    return json.dumps({"status": "error", "detail": result})


def _get_devices_handler(args: Dict[str, Any], token: str, location_id: str, api_client: Any) -> str:
    """Retrieve a list of available smart devices via the active integrator client.

    Args:
        args (Dict[str, Any]): Optional key: "excluded_categories" (ignored by mock; accepted for compatibility).
        token (str): Authentication token for the integrator.
        location_id (str): Logical location identifier.
        api_client (Any): The injected provider client instance.

    Returns:
        str: A JSON string representing the list of devices or an error message.
    """
    logger.info(
        f"[_get_devices_handler] location_id={location_id}, excluded_categories={args.get('excluded_categories')}\n"
    )
    devices = api_client.get_devices(
        token=token,
        service_location_id=location_id,
    )
    return json.dumps(devices)

# ---------------------------------------------------------------------------
# Utility/stub tools --------------------------------------------------------
# ---------------------------------------------------------------------------

def _get_current_server_time_handler(args: Dict[str, Any], token: str, location_id: str, api_client: Any) -> str:
    """
    Handler function to get the current local date and time string from the server.

    This tool helps resolve relative time expressions in user requests.
    """
    now_local = datetime.datetime.now()
    current_time_str_local = now_local.isoformat(timespec='seconds')
    logger.info(
        f"[_get_current_server_time_handler] returning local time string: {current_time_str_local}\n"
    )
    return current_time_str_local


def _get_car_current_charge_handler(args: Dict[str, Any], token: str, location_id: str, api_client: Any) -> str:
    """Return a placeholder for the car's current battery charge."""
    return "12 Kilowatt hours"


def _get_current_schedules_handler(args: Dict[str, Any], token: str, location_id: str, api_client: Any) -> str:
    """Return a placeholder describing upcoming charging schedules."""
    return "Your car is scheduled to charge tomorrow from 6 AM to 12 AM"


def _get_weather_forecast_handler(args: Dict[str, Any], token: str, location_id: str, api_client: Any) -> str:
    """Return a placeholder 24‑hour weather forecast string."""
    return (
        "Weather forecast for next 24 h: sunny with a chance of rain (stub value)."
    )


def _get_dynamic_energy_prices_handler(args: Dict[str, Any], token: str, location_id: str, api_client: Any) -> str:
    """Return a placeholder for dynamic energy prices (stub)."""
    return (
        "Dynamic energy prices for next 24 h: flat 0.15 EUR/kWh (stub value)."
    )

# ---------------------------------------------------------------------------
# Registration logic, to be called from __init__.py
# ---------------------------------------------------------------------------

def register_all_tools(tool_manager: ToolManager) -> None:
    """Registers all tools defined in this file with the provided ToolManager."""
    tool_manager.register(
        Tool(
            name="control_device",
            handler=_control_device_handler,
            description=(
                "Turn a smart device on or off immediately. Use this for direct, real-time control "
                "when the user asks to perform an action now."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": (
                        "The unique identifier (ID) of the device to control. If the user provides a name "
                        "(e.g., 'living room light'), you should use the 'get_devices' tool first to find the "
                        "specific ID for that named device."
                    )},
                    "action": {"type": "string", "description": (
                        "The action to perform on the device. Must be either 'on' or 'off'."
                    )},
                },
                "required": ["device_id", "action"],
            },
        )
    )

    tool_manager.register(
        Tool(
            name="get_devices",
            handler=_get_devices_handler,
            description=(
                "Retrieve a list of all available smart devices at the user's location. Use this when the user "
                "asks to see their devices or wants to know what devices are available."
            ),
            parameters={"type": "object", "properties": {}}  # No parameters needed for this tool
        )
    )

    tool_manager.register(
        Tool(
            name="get_current_server_time",
            handler=_get_current_server_time_handler,
            description=(
                "Retrieves the current date and time from the server. Use this tool when a user's scheduling "
                "request involves relative time expressions (e.g., 'tomorrow', 'in 2 hours', 'next Monday') to "
                "get an accurate anchor point for calculating the target schedule."
            ),
            parameters={"type": "object", "properties": {}}  # No parameters needed
        )
    )

    tool_manager.register(
        Tool(
            name="get_car_current_charge",
            handler=_get_car_current_charge_handler,
            description="Return the current battery charge for the electric car.",
            parameters={"type": "object", "properties": {}},
        )
    )

    tool_manager.register(
        Tool(
            name="get_current_schedules",
            handler=_get_current_schedules_handler,
            description="Return upcoming charging schedules for the car.",
            parameters={"type": "object", "properties": {}},
        )
    )

    tool_manager.register(
        Tool(
            name="get_weather_forecast",
            handler=_get_weather_forecast_handler,
            description="Return a 24-hour weather forecast (stub).",
            parameters={"type": "object", "properties": {}},
        )
    )

    # Note: spot_prices tool removed in favor of provider-agnostic design without pricing in this layer.
    logger.info("[register_all_tools] All tools registered (including utility stubs).\n")
