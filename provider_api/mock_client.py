"""
Deterministic mock Smart Home Integrator client for local runs, demos, and tests.

This module provides a reference implementation of the provider-agnostic interface
so the project can be executed end-to-end without any external credentials, secrets,
or network access. The mock returns a small but realistic set of devices and supports
simple on/off control that updates internal state. Because behavior and outputs are
stable, unit tests and manual demonstrations are reproducible across machines.

Abbreviations:
- API: Application Programming Interface, the boundary between components.
- env: Environment variables used to influence behavior without code changes.

Usage:
- The mock is intended to be the default Smart Home Integrator in public repositories.
  Real integrations can be added as separate modules that implement the same interface
  defined in `provider_api.base` and selected at runtime by a small factory (for
  example, via an environment variable named ENERGY_PROVIDER).
"""

from typing import Any, Dict, List, Optional
from .base import ProviderClient


class MockProviderClient(ProviderClient):
    """
    In-memory mock implementation of `ProviderClient` with deterministic behavior.

    This class stores a compact dataset of devices and simulates device control by
    mutating that in-memory state. The intent is not to be exhaustive but to exercise
    all critical code paths in the surrounding application so that development and
    testing do not require external dependencies. Because the output is stable, it is
    straightforward to write unit tests and reason about behavior without flakiness.

    Args:
        None

    Returns:
        None
    """

    def __init__(self) -> None:
        """
        Initialize the mock Smart Home Integrator with a small, realistic device inventory.

        The dataset intentionally includes devices from different categories so that
        both listing and control behaviors can be demonstrated. State is stored in a
        simple dictionary keyed by device identifier. Each device entry contains a
        minimal set of fields that higher layers depend upon, which keeps interfaces
        stable and understandable.

        Args:
            None

        Returns:
            None
        """
        self._devices: Dict[str, Dict[str, Any]] = {
            "dev-1": {"id": "dev-1", "name": "Living Room Light", "category": "LIGHT", "state": "off"},
            "dev-2": {"id": "dev-2", "name": "Heat Pump", "category": "HVAC", "state": "on"},
        }

    def get_devices(
        self, token: str, service_location_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Return the list of known devices in a normalized, integrator-agnostic format.

        The normalized schema mirrors the original Smart Home Integrator helper output for maximum
        compatibility with existing handlers and UIs:
        - id (str)
        - name (str)
        - actions (List[str]) -> ["on", "off"]

        Args:
            token (str): Unused placeholder for parity with real integrations.
            service_location_id (Optional[str]): Optional location identifier; unused
                in the mock but accepted for interface compatibility.

        Returns:
            List[Dict[str, Any]]: Normalized device items.
        """
        normalized: List[Dict[str, Any]] = []
        for device in self._devices.values():
            normalized.append({
                "id": device["id"],
                "name": device["name"],
                "actions": ["on", "off"],
            })
        return normalized

    def control_device(self, token: str, device_id: str, action: str) -> Dict[str, Any]:
        """
        Simulate executing a control action on a device and return the resulting state.

        For realism, this method validates the requested action and checks that the
        target device exists. Supported actions in the mock are limited to "on" and
        "off". When successful, the device's state is updated and the modified device
        is returned. When the device is not found or the action is unsupported, a
        structured error object is returned instead of raising, which simplifies tool
        handling and LLM reasoning.

        Args:
            token (str): Unused placeholder for parity with real integrations.
            device_id (str): Identifier of the device to control.
            action (str): Desired action, typically "on" or "off".

        Returns:
            Dict[str, Any]: {"ok": True, "device": {...}} on success or
            {"ok": False, "error": "reason"} on failure.
        """
        device = self._devices.get(device_id)
        if not device:
            return {"ok": False, "error": "device_not_found", "device_id": device_id}

        normalized = action.strip().lower()
        if normalized in {"on", "off"}:
            device["state"] = normalized
            return {"ok": True, "device": device}

        return {"ok": False, "error": "unsupported_action", "action": action} 