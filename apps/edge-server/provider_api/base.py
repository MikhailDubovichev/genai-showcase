"""
Provider-agnostic client interface for Smart Home Integrator integrations.

This module defines the abstract contract that any concrete Smart Home Integrator client
must fulfill in order to be used by the rest of the system. The design uses the
adapter pattern to separate application logic from provider-specific concerns like
authentication, HTTP transport, rate limiting, and response normalization. By
constraining the interface to just the capabilities that the application needs
right now—device listing and device control—we enable the orchestrator, pipelines,
and API layers to remain stable and provider-agnostic.

Key concepts and abbreviations:
- API: Application Programming Interface, the surface by which software components
  communicate. Here, it refers to the minimal set of methods that integrators expose.
- ABC: Abstract Base Class, a Python mechanism for defining interfaces via abstract
  methods that subclasses must implement.
- env: Environment variables, process-level configuration used to select the active
  integrator and pass secrets in a secure way (not committed to source control).

A mock implementation is provided in `provider_api.mock_client` so the repository can
be executed end-to-end without external credentials or network access. To integrate a
real Smart Home Integrator, create a new subclass that implements all abstract methods
and wire a small factory to select it at runtime based on configuration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ProviderClient(ABC):
    """
    Abstract client defining required operations for Smart Home Integrator integrations.

    This class captures the stable contract that the rest of the codebase depends on.
    Implementations are responsible for handling all integrator-specific details such as
    authentication flows, HTTP transport, retry logic, and translating raw responses
    into simplified, application-friendly structures. The goal is to isolate changes
    in integrator behavior to a single module while keeping the rest of the system
    unaffected. A deterministic mock implementation ships with the repository so that
    demos and tests can run without any secrets or internet connectivity, while still
    exercising the same control paths and data shapes as production code would.

    Note:
        Each abstract method below includes a detailed docstring describing expected
        inputs and outputs so that implementers have clear guidance. The calling code
        only relies on these shapes, which should remain stable over time.
    """

    @abstractmethod
    def get_devices(
        self,
        token: str,
        service_location_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a normalized list of devices available at a service location.

        Normalized schema recommendation (kept intentionally simple and stable):
        - id (str): Unique device identifier.
        - name (str): Human-friendly device name.
        - actions (List[str]): Supported actions, e.g., ["on", "off"].
        - Optional fields such as category (e.g., "LIGHT", "HVAC") and state (e.g., "on"/"off")
          may be included for richer client UX, but callers should only rely on the
          required fields above unless negotiated otherwise.

        Args:
            token (str): An authentication token or credential required by the integrator.
            service_location_id (Optional[str]): Logical location or site identifier. If
                omitted, implementations may use a default from configuration.

        Returns:
            List[Dict[str, Any]]: A list of device dictionaries with normalized fields
            suitable for direct consumption by higher-level components.
        """
        raise NotImplementedError

    @abstractmethod
    def control_device(self, token: str, device_id: str, action: str) -> Dict[str, Any]:
        """
        Execute a control action (such as turning a device on or off) on a target device.

        Implementations should validate the requested action against the device's
        capabilities and perform any necessary integrator-specific translation. The
        response should be a concise dictionary indicating success or failure and, when
        successful, the updated device state. Keeping this payload small and predictable
        simplifies reasoning for the LLM and makes it easy to log and test.

        Args:
            token (str): Authentication token or credential required by the integrator.
            device_id (str): Integrator-specific identifier of the target device.
            action (str): The desired control action, commonly "on" or "off".

        Returns:
            Dict[str, Any]: A result dictionary such as {"ok": True, "device": {...}} on
            success, or {"ok": False, "error": "reason"} on failure.
        """
        raise NotImplementedError 