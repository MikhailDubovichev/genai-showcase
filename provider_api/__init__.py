"""
provider_api package: provider-agnostic Smart Home Integrator integrations.

This package contains the abstractions and concrete implementations that allow the
application to interact with different Smart Home Integrators through a consistent,
stable interface. The design follows the adapter pattern, where the rest of the
codebase speaks to a small, carefully designed contract while provider-specific
details are encapsulated behind that boundary. This means the orchestration,
pipelines, and API layers do not need to change when swapping integrators, which
preserves maintainability and testability.

Included modules:
- base: Defines the abstract interface (Application Programming Interface, often
  abbreviated as API) that every integrator client must implement. It uses Python's
  Abstract Base Class (ABC) facilities to declare required methods for listing
  devices and controlling devices.
- mock_client: A deterministic, in-memory implementation of the integrator interface
  used for local development, tests, and demos. It enables running the entire project
  without any external credentials (often called "env" variables for environment
  variables) or network access, while still exercising realistic behavior flows.

Public exports:
- ProviderClient: The abstract interface for integrators
- MockProviderClient: The default mock implementation used by the public repository

The goal of this package is to keep integrator concerns isolated. Adding a real
integrator later simply means creating a new module that implements `ProviderClient`
and wiring a small factory to select it based on configuration.
"""

from .base import ProviderClient
from .mock_client import MockProviderClient

__all__ = [
    "ProviderClient",
    "MockProviderClient",
] 