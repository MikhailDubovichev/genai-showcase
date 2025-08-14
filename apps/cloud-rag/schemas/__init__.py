"""
Schemas package for the Cloud RAG service.

This package groups Pydantic models that define the public response contracts
for the cloud APIs. Keeping response schemas colocated and importable from a
single package helps ensure both the edge and frontend clients can rely on a
stable shape while we iterate on the service internals. For the MVP we export
the energy efficiency response schema, mirroring the edge server, so that both
services return identical JSON structures and the consumer code remains
agnostic to where the response originated.
"""

from .energy_efficiency import EnergyEfficiencyResponse

__all__ = ["EnergyEfficiencyResponse"]


