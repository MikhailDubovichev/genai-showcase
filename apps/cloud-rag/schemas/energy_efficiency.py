"""
Energy efficiency response schema for the Cloud RAG service.

This module defines a Pydantic model that exactly mirrors the response format
used by the edge server for energy efficiency answers. The goal is to maintain
an identical JSON contract across services so that calling clients (edge app,
frontend, integrations) can parse responses uniformly without branching on the
origin of the data. By centralizing the response shape in this schemas package
we make it straightforward to validate outputs and to generate example payloads
for tests or documentation.

The schema is intentionally minimal for the MVP milestone. It codifies the base
fields promised by the prompt/system configuration and leaves room for future
extensions without breaking existing clients. In later milestones, additional
schemas (e.g., for RAG citations or feedback acknowledgments) can follow the
same pattern: a focused Pydantic model with clear docstrings and sensible
defaults that keep the surface area stable across deployments.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class EnergyEfficiencyResponse(BaseModel):
    """
    Validate the JSON structure of energy efficiency responses.

    This model enforces the exact field names and defaults expected by clients
    that already consume the edge server's responses. It keeps validation
    intentionally light-weight: the contract guarantees that the payload will
    include a textual message, a unique interaction identifier, a type field
    that defaults to "text", and a content list which is empty for this
    category. The model can be expanded in the future if the response format
    grows richer, but for the MVP stage we prioritize compatibility and
    simplicity.

    Fields:
        message (str): Main textual answer content to display to the user.
        interactionId (str): Unique identifier correlating this response with a
            specific interaction cycle, used for tracing and analytics.
        type (str): Response type. Defaults to "text" for energy efficiency
            content and enables a unified client-side rendering path.
        content (List): An empty list by default. Reserved for future structured
            elements related to the response (e.g., references).
    """

    message: str = Field(..., description="The main response text")
    interactionId: str = Field(..., description="Unique identifier for this interaction")
    type: str = Field("text", description="Response type, always 'text' for energy efficiency")
    content: List = Field(default_factory=list, description="Empty list as per contract")

    @staticmethod
    def example() -> "EnergyEfficiencyResponse":
        """
        Construct a simple example instance for documentation and testing.

        Returns:
            EnergyEfficiencyResponse: A valid example object that mirrors the
            expected default shape and is suitable for quick demonstrations or
            unit test fixtures.
        """
        return EnergyEfficiencyResponse(
            message="Energy efficiency tip example.",
            interactionId="example-id-123",
            type="text",
            content=[],
        )


__all__ = ["EnergyEfficiencyResponse"]


