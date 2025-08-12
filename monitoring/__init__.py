"""
Monitoring package initializer.

This package exposes Prometheus metrics and helper decorators for tracking application
performance in an integrator‑agnostic way.
"""

from .metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ERROR_COUNT,
    PIPELINE_PROCESSING_TIME,
    TOOL_EXECUTION_TIME,
    LLM_REQUEST_TIME,
    INTEGRATOR_REQUEST_TIME,
    track_latency,
    track_errors,
)

__all__ = [
    'REQUEST_COUNT',
    'REQUEST_LATENCY',
    'ERROR_COUNT',
    'PIPELINE_PROCESSING_TIME',
    'TOOL_EXECUTION_TIME',
    'LLM_REQUEST_TIME',
    'INTEGRATOR_REQUEST_TIME',
    'track_latency',
    'track_errors',
] 