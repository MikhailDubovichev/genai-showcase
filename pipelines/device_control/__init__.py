"""
pipelines/device_control/__init__.py

Device control pipeline for smart device management and automation.

This pipeline handles all device-related queries including:
- Direct device control (turn on/off)
- Device listing and status queries
- Scheduling and automation commands
- Energy consumption monitoring
- Smart Home Integrator API integration

The pipeline uses the existing tool ecosystem and LLM interactions
optimized for device control operations.
"""

from .pipeline_device_control import DeviceControlPipeline

__all__ = ['DeviceControlPipeline'] 