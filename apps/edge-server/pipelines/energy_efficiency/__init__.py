"""
pipelines/energy_efficiency/__init__.py

Energy efficiency pipeline for providing energy saving advice and education.

This pipeline handles all energy efficiency related queries including:
- Energy saving tips and strategies
- Energy bill reduction advice
- Energy efficiency best practices
- Educational content about energy consumption
- Energy optimization recommendations

The pipeline provides conversational energy efficiency advice without
requiring real-time device data or tool interactions.
"""

from .pipeline_energy_efficiency import EnergyEfficiencyPipeline

__all__ = ['EnergyEfficiencyPipeline'] 