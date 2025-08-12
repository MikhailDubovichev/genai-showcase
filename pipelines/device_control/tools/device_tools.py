"""
pipelines/device_control/tools/device_tools.py

Device control pipeline tools management.

This module provides a filtered view of device-specific tools from the
global tool manager, organizing them for device control operations.
"""

import logging
from typing import List
from llm_cloud.tools import tool_manager

logger = logging.getLogger(__name__)

class DeviceControlToolManager:
    """
    Tool manager specifically for device control pipeline.
    
    This class provides a filtered view of tools that are relevant for
    device control operations, using the global tool manager instead of
    creating new instances.
    """
    
    # Define which tools are relevant for device control
    DEVICE_CONTROL_TOOLS = {
        "control_device",      # Direct device control
        "get_devices",         # Device listing
        "get_current_server_time",  # For scheduling context
        "get_spot_prices"      # For energy cost optimization
    }
    
    def __init__(self):
        """
        Initialize the device control tool manager by referencing the global tool registry.
        
        Rather than creating a new registry, this class wraps the shared `tool_manager` so that
        tool definitions remain centralized and consistent across the application. This ensures the
        same function schemas are exposed to the LLM regardless of where tool calls originate and
        avoids duplicate state. The manager only exposes device‑relevant tools via filtering methods.
        """
        self.global_manager = tool_manager  # Use global instance
        logger.info("[DeviceControlToolManager] Device control tools initialized")
    
    def get_tool_definitions(self) -> List[dict]:
        """
        Get tool definitions for LLM function calling.
        
        Returns:
            List[dict]: List of tool definitions for device control operations
            
        This filters the global tool definitions to only include tools
        that are relevant for device control operations.
        """
        all_tools = self.global_manager.get_definitions()
        device_tools = [
            tool for tool in all_tools 
            if tool["function"]["name"] in self.DEVICE_CONTROL_TOOLS
        ]
        
        logger.debug(f"[DeviceControlToolManager] Providing {len(device_tools)} device control tools")
        return device_tools
    
    def get_tool_names(self) -> List[str]:
        """
        Get list of registered tool names.
        
        Returns:
            List[str]: List of tool names available for device control
        """
        return list(self.DEVICE_CONTROL_TOOLS)
    
    def get_tool_manager(self):
        """
        Get the underlying tool manager instance.
        
        Returns:
            ToolManager: Global tool manager instance
            
        Note: This is maintained for compatibility but ideally should
        not be used directly - prefer get_tool_definitions() instead.
        """
        return self.global_manager 