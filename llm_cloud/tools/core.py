# llm_cloud/tools/core.py
"""
core.py – Defines the core data structures and classes for tool management and execution.
--------------------------------------------------------------------------------------
This module provides the fundamental building blocks for the tool system:
- Tool: Represents a single tool that the LLM can call.
- ToolManager: Manages the registration and retrieval of tool definitions.
- ToolExecutor: Handles the execution of tools based on LLM requests.

Design notes:
1. Dependency Injection:
   - Smart Home Integrator client is passed to tool handlers (not imported directly)
   - ToolManager is passed to ToolExecutor (not hardcoded)
   This makes testing easier and dependencies explicit.

2. Single Responsibility (We split the original ToolRegistry into two classes following the Single Responsibility Principle):
   - ToolManager: handles registration and metadata
   - ToolExecutor: handles execution and error handling
"""

import json
import logging
from typing import Callable, Any, Dict, List

from provider_api import ProviderClient  # generic interface for Smart Home Integrators

# Added: Initialize logger for the module
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core data-structures (Moved from original tools.py)
# ---------------------------------------------------------------------------

class Tool:
    """Metadata wrapper around a callable tool. Represents a tool that can be called by the LLM.
    
    The Tool class encapsulates:
    1. The function to execute (which now receives an API client)
    2. Required parameters
    3. Description for the LLM

    Args:
        name:     Name of the tool. Human-readable identifier, must be unique.
        handler:  Function that performs the work.  Signature must accept
                  (args: dict, token: str, location_id: str, api_client: ProviderClient) and return str.
        description:  Short text shown to the LLM.
        parameters:   JSON schema describing *args* for the handler.
    """

    def __init__(self, name, handler, description, parameters):
        self.name = name
        self.handler = handler
        self.description = description
        self.parameters = parameters


class ToolManager:
    """Manages tool registration and metadata retrieval."""

    def __init__(self) -> None:
        """this method doesn't return a value.
        Its job is to modify the state of the ToolManager (by adding a tool to the _tools dictionary)"""
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add or replace a tool in the registry.
        Uses the name attribute of the tool as the KEY in the dictionary (self._tools).
        And the VALUE is the tool object itself."""
        self._tools[tool.name] = tool

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Return the JSON schema list of function descriptions expected by the LLM chat endpoint.
        This method's crucial role is to prepare the list of tool descriptions in a format that the LLM
        understands when you're using "function calling" or "tool calling." It essentially tells the LLM,
        "Here are the tools you can use, here's what they do, and here's what information they need." """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def get_tool(self, tool_name: str) -> Tool:
        """Retrieve a specific tool instance from the manager by its unique name.

        This method allows other parts of the system, particularly the ToolExecutor,
        to fetch the complete Tool object (which includes its handler function and metadata)
        when the LLM indicates it wants to use a tool with a particular name.

        Args:
            tool_name (str): The unique name of the tool to retrieve. This name
                             corresponds to the 'name' attribute of a registered Tool object.

        Returns:
            Tool: The Tool object associated with the provided tool_name.
        
        Raises:
            KeyError: If no tool with the given tool_name is found in the registry.
                      The caller (e.g., ToolExecutor) is responsible for handling this.
        """
        return self._tools[tool_name]


class ToolExecutor:
    """Handles the execution of tools requested by the LLM, including argument parsing and error handling.
    
    The ToolExecutor acts as the component that takes a tool call request (typically from an LLM),
    retrieves the appropriate tool definition from a ToolManager, parses the arguments, 
    and then invokes the tool's handler function with those arguments and necessary context 
    (like an API client, authentication token, and location ID).
    It ensures that tool execution is robust by catching common errors.
    
    Key functionalities:
    1. Receives tool call requests.
    2. Uses a ToolManager to find the requested tool.
    3. Parses JSON string arguments provided by the LLM into a Python dictionary.
    4. Invokes the tool's handler function, injecting dependencies like the provider client.
    5. Catches and logs errors during tool execution, returning a user-friendly error message.
    """

    def __init__(self, tool_manager: ToolManager, api_client: ProviderClient) -> None:
        """Initializes the ToolExecutor with its dependencies.

        This constructor sets up the ToolExecutor by storing references to the
        ToolManager (which provides access to tool definitions) and the provider client
        (which tool handlers will use to interact with the Smart Home Integrator).
        Dependency injection here makes the ToolExecutor more testable and its dependencies explicit.

        Args:
            tool_manager (ToolManager): An instance of ToolManager that holds all registered
                                       tool definitions. This is used to look up tools by name.
            api_client (ProviderClient): An instance of the provider client that will be passed
                                     to the tool handler functions, allowing them to make
                                     calls to the integrator backend.
        """
        self.tool_manager = tool_manager
        self.api_client: ProviderClient = api_client

    def run_tool(self, tool_call: Any, token: str, location_id: str) -> str:
        """Executes a specific tool based on a tool call request from the LLM.

        This method is the primary entry point for running a tool. It takes the
        raw tool_call object (as provided by the LLM interaction library),
        extracts the tool name and arguments, retrieves the tool from the
        ToolManager, and then executes the tool's handler function.

        It handles potential errors such as:
        - The requested tool not being found (KeyError).
        - Arguments not being valid JSON.
        - Any other exceptions during the tool's handler execution.

        Args:
            tool_call (Any): The tool call object, typically provided by an LLM client library.
                             It's expected to have a 'function' attribute, which in turn has
                             'name' (string) and 'arguments' (JSON string) attributes.
            token (str): The authentication token required for API calls made by the tool.
            location_id (str): The service location ID relevant for the API calls, passed to the tool.

        Returns:
            str: A string representing the result of the tool's execution. This could be
                 a JSON string for structured data, a confirmation message, or an error message.
        """
        tool_name = tool_call.function.name # Get tool_name early for logging
        raw_arguments = tool_call.function.arguments or "{}"

        # Log the attempt to run the tool with raw arguments.
        logger.info(f"[run_tool] Attempting to run tool: '{tool_name}' for location_id: '{location_id}' with raw arguments: {raw_arguments}\n")

        try:
            args = json.loads(raw_arguments)
            # Log parsed arguments at DEBUG level.
            logger.debug(f"[run_tool] Successfully parsed arguments for tool '{tool_name}' (location_id: '{location_id}'): {args}\n")

            tool = self.tool_manager.get_tool(tool_name)

            # Log before calling the handler.
            logger.debug(f"[run_tool] Executing handler for tool '{tool_name}' (location_id: '{location_id}')\n")
            result = tool.handler(args, token, location_id, self.api_client)

            # Log successful execution and the result.
            # Consider result length; if potentially very long, use DEBUG or truncate for INFO.
            # For now, assuming results are reasonably sized for INFO.
            bold = '\033[1m'
            reset = '\033[0m'
            logger.info(f"[run_tool] Successfully executed tool: '{bold}{tool_name}{reset}' for location_id: '{location_id}'.Result:\n")
            logger.info(f"{bold}{result}\n")
            return result
        except KeyError: # Specific case: Tool not found by ToolManager
            logger.error(f"[run_tool] Unknown tool requested: '{tool_name}' by LLM for location_id: '{location_id}'. Raw arguments: {raw_arguments}\n", exc_info=False) # exc_info=False as stack trace isn't super helpful for a KeyError here.
            return f"Unknown tool '{tool_name}'"
        except json.JSONDecodeError as exc: # Specific case: LLM provided malformed JSON for arguments
            logger.error(f"[run_tool] Failed to parse JSON arguments for tool '{tool_name}' (location_id: '{location_id}'). Error: {exc}. Raw arguments: {raw_arguments}\n", exc_info=True)
            return f"Error: Malformed arguments provided for tool '{tool_name}'. Arguments must be a valid JSON string."
        except Exception as exc: # Catch-all for other errors during tool.handler execution
            # Existing logging.exception is good as it includes stack trace.
            logger.exception(f"[run_tool] Error executing tool '{tool_name}' (location_id: '{location_id}'). Parsed args (if available): {args if 'args' in locals() else 'N/A due to earlier error'}\n")
            return f"Error executing tool '{tool_name}': {exc}"
