"""
shared/utils.py

Shared utility functions used across multiple modules.

This module contains common helper functions that are used by various
components of the AI assistant to avoid code duplication and maintain
consistency across the system.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from config import CONFIG

logger = logging.getLogger(__name__)

def safe_json_loads(json_string: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Safely parse JSON string with fallback handling.
    
    Args:
        json_string (str): JSON string to parse
        fallback (Optional[Dict[str, Any]]): Fallback value if parsing fails
        
    Returns:
        Dict[str, Any]: Parsed JSON dictionary or fallback value
        
    This function safely handles JSON parsing errors that can occur when
    processing LLM responses or configuration data.
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {e}. Using fallback value.")
        return fallback or {}

def create_error_response(message: str, interaction_id: str, error_type: str = "error") -> str:
    """
    Create standardized error response in JSON format.
    
    Args:
        message (str): Error message to include in response
        interaction_id (str): Interaction identifier
        error_type (str): Type of error (default: "error")
        
    Returns:
        str: JSON formatted error response
        
    This creates consistent error response format across all pipelines.
    """
    return json.dumps({
        "message": message,
        "interactionId": interaction_id,
        "type": error_type,
        "content": []
    })

def create_text_response(message: str, interaction_id: str) -> str:
    """
    Create standardized text response in JSON format.
    
    Args:
        message (str): Response message content
        interaction_id (str): Interaction identifier
        
    Returns:
        str: JSON formatted text response
        
    This creates consistent text response format across all pipelines.
    """
    return json.dumps({
        "message": message,
        "interactionId": interaction_id,
        "type": "text",
        "content": []
    })

def truncate_message_for_logging(message: str, max_length: int = 100) -> str:
    """
    Truncate long messages for logging purposes.
    
    Args:
        message (str): Message to truncate
        max_length (int): Maximum length before truncation (default: 100)
        
    Returns:
        str: Truncated message with ellipsis if needed
        
    Used for logging to avoid extremely long log entries while preserving
    the beginning of the message for debugging purposes.
    """
    if len(message) <= max_length:
        return message
    return message[:max_length] + "..."

# Session Context utility - moved from deleted llm_cloud.chat.context
class SessionContext:
    """
    Session context for managing user‑specific conversation history and safe message loading.
    
    This lightweight utility encapsulates how conversation history is retrieved and sanitized for use with
    the LLM client. It exists to prevent code duplication across pipelines and to centralize defensive
    handling for malformed records that might appear in history files (for example, non‑string contents).
    By keeping the concerns here, pipeline code can assume it receives a clean list of role/content pairs
    suitable for API calls, while still benefiting from per‑user session isolation via e‑mail.
    """
    
    def __init__(self, user_email: Optional[str] = None):
        """
        Initialize session context.
        
        Args:
            user_email (Optional[str]): User email for session isolation
        """
        self.user_email = user_email
    
    def load_conversation_history(self) -> List[Dict[str, str]]:
        """
        Load and sanitize conversation history for the current session.
        
        This method reads the user‑specific history file (or legacy global session) and performs defensive
        normalization so the result is always a list of dictionaries with 'role' and 'content' as strings.
        Any malformed entries are skipped or coerced to preserve API compatibility and avoid runtime errors
        during LLM calls.
        
        Returns:
            List[Dict[str, str]]: Cleaned messages in chronological order, each with 'role' and 'content' keys.
        """
        from services.history_manager import load_conversation_history
        history = load_conversation_history(user_email=self.user_email)
        
        # Validate and clean conversation history to prevent API errors
        cleaned_history = []
        for msg in history:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                # Ensure content is always a string
                content = msg['content']
                if not isinstance(content, str):
                    if content is None:
                        content = ""
                    else:
                        content = str(content)
                
                cleaned_history.append({
                    'role': str(msg['role']),
                    'content': content
                })
        
        return cleaned_history

# Tool call handler - simplified version of what was in llm_cloud.chat.tool_handler
def handle_tool_call(response, interaction_id: str, token: str, location_id: str, 
                     system_prompt: str, client, session_context) -> str:
    """
    Execute tool calls returned by the LLM and perform a follow‑up completion with tool results.
    
    The function iterates over tool calls produced by the initial model response, executes each handler via
    the shared `ToolExecutor`, collects their outputs as tool messages, and then issues a second LLM request
    that includes the original system prompt, the assistant's tool call message, and the tool outputs. This
    mirrors standard tool‑use patterns of OpenAI‑compatible APIs and consolidates the logic so pipelines do not
    need to manage the mechanics themselves.
    
    Args:
        response: The initial LLM response object containing potential `tool_calls`.
        interaction_id (str): Unique identifier associated with this conversation turn.
        token (str): Smart Home Integrator API token that tools require to authenticate downstream calls.
        location_id (str): Smart Home Integrator service location identifier used by tools.
        system_prompt (str): The system prompt (augmented with the interaction id) to anchor the follow‑up call.
        client: The OpenAI‑compatible client used to perform the follow‑up completion.
        session_context: Session context object retained for future extensions that need prior messages.
    
    Returns:
        str: The final assistant message content after tool execution and follow‑up completion.
    
    Notes:
        - Each tool call is run independently and errors from a single tool are captured and surfaced as tool
          messages so that the model can react gracefully.
        - Follow‑up uses the device control model configuration to remain consistent with the initiating pipeline.
    """
    from llm_cloud.tools import tool_executor
    
    # Process tool calls
    tool_results = []
    for tool_call in response.choices[0].message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        try:
            result = tool_executor.run_tool(tool_call, token, location_id)
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": result
            })
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool", 
                "name": tool_name,
                "content": f"Error: {str(e)}"
            })
    
    # Follow-up LLM call with tool results
    messages = [
        {"role": "system", "content": system_prompt},
        response.choices[0].message.model_dump(),
    ] + tool_results
    
    model_cfg = CONFIG["llm"]["models"]["device_control"]
    provider = (CONFIG.get("llm", {}) or {}).get("provider", "nebius").lower()
    extra_kwargs = {}
    if provider != "openai":
        extra_kwargs["extra_body"] = {"top_k": model_cfg["settings"]["top_k"]}

    follow_up_response = client.chat.completions.create(
        model=model_cfg["name"],
        messages=messages,
        max_tokens=model_cfg["settings"]["max_tokens"],
        temperature=model_cfg["settings"]["temperature"],
        top_p=model_cfg["settings"]["top_p"],
        **extra_kwargs,
    )
    
    return follow_up_response.choices[0].message.content 