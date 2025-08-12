"""
api/prompt.py (both PROMPT and RESET endpoints)

Handles all API endpoints related to direct Large Language Model (LLM) interactions,
including processing user prompts and managing conversation history.
We manage PROMPT and RESET endpoints in one file, as they are closely related.
Reset is used to clear the conversation history, which is used by the PROMPT endpoint.

Endpoints:
  - POST /prompt: Receives a user message, interacts with the LLM (via llm_cloud module),
                  and returns the LLM's response. Manages ongoing conversation history.
  - POST /reset: Clears the current conversation history used by the /prompt endpoint.
"""

from typing import Optional
import json
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from core.orchestrator import PipelineOrchestrator
from config import CONFIG # Import the global CONFIG object
from services.history_manager import archive_active_conversation

# Get a logger instance for this module
logger = logging.getLogger(__name__)

router = APIRouter()

# TODO: in production we should make sure that the user_email is always provided. It shouldn't be optional.
@router.post("/prompt")
async def handle_prompt(
    message: str, 
    token: str, 
    location_id: str,
    user_email: Optional[str] = None
):
    """
    Process a user prompt by orchestrating the pipeline flow and returning a standardized JSON response.
    
    This endpoint serves as the primary entry point for conversational turns. It logs the request, constructs
    the pipeline orchestrator, classifies the message, and routes it to the appropriate pipeline implementation
    (device control or energy efficiency). Conversation history is updated with both the user message and the
    assistant reply, enabling multi‑turn context. The function also gracefully handles cases where the LLM returns
    non‑JSON content by wrapping it into a standardized structure so the frontend can reliably parse responses.
    
    Args:
        message (str): Raw user message to be processed by the assistant.
        token (str): Integrator authentication token used by downstream tools during device operations.
        location_id (str): Integrator service location identifier for scoping device queries.
        user_email (Optional[str]): E‑mail address used for per‑user session isolation. If omitted, the system
            falls back to a legacy global session for backward compatibility.
    
    Returns:
        JSONResponse: A response containing at least 'message', 'type', 'interactionId', and 'content'. For JSON
            outputs produced by the LLM, the content is returned as‑is. For plain text outputs, the function wraps
            the text in a consistent 'text' structure to keep frontend parsing stable across models.
    
    Raises:
        Exception: Any unexpected exception is caught and converted into a standardized error JSON and HTTP 500.
    
    Side effects:
        - Persists the user message and assistant response to the user‑specific conversation history on disk.
        - Generates a unique 'interaction_id' for each turn, which is logged for observability and tracing.
        - Reads global configuration to initialize the orchestrator and select the configured LLM/model settings.
    
    Note:
        - Prefer providing 'user_email' in production to avoid session collisions when multiple users share the same
          location. This endpoint is designed to be robust for macOS/iOS/browser clients by always returning JSON.
    """

    if user_email:
        logger.info(f"[handle_prompt] Received prompt for user: {user_email}, location_id: {location_id} - Message: '{message}' - Token: {token[:10]}... (truncated)\n")
    else:
        logger.warning(f"[handle_prompt] Received prompt WITHOUT user_email for location_id: {location_id} - Using global session - Message: '{message}' - Token: {token[:10]}... (truncated)\n")

    # Use new modular pipeline architecture
    orchestrator = PipelineOrchestrator(CONFIG)
    llm_output = orchestrator.process_query(
        message=message,
        token=token, 
        location_id=location_id,
        user_email=user_email
    )
    response_content = llm_output["response_content"]
    interaction_id = llm_output["interaction_id"]

    logger.info(f"[handle_prompt] LLM Raw Response for location_id {location_id}:")
    bold = '\033[1m'
    logger.info(f"{bold}{response_content}\n")

    try:
        response_dict = json.loads(response_content)
        if isinstance(response_dict, dict):
            # No need to manually add interaction_id - it's now included by the LLM in its JSON response
            pass
        else:
            # Handle non-dict responses by wrapping them, still including interaction_id for consistency
            response_dict = {"data": response_dict, "interactionId": interaction_id}
        logger.debug(f"[handle_prompt] Successfully parsed LLM response for location_id {location_id}: {response_dict}\n")
        return JSONResponse(response_dict)
    except json.JSONDecodeError as e:
        logger.error(f"[handle_prompt] LLM response_content was not valid JSON for location_id {location_id}: {e}. Returning as plain text. Response: \n{response_content}\n", exc_info=False)
        
        text_fallback_response = {
            "message": response_content,
            "type": "text",
            "interactionid": interaction_id
        }
        return JSONResponse(text_fallback_response)
    except Exception as e:
        logger.error(f"[handle_prompt] An unexpected error occurred for location_id {location_id}: {e} - Response: {response_content}\n", exc_info=True)
        default_fallback_msg = "I apologize, but I encountered an unexpected issue processing your request. Please try again."
        error_response = {
            "message": default_fallback_msg,
            "type": "error",
            "interactionid": interaction_id
        }
        return JSONResponse(content=error_response, status_code=500)

@router.post("/reset")
async def reset_conversation(user_email: Optional[str] = None):
    """
    Archive and reset the active conversation for a given user session.
    
    This endpoint rotates the current active conversation file into an archived, timestamped file and prepares a
    clean slate for the next turn. It supports per‑user isolation via a stable e‑mail hash so that multiple people
    at the same location do not overwrite each other's histories. If no active conversation exists or the file is
    empty, the operation is treated as a no‑op and still returns success to keep the user experience smooth.
    
    Args:
        user_email (Optional[str]): E‑mail address used to select the user‑specific conversation file. If None,
            the legacy global conversation file is archived instead for backward compatibility.
    
    Returns:
        JSONResponse: A payload containing 'response' and a human‑readable 'message' describing the result. Returns
            HTTP 200 on success and HTTP 500 on failure.
    
    Notes:
        - File operations are performed under the configured 'user_data' path on disk.
        - Archive filenames include a timestamp (YYYYMMDD_HHMMSS) and, for user sessions, a deterministic e‑mail hash.
    """

    if user_email:
        logger.info(f"[reset_conversation] Received request to reset conversation for user: {user_email}\n")
    else:
        logger.warning("[reset_conversation] Received request to reset conversation WITHOUT user_email - Using global session\n")

    success, message = archive_active_conversation(user_email=user_email)
    if success:
        logger.info(f"[reset_conversation] Conversation archived successfully: {message}\n")
        return JSONResponse({"response": "ok", "message": message})
    else:
        logger.error(f"[reset_conversation] Failed to archive conversation: {message}\n")
        return JSONResponse({"response": "error", "message": message}, status_code=500) 