""" api/context.py: This file contains the API endpoint for updating the location context.
It fetches the device list from the active Smart Home Integrator using the provided token and location_id,
and then saves this list to a context file for later use by the LLM. 

Enhanced with daily digest functionality to inject energy efficiency tips
when users open the chat window.
"""

import os
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
# Use the shared provider client instance from llm_cloud.tools
from llm_cloud.tools import provider_client_instance
from config import CONFIG # Import the global CONFIG object

# Import daily digest functionality
from services.daily_digest import generate_daily_digest, should_show_daily_digest, format_digest_for_injection
from services.history_manager import generate_interaction_id, save_message

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Configuration for Storing User-Specific Context ---
# Get the absolute path to the contexts directory directly from CONFIG
# This path is pre-calculated in config/__init__.py
CONTEXTS_DIR = CONFIG['paths']['contexts_full_path']

# Ensure the contexts directory exists when this module is loaded.
# This could also be centralized at application startup.
os.makedirs(CONTEXTS_DIR, exist_ok=True)

router = APIRouter() # Create an APIRouter instance

# --- API Endpoint for Updating Location Context ---
# In current frontend-backend architecture, this endpoint is hit when the user opens the chat window.
# It fetches the device list from the Smart Home Integrator using the provided token and location_id,
# and then saves this list to a context file for later use by the LLM.
# 
# Enhanced to also inject daily energy efficiency digest as the first message in conversation.
# NOTE: Now requires user_email parameter for per-user daily digest tracking.
@router.post("/context") # Note: Path is relative to the prefix defined in main.py when including the router
async def update_location_context(token: str, location_id: str, user_email: str = None):
    """
    Fetches the current device list for the given location_id using the integrator token,
    saves it to a local context file, and optionally injects a daily energy digest
    into the conversation history.
    
    Args:
        token (str): Integrator authentication token
        location_id (str): Logical service location identifier  
        user_email (str, optional): User's email for session isolation and daily digest tracking
        
    Note:
        The device list is shared by location (same for all users in household).
        The daily digest is personal (once per day per user).
        Frontend should now provide user_email for proper digest tracking.
    """
    logger.info(f"[update_location_context] Received request to update context for location_id: {location_id}, user: {user_email}\n")

    context_file_path = os.path.join(CONTEXTS_DIR, f"{location_id}_context.json")

    try:
        # --- Step 1: Fetch and save device list (existing functionality - location-based) ---
        logger.info(f"[update_location_context] Fetching device list from provider for location_id: {location_id}.\n")
        devices_list = provider_client_instance.get_devices(token=token, service_location_id=location_id)

        # Map provider error object to HTTP 502 to mirror previous behavior
        if isinstance(devices_list, dict) and 'error' in devices_list:
            error_message = devices_list['error']
            logger.error(f"[update_location_context] Provider returned an error for location_id {location_id}: {error_message}\n")
            raise HTTPException(status_code=502, detail=f"Provider error: {error_message}")

        if not isinstance(devices_list, list):
            logger.error(f"[update_location_context] Fetched devices data is not a list for location_id {location_id}. Data: {devices_list}\n")
            raise HTTPException(status_code=500, detail="Unexpected format for device list.")

        logger.info(f"[update_location_context] Successfully fetched {len(devices_list)} devices for location_id: {location_id}\n")

        data_to_store = {"device_list": devices_list}
        with open(context_file_path, "w") as f:
            json.dump(data_to_store, f, indent=4)
        
        logger.info(f"[update_location_context] Successfully saved device list context for location_id: {location_id} to {context_file_path}\n")
        
        # --- Step 2: Generate and inject daily digest (new functionality - user-based) ---
        if should_show_daily_digest(location_id, user_email):
            try:
                # Generate the daily digest
                daily_digest = generate_daily_digest()
                
                # Create interaction ID for the digest message
                digest_interaction_id = generate_interaction_id()
                
                # Format digest as JSON string for conversation history
                digest_json = format_digest_for_injection(daily_digest, digest_interaction_id)
                
                # Inject digest as assistant message at the beginning of conversation
                save_message(
                    interaction_id=digest_interaction_id,
                    role="assistant", 
                    content=digest_json,
                    user_email=user_email
                )
                
                logger.info(f"[update_location_context] Successfully injected daily digest for user: {user_email} at location_id: {location_id}\n")
                
                return JSONResponse(daily_digest)
                
            except Exception as digest_error:
                # If digest injection fails, don't fail the whole request
                # The device list update was successful, so return success but log the digest error
                logger.error(f"[update_location_context] Failed to inject daily digest for user {user_email} at location_id {location_id}: {digest_error}\n", exc_info=True)
                
                return JSONResponse({"error": "digest_generation_failed"})
        else:
            # Digest already shown today for this user
            logger.debug(f"[update_location_context] Daily digest already shown today for user: {user_email} at location_id: {location_id}\n")
            return JSONResponse({"status": "no_digest_today"})

    except json.JSONDecodeError as e:
        # Should not occur now because provider client returns Python lists, not JSON strings
        logger.error(f"[update_location_context] Failed to parse device list for location_id {location_id}: {e}\n", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to parse device list response")
    except HTTPException: # Re-raise HTTPException to let FastAPI handle it
        raise
    except Exception as e:
        logger.error(f"[update_location_context] An unexpected error occurred while updating context for location_id {location_id}: {e}\n", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}") 