"""
Feedback API Endpoints

This module handles all API endpoints related to user feedback collection.
Supports both negative feedback (thumb-down) and positive feedback (thumb-up) submission 
for assistant responses.

The feedback system helps identify problematic interactions and measure satisfaction levels
to improve the AI assistant by collecting user feedback signals tied to specific interaction IDs.

Endpoints:
  - POST /feedback/negative: Submit negative feedback for a specific interaction
  - POST /feedback/positive: Submit positive feedback for a specific interaction  
  - GET /feedback/negative/stats: Get negative feedback statistics (for monitoring)
  - GET /feedback/positive/stats: Get positive feedback statistics (for monitoring)
"""
from typing import Optional
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from services.feedback_manager import (
    save_negative_feedback, 
    save_positive_feedback,
    validate_interaction_exists,
    get_negative_feedback_statistics,
    get_positive_feedback_statistics
)

# Get a logger instance for this module
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/feedback/negative")
async def submit_negative_feedback(
    interaction_id: str,
    user_email: Optional[str] = None
):
    """
    Submit negative feedback (thumb-down) for a specific interaction with optional user session isolation.
    
    This endpoint allows users to indicate dissatisfaction with an assistant response
    by providing the interaction ID. The system automatically captures the context
    (user message and assistant response) from the user-specific conversation history for later analysis.
    
    Args:
        interaction_id (str): The unique ID of the interaction receiving negative feedback
        user_email (Optional[str]): User's email for session isolation (optional for backward compatibility)
        
    Returns:
        JSONResponse: Success confirmation with feedback ID, or error details
        
    HTTP Status Codes:
        200: Feedback successfully recorded
        400: Invalid interaction ID (not found in user's conversation history)
        500: Server error (unable to save feedback)
        
    Example Request:
        POST /api/feedback/negative
        Form data: interaction_id=71653d96-ae38-480d-9e73-c5a87dd45b38&user_email=user@example.com
        
    Example Success Response:
        {
            "response": "success",
            "message": "Negative feedback recorded successfully",
            "feedback_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "interactionid": "71653d96-ae38-480d-9e73-c5a87dd45b38"
        }
        
    Note:
        - Validates interaction exists in user-specific conversation history before accepting feedback
        - Automatically extracts context from user-specific conversation history
        - Creates feedback storage file if it doesn't exist
        - Uses UTC timestamps for consistency
    """
    if user_email:
        logger.info(f"[submit_negative_feedback] Received negative feedback request for user: {user_email}, interaction: {interaction_id}\n")
    else:
        logger.warning(f"[submit_negative_feedback] Received negative feedback request WITHOUT user_email for interaction: {interaction_id} - Searching in global session\n")
    
    # Validate that the interaction exists in user-specific conversation history
    if not validate_interaction_exists(interaction_id, user_email):
        if user_email:
            logger.warning(f"[submit_negative_feedback] Invalid interaction_id: {interaction_id} for user: {user_email} - not found in user-specific conversation history\n")
            error_message = "Invalid interaction ID. The specified interaction was not found in your conversation history."
        else:
            logger.warning(f"[submit_negative_feedback] Invalid interaction_id: {interaction_id} - not found in global conversation history. Possible session mismatch: frontend may be using user-specific session while backend received no user_email parameter.\n")
            error_message = "Invalid interaction ID. The specified interaction was not found in conversation history."
            
        return JSONResponse(
            content={
                "response": "error", 
                "message": error_message
            }, 
            status_code=400
        )
    
    try:
        # Save the negative feedback with user context
        feedback_record = save_negative_feedback(interaction_id, user_email)
        
        logger.info(f"[submit_negative_feedback] Successfully saved negative feedback. Feedback ID: {feedback_record['feedback_id']} for interaction: {interaction_id}\n")
        
        return JSONResponse({
            "response": "ok",
            "message": "Negative feedback recorded successfully.",
            "feedback_id": feedback_record["feedback_id"]
        })
        
    except Exception as e:
        logger.error(f"[submit_negative_feedback] Error saving negative feedback for interaction {interaction_id}: {e}\n", exc_info=True)
        
        return JSONResponse(
            content={"response": "error", "message": "Failed to save negative feedback."}, 
            status_code=500
        )

@router.get("/feedback/stats")
async def get_feedback_stats():
    """
    Get basic statistics about collected negative feedback.
    
    This endpoint provides summary information about the negative feedback
    collected so far. Useful for monitoring system health and understanding
    user satisfaction patterns.
    
    Returns:
        JSONResponse: Feedback statistics including total count, daily count, etc.
        
    HTTP Status Codes:
        200: Statistics successfully retrieved
        500: Server error (unable to read feedback data)
        
    Example Response:
        {
            "response": "success",
            "statistics": {
                "total_negative_feedback": 15,
                "feedback_today": 3,
                "latest_feedback_time": "2024-12-06T14:30:45.123456+00:00",
                "unique_interactions": 12
            }
        }
        
    Note:
        - Returns zeros/null values if no feedback has been collected
        - Uses UTC timezone for timestamp consistency
        - Counts unique interactions that received negative feedback
    """
    logger.info("[get_feedback_stats] Feedback statistics requested\n")
    
    try:
        stats = get_negative_feedback_statistics()
        
        logger.info(f"[get_feedback_stats] Successfully retrieved feedback statistics: {stats['total_negative_feedback']} total negative feedback entries\n")
        
        return JSONResponse(content={
            "response": "ok",
            "data": stats
        })
        
    except Exception as e:
        logger.error(f"[get_feedback_stats] Error retrieving feedback statistics: {e}\n", exc_info=True)
        
        return JSONResponse(
            content={"response": "error", "message": "Failed to retrieve feedback statistics."}, 
            status_code=500
        )

# ---------------------------------------------------------------------------
# POSITIVE FEEDBACK ENDPOINTS - Mirror of negative feedback system  
# ---------------------------------------------------------------------------

@router.post("/feedback/positive")
async def submit_positive_feedback(
    interaction_id: str,
    user_email: Optional[str] = None
):
    """
    Submit positive feedback (thumb-up) for a specific interaction with optional user session isolation.
    
    This endpoint allows users to indicate satisfaction with an assistant response
    by providing the interaction ID. The system automatically captures the context
    (user message and assistant response) from the user-specific conversation history for later analysis.
    
    Args:
        interaction_id (str): The unique ID of the interaction receiving positive feedback
        user_email (Optional[str]): User's email for session isolation (optional for backward compatibility)
        
    Returns:
        JSONResponse: Success confirmation with feedback ID, or error details
        
    HTTP Status Codes:
        200: Feedback successfully recorded
        400: Invalid interaction ID (not found in user's conversation history)
        500: Server error (unable to save feedback)
        
    Example Request:
        POST /api/feedback/positive
        Form data: interaction_id=71653d96-ae38-480d-9e73-c5a87dd45b38&user_email=user@example.com
        
    Example Success Response:
        {
            "response": "success",
            "message": "Positive feedback recorded successfully",
            "feedback_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "interactionId": "71653d96-ae38-480d-9e73-c5a87dd45b38"
        }
        
    Note:
        - Validates interaction exists in user-specific conversation history before accepting feedback
        - Automatically extracts context from user-specific conversation history
        - Creates feedback storage file if it doesn't exist
        - Uses UTC timestamps for consistency
    """
    if user_email:
        logger.info(f"[submit_positive_feedback] Received positive feedback request for user: {user_email}, interaction: {interaction_id}\n")
    else:
        logger.warning(f"[submit_positive_feedback] Received positive feedback request WITHOUT user_email for interaction: {interaction_id} - Searching in global session\n")
    
    # Validate that the interaction exists in user-specific conversation history
    if not validate_interaction_exists(interaction_id, user_email):
        if user_email:
            logger.warning(f"[submit_positive_feedback] Invalid interaction_id: {interaction_id} for user: {user_email} - not found in user-specific conversation history\n")
            error_message = "Invalid interaction ID. The specified interaction was not found in your conversation history."
        else:
            logger.warning(f"[submit_positive_feedback] Invalid interaction_id: {interaction_id} - not found in global conversation history. Possible session mismatch: frontend may be using user-specific session while backend received no user_email parameter.\n")
            error_message = "Invalid interaction ID. The specified interaction was not found in conversation history."
            
        return JSONResponse(
            content={
                "response": "error", 
                "message": error_message
            }, 
            status_code=400
        )
    
    try:
        # Save the positive feedback with user context
        feedback_record = save_positive_feedback(interaction_id, user_email)
        
        logger.info(f"[submit_positive_feedback] Successfully saved positive feedback. Feedback ID: {feedback_record['feedback_id']} for interaction: {interaction_id}\n")
        
        return JSONResponse({
            "response": "ok",
            "message": "Positive feedback recorded successfully.",
            "feedback_id": feedback_record["feedback_id"]
        })
        
    except Exception as e:
        logger.error(f"[submit_positive_feedback] Error saving positive feedback for interaction {interaction_id}: {e}\n", exc_info=True)
        
        return JSONResponse(
            content={"response": "error", "message": "Failed to save positive feedback."}, 
            status_code=500
        )

@router.get("/feedback/negative/stats")
async def get_negative_feedback_stats():
    """
    Get basic statistics about collected negative feedback.
    
    This endpoint provides summary information about the negative feedback
    collected so far. Useful for monitoring system health and understanding
    user dissatisfaction patterns.
    
    Returns:
        JSONResponse: Negative feedback statistics including total count, daily count, etc.
        
    HTTP Status Codes:
        200: Statistics successfully retrieved
        500: Server error (unable to read feedback data)
        
    Example Response:
        {
            "response": "success",
            "statistics": {
                "total_negative_feedback": 15,
                "feedback_today": 3,
                "latest_feedback_time": "2024-12-06T14:30:45.123456+00:00",
                "unique_interactions": 12
            }
        }
        
    Note:
        - Returns zeros/null values if no feedback has been collected
        - Uses UTC timezone for timestamp consistency
        - Counts unique interactions that received negative feedback
    """
    logger.info("[get_negative_feedback_stats] Negative feedback statistics requested\n")
    
    try:
        stats = get_negative_feedback_statistics()
        
        logger.info(f"[get_negative_feedback_stats] Successfully retrieved negative feedback statistics: {stats['total_negative_feedback']} total negative feedback entries\n")
        
        return JSONResponse(content={
            "response": "ok",
            "data": stats
        })
        
    except Exception as e:
        logger.error(f"[get_negative_feedback_stats] Error retrieving negative feedback statistics: {e}\n", exc_info=True)
        
        return JSONResponse(
            content={"response": "error", "message": "Failed to retrieve negative feedback statistics."}, 
            status_code=500
        )

@router.get("/feedback/positive/stats")
async def get_positive_feedback_stats():
    """
    Get basic statistics about collected positive feedback.
    
    This endpoint provides summary information about the positive feedback
    collected so far. Useful for monitoring system health and understanding
    user satisfaction patterns.
    
    Returns:
        JSONResponse: Positive feedback statistics including total count, daily count, etc.
        
    HTTP Status Codes:
        200: Statistics successfully retrieved
        500: Server error (unable to read feedback data)
        
    Example Response:
        {
            "response": "success",
            "statistics": {
                "total_positive_feedback": 25,
                "feedback_today": 8,
                "latest_feedback_time": "2024-12-06T15:45:30.987654+00:00",
                "unique_interactions": 20
            }
        }
        
    Note:
        - Returns zeros/null values if no feedback has been collected
        - Uses UTC timezone for timestamp consistency
        - Counts unique interactions that received positive feedback
    """
    logger.info("[get_positive_feedback_stats] Positive feedback statistics requested\n")
    
    try:
        stats = get_positive_feedback_statistics()
        
        logger.info(f"[get_positive_feedback_stats] Successfully retrieved positive feedback statistics: {stats['total_positive_feedback']} total positive feedback entries\n")
        
        return JSONResponse(content={
            "response": "ok",
            "data": stats
        })
        
    except Exception as e:
        logger.error(f"[get_positive_feedback_stats] Error retrieving positive feedback statistics: {e}\n", exc_info=True)
        
        return JSONResponse(
            content={"response": "error", "message": "Failed to retrieve positive feedback statistics."}, 
            status_code=500
        )








