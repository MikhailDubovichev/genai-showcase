"""
Feedback Management Module

This module handles the collection and storage of user feedback for AI assistant responses.
Supports both negative feedback (thumb-down) and positive feedback (thumb-up) collection 
to help identify problematic interactions and measure satisfaction levels for system improvement.

The module stores feedback in JSON format for easy analysis and reporting.
Each feedback entry is linked to a specific interaction through interaction_id.
"""
import json
import uuid
import os
import datetime
from typing import List, Dict, Any, Optional
from services.history_manager import load_conversation_history
from config import CONFIG

# Define feedback storage paths using CONFIG
USER_DATA_DIR = CONFIG['paths']['user_data_full_path']
FEEDBACK_DIR = os.path.join(
    USER_DATA_DIR,
    CONFIG['paths'].get('feedback_subdir_name', 'feedback')
)
NEGATIVE_FEEDBACK_FILENAME = CONFIG['paths'].get('negative_feedback_filename', 'negative_feedback.json')
POSITIVE_FEEDBACK_FILENAME = CONFIG['paths'].get('positive_feedback_filename', 'positive_feedback.json')

# Ensure feedback directory exists
os.makedirs(FEEDBACK_DIR, exist_ok=True)

def get_negative_feedback_path() -> str:
    """
    Returns the full path to the negative feedback storage file.
    
    This function provides the absolute path to the JSON file where
    all negative feedback is stored for analysis and system improvement.
    
    Returns:
        str: The absolute path to 'negative_feedback.json' file
        
    Note:
        The file is created automatically when first feedback is saved
    """
    return os.path.join(FEEDBACK_DIR, NEGATIVE_FEEDBACK_FILENAME)

def load_negative_feedback() -> List[Dict[str, Any]]:
    """
    Loads all negative feedback records from the storage file.
    
    This function reads the negative feedback JSON file and returns
    all feedback entries. If the file doesn't exist or is corrupted,
    returns an empty list to allow graceful handling.
    
    Returns:
        List[Dict[str, Any]]: List of negative feedback records, where each record contains:
            - feedback_id: Unique identifier for the feedback entry
            - interaction_id: ID of the interaction that received negative feedback  
            - feedback_type: Always "negative" for this function
            - timestamp: When the feedback was submitted
            - context: User message and assistant response for analysis
            
    Note:
        - Returns empty list if file doesn't exist (first run scenario)
        - Handles JSON parsing errors gracefully
        - Each feedback record is self-contained for analysis
    """
    feedback_path = get_negative_feedback_path()
    
    if not os.path.exists(feedback_path):
        return []
    
    try:
        with open(feedback_path, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)
            if not isinstance(feedback_data, list):
                return []
            return feedback_data
    except json.JSONDecodeError:
        # If the file is corrupted or contains invalid JSON, treat as empty
        return []
    except Exception:
        # Catch any other file reading errors (permissions, etc.)
        return []

def save_negative_feedback(interaction_id: str, user_email: Optional[str] = None) -> Dict[str, Any]:
    """
    Saves a negative feedback record for a specific interaction.
    
    This function records that a user gave negative feedback (thumb-down)
    to a specific assistant response. It automatically extracts the context
    (user message and assistant response) from user-specific conversation history for analysis.
    
    Args:
        interaction_id (str): The unique ID of the interaction receiving negative feedback
        user_email (Optional[str]): User's email for session isolation (optional for backward compatibility)
        
    Returns:
        Dict[str, Any]: The feedback record that was saved, containing:
            - feedback_id: Unique identifier for this feedback entry
            - interaction_id: The interaction that received negative feedback
            - feedback_type: "negative" 
            - timestamp: When feedback was submitted (ISO format)
            - context: Dictionary with user_message and assistant_response
            
    Raises:
        Exception: If there's an error saving the feedback to file
        
    Note:
        - Automatically extracts context from user-specific conversation history
        - Creates the feedback file if it doesn't exist
        - Appends to existing feedback records
        - Uses UTC timestamps for consistency
    """
    feedback_list = load_negative_feedback()
    
    # Extract context from user-specific conversation history
    context = extract_interaction_context(interaction_id, user_email)
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    feedback_record = {
        "feedback_id": str(uuid.uuid4()),
        "interaction_id": interaction_id,
        "feedback_type": "negative",
        "timestamp": timestamp,
        "context": context
    }
    
    feedback_list.append(feedback_record)
    
    try:
        feedback_path = get_negative_feedback_path()
        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump(feedback_list, f, indent=4)
    except Exception as e:
        # Re-raise the exception so the API can handle the error appropriately
        raise Exception(f"Failed to save negative feedback: {e}")
    
    return feedback_record

def extract_interaction_context(interaction_id: str, user_email: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts all messages related to a given interaction ID from user-specific conversation history.
    
    This function searches through the user-specific conversation history to find ALL messages
    associated with the specified interaction ID, including user messages, assistant
    responses, and tool calls. This complete context is crucial for analyzing
    what went wrong when users provide negative feedback.
    
    Args:
        interaction_id (str): The interaction ID to search for in conversation history
        user_email (Optional[str]): User's email for session isolation (optional for backward compatibility)
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - user_message: The original user request (empty string if not found)
            - assistant_response: The final assistant response (empty string if not found)
            - all_messages: List of ALL messages for this interaction_id in chronological order
            
    Note:
        - Searches the user-specific conversation history
        - Returns empty values if messages not found (graceful degradation)
        - Could be extended to search archived conversations if needed
        - Captures ALL messages for the interaction_id for complete feedback analysis
    """
    conversation_history = load_conversation_history(user_email=user_email)
    
    user_message = ""
    assistant_response = ""
    all_messages = []
    
    # Find all messages with matching interaction_id
    for message in conversation_history:
        if message.get("interaction_id") == interaction_id:
            all_messages.append(message)
            
            role = message.get("role")
            content = message.get("content", "")
            
            if role == "user":
                user_message = content
            elif role == "assistant" and content:
                # Only capture non-null assistant responses
                assistant_response = content
    
    return {
        "user_message": user_message,
        "assistant_response": assistant_response,
        "all_messages": all_messages
    }

def validate_interaction_exists(interaction_id: str, user_email: Optional[str] = None) -> bool:
    """
    Validates that an interaction ID exists in the user-specific conversation history.
    
    This function checks if the provided interaction ID corresponds to
    an actual conversation interaction in the user's session, preventing invalid feedback submissions.
    It's important to validate this to maintain data integrity.
    
    Args:
        interaction_id (str): The interaction ID to validate
        user_email (Optional[str]): User's email for session isolation (optional for backward compatibility)
        
    Returns:
        bool: True if the interaction ID exists in user's conversation history, False otherwise
        
    Note:
        - Searches through user-specific conversation history
        - Prevents feedback for non-existent interactions
        - Could be extended to search archived conversations
        - Used by API endpoint to validate requests
    """
    conversation_history = load_conversation_history(user_email=user_email)
    
    for message in conversation_history:
        if message.get("interaction_id") == interaction_id:
            return True
    
    return False

def get_negative_feedback_statistics() -> Dict[str, Any]:
    """
    Provides basic statistics about collected negative feedback.
    
    This function generates summary statistics about the negative feedback
    collected so far. Useful for quick analysis and monitoring system health.
    
    Returns:
        Dict[str, Any]: Statistics dictionary containing:
            - total_negative_feedback: Total number of negative feedback entries
            - feedback_today: Number of negative feedback entries from today
            - latest_feedback_time: Timestamp of most recent feedback
            - unique_interactions: Number of unique interactions with negative feedback
            
    Note:
        - Provides quick overview without detailed analysis
        - Uses UTC timezone for consistency
        - Returns zeros/None for empty feedback data
    """
    feedback_data = load_negative_feedback()
    
    if not feedback_data:
        return {
            "total_negative_feedback": 0,
            "feedback_today": 0,
            "latest_feedback_time": None,
            "unique_interactions": 0
        }
    
    # Calculate today's feedback count
    today = datetime.date.today()
    feedback_today = 0
    
    for feedback in feedback_data:
        try:
            feedback_time = datetime.datetime.fromisoformat(feedback.get("timestamp", ""))
            if feedback_time.date() == today:
                feedback_today += 1
        except Exception:
            continue
    
    # Get unique interaction IDs
    unique_interactions = len(set(f.get("interaction_id") for f in feedback_data))
    
    # Get latest feedback time
    latest_feedback_time = None
    if feedback_data:
        try:
            latest_feedback_time = max(f.get("timestamp") for f in feedback_data)
        except Exception:
            pass
    
    return {
        "total_negative_feedback": len(feedback_data),
        "feedback_today": feedback_today,
        "latest_feedback_time": latest_feedback_time,
        "unique_interactions": unique_interactions
    }

# ---------------------------------------------------------------------------
# POSITIVE FEEDBACK FUNCTIONS - Mirror of negative feedback system
# ---------------------------------------------------------------------------

def get_positive_feedback_path() -> str:
    """
    Returns the full path to the positive feedback storage file.
    
    This function provides the absolute path to the JSON file where
    all positive feedback is stored for analysis and system improvement.
    
    Returns:
        str: The absolute path to 'positive_feedback.json' file
        
    Note:
        The file is created automatically when first feedback is saved
    """
    return os.path.join(FEEDBACK_DIR, POSITIVE_FEEDBACK_FILENAME)

def load_positive_feedback() -> List[Dict[str, Any]]:
    """
    Loads all positive feedback records from the storage file.
    
    This function reads the positive feedback JSON file and returns
    all feedback entries. If the file doesn't exist or is corrupted,
    returns an empty list to allow graceful handling.
    
    Returns:
        List[Dict[str, Any]]: List of positive feedback records, where each record contains:
            - feedback_id: Unique identifier for the feedback entry
            - interaction_id: ID of the interaction that received positive feedback  
            - feedback_type: Always "positive" for this function
            - timestamp: When the feedback was submitted
            - context: User message and assistant response for analysis
            
    Note:
        - Returns empty list if file doesn't exist (first run scenario)
        - Handles JSON parsing errors gracefully
        - Each feedback record is self-contained for analysis
    """
    feedback_path = get_positive_feedback_path()
    
    if not os.path.exists(feedback_path):
        return []
    
    try:
        with open(feedback_path, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)
            if not isinstance(feedback_data, list):
                return []
            return feedback_data
    except json.JSONDecodeError:
        # If the file is corrupted or contains invalid JSON, treat as empty
        return []
    except Exception:
        # Catch any other file reading errors (permissions, etc.)
        return []

def save_positive_feedback(interaction_id: str, user_email: Optional[str] = None) -> Dict[str, Any]:
    """
    Saves a positive feedback record for a specific interaction.
    
    This function records that a user gave positive feedback (thumb-up)
    to a specific assistant response. It automatically extracts the context
    (user message and assistant response) from user-specific conversation history for analysis.
    
    Args:
        interaction_id (str): The unique ID of the interaction receiving positive feedback
        user_email (Optional[str]): User's email for session isolation (optional for backward compatibility)
        
    Returns:
        Dict[str, Any]: The feedback record that was saved, containing:
            - feedback_id: Unique identifier for this feedback entry
            - interaction_id: The interaction that received positive feedback
            - feedback_type: "positive" 
            - timestamp: When feedback was submitted (ISO format)
            - context: Dictionary with user_message and assistant_response
            
    Raises:
        Exception: If there's an error saving the feedback to file
        
    Note:
        - Automatically extracts context from user-specific conversation history
        - Creates the feedback file if it doesn't exist
        - Appends to existing feedback records
        - Uses UTC timestamps for consistency
    """
    feedback_list = load_positive_feedback()
    
    # Extract context from user-specific conversation history
    context = extract_interaction_context(interaction_id, user_email)
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    feedback_record = {
        "feedback_id": str(uuid.uuid4()),
        "interaction_id": interaction_id,
        "feedback_type": "positive",
        "timestamp": timestamp,
        "context": context
    }
    
    feedback_list.append(feedback_record)
    
    try:
        feedback_path = get_positive_feedback_path()
        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump(feedback_list, f, indent=4)
    except Exception as e:
        # Re-raise the exception so the API can handle the error appropriately
        raise Exception(f"Failed to save positive feedback: {e}")
    
    return feedback_record

def get_positive_feedback_statistics() -> Dict[str, Any]:
    """
    Provides basic statistics about collected positive feedback.
    
    This function generates summary statistics about the positive feedback
    collected so far. Useful for measuring user satisfaction and system health.
    
    Returns:
        Dict[str, Any]: Statistics dictionary containing:
            - total_positive_feedback: Total number of positive feedback entries
            - feedback_today: Number of positive feedback entries from today
            - latest_feedback_time: Timestamp of most recent feedback
            - unique_interactions: Number of unique interactions with positive feedback
            
    Note:
        - Provides quick overview without detailed analysis
        - Uses UTC timezone for consistency
        - Returns zeros/None for empty feedback data
    """
    feedback_data = load_positive_feedback()
    
    if not feedback_data:
        return {
            "total_positive_feedback": 0,
            "feedback_today": 0,
            "latest_feedback_time": None,
            "unique_interactions": 0
        }
    
    # Calculate today's feedback count
    today = datetime.date.today()
    feedback_today = 0
    
    for feedback in feedback_data:
        try:
            feedback_time = datetime.datetime.fromisoformat(feedback.get("timestamp", ""))
            if feedback_time.date() == today:
                feedback_today += 1
        except Exception:
            continue
    
    # Get unique interaction IDs
    unique_interactions = len(set(f.get("interaction_id") for f in feedback_data))
    
    # Get latest feedback time
    latest_feedback_time = None
    if feedback_data:
        try:
            latest_feedback_time = max(f.get("timestamp") for f in feedback_data)
        except Exception:
            pass
    
    return {
        "total_positive_feedback": len(feedback_data),
        "feedback_today": feedback_today,
        "latest_feedback_time": latest_feedback_time,
        "unique_interactions": unique_interactions
    }

