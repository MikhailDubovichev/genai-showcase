"""
Manages conversation history, including interaction IDs, saving, loading, and archiving.

This module provides the core functionalities for persisting chat conversations
to the file system, ensuring that each interaction is uniquely identified and
that conversations can be managed over time.

Enhanced with user-specific session support to prevent conversation interference
between multiple users in the same location.
"""
import json
import uuid
import os
import datetime
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from config import CONFIG

# Define the base directory for user data and specific subdirectories using CONFIG
USER_DATA_DIR = CONFIG['paths']['user_data_full_path']
CONVERSATIONS_DIR = os.path.join(
    USER_DATA_DIR,
    CONFIG['paths'].get('conversations_subdir_name', 'conversations')
)
ACTIVE_CONVERSATION_FILENAME = CONFIG['paths'].get('active_conversation_filename', 'active_conversation.json')

# Ensure the conversation directory exists
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

def generate_interaction_id() -> str:
    """
    Generates a unique interaction ID using UUID4.

    This function creates a universally unique identifier (UUID) for each
    user-assistant interaction. The UUID4 algorithm generates random UUIDs
    that are extremely unlikely to collide.

    Returns:
        str: A UUID4 string (e.g., "a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    """
    return str(uuid.uuid4())

def get_user_hash(email: str) -> str:
    """
    Generate a secure, consistent hash from user email for file naming.
    
    This function creates a short, URL-safe hash from the user's email address
    that can be used in filenames. The hash is deterministic (same email always
    produces the same hash) but doesn't reveal the original email.
    
    Args:
        email (str): User's email address
        
    Returns:
        str: A 16-character hexadecimal hash string
        
    Examples:
        get_user_hash("user@example.com") -> "a1b2c3d4e5f6789a"
        get_user_hash("USER@EXAMPLE.COM") -> "a1b2c3d4e5f6789a" (case insensitive)
    """
    # Convert to lowercase for case-insensitive hashing
    # Use SHA-256 for security, then take first 16 characters for brevity
    return hashlib.sha256(email.lower().encode()).hexdigest()[:16]

def get_active_conversation_path(user_email: Optional[str] = None) -> str:
    """
    Returns the full path to the active conversation file.
    
    This function supports both user-specific sessions (when user_email is provided)
    and the legacy global session (when user_email is None) for backward compatibility.

    Args:
        user_email (Optional[str]): User's email for session isolation.
                                   If None, uses legacy global conversation file.

    Returns:
        str: The absolute path to the appropriate active conversation file.
        
    Examples:
        get_active_conversation_path(None) -> "user_data/conversations/active_conversation.json"
        get_active_conversation_path("user@example.com") -> "user_data/conversations/a1b2c3d4e5f6789a_active_conversation.json"
    """
    if user_email:
        user_hash = get_user_hash(user_email)
        filename = f"{user_hash}_active_conversation.json"
    else:
        # Fallback to legacy behavior for backward compatibility
        filename = ACTIVE_CONVERSATION_FILENAME
    
    return os.path.join(CONVERSATIONS_DIR, filename)

def load_conversation_history(user_email: Optional[str] = None, conversation_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads the conversation history from a specified JSON file.

    If no path is provided, it defaults to the user-specific active conversation file
    (or legacy global file if user_email is None). If the file doesn't exist or is 
    empty, an empty list is returned.

    Args:
        user_email (Optional[str]): User's email for session isolation.
                                   If None, uses legacy global conversation file.
        conversation_path (Optional[str]): The path to the conversation file.
                                          If None, uses get_active_conversation_path().

    Returns:
        List[Dict[str, Any]]: A list of message objects, where each object
                              contains 'role', 'content', 'timestamp', and
                              'interaction_id'. Returns an empty list if
                              the file doesn't exist or is invalid.
    """
    if conversation_path is None:
        conversation_path = get_active_conversation_path(user_email)

    if not os.path.exists(conversation_path):
        return []
    try:
        with open(conversation_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
            if not isinstance(history, list): # Basic validation
                return []
            return history
    except json.JSONDecodeError:
        # If the file is corrupted or not valid JSON, treat as empty
        return []
    except Exception:
        # Catch any other file reading errors
        return []


def save_message(
    interaction_id: str,
    role: str,
    content: str,
    user_email: Optional[str] = None,
    conversation_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Saves a single message to the conversation history.

    Appends the new message to the specified conversation file (or user-specific
    active conversation file by default). Each message includes the interaction ID,
    role, content, and a timestamp.

    Args:
        interaction_id (str): The unique ID for this user-assistant interaction.
        role (str): The role of the message sender (e.g., 'user', 'assistant').
        content (str): The text content of the message.
        user_email (Optional[str]): User's email for session isolation.
                                   If None, uses legacy global conversation file.
        conversation_path (Optional[str]): The path to the conversation file.
                                          If None, uses get_active_conversation_path().

    Returns:
        Dict[str, Any]: The message dictionary that was saved.
    """
    if conversation_path is None:
        conversation_path = get_active_conversation_path(user_email)

    history = load_conversation_history(user_email, conversation_path)
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    message = {
        "interaction_id": interaction_id,
        "role": role,
        "content": content,
        "timestamp": timestamp
    }
    history.append(message)
    
    try:
        with open(conversation_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        # In a production application, you'd want more sophisticated error logging here
        print(f"Error saving conversation: {e}")
        # Potentially re-raise or handle so the app knows saving failed
    
    return message


def archive_active_conversation(user_email: Optional[str] = None) -> Tuple[bool, str]:
    """
    Archives the current active conversation file by renaming it with a timestamp.

    If the user-specific active conversation file exists and is not empty, it's renamed to
    include the user hash and timestamp. For legacy support, when user_email is None,
    it uses the original global conversation file behavior.

    Args:
        user_email (Optional[str]): User's email for session isolation.
                                   If None, uses legacy global conversation file.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if archiving was successful or not needed, False otherwise.
            - str: A message indicating the result of the operation.
    """
    active_path = get_active_conversation_path(user_email)
    if not os.path.exists(active_path) or os.path.getsize(active_path) == 0:
        if user_email:
            return True, f"No active conversation to archive for user or it's empty."
        else:
            return True, "No active conversation to archive or it's empty."

    try:
        # Create a timestamped filename for the archive
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if user_email:
            user_hash = get_user_hash(user_email)
            archive_filename = f"{user_hash}_conversation_{timestamp_str}.json"
        else:
            # Legacy behavior for backward compatibility
            archive_filename = f"conversation_{timestamp_str}.json"
            
        archive_path = os.path.join(CONVERSATIONS_DIR, archive_filename)
        
        os.rename(active_path, archive_path)
        
        # After archiving, the next call to load_conversation_history() for the
        # active path will return an empty list if the active conversation file
        # doesn't exist, effectively starting a new conversation.
        # A new active conversation file will be created on the next save_message.
        
        return True, f"Active conversation archived to {archive_filename}"
    except Exception as e:
        return False, f"Error archiving conversation: {e}"
