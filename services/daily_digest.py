"""
Daily Digest Service
-------------------
This module generates simple daily energy efficiency reports that appear when users open the chat.
The digest is injected as the first message in the conversation to provide proactive energy advice.

For now, this uses static content to validate the concept. Later it can be enhanced with:
- Dynamic spot price analysis
- Personalized recommendations based on user behavior
- LLM-generated content for variety
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Import the existing user hash function to maintain consistency in naming
from services.history_manager import get_user_hash
from config import CONFIG

# Create a logger for this module
logger = logging.getLogger(__name__)

def generate_daily_digest() -> Dict[str, Any]:
    """
    Generate a simple daily energy efficiency digest.
    
    This function creates a static daily report with basic energy saving advice.
    The report follows the JSON schema expected by the frontend for 'dailyReport' type messages.
    
    Returns:
        Dict[str, Any]: A dictionary containing the daily digest in the format expected by the LLM:
        {
            "message": "Friendly introduction to the daily digest",
            "type": "dailyReport", 
            "content": [
                {
                    "title": "Daily Energy Tip",
                    "tip": "Specific energy saving advice",
                    "potentialSavings": "Estimated savings information",
                    "date": "Current date",
                    "tipNumber": 1,
                    "totalTips": 5
                }
            ]
        }
    """
    
    # Get current date for the digest
    current_date = datetime.now().strftime("%B %d, %Y")  # e.g., "December 13, 2025"
    
    # Static energy tips for now - later this can be dynamic/LLM-generated
    energy_tips = [
        {
            "tip": "When boiling water, only fill your kettle with the amount you actually need. Most people boil 2-3 times more water than necessary.",
            "potential_savings": "This simple habit can save up to €50 per year on your electricity bill."
        },
        {
            "tip": "Check for 'phantom loads' - devices that consume power even when turned off. Common culprits include TVs, coffee makers, and phone chargers.",
            "potential_savings": "Eliminating phantom loads can reduce your electricity consumption by 5-10%."
        },
        {
            "tip": "Use your dishwasher's eco mode and only run it when it's full. The eco mode uses less water and energy, even though it takes longer.",
            "potential_savings": "This can save up to €40 per year compared to normal wash cycles."
        },
        {
            "tip": "Set your water heater temperature to 60°C (140°F). Higher temperatures waste energy and can be dangerous.",
            "potential_savings": "Lowering from 70°C to 60°C can save 6-10% on water heating costs."
        },
        {
            "tip": "Close curtains and blinds during hot summer days to keep your home cooler naturally, reducing air conditioning needs.",
            "potential_savings": "This simple step can reduce cooling costs by up to 15% during summer months."
        }
    ]
    
    # Select today's tip based on the day of year (simple rotation)
    day_of_year = datetime.now().timetuple().tm_yday
    # The % operator (modulo) gives the remainder after division, so it will always be between 0 and len(energy_tips) - 1
    selected_tip = energy_tips[day_of_year % len(energy_tips)]
    
    # Create the digest content
    digest_content = {
        "message": f"Good morning! Here's your daily energy efficiency digest for {current_date}.",
        "type": "dailyReport",
        "content": [
            {
                "title": "Daily Energy Tip",
                "tip": selected_tip["tip"],
                "potentialSavings": selected_tip["potential_savings"],
                "date": current_date,
                "tipNumber": (day_of_year % len(energy_tips)) + 1,
                "totalTips": len(energy_tips)
            }
        ]
    }
    
    logger.info(f"[generate_daily_digest] Generated daily digest for {current_date} with tip #{(day_of_year % len(energy_tips)) + 1}\n")
    return digest_content

def should_show_daily_digest(location_id: str, user_email: Optional[str] = None) -> bool:
    """
    Determine if the daily digest should be shown for this user today.
    
    This function checks if a digest has already been shown today for this specific user
    to avoid showing multiple digests in the same day, while ensuring each user in a 
    household gets their own daily digest.
    
    Args:
        location_id (str): The Smart Home Integrator location identifier
        user_email (Optional[str]): User's email for per-user tracking
        
    Returns:
        bool: True if digest should be shown, False if already shown today to this user
        
    Note:
        Uses file-based tracking with the same user hash system as conversation history.
        If user_email is None (legacy behavior), falls back to always showing digest.
    """
    
    # If no user email provided, fall back to always showing (legacy behavior)
    if not user_email:
        logger.debug(f"[should_show_daily_digest] No user_email provided for location_id: {location_id} - showing digest (legacy mode)\n")
        return True
    
    # Create digest tracking directory if it doesn't exist
    digest_tracking_dir = os.path.join(
        CONFIG['paths']['user_data_full_path'],
        CONFIG['paths'].get('digest_tracking_subdir_name', 'digest_tracking')
    )
    os.makedirs(digest_tracking_dir, exist_ok=True)
    
    # Use the same user hash function as conversation history for consistency
    user_hash = get_user_hash(user_email)
    tracking_file = os.path.join(digest_tracking_dir, f"{user_hash}_digest_log.json")
    
    # Get today's date string for comparison
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Try to load existing tracking data
        if os.path.exists(tracking_file):
            with open(tracking_file, 'r', encoding='utf-8') as f:
                tracking_data = json.load(f)
            
            # Check if digest was already shown today
            last_digest_date = tracking_data.get("last_digest_date")
            if last_digest_date == today:
                logger.debug(f"[should_show_daily_digest] Daily digest already shown today ({today}) for user: {user_email}\n")
                return False
        else:
            # File doesn't exist, this is the first time for this user
            tracking_data = {}
        
        # Update tracking data with today's date
        tracking_data["last_digest_date"] = today
        tracking_data["location_id"] = location_id
        tracking_data["user_email"] = user_email
        tracking_data["user_hash"] = user_hash
        tracking_data["last_updated"] = datetime.now().isoformat()
        
        # Save updated tracking data
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(tracking_data, f, indent=2)
        
        logger.info(f"[should_show_daily_digest] Daily digest approved for user: {user_email} (hash: {user_hash}) on {today}\n")
        return True
        
    except Exception as e:
        # If anything goes wrong with file operations, err on the side of showing the digest
        logger.error(f"[should_show_daily_digest] Error checking digest tracking for user {user_email}: {e}\n", exc_info=True)
        logger.info(f"[should_show_daily_digest] Showing digest due to tracking error for user: {user_email}\n")
        return True

def format_digest_for_injection(digest: Dict[str, Any], interaction_id: str) -> str:
    """
    Format the daily digest for injection into the conversation history.
    
    This function takes the digest content and formats it as a JSON string
    that can be injected as an assistant message in the conversation history.
    
    The injected digest will be automatically loaded by get_llm_response() when
    processing user messages, making the digest content available to the LLM
    for answering follow-up questions (e.g., "Tell me more about that water heater tip").
    
    Args:
        digest (Dict[str, Any]): The digest content from generate_daily_digest()
        interaction_id (str): The interaction ID for this message
        
    Returns:
        str: JSON string formatted for conversation history injection
    """
    # Add the interaction ID to the digest
    digest_with_id = digest.copy()
    digest_with_id["interactionId"] = interaction_id
    
    # Convert to JSON string for conversation history
    return json.dumps(digest_with_id, ensure_ascii=False, indent=2) 