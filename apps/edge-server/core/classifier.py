"""
core/classifier.py

Message classification for pipeline routing.

This module provides centralized message classification that determines
which pipeline should handle each user request. This is now the single
source of truth for classification logic.
"""

import logging
from shared.models import MessageCategory
from llm_cloud.provider import get_client
from config import CONFIG
import os

logger = logging.getLogger(__name__)

class MessageClassifier:
    """
    Central message classifier that assigns user messages to pipeline categories for routing.
    
    Responsibilities:
    - Load and cache the classification system prompt and fallback responses from the configuration directory
    - Use the configured LLM to infer the most appropriate category for a message
    - Provide a direct rejection response for messages that fall outside supported domains
    
    Design notes:
    - This class encapsulates model access so that other modules do not need to know about client
      creation or prompt loading. It also centralizes category mapping and logging.
    - Classification is kept intentionally lightweight to minimize latency before pipeline execution.
    """
    
    def __init__(self):
        """
        Initialize the classifier by creating an LLM client and loading prompt templates from disk.
        
        The constructor builds a client via `get_client()` and reads two text files under `config/`:
        - `classification_system_prompt.txt`: The instruction template the model uses to classify input
        - `other_queries_response.txt`: A predefined textual response for unsupported queries
        These resources are read once at initialization to avoid per‑request filesystem overhead.
        """
        self.client = get_client()
        
        # Load classification prompt
        classification_prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'classification_system_prompt.txt')
        with open(classification_prompt_path, 'r', encoding='utf-8') as f:
            self.CLASSIFICATION_PROMPT = f.read().strip()
            
        # Load other queries response template
        other_queries_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'other_queries_response.txt')
        with open(other_queries_path, 'r', encoding='utf-8') as f:
            self.OTHER_QUERIES_RESPONSE = f.read().strip()
            
        logger.info("[MessageClassifier] Initialized core classifier")
    
    def classify_message(self, message: str) -> MessageCategory:
        """
        Classify a user message into one of the supported categories for pipeline routing.
        
        The model receives a system prompt that embeds the user's message and returns a textual label which is
        normalized and mapped to the `MessageCategory` enum. If the model output cannot be interpreted, the method
        falls back to `MessageCategory.OTHER_QUERIES` to keep the system resilient.
        
        Args:
            message (str): The raw user input to analyze.
        
        Returns:
            MessageCategory: The category used by the orchestrator to choose a pipeline. One of
            DEVICE_CONTROL, ENERGY_EFFICIENCY, or OTHER_QUERIES.
        
        Note:
            Any exception during model invocation or parsing is handled by returning OTHER_QUERIES as a
            safe default to avoid user‑visible errors during classification.
        """
        try:
            model_config = CONFIG["llm"]["models"]["classification"]
            response = self.client.chat.completions.create(
                model=model_config["name"],
                messages=[{
                    "role": "system", 
                    "content": self.CLASSIFICATION_PROMPT.format(message=message)
                }],
                max_tokens=model_config["settings"]["max_tokens"],
                temperature=model_config["settings"]["temperature"],
                top_p=model_config["settings"]["top_p"],
                extra_body={"top_k": model_config["settings"]["top_k"]}
            )
            
            category_str = response.choices[0].message.content.strip().upper()
            
            # Map to enum
            logger.info(f"[MessageClassifier] Raw model output: '{response.choices[0].message.content.strip()}'")
            logger.info(f"[MessageClassifier] Normalized: '{category_str}'")
            
            if "DEVICE_CONTROL" in category_str:
                result = MessageCategory.DEVICE_CONTROL
            elif "ENERGY_EFFICIENCY" in category_str:
                result = MessageCategory.ENERGY_EFFICIENCY
            else:
                result = MessageCategory.OTHER_QUERIES
                
            logger.debug(f"[MessageClassifier] Classified '{message[:50]}...' as {result.value}")
            return result
            
        except Exception as e:
            logger.error(f"[MessageClassifier] Classification error: {e}")
            return MessageCategory.OTHER_QUERIES  # Safe fallback
    
    def get_direct_rejection_response(self, interaction_id: str) -> str:
        """
        Build a standardized JSON string for immediate rejection of unsupported queries.
        
        This helper returns a minimal, user‑friendly message that informs the user the request is outside the
        system's supported scope. The response includes the provided `interaction_id` so it can be stored in the
        conversation history and correlated in logs. The JSON structure matches the assistant's response schema
        expected by the frontend (message, interactionId, type, content).
        
        Args:
            interaction_id (str): The unique identifier to include in the response payload.
        
        Returns:
            str: A JSON‑encoded string representing the rejection response.
        """
        import json
        return json.dumps({
            'message': self.OTHER_QUERIES_RESPONSE,
            'interactionId': interaction_id,
            'type': 'text',
            'content': []
        }) 