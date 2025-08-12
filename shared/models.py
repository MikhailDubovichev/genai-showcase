"""
shared/models.py

Common data models and type definitions used across pipeline implementations.

This module contains shared data structures that standardize communication
between different components of the AI assistant system.

TODO: Add Pydantic validation for device_control pipeline responses.
Key points for implementation:
1. Base response structure is always the same:
   {
     "message": str,
     "interactionId": str,
     "type": str,  # "text", "devices", "schedule", "dynamicPrices", or "dailyReport"
     "content": list  # structure depends on type
   }

2. Validation strategy:
   - Start with just the base structure validation (like EnergyEfficiencyResponse)
   - Test with simple "text" responses (empty content list)
   - Then gradually add content validation for each type as needed
   - No need to create separate classes for each content type initially

3. Implementation steps (when Smart Home Integrator token is available):
   a. Create DeviceControlResponse model (similar to EnergyEfficiencyResponse)
   b. Add type validation to ensure type is one of the allowed values
   c. Test with basic responses first
   d. Add content validation only for the most critical response types

4. Keep it simple:
   - Don't validate every possible field initially
   - Focus on required fields that could break the frontend if missing
   - Add more detailed validation only if we encounter specific issues
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class MessageCategory(Enum):
    """
    Message categories for routing to appropriate pipeline handlers.
    
    These categories determine which pipeline will process a user's message:
    - DEVICE_CONTROL: Commands for device management, scheduling, automation
    - ENERGY_EFFICIENCY: Questions about energy saving and efficiency advice  
    - OTHER_QUERIES: Unsupported topics that get immediate rejection response
    """
    DEVICE_CONTROL = "device_control"
    ENERGY_EFFICIENCY = "energy_efficiency" 
    OTHER_QUERIES = "other_queries"

@dataclass
class SessionContext:
    """
    User session context containing all information needed for processing.
    
    This class encapsulates all the contextual information that pipelines
    need to process user requests, including authentication, conversation
    history, and user-specific data.
    """
    user_email: Optional[str]
    token: str
    location_id: str
    conversation_history: List[Dict[str, str]]
    interaction_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the session context into a plain dictionary for serialization and logging.
        
        Returns:
            Dict[str, Any]: A dictionary with keys 'user_email', 'token', 'location_id',
            'conversation_history', and 'interaction_id'. This shape is convenient for structured
            logging and for passing compact summaries across layers without exposing implementation details.
        """
        return {
            'user_email': self.user_email,
            'token': self.token,
            'location_id': self.location_id,
            'conversation_history': self.conversation_history,
            'interaction_id': self.interaction_id
        }

@dataclass 
class ProcessingResult:
    """
    Result of pipeline message processing.
    
    Standardizes the response format from pipeline processing operations,
    including success status, response content, and any error information.
    """
    success: bool
    response_content: str
    interaction_id: str
    pipeline_name: str
    error_message: Optional[str] = None
    
    def to_api_response(self) -> Dict[str, str]:
        """
        Render the processing result in the minimal dictionary form expected by API endpoints.
        
        Returns:
            Dict[str, str]: A mapping with 'response_content' and 'interaction_id'. This method exists to
            keep controller logic simple and to ensure a consistent API contract across pipelines regardless
            of internal differences in how responses are produced.
        """
        return {
            "response_content": self.response_content,
            "interaction_id": self.interaction_id
        }

class EnergyEfficiencyResponse(BaseModel):
    """
    Validate the JSON format of energy efficiency advice returned by the LLM.
    
    This schema enforces the minimal structure promised by the energy efficiency system prompt so the frontend
    can rely on consistent fields. It intentionally keeps validation light‑weight to avoid over‑constraining the
    LLM while still catching shape regressions early during testing. Over time, additional constraints can be
    introduced if new response types are added.
    """
    message: str = Field(..., description="The main response text")
    interactionId: str = Field(..., description="Unique identifier for this interaction")
    type: str = Field("text", description="Response type, always 'text' for energy efficiency")
    content: List = Field(default_factory=list, description="Empty list as per system prompt") 