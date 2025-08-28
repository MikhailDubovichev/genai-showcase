"""
pipelines/device_control/pipeline_device_control.py

Device control pipeline implementation.

This module contains the main DeviceControlPipeline class that handles
device control, automation, and smart home management queries.
"""

from typing import Dict, Any
import logging
from config.logging_config import get_logger

from ..base import BasePipeline
from .tools.device_tools import DeviceControlToolManager
from llm_cloud.provider import get_client
from shared.utils import create_error_response, handle_tool_call, SessionContext

logger = get_logger(__name__)

class DeviceControlPipeline(BasePipeline):
    """
    Pipeline for handling device control and smart home automation queries.
    
    This pipeline processes requests related to:
    - Device control (turn on/off, dimming, etc.)
    - Device listing and status queries
    - Scheduling and automation
    - Energy consumption monitoring
    - Smart Home Integrator API operations
    
    Features:
    - Full tool ecosystem for device operations
    - Smart Home Integrator API integration
    - Conversation history management
    - Tool call handling and execution
    """
    
    def setup(self) -> None:
        """
        Initialize device control pipeline resources.
        
        Sets up:
        - LLM client for device control model
        - Device control tool manager
        - System prompt for device operations
        """
        # Initialize LLM client
        self.client = get_client()
        
        # Initialize device control tool manager
        self.tool_manager = DeviceControlToolManager()
        
        # Load device control system prompt from config
        self.system_prompt = self.config['device_control_message']
        
        # Get device control model configuration
        self.model_config = self.config["llm"]["models"]["device_control"]
        
        logger.info(
            "Pipeline setup complete",
            extra={
                'pipeline_name': self.get_pipeline_name(),
                'tool_count': len(self.tool_manager.get_tool_names()),
                'model_name': self.model_config["name"]
            }
        )
    
    def _process_message_internal(
        self,
        message: str,
        token: str,
        location_id: str,
        user_email: str,
        interaction_id: str
    ) -> Dict[str, Any]:
        """
        Process device control message and return response.
        
        Args:
            message (str): User's device control request
            token (str): Authentication token for external services
            location_id (str): The user's location identifier
            user_email (str): The user's email for session management
            interaction_id (str): Unique identifier for this interaction
            
        Returns:
            Dict[str, Any]: The processed response
        """
        try:
            # Update logger context with interaction_id
            logger.extra.update({
                'interaction_id': interaction_id,
                'pipeline_name': self.get_pipeline_name()
            })
            
            # Create session context for tool handler
            session_context = SessionContext(user_email)
            
            # Inject interaction_id into system prompt for LLM response
            system_prompt_with_id = self.system_prompt + f"\n\nFor this conversation turn, use this interactionId in your JSON response: {interaction_id}\n"
            
            # Prepare messages for LLM: include both system and user messages
            messages = [
                {"role": "system", "content": system_prompt_with_id},
                {"role": "user", "content": message},
            ]
            
            logger.info(
                "Making initial LLM request",
                extra={
                    'model': self.model_config["name"],
                    'max_tokens': self.model_config["settings"]["max_tokens"],
                    'message_preview': message[:50],
                    'tool_count': len(self.tool_manager.get_tool_definitions())
                }
            )
            
            # Make initial LLM request with tools enabled
            from config import CONFIG as EDGE_CONFIG
            provider = (EDGE_CONFIG.get("llm", {}) or {}).get("provider", "nebius").lower()
            extra_kwargs = {}
            if provider != "openai":
                extra_kwargs["extra_body"] = {"top_k": self.model_config["settings"]["top_k"]}

            response = self.client.chat.completions.create(
                model=self.model_config["name"],
                max_tokens=self.model_config["settings"]["max_tokens"],
                temperature=self.model_config["settings"]["temperature"],
                top_p=self.model_config["settings"]["top_p"],
                messages=messages,
                tools=self.tool_manager.get_tool_definitions(),
                tool_choice="auto",
                **extra_kwargs,
            )
            
            # Handle tool calls if present
            if response.choices[0].message.tool_calls:
                logger.info(
                    "Processing tool calls",
                    extra={
                        'tool_call_count': len(response.choices[0].message.tool_calls)
                    }
                )
                
                # Handle tool calls
                assistant_response = handle_tool_call(
                    response, 
                    interaction_id, 
                    token, 
                    location_id, 
                    system_prompt_with_id, 
                    self.client, 
                    session_context
                )
            else:
                logger.info("No tool calls in response")
                assistant_response = response.choices[0].message.content
            
            logger.info(
                "Successfully processed message",
                extra={
                    'response_length': len(assistant_response),
                    'had_tool_calls': bool(response.choices[0].message.tool_calls)
                }
            )
            
            return {
                "response_content": assistant_response,
                "interaction_id": interaction_id
            }
            
        except Exception as e:
            logger.error(
                "Error processing message",
                exc_info=True,
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            error_response = create_error_response(
                "I apologize, but I encountered an issue processing your device control request. Please try again.",
                interaction_id
            )
            return {
                "response_content": error_response,
                "interaction_id": interaction_id
            }
    
    def get_pipeline_name(self) -> str:
        """
        Return the canonical identifier for this pipeline used in logs and metrics.
        
        Returns:
            str: The string literal "device_control". This value is consumed by logging adapters,
            Prometheus label helpers, and the orchestrator when building diagnostic payloads.
        """
        return "device_control" 