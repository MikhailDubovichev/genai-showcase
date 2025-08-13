"""
core/orchestrator.py

Central pipeline orchestrator for routing user requests.

This module contains the main coordination logic that:
1. Classifies incoming user messages
2. Routes them to appropriate pipeline processors
3. Manages session context and conversation history
4. Handles errors and fallbacks
"""

from typing import Dict, Any, Optional
import logging
from config.logging_config import get_logger

from .classifier import MessageClassifier
from shared.models import MessageCategory, SessionContext, ProcessingResult
from pipelines.device_control import DeviceControlPipeline
from pipelines.energy_efficiency import EnergyEfficiencyPipeline
from services.history_manager import generate_interaction_id, save_message
from shared.utils import SessionContext

logger = get_logger(__name__)

class PipelineOrchestrator:
    """
    Central orchestrator that routes classified messages to appropriate pipelines.
    
    Responsibilities:
    - Message classification using the core classifier
    - Pipeline selection and routing based on classification
    - Session context management and conversation history
    - Error handling and fallback responses
    - Integration with existing services (history_manager, context)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator with configuration and pipelines.
        
        Args:
            config (Dict[str, Any]): Global configuration dictionary
        """
        self.config = config
        self.classifier = MessageClassifier()
        
        # Initialize pipelines
        self.pipelines = {
            MessageCategory.DEVICE_CONTROL: DeviceControlPipeline(),
            MessageCategory.ENERGY_EFFICIENCY: EnergyEfficiencyPipeline()
        }
        
        logger.info("Initialized with %d pipelines", len(self.pipelines))
    
    def process_query(
        self, 
        message: str, 
        token: str, 
        location_id: str, 
        user_email: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Main entry point for query processing through the pipeline architecture.
        
        Args:
            message (str): User input message
            token (str): Smart Home Integrator API authentication token
            location_id (str): Smart Home Integrator service location identifier
            user_email (Optional[str]): User identifier for session management
            
        Returns:
            Dict[str, str]: Dictionary containing response_content and interaction_id
        """
        # Generate unique interaction ID
        interaction_id = generate_interaction_id()
        
        # Set interaction_id in logger context
        logger.extra.update({'interaction_id': interaction_id})
        
        logger.info(
            "Processing query for location %s", 
            location_id,
            extra={
                'message_preview': message[:50],
                'location_id': location_id,
                'user_email': user_email
            }
        )
        
        # Save user message to history
        save_message(
            interaction_id=interaction_id,
            role="user",
            content=message,
            user_email=user_email
        )
        
        try:
            # 1. Classify the message
            category = self.classifier.classify_message(message)
            logger.info(
                "Message classified", 
                extra={
                    'category': category.value,
                    'message_preview': message[:50]
                }
            )
            
            # 2. Handle OTHER_QUERIES immediately without pipeline processing
            if category == MessageCategory.OTHER_QUERIES:
                response_content = self.classifier.get_direct_rejection_response(interaction_id)
                
                # Save response to history
                save_message(
                    interaction_id=interaction_id,
                    role="assistant",
                    content=response_content,
                    user_email=user_email
                )
                
                logger.info(
                    "Handled OTHER_QUERIES category", 
                    extra={'response_type': 'direct_rejection'}
                )
                
                return {
                    "response_content": response_content,
                    "interaction_id": interaction_id
                }
            
            # 3. Route to appropriate pipeline
            pipeline = self.pipelines[category]
            
            # Update logger context with pipeline info
            logger.extra.update({'pipeline_name': pipeline.get_pipeline_name()})
            
            logger.info(
                "Routing to pipeline", 
                extra={
                    'pipeline_name': pipeline.get_pipeline_name(),
                    'category': category.value
                }
            )
            
            result = pipeline.process_message(
                message=message,
                token=token,
                location_id=location_id,
                user_email=user_email,
                interaction_id=interaction_id
            )
            
            # 4. Save assistant response to history
            save_message(
                interaction_id=interaction_id,
                role="assistant",
                content=result["response_content"],
                user_email=user_email
            )
            
            logger.info(
                "Successfully processed query",
                extra={
                    'pipeline_name': pipeline.get_pipeline_name(),
                    'response_length': len(result["response_content"])
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Error processing query", 
                exc_info=True,
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            
            # Fallback error response
            error_response = {
                "message": "I apologize, but I encountered an unexpected issue processing your request. Please try again.",
                "interactionId": interaction_id,
                "type": "error",
                "content": []
            }
            
            import json
            error_response_str = json.dumps(error_response)
            
            # Save error response to history
            save_message(
                interaction_id=interaction_id,
                role="assistant",
                content=error_response_str,
                user_email=user_email
            )
            
            return {
                "response_content": error_response_str,
                "interaction_id": interaction_id
            }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about registered pipelines.
        
        Returns:
            Dict[str, Any]: Information about available pipelines
        """
        return {
            pipeline_category.value: {
                "name": pipeline.get_pipeline_name(),
                "class": pipeline.__class__.__name__
            }
            for pipeline_category, pipeline in self.pipelines.items()
        } 