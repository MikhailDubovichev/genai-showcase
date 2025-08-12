"""
Base class for all pipelines in the AI Assistant.

This module defines the BasePipeline abstract base class that all specific
pipelines must implement. It enforces a common interface and provides shared
functionality for pipeline setup and message processing.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any
from monitoring.metrics import track_latency, track_errors, PIPELINE_PROCESSING_TIME
from config import CONFIG

logger = logging.getLogger(__name__)

class BasePipeline(ABC):
    """
    Abstract base class for all AI Assistant pipelines.
    
    This class defines the interface that all pipelines must implement and provides shared
    functionality for pipeline setup, logging, configuration access, metrics instrumentation,
    and message processing orchestration. Concrete pipelines should override `setup`,
    `get_pipeline_name`, and `_process_message_internal` to provide domain‑specific behavior.
    """
    
    def __init__(self):
        """
        Initialize common pipeline state and invoke pipeline‑specific setup.
        
        The constructor configures a per‑pipeline namespaced logger and loads the global application
        configuration so subclasses can access settings without re‑reading configuration files. It then
        calls `self.setup()` to allow the concrete pipeline to initialize its own resources (for example,
        an LLM client, tool manager, and system prompt). Keeping this in the base class ensures all pipelines
        follow a consistent lifecycle and logging pattern.
        """
        # Set up logger and config before calling setup
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = CONFIG
        
        # Initialize pipeline-specific resources
        self.setup()
    
    @abstractmethod
    def setup(self) -> None:
        """
        Set up the pipeline with necessary resources.
        
        This method should be implemented by each pipeline to initialize
        any required resources (e.g., LLM client, tool manager).
        """
        pass
    
    @abstractmethod
    def get_pipeline_name(self) -> str:
        """
        Get the name of the pipeline.
        
        Returns:
            str: The pipeline's name for use in logging and metrics.
        """
        pass
    
    @track_latency(PIPELINE_PROCESSING_TIME, lambda self: {'pipeline_name': self.get_pipeline_name()})
    @track_errors('pipeline', lambda self: self.get_pipeline_name())
    def process_message(self, 
                       message: str,
                       token: str,
                       location_id: str,
                       user_email: str,
                       interaction_id: str
                       ) -> Dict[str, Any]:
        """
        Process a user message through the pipeline.
        
        This is the main entry point for message processing. It should be
        implemented by each pipeline to handle messages according to its
        specific logic.
        
        Args:
            message (str): The user's message to process
            token (str): Authentication token for external services
            location_id (str): The user's location identifier
            user_email (str): The user's email for session management
            interaction_id (str): Unique identifier for this interaction
            
        Returns:
            Dict[str, Any]: The processed response
        """
        pipeline_name = self.get_pipeline_name()
        self._log_processing_start(message, interaction_id)
        
        try:
            response = self._process_message_internal(
                message=message,
                token=token,
                location_id=location_id,
                user_email=user_email,
                interaction_id=interaction_id
            )
            self._log_processing_end(interaction_id, success=True)
            return response
        except Exception as e:
            self._log_processing_end(interaction_id, success=False)
            raise
    
    @abstractmethod
    def _process_message_internal(self,
                                message: str,
                                token: str,
                                location_id: str,
                                user_email: str,
                                interaction_id: str
                                ) -> Dict[str, Any]:
        """
        Internal message processing implementation.
        
        This method should be implemented by each pipeline to define its
        specific message processing logic.
        
        Args:
            message (str): The user's message to process
            token (str): Authentication token for external services
            location_id (str): The user's location identifier
            user_email (str): The user's email for session management
            interaction_id (str): Unique identifier for this interaction
            
        Returns:
            Dict[str, Any]: The processed response
        """
        pass
    
    def _log_processing_start(self, message: str, interaction_id: str) -> None:
        """
        Log the start of message processing with contextual metadata.
        
        This helper emits an informational log entry that includes the pipeline name, a truncated preview of
        the user's message, and the unique `interaction_id`. Centralizing the log format here keeps pipeline
        implementations concise and ensures observability is consistent across all pipelines.
        
        Args:
            message (str): The message being processed. Potentially long; only a prefix is logged.
            interaction_id (str): Unique identifier to correlate all logs and messages from this turn.
        """
        pipeline_name = self.get_pipeline_name()
        self.logger.info(f"[{pipeline_name}] Starting message processing for interaction {interaction_id}: '{message[:50]}...'")
    
    def _log_processing_end(self, interaction_id: str, success: bool = True) -> None:
        """
        Log the end of message processing indicating success or failure.
        
        This helper records a standardized completion message that includes the pipeline name and whether
        processing succeeded. Logging both success and failure in a consistent format simplifies dashboards
        and makes it easier to trace errors when combined with metrics emitted by decorators.
        
        Args:
            interaction_id (str): Unique identifier associated with the processed message.
            success (bool): Set to True when processing completed normally; False when an exception occurred.
        """
        pipeline_name = self.get_pipeline_name()
        status = "completed successfully" if success else "failed"
        self.logger.info(f"[{pipeline_name}] Message processing {status} for interaction {interaction_id}") 