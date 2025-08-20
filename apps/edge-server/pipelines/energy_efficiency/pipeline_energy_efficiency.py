"""
pipelines/energy_efficiency/pipeline_energy_efficiency.py

Energy efficiency pipeline implementation (cloud‑first with graceful fallback).

This module contains the main `EnergyEfficiencyPipeline` which now supports a
cloud‑first retrieval‑augmented generation path guarded by a feature flag, while
preserving the original local LLM behavior as a seamless fallback. When the edge
configuration flag `CONFIG["features"]["energy_efficiency_rag_enabled"]` is true,
the pipeline first attempts to call the Cloud RAG service via a short‑timeout HTTP
request and, on success, validates the strict JSON (JavaScript Object Notation)
response against the shared schema before returning it to the caller. If the cloud
request times out, fails, or produces invalid JSON, the pipeline logs a concise
warning and automatically falls back to the original local LLM path so API behavior
and schema remain stable. When the flag is false, the pipeline behaves exactly as
before, using only the local LLM path with no network dependency.
"""

from typing import Dict, Any
import logging
import json
from config.logging_config import get_logger
from config import CONFIG

from ..base import BasePipeline
from llm_cloud.provider import get_client
from shared.utils import create_error_response
from shared.models import EnergyEfficiencyResponse
from shared.rag_client import post_answer_from_config, RAGClientError, RAGClientTimeoutError

logger = get_logger(__name__)

class EnergyEfficiencyPipeline(BasePipeline):
    """
    Pipeline for handling energy efficiency advice and education queries.
    
    This pipeline processes requests related to:
    - General energy saving tips and strategies
    - Energy bill reduction advice
    - Energy efficiency best practices
    - Educational content about energy consumption
    - Energy optimization recommendations
    
    Features:
    - Conversational energy efficiency advice
    - Educational content generation
    - Energy saving strategy recommendations
    - Future: RAG integration with energy efficiency knowledge base
    
    Notes:
    - Cloud‑first behavior: When `features.energy_efficiency_rag_enabled` is true,
      the pipeline posts the user question to the Cloud RAG endpoint with a short
      timeout. On success, the validated JSON is returned immediately. On timeout
      or error, the pipeline logs a warning and falls back to the local LLM path,
      ensuring the response schema and user experience remain consistent.
    - No device tools: This pipeline does not execute device tools; it remains
      knowledge‑based and focuses on educational and advisory responses.
    """
    
    def setup(self) -> None:
        """
        Initialize energy efficiency pipeline resources.
        
        Sets up:
        - LLM client for energy efficiency model
        - System prompt for energy efficiency advice
        - Energy efficiency model configuration
        """
        # Initialize LLM client
        self.client = get_client()
        
        # Load energy efficiency system prompt from config
        self.system_prompt = self.config['energy_efficiency_message']
        
        # Get energy efficiency model configuration
        self.model_config = self.config["llm"]["models"]["energy_efficiency"]
        
        logger.info(
            "Pipeline setup complete",
            extra={
                'pipeline_name': self.get_pipeline_name(),
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
        Process energy efficiency message and return advice response.
        
        Args:
            message (str): User's energy efficiency question or request
            token (str): Authentication token for external services
            location_id (str): The user's location identifier
            user_email (str): The user's email for session management
            interaction_id (str): Unique identifier for this interaction
            
        Returns:
            Dict[str, Any]: The processed response with validated JSON structure
        """
        try:
            # Update logger context with interaction_id
            logger.extra.update({
                'interaction_id': interaction_id,
                'pipeline_name': self.get_pipeline_name()
            })
            
            # Inject interaction_id into system prompt for LLM response
            system_prompt_with_id = self.system_prompt + f"\n\nFor this conversation turn, use this interactionId in your JSON response: {interaction_id}\n"
            
            # Cloud-first RAG (feature-flagged) with graceful fallback
            use_cloud = bool((CONFIG.get("features", {}) or {}).get("energy_efficiency_rag_enabled", False))
            if use_cloud:
                try:
                    # Use pipeline model setting if present; otherwise default to 3
                    top_k = 3
                    try:
                        top_k = int(self.model_config.get("settings", {}).get("top_k", 3))
                    except Exception:
                        pass

                    cloud_obj = post_answer_from_config(
                        question=message,
                        interaction_id=interaction_id,
                        top_k=top_k,
                        # TODO: Move hardcoded timeout to CONFIG["cloud_rag"]["timeout_s"]
                        timeout_s=1.5,  # short timeout to keep edge responsive
                    )

                    # Validate cloud response and return immediately on success
                    validated = EnergyEfficiencyResponse(**cloud_obj)
                    return {
                        "response_content": validated.model_dump_json(),
                        "interaction_id": interaction_id,
                    }

                except (RAGClientTimeoutError, RAGClientError) as cloud_err:
                    logger.warning(
                        "Cloud RAG unavailable or failed; falling back to local path",
                        extra={
                            "error_type": type(cloud_err).__name__,
                            "pipeline_name": self.get_pipeline_name(),
                            "interaction_id": interaction_id,
                        },
                    )
                    # Fallback to local path below
                except Exception as cloud_unexpected:
                    logger.warning(
                        "Unexpected cloud error; falling back to local path",
                        extra={
                            "error_type": type(cloud_unexpected).__name__,
                            "pipeline_name": self.get_pipeline_name(),
                            "interaction_id": interaction_id,
                        },
                    )
                    # Fallback to local path below

            # Prepare messages for LLM (no tools needed for energy efficiency)
            # Include the user's message so the model has the actual question context.
            messages = [
                {"role": "system", "content": system_prompt_with_id},
                {"role": "user", "content": message},
            ]
            
            logger.info(
                "Making LLM request",
                extra={
                    'model': self.model_config["name"],
                    'max_tokens': self.model_config["settings"]["max_tokens"],
                    'message_preview': message[:50]
                }
            )
            
            # Make LLM request for energy efficiency advice
            response = self.client.chat.completions.create(
                model=self.model_config["name"],
                max_tokens=self.model_config["settings"]["max_tokens"],
                temperature=self.model_config["settings"]["temperature"],
                top_p=self.model_config["settings"]["top_p"],
                # Ask the model to return a strict JSON object (OpenAI-compatible parameter)
                response_format={"type": "json_object"},
                extra_body={"top_k": self.model_config["settings"]["top_k"]},
                messages=messages,
                # Note: No tools parameter - energy efficiency is purely conversational
            )
            
            # Get the assistant response
            assistant_response = response.choices[0].message.content
            
            try:
                # Parse LLM response as JSON and validate with Pydantic
                response_dict = json.loads(assistant_response)
                validated_response = EnergyEfficiencyResponse(**response_dict)
                
                logger.info(
                    "Successfully validated response",
                    extra={
                        'response_length': len(assistant_response)
                    }
                )
                
                return {
                    "response_content": validated_response.model_dump_json(),
                    "interaction_id": interaction_id
                }
                
            except json.JSONDecodeError as e:
                logger.error(
                    "LLM response was not valid JSON",
                    exc_info=True,
                    extra={
                        'error_type': 'JSONDecodeError',
                        'error_message': str(e),
                        'response_preview': assistant_response[:100]
                    }
                )
                error_response = create_error_response(
                    "I apologize, but I received an invalid response format. Please try again.",
                    interaction_id
                )
                return {
                    "response_content": error_response,
                    "interaction_id": interaction_id
                }
            except ValueError as e:
                logger.error(
                    "Pydantic validation failed",
                    exc_info=True,
                    extra={
                        'error_type': 'ValueError',
                        'error_message': str(e),
                        'response_preview': assistant_response[:100]
                    }
                )
                error_response = create_error_response(
                    "I apologize, but my response format was incorrect. Please try again.",
                    interaction_id
                )
                return {
                    "response_content": error_response,
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
                "I apologize, but I encountered an issue providing energy efficiency advice. Please try again.",
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
            str: The string literal "energy_efficiency". Used by the orchestrator and monitoring
            utilities to label requests, errors, and durations for this pipeline.
        """
        return "energy_efficiency" 