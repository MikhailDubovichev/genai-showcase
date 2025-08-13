"""
Unit tests for `core/orchestrator.py` – PipelineOrchestrator routing behavior.

This test module verifies the orchestration logic that classifies a message and routes it to the
appropriate pipeline, while ensuring history is stored and fallbacks are applied as designed. The tests
mock all external collaborators, including the classifier, pipeline classes, and history manager. By
replacing these with controlled fakes, we validate only the logic in `PipelineOrchestrator` and avoid
real network calls, file I/O, or heavy pipeline execution. This mirrors best practices in unit testing,
where a unit is exercised in isolation and collaborators are mocked.
"""

import os
import json
import unittest
from unittest.mock import MagicMock, patch

# Ensure required env var exists before any module under `config` is imported
os.environ.setdefault("NEBIUS_API_KEY", "test-key")

from core.orchestrator import PipelineOrchestrator
from shared.models import MessageCategory
from config import CONFIG


class TestPipelineOrchestrator(unittest.TestCase):
    """
    Unit tests for the `PipelineOrchestrator` class routing and error handling.

    These tests cover:
    - Classification flow and routing to the correct pipeline
    - Immediate handling of OTHER_QUERIES with direct rejection response
    - History persistence calls for user and assistant messages
    - Error fallback behavior when pipeline processing raises exceptions

    All external collaborators are mocked so that the tests are deterministic, fast, and focused on
    the orchestration control flow rather than downstream implementations or infrastructure calls.
    """

    @patch("core.orchestrator.save_message")
    @patch("core.orchestrator.MessageClassifier")
    @patch("core.orchestrator.DeviceControlPipeline")
    @patch("core.orchestrator.EnergyEfficiencyPipeline")
    def test_routes_to_device_control_pipeline(
        self,
        mock_energy_pipeline_cls,
        mock_device_pipeline_cls,
        mock_classifier_cls,
        mock_save_message,
    ):
        """
        Verify that when the classifier returns `MessageCategory.DEVICE_CONTROL`, the orchestrator
        invokes the DeviceControlPipeline and returns its result. The test also confirms that
        conversation history is saved both for the user message and the assistant response.
        """
        # Arrange classifier mock
        mock_classifier = MagicMock()
        mock_classifier.classify_message.return_value = MessageCategory.DEVICE_CONTROL
        mock_classifier.get_direct_rejection_response.side_effect = AssertionError(
            "Should not be called for device control"
        )
        mock_classifier_cls.return_value = mock_classifier

        # Arrange pipeline mock
        mock_device_pipeline = MagicMock()
        mock_device_pipeline.get_pipeline_name.return_value = "device_control"
        mock_device_pipeline.process_message.return_value = {
            "response_content": "{\"message\": \"ok\", \"interactionId\": \"id-1\", \"type\": \"text\", \"content\": []}",
            "interaction_id": "id-1",
        }
        mock_device_pipeline_cls.return_value = mock_device_pipeline

        mock_energy_pipeline_cls.return_value = MagicMock()

        orchestrator = PipelineOrchestrator(config=CONFIG)

        # Act
        result = orchestrator.process_query(
            message="Turn on lamp",
            token="tkn",
            location_id="loc-1",
            user_email="user@example.com",
        )

        # Assert
        self.assertIn("response_content", result)
        self.assertIn("interaction_id", result)
        mock_classifier.classify_message.assert_called_once_with("Turn on lamp")
        mock_device_pipeline.process_message.assert_called_once()
        # Ensure history saved for both user and assistant
        self.assertGreaterEqual(mock_save_message.call_count, 2)

    @patch("core.orchestrator.save_message")
    @patch("core.orchestrator.MessageClassifier")
    @patch("core.orchestrator.DeviceControlPipeline")
    @patch("core.orchestrator.EnergyEfficiencyPipeline")
    def test_other_queries_direct_rejection(
        self,
        mock_energy_pipeline_cls,
        mock_device_pipeline_cls,
        mock_classifier_cls,
        mock_save_message,
    ):
        """
        Verify that when the classifier returns `MessageCategory.OTHER_QUERIES`, the orchestrator
        does not call any pipeline and returns the classifier's direct rejection response, while
        persisting the user message and assistant response to history.
        """
        # Arrange classifier
        mock_classifier = MagicMock()
        mock_classifier.classify_message.return_value = MessageCategory.OTHER_QUERIES
        mock_classifier.get_direct_rejection_response.return_value = json.dumps(
            {"message": "nope", "interactionId": "id-2", "type": "text", "content": []}
        )
        mock_classifier_cls.return_value = mock_classifier

        orchestrator = PipelineOrchestrator(config=CONFIG)

        # Act
        result = orchestrator.process_query(
            message="Unsupported",
            token="tkn",
            location_id="loc-1",
            user_email="user@example.com",
        )

        # Assert
        self.assertEqual(json.loads(result["response_content"])['message'], "nope")
        mock_device_pipeline_cls.assert_called_once()  # constructed at init
        mock_energy_pipeline_cls.assert_called_once()  # constructed at init
        # But no pipeline.process_message call should have occurred
        self.assertFalse(mock_device_pipeline_cls.return_value.process_message.called)
        self.assertFalse(mock_energy_pipeline_cls.return_value.process_message.called)
        # Save history for both user and assistant
        self.assertGreaterEqual(mock_save_message.call_count, 2)

    @patch("core.orchestrator.save_message")
    @patch("core.orchestrator.MessageClassifier")
    @patch("core.orchestrator.DeviceControlPipeline")
    @patch("core.orchestrator.EnergyEfficiencyPipeline")
    def test_pipeline_error_triggers_fallback(
        self,
        mock_energy_pipeline_cls,
        mock_device_pipeline_cls,
        mock_classifier_cls,
        mock_save_message,
    ):
        """
        Ensure that when a pipeline raises an exception, the orchestrator catches it and returns a
        standardized error payload, also persisting that payload to history. This validates robust
        error handling and user-facing resilience of the orchestration layer.
        """
        mock_classifier = MagicMock()
        mock_classifier.classify_message.return_value = MessageCategory.DEVICE_CONTROL
        mock_classifier_cls.return_value = mock_classifier

        mock_device_pipeline = MagicMock()
        mock_device_pipeline.get_pipeline_name.return_value = "device_control"
        mock_device_pipeline.process_message.side_effect = RuntimeError("boom")
        mock_device_pipeline_cls.return_value = mock_device_pipeline

        mock_energy_pipeline_cls.return_value = MagicMock()

        orchestrator = PipelineOrchestrator(config=CONFIG)

        # Act
        result = orchestrator.process_query(
            message="Turn on lamp",
            token="tkn",
            location_id="loc-1",
            user_email="user@example.com",
        )

        # Assert
        payload = json.loads(result["response_content"])
        self.assertEqual(payload["type"], "error")
        self.assertIn("interactionId", payload)
        self.assertIn("message", payload)
        # History saved for both user and assistant
        self.assertGreaterEqual(mock_save_message.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2) 