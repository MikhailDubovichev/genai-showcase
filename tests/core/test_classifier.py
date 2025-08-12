"""
Unit tests for `core/classifier.py` – MessageClassifier behavior in isolation.

This test module focuses on verifying the logic implemented inside the `MessageClassifier` class
without involving real network calls or filesystem dependencies. To achieve this, we apply mocking
(a software testing technique where external collaborators are replaced with controllable test doubles)
for the following collaborators:

- get_client (from `llm_cloud.provider`): This normally returns a real OpenAI (Open Artificial Intelligence) SDK client.
  We replace it with a lightweight mock so that no external HTTP requests are performed and we can predetermine the
  model outputs to exercise different code paths in a repeatable way.
- builtins.open (the Python built-in function): We replace reading prompt files with fake in-memory strings to avoid
  coupling tests to the developer's file system and to make the unit test purely logic-focused.

By isolating `MessageClassifier` from these external dependencies, the tests become fast, deterministic, and focused on
verifying classification mapping, error handling, and JSON response generation. This aligns with the separation of
concerns principle, where a unit test only validates the code in the unit under test and not the behavior of dependencies.
"""

import os
import json
import unittest
from unittest.mock import MagicMock, patch

# Ensure required env var exists before any module under `config` is imported
os.environ.setdefault("NEBIUS_API_KEY", "test-key")

from shared.models import MessageCategory
from core.classifier import MessageClassifier


class TestMessageClassifier(unittest.TestCase):
    """
    Unit tests for the `MessageClassifier` class.

    These tests validate how user messages are mapped to `MessageCategory` values and ensure that
    the class gracefully handles unexpected conditions. External dependencies such as the Large Language
    Model (LLM) client and prompt file loading are mocked, which keeps the tests independent from network
    connectivity and the local filesystem layout. This strategy ensures tests run quickly and reliably on any
    machine (developer laptops or Continuous Integration), while focusing solely on the classification logic,
    the transformation from raw LLM output to enum categories, and the structure of the rejection response.
    """

    @patch("core.classifier.get_client")
    @patch("builtins.open")
    def test_classify_message_as_device_control(self, mock_open, mock_get_client):
        """
        Verify that when the mocked LLM responds with the string that contains "DEVICE_CONTROL",
        the classifier maps it to `MessageCategory.DEVICE_CONTROL`. The test sets up the mocked client
        to return a synthetic response object, avoiding any network call. It also mocks file reads
        to avoid touching disk. This checks the positive path where the category is recognized and
        ensures the returned enum matches the expected routing category for downstream pipelines.
        """
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "DEVICE_CONTROL"
        mock_llm_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_llm_client

        # Simulate file contents for prompts
        mock_open.return_value.__enter__.return_value.read.return_value = "Fake prompt"

        classifier = MessageClassifier()
        result = classifier.classify_message("Turn on the lights")
        self.assertEqual(result, MessageCategory.DEVICE_CONTROL)

    @patch("core.classifier.get_client")
    @patch("builtins.open")
    def test_classify_message_as_energy_efficiency(self, mock_open, mock_get_client):
        """
        Verify that when the mocked LLM responds with the string that contains "ENERGY_EFFICIENCY",
        the classifier maps it to `MessageCategory.ENERGY_EFFICIENCY`. This validates the second branch
        in the mapping logic and ensures that the enum conversion works consistently across supported
        categories, preserving expected routing behavior for energy efficiency requests.
        """
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ENERGY_EFFICIENCY"
        mock_llm_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_llm_client

        mock_open.return_value.__enter__.return_value.read.return_value = "Fake prompt"

        classifier = MessageClassifier()
        result = classifier.classify_message("How can I reduce my bill?")
        self.assertEqual(result, MessageCategory.ENERGY_EFFICIENCY)

    @patch("core.classifier.get_client")
    @patch("builtins.open")
    def test_classify_message_defaults_to_other_queries(self, mock_open, mock_get_client):
        """
        Ensure that if the mocked LLM returns a string without a known category, the classifier defaults
        to `MessageCategory.OTHER_QUERIES`. This protects the system from unpredictable or malformed model
        outputs by routing unsupported queries to a safe, non-destructive fallback path without invoking
        heavy pipeline logic. The behavior contributes to robust error containment.
        """
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "UNKNOWN_STUFF"
        mock_llm_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_llm_client

        mock_open.return_value.__enter__.return_value.read.return_value = "Fake prompt"

        classifier = MessageClassifier()
        result = classifier.classify_message("Blah blah")
        self.assertEqual(result, MessageCategory.OTHER_QUERIES)

    @patch("core.classifier.get_client")
    @patch("builtins.open")
    def test_classify_message_handles_llm_error(self, mock_open, mock_get_client):
        """
        Validate that exceptions raised by the LLM client are caught and the safe fallback category
        `MessageCategory.OTHER_QUERIES` is returned. This ensures the method is resilient to transient
        infrastructure problems (such as network outages) and continues to provide a consistent behavior
        to callers without propagating raw exceptions from external services.
        """
        mock_llm_client = MagicMock()
        mock_llm_client.chat.completions.create.side_effect = Exception("LLM is down")
        mock_get_client.return_value = mock_llm_client

        mock_open.return_value.__enter__.return_value.read.return_value = "Fake prompt"

        classifier = MessageClassifier()
        result = classifier.classify_message("This will error")
        self.assertEqual(result, MessageCategory.OTHER_QUERIES)

    @patch("core.classifier.get_client")
    @patch("builtins.open")
    def test_get_direct_rejection_response_structure(self, mock_open, mock_get_client):
        """
        Check that `get_direct_rejection_response` returns a JSON string with the expected structure
        and that the message field is populated from the loaded template content. The test decodes the
        JSON output and verifies required keys; it does not rely on real files due to mocking, which
        simplifies the test while preserving the behavioral contract of the method.
        """
        mock_get_client.return_value = MagicMock()
        mock_open.return_value.__enter__.return_value.read.return_value = "Template: Unsupported query"

        classifier = MessageClassifier()
        payload = json.loads(classifier.get_direct_rejection_response("abc-123"))

        self.assertIn("message", payload)
        self.assertIn("interactionId", payload)
        self.assertIn("type", payload)
        self.assertIn("content", payload)
        self.assertEqual(payload["interactionId"], "abc-123")
        self.assertEqual(payload["type"], "text")
        self.assertEqual(payload["content"], [])
        self.assertEqual(payload["message"], "Template: Unsupported query")


if __name__ == "__main__":
    unittest.main(verbosity=2) 