"""
Unit tests for `pipelines/energy_efficiency/pipeline_energy_efficiency.py`.

Covers `_process_message_internal` core behavior in isolation by mocking the LLM client:
- Success path: valid JSON matching `EnergyEfficiencyResponse` schema returns validated JSON
- JSON decode error path: non-JSON text returns standardized error response
- Validation error path: JSON that fails Pydantic validation returns standardized error response

We call `_process_message_internal` directly to avoid decorator side effects on `process_message` and
focus on business logic. The `get_client` function is patched during `setup()` to prevent real
network client creation.
"""

import json
from unittest.mock import MagicMock, patch

from config import CONFIG
from pipelines.energy_efficiency import EnergyEfficiencyPipeline


@patch("pipelines.energy_efficiency.pipeline_energy_efficiency.get_client")
def test_success_valid_json_returns_validated_json(mock_get_client):
    # Arrange: mock client and response
    mock_client = MagicMock()
    mock_response = MagicMock()
    assistant_payload = {
        "message": "Advice",
        "interactionId": "id-123",
        "type": "text",
        "content": []
    }
    mock_response.choices[0].message.content = json.dumps(assistant_payload)
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    pipeline = EnergyEfficiencyPipeline()

    # Act
    result = pipeline._process_message_internal(
        message="How to save energy?",
        token="tkn",
        location_id="loc-1",
        user_email="user@example.com",
        interaction_id="id-123",
    )

    # Assert
    assert "response_content" in result and "interaction_id" in result
    assert result["interaction_id"] == "id-123"
    # The pipeline returns the validated model as JSON; compare content semantically
    validated = json.loads(result["response_content"])
    assert validated == assistant_payload

    # Also assert the call used the configured model and injected interaction id into system prompt
    args, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == CONFIG["llm"]["models"]["energy_efficiency"]["name"]
    messages = kwargs["messages"]
    assert isinstance(messages, list) and messages and messages[0]["role"] == "system"
    assert "interactionId in your JSON response: id-123" in messages[0]["content"]


@patch("pipelines.energy_efficiency.pipeline_energy_efficiency.get_client")
def test_non_json_response_returns_error(mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "not json"
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    pipeline = EnergyEfficiencyPipeline()

    result = pipeline._process_message_internal(
        message="Question",
        token="tkn",
        location_id="loc-1",
        user_email="user@example.com",
        interaction_id="id-x",
    )

    body = json.loads(result["response_content"])
    assert body["type"] == "error"
    assert body["interactionId"] == "id-x"
    # specific message text is defined in implementation for JSONDecodeError branch
    assert "invalid response format" in body["message"].lower()


@patch("pipelines.energy_efficiency.pipeline_energy_efficiency.get_client")
def test_validation_error_returns_error(mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Missing required field "message" to trigger validation error
    invalid_payload = {
        "interactionId": "id-y",
        "type": "text",
        "content": []
    }
    mock_response.choices[0].message.content = json.dumps(invalid_payload)
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    pipeline = EnergyEfficiencyPipeline()

    result = pipeline._process_message_internal(
        message="Question",
        token="tkn",
        location_id="loc-1",
        user_email="user@example.com",
        interaction_id="id-y",
    )

    body = json.loads(result["response_content"])
    assert body["type"] == "error"
    assert body["interactionId"] == "id-y"
    # message indicates incorrect response format per ValueError branch
    assert "incorrect" in body["message"].lower() 