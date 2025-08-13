"""
API tests for `api/prompt.py` endpoints using FastAPI's TestClient.

Covers:
- POST /api/prompt: success path (valid JSON), non-JSON fallback, and unexpected exception handling
- POST /api/reset: success and failure responses based on archive result

We mock dependencies to avoid network and filesystem I/O:
- `api.prompt.PipelineOrchestrator` to prevent real pipeline execution
- `api.prompt.archive_active_conversation` to simulate reset outcomes
"""

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@patch("api.prompt.PipelineOrchestrator")
def test_prompt_success_json(mock_orchestrator_cls):
    """
    When the orchestrator returns a JSON string in `response_content`, the endpoint should parse it
    and return the JSON object with a 200 status.
    """
    mock_orchestrator = MagicMock()
    mock_orchestrator.process_query.return_value = {
        "response_content": json.dumps({
            "message": "ok",
            "interactionId": "id-123",
            "type": "text",
            "content": []
        }),
        "interaction_id": "id-123",
    }
    mock_orchestrator_cls.return_value = mock_orchestrator

    resp = client.post(
        "/api/prompt",
        params={
            "message": "hello",
            "token": "tkn",
            "location_id": "loc-1",
            "user_email": "user@example.com",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "ok"
    assert body["interactionId"] == "id-123"
    assert body["type"] == "text"
    assert body["content"] == []


@patch("api.prompt.PipelineOrchestrator")
def test_prompt_non_json_fallback(mock_orchestrator_cls):
    """
    If the orchestrator returns non-JSON text, the endpoint should respond with a text fallback
    containing the original text and the interaction ID.
    """
    mock_orchestrator = MagicMock()
    mock_orchestrator.process_query.return_value = {
        "response_content": "plain text response",
        "interaction_id": "id-999",
    }
    mock_orchestrator_cls.return_value = mock_orchestrator

    resp = client.post(
        "/api/prompt",
        params={
            "message": "hello",
            "token": "tkn",
            "location_id": "loc-1",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    # The fallback branch should use the same camelCase key as JSON responses
    assert body["message"] == "plain text response"
    assert body["interactionId"] == "id-999"
    assert body["type"] == "text"


@patch("api.prompt.json.loads")
@patch("api.prompt.PipelineOrchestrator")
def test_prompt_unexpected_exception_returns_500(mock_orchestrator_cls, mock_json_loads):
    """
    If an unexpected exception occurs while handling the orchestrator's response (e.g., json.loads raises
    RuntimeError instead of JSONDecodeError), the endpoint should return a standardized 500 error payload.
    """
    mock_orchestrator = MagicMock()
    mock_orchestrator.process_query.return_value = {
        "response_content": "will cause runtime error",
        "interaction_id": "id-err",
    }
    mock_orchestrator_cls.return_value = mock_orchestrator

    # Force a generic exception to hit the broad except block
    mock_json_loads.side_effect = RuntimeError("boom")

    resp = client.post(
        "/api/prompt",
        params={
            "message": "hello",
            "token": "tkn",
            "location_id": "loc-1",
        },
    )

    assert resp.status_code == 500
    # Avoid resp.json() because we patched json.loads globally; assert on raw text instead
    text = resp.text
    assert '"type":"error"' in text
    assert '"interactionId"' in text


@patch("api.prompt.archive_active_conversation")
def test_reset_success(mock_archive):
    """
    /api/reset returns 200 with a success message when archiving succeeds.
    """
    mock_archive.return_value = (True, "archived")

    resp = client.post("/api/reset", data={"user_email": "user@example.com"})

    assert resp.status_code == 200
    assert resp.json()["response"] == "ok"


@patch("api.prompt.archive_active_conversation")
def test_reset_failure(mock_archive):
    """
    /api/reset returns 500 with an error message when archiving fails.
    """
    mock_archive.return_value = (False, "problem")

    resp = client.post("/api/reset")

    assert resp.status_code == 500
    assert resp.json()["response"] == "error" 