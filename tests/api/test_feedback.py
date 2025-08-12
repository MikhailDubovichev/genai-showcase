"""
API tests for `api/feedback.py` using FastAPI's TestClient.

Covers:
- POST /api/feedback/negative: interaction not found (400) and success (200)
- POST /api/feedback/positive: interaction not found (400) and success (200)
- GET /api/feedback/stats: negative stats success (200)
- GET /api/feedback/negative/stats and /api/feedback/positive/stats: success (200)

Mocks:
- `api.feedback.validate_interaction_exists`
- `api.feedback.save_negative_feedback` / `save_positive_feedback`
- `api.feedback.get_negative_feedback_statistics` / `get_positive_feedback_statistics`
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@patch("api.feedback.validate_interaction_exists", return_value=False)
def test_negative_feedback_invalid_interaction(mock_validate):
    resp = client.post(
        "/api/feedback/negative",
        params={"interaction_id": "bad", "user_email": "user@example.com"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["response"] == "error"


@patch("api.feedback.save_negative_feedback")
@patch("api.feedback.validate_interaction_exists", return_value=True)
def test_negative_feedback_success(mock_validate, mock_save):
    mock_save.return_value = {"feedback_id": "fid-1"}
    resp = client.post(
        "/api/feedback/negative",
        params={"interaction_id": "good", "user_email": "user@example.com"},
    )
    assert resp.status_code == 200
    assert resp.json()["response"] == "ok"
    assert resp.json()["feedback_id"] == "fid-1"


@patch("api.feedback.get_negative_feedback_statistics", return_value={"total_negative_feedback": 1})
def test_feedback_stats_alias(mock_stats):
    resp = client.get("/api/feedback/stats")
    assert resp.status_code == 200
    assert resp.json()["response"] == "ok"


@patch("api.feedback.validate_interaction_exists", return_value=False)
def test_positive_feedback_invalid_interaction(mock_validate):
    resp = client.post(
        "/api/feedback/positive",
        params={"interaction_id": "bad"},
    )
    assert resp.status_code == 400
    assert resp.json()["response"] == "error"


@patch("api.feedback.save_positive_feedback")
@patch("api.feedback.validate_interaction_exists", return_value=True)
def test_positive_feedback_success(mock_validate, mock_save):
    mock_save.return_value = {"feedback_id": "fid-2"}
    resp = client.post(
        "/api/feedback/positive",
        params={"interaction_id": "good"},
    )
    assert resp.status_code == 200
    assert resp.json()["response"] == "ok"
    assert resp.json()["feedback_id"] == "fid-2"


@patch("api.feedback.get_negative_feedback_statistics", return_value={"total_negative_feedback": 2})
def test_negative_feedback_stats(mock_stats):
    resp = client.get("/api/feedback/negative/stats")
    assert resp.status_code == 200
    assert resp.json()["response"] == "ok"


@patch("api.feedback.get_positive_feedback_statistics", return_value={"total_positive_feedback": 3})
def test_positive_feedback_stats(mock_stats):
    resp = client.get("/api/feedback/positive/stats")
    assert resp.status_code == 200
    assert resp.json()["response"] == "ok" 