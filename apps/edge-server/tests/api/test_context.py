"""
API tests for `api/context.py` using FastAPI's TestClient.

Covers:
- POST /api/context success with daily digest injection
- POST /api/context success without daily digest (no digest today)
- Provider client error mapped to HTTP 502
- Unexpected format (non-list) mapped to HTTP 500
- JSON decode error mapped to HTTP 500

Mocks:
- `api.context.provider_client_instance.get_devices` to avoid real integrator calls
- `api.context.should_show_daily_digest`, `generate_daily_digest`, `format_digest_for_injection`, `save_message` for digest flow
- `builtins.open` to avoid filesystem writes
"""

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@patch("builtins.open")
@patch("api.context.save_message")
@patch("api.context.format_digest_for_injection")
@patch("api.context.generate_daily_digest")
@patch("api.context.should_show_daily_digest", return_value=True)
@patch("api.context.provider_client_instance.get_devices")
def test_context_success_with_digest(mock_get_devices, mock_should_digest, mock_gen_digest, mock_fmt_digest, mock_save_msg, mock_open):
    """
    When the provider returns a valid device list (Python list) and digest should be shown,
    the endpoint saves context, injects digest, and returns the digest JSON.
    """
    mock_get_devices.return_value = []
    mock_gen_digest.return_value = {"tips": ["a", "b"]}
    mock_fmt_digest.return_value = json.dumps({"message": "digest", "content": []})
    # Mock file write
    mock_open.return_value.__enter__.return_value.write.return_value = None

    resp = client.post(
        "/api/context",
        params={
            "token": "tkn",
            "location_id": "loc-1",
            "user_email": "user@example.com",
        },
    )

    assert resp.status_code == 200
    # response is daily_digest dict; in our mock, we returned a dict from generate_daily_digest
    # but endpoint returns JSONResponse(daily_digest) directly
    assert resp.json() == {"tips": ["a", "b"]}
    mock_save_msg.assert_called_once()


@patch("builtins.open")
@patch("api.context.should_show_daily_digest", return_value=False)
@patch("api.context.provider_client_instance.get_devices")
def test_context_success_no_digest(mock_get_devices, mock_should_digest, mock_open):
    """
    When the provider returns a valid device list but digest is not shown today,
    the endpoint should return a no-digest status.
    """
    mock_get_devices.return_value = []
    mock_open.return_value.__enter__.return_value.write.return_value = None

    resp = client.post(
        "/api/context",
        params={
            "token": "tkn",
            "location_id": "loc-1",
            "user_email": "user@example.com",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"status": "no_digest_today"}


@patch("builtins.open")
@patch("api.context.provider_client_instance.get_devices")
def test_context_provider_error_502(mock_get_devices, mock_open):
    """
    If the provider returns an error object, the endpoint should respond with 502.
    """
    mock_get_devices.return_value = {"error": "bad"}
    mock_open.return_value.__enter__.return_value.write.return_value = None

    resp = client.post(
        "/api/context",
        params={
            "token": "tkn",
            "location_id": "loc-1",
        },
    )

    assert resp.status_code == 502


@patch("builtins.open")
@patch("api.context.provider_client_instance.get_devices")
def test_context_unexpected_format_500(mock_get_devices, mock_open):
    """
    If fetched devices data is not a list (and not an error dict), return 500.
    """
    mock_get_devices.return_value = {"not": "a list"}
    mock_open.return_value.__enter__.return_value.write.return_value = None

    resp = client.post(
        "/api/context",
        params={
            "token": "tkn",
            "location_id": "loc-1",
        },
    )

    assert resp.status_code == 500


@patch("builtins.open")
@patch("api.context.provider_client_instance.get_devices")
def test_context_json_decode_error_500(mock_get_devices, mock_open):
    """
    If a JSONDecodeError arises while parsing devices data, return 500.
    """
    mock_get_devices.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    mock_open.return_value.__enter__.return_value.write.return_value = None

    resp = client.post(
        "/api/context",
        params={
            "token": "tkn",
            "location_id": "loc-1",
        },
    )

    assert resp.status_code == 500 