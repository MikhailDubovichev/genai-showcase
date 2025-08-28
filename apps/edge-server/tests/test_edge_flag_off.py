"""
Smoke test for edge server local LLM path when RAG flag is off.

This module contains integration tests for the edge server's `/api/prompt` endpoint
to ensure it uses the local LLM path when the RAG feature flag is disabled. The tests
validate the default behavior and flag control mechanism that routes requests to the
local language model instead of attempting cloud RAG calls.

The tests use dynamic URL configuration by reading from the app's config.json file
or falling back to environment variables, ensuring consistency with the actual
application setup and allowing flexible deployment configurations.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import requests


def get_base_url() -> str:
    """
    Get the base URL for the edge server using dynamic configuration.

    This helper function reads the service port from the application configuration
    to ensure tests use the same port as the running service. It first checks for
    a port setting in the config.json server section, then falls back to environment variables,
    and finally uses a default port of 8080.

    The dynamic URL resolution ensures that tests work correctly whether the service
    is running on the default port or a custom port specified in configuration or
    environment variables.

    Returns:
        str: Base URL for the edge server (e.g., "http://localhost:8080")

    Example:
        >>> get_base_url()
        "http://localhost:8080"
    """
    # Try to read port from config.json server section
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            # Look for port in server section
            port = config.get("server", {}).get("port", 8080)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        # Fall back to environment variable or default
        port = int(os.getenv("EDGE_SERVER_PORT", "8080"))

    return f"http://localhost:{port}"


def create_test_payload() -> Dict[str, Any]:
    """
    Create a test payload for the edge server prompt endpoint.

    This helper generates a realistic test payload that matches the expected
    format for the `/api/prompt` endpoint. The payload includes a typical energy
    efficiency question and a unique interaction ID for tracking.

    The payload structure matches what the edge server expects and is designed
    to trigger the energy efficiency pipeline when the RAG flag is disabled.

    Returns:
        Dict[str, Any]: Test payload dictionary ready for JSON serialization
    """
    return {
        "message": "How to reduce energy consumption at home?",
        "interactionId": "test-edge-flag-off-123",
        "token": "dummy",
        "location_id": "1000299",
    }


def load_config() -> Dict[str, Any]:
    """Load the current edge server configuration."""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config: Dict[str, Any]) -> None:
    """Save the configuration back to the config file."""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


@pytest.fixture
def temp_config():
    """
    Fixture to temporarily modify the edge server configuration.

    This fixture saves the original configuration, allows tests to modify it,
    and automatically restores the original configuration after the test completes.
    This ensures that configuration changes made during testing don't persist
    and don't affect other tests or the development environment.

    The fixture yields control to the test function, which can then modify
    the configuration as needed. After the test completes, the original
    configuration is automatically restored.

    Yields:
        Dict[str, Any]: The current configuration that can be modified
    """
    # Load original config
    original_config = load_config()

    # Yield control to test (allows modification)
    yield original_config

    # Restore original config after test
    save_config(original_config)


def test_edge_local_path_when_flag_off(temp_config):
    """
    Test that edge server uses local LLM path when RAG flag is off.

    This integration test validates the edge server's feature flag control mechanism
    by ensuring that when the RAG feature flag is disabled, requests are routed to
    the local LLM path instead of attempting cloud RAG calls. The test temporarily
    modifies the configuration to disable the RAG feature, sends a test request,
    and verifies that the response comes from the local language model.

    The test is crucial for validating the default behavior and ensuring that the
    feature flag mechanism works correctly to control routing between local and
    cloud processing paths. This ensures users can reliably disable cloud features
    and fall back to local processing when needed.

    The test temporarily sets `features.energy_efficiency_rag_enabled` to false,
    sends an energy efficiency question, and expects a response that indicates
    local LLM processing (no cloud-specific fields like "sourceId" in content).

    The test will pass if:
    - The endpoint returns HTTP 200 status
    - The response is valid JSON matching EnergyEfficiencyResponse schema
    - The response contains expected fields: "message", "type", "content"
    - The response lacks cloud-specific indicators (no "sourceId" fields)
    - The local LLM generates a reasonable response

    This test provides confidence that the feature flag control works correctly
    and that users can reliably disable cloud RAG features when needed.

    After the test, the original configuration is automatically restored.
    """
    # Disable RAG flag temporarily
    temp_config["features"]["energy_efficiency_rag_enabled"] = False
    save_config(temp_config)

    # Build the full endpoint URL
    base_url = get_base_url()
    endpoint = f"{base_url}/api/prompt"

    # Create test payload
    payload = create_test_payload()

    # Send POST request to the edge prompt endpoint
    # /api/prompt expects query parameters (FastAPI Query), not a JSON body
    response = requests.post(endpoint, params=payload, timeout=30)

    # Validate HTTP status
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Validate response is valid JSON
    try:
        result = response.json()
    except json.JSONDecodeError as e:
        pytest.fail(f"Response is not valid JSON: {e}")

    # Validate required fields exist (matches EnergyEfficiencyResponse)
    assert "message" in result, "Response missing 'message' field"
    assert "type" in result, "Response missing 'type' field"
    assert "content" in result, "Response missing 'content' field"

    # Validate data types
    assert isinstance(result["message"], str), "message should be a string"
    assert isinstance(result["type"], str), "type should be a string"
    assert isinstance(result["content"], list), "content should be a list"

    # Validate that this is from local LLM (not cloud RAG)
    # Local responses should not have cloud-specific fields like "sourceId"
    if result["content"]:
        for i, item in enumerate(result["content"]):
            # Local LLM responses typically don't have structured content with sourceId
            # If there are content items, they should be simple strings or basic objects
            assert isinstance(item, (str, dict)), f"content[{i}] should be string or dict"

            # If it's a dict, it shouldn't have cloud-specific fields like "sourceId"
            if isinstance(item, dict):
                # Local LLM might return simple dicts, but not with cloud fields
                assert "sourceId" not in item, f"content[{i}] should not have 'sourceId' (indicates cloud RAG usage)"

    # Validate the response contains actual content (not empty)
    assert len(result["message"]) > 0, "Response message should not be empty"

    # Validate interactionId is preserved (if present in response)
    if "interactionId" in result:
        assert isinstance(result["interactionId"], str), "interactionId should be a string"


# Standalone execution for manual testing
if __name__ == "__main__":
    print("Running edge server flag-off smoke test...")
    print("=" * 50)

    # Load current config
    config = load_config()
    original_flag = config["features"]["energy_efficiency_rag_enabled"]

    print(f"Current RAG flag setting: {original_flag}")
    print("Temporarily setting flag to False for testing...")

    # Temporarily disable flag
    config["features"]["energy_efficiency_rag_enabled"] = False
    save_config(config)

    try:
        base_url = get_base_url()
        endpoint = f"{base_url}/api/prompt"
        payload = create_test_payload()

        print(f"Testing endpoint: {endpoint}")
        print(f"Test payload: {json.dumps(payload, indent=2)}")

        response = requests.post(endpoint, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            print("\n✅ Response received:")
            print(f"   Status: {response.status_code}")
            print(f"   Message: {result.get('message', 'N/A')[:100]}...")
            print(f"   Type: {result.get('type', 'N/A')}")
            print(f"   Content items: {len(result.get('content', []))}")

            # Check for cloud indicators
            has_source_id = any(
                isinstance(item, dict) and "sourceId" in item
                for item in result.get("content", [])
            )

            if not has_source_id:
                print("   ✅ No cloud RAG indicators found (local LLM path confirmed)")
            else:
                print("   ⚠️  Cloud RAG indicators found (unexpected for flag off)")

        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the edge server running?")
        print("Start with: poetry run python apps/edge-server/main.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")

    finally:
        # Always restore original config
        print(f"\nRestoring original RAG flag setting: {original_flag}")
        config["features"]["energy_efficiency_rag_enabled"] = original_flag
        save_config(config)
        print("✅ Original configuration restored")

    print("\n" + "=" * 50)
    print("Edge server flag-off test completed!")
    print("\nNote: Make sure the edge server is running before testing.")
    print("Start with: poetry run python apps/edge-server/main.py")
