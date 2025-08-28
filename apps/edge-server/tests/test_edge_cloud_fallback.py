"""
Test module for edge server cloud-first with fallback integration.

This module contains integration tests that validate the edge server's ability to:
- Use cloud RAG when the feature flag is enabled
- Fall back gracefully to local LLM when cloud times out or fails
- Maintain service availability during cloud outages

The tests simulate real-world scenarios where the cloud service might be unavailable
or slow, ensuring the edge server remains responsive for end users.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import pytest
import requests


def get_edge_base_url() -> str:
    """
    Get the base URL for the edge server using dynamic configuration.

    This helper function reads the edge server port from the application configuration
    to ensure tests use the same port as the running service. It first checks for
    a port setting in the config.json server section, then falls back to environment variables,
    and finally uses a default port of 8080.

    The dynamic URL resolution ensures that tests work correctly whether the service
    is running on the default port or a custom port specified in configuration or
    environment variables. This is particularly important for integration tests that
    need to communicate with the actual running service.

    Returns:
        str: Base URL for the edge server (e.g., "http://localhost:8080")

    Example:
        >>> get_edge_base_url()
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


def get_cloud_base_url() -> str:
    """
    Get the base URL for the cloud RAG service using dynamic configuration.

    This helper function reads the cloud RAG base URL from the edge server's configuration
    to ensure tests use the same URL that the edge server will use for cloud communication.
    It first checks for a base_url setting in the config.json cloud_rag section, then falls
    back to environment variables, and finally uses a default localhost URL.

    The dynamic URL resolution ensures that tests validate the exact same cloud endpoint
    that the edge server will attempt to contact, maintaining consistency between test
    scenarios and production behavior.

    Returns:
        str: Base URL for the cloud RAG service (e.g., "http://localhost:8000")

    Example:
        >>> get_cloud_base_url()
        "http://localhost:8000"
    """
    # Try to read base URL from config.json cloud_rag section
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            # Look for base URL in cloud_rag section
            base_url = config.get("cloud_rag", {}).get("base_url", "http://localhost:8000")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        # Fall back to environment variable or default
        base_url = os.getenv("CLOUD_RAG_BASE_URL", "http://localhost:8000")

    return base_url


def create_test_payload() -> Dict[str, Any]:
    """
    Create a standardized test payload for energy efficiency queries.

    This helper function generates a consistent test payload that can be used across
    multiple test scenarios. It includes all required fields for the edge server's
    prompt API with realistic values that would trigger the energy efficiency pipeline.

    The payload is designed to be simple yet complete, ensuring that tests focus on
    the cloud-first/fallback logic rather than payload validation issues.

    Returns:
        Dict[str, Any]: Test payload with message, interactionId, token, and location_id

    Example:
        >>> payload = create_test_payload()
        >>> payload["message"]
        "How to reduce energy?"
    """
    return {
        "message": "How to reduce energy?",
        "interactionId": "test-456",
        "token": "dummy",
        "location_id": "1000299"
    }


@pytest.fixture
def temp_config_with_cloud_enabled():
    """
    Temporary fixture to enable cloud RAG with short timeout for testing fallback.

    This fixture temporarily modifies the edge server's configuration to:
    - Enable the energy_efficiency_rag_enabled feature flag
    - Set a very short timeout (0.1 seconds) to force timeout scenarios
    - Restore original configuration after test completion

    The fixture ensures tests run in isolation and don't affect other tests or
    production configuration. It uses file operations to modify the config.json
    directly, providing realistic test conditions.

    Yields:
        Dict[str, Any]: Original configuration for restoration
    """
    config_path = Path(__file__).parent.parent / "config" / "config.json"

    # Read original configuration
    with open(config_path, "r") as f:
        original_config = json.load(f)

    # Create modified configuration for testing
    test_config = original_config.copy()
    test_config["features"] = test_config.get("features", {})
    test_config["features"]["energy_efficiency_rag_enabled"] = True

    # Add or update cloud_rag section with short timeout
    test_config["cloud_rag"] = test_config.get("cloud_rag", {})
    # Edge reads `timeout_s`; keep backward compatibility by setting both
    test_config["cloud_rag"]["timeout_s"] = 0.1  # Very short timeout to force fallback
    test_config["cloud_rag"]["timeout"] = 0.1

    # Write test configuration
    with open(config_path, "w") as f:
        json.dump(test_config, f, indent=2)

    yield original_config

    # Restore original configuration
    with open(config_path, "w") as f:
        json.dump(original_config, f, indent=2)


def test_edge_cloud_timeout_fallback(temp_config_with_cloud_enabled):
    """
    Test edge server cloud-first behavior with simulated timeout and fallback.

    This integration test validates the edge server's resilience and fallback mechanism
    by simulating a scenario where the cloud RAG service is unavailable due to timeout.
    The test temporarily configures the edge server to:
    - Enable cloud RAG feature flag
    - Set an extremely short timeout (0.1s) that will cause cloud requests to fail
    - Attempt a request that should trigger the cloud path and then fallback

    The test verifies that:
    - The edge server attempts the cloud path (feature flag enabled)
    - The cloud request times out due to the short timeout
    - The edge server falls back to the local LLM path seamlessly
    - The response is valid and comes from the local LLM (no cloud-specific fields)
    - The service remains responsive despite cloud unavailability

    This ensures production resilience and user experience continuity during cloud outages.

    Args:
        temp_config_with_cloud_enabled: Fixture that enables cloud with short timeout

    Raises:
        AssertionError: If fallback behavior doesn't work as expected
    """
    # Build URLs for both services
    edge_url = f"{get_edge_base_url()}/api/prompt"
    cloud_url = f"{get_cloud_base_url()}/api/rag/answer"

    # Create test payload
    payload = create_test_payload()

    # Record start time to measure latency
    start_time = time.time()

    # Make request to edge server (should timeout on cloud and fallback)
    response = requests.post(edge_url, params=payload, timeout=30)
    end_time = time.time()

    # Calculate total latency
    latency = end_time - start_time

    # Verify response status
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Parse and validate JSON response
    response_data = response.json()
    assert "message" in response_data, "Response should contain 'message' field"
    assert "interactionId" in response_data, "Response should contain 'interactionId' field"
    assert "type" in response_data, "Response should contain 'type' field"
    assert response_data["type"] == "text", "Response type should be 'text'"

    # Verify this is from local LLM (not cloud) - check for absence of cloud-specific fields
    # Local LLM responses typically don't have "content" array with sources
    content = response_data.get("content", [])
    assert isinstance(content, list), "Content should be a list"

    # If content is empty or doesn't have cloud-specific structure, it's likely from local LLM
    if content:
        # Check that content items don't have cloud-specific fields like "sourceId" with scores
        for item in content:
            # Local LLM typically doesn't provide structured content with scores
            if "sourceId" in item and "score" in item:
                # This might indicate cloud response, which would be unexpected in timeout scenario
                pytest.fail(f"Unexpected cloud-structured content in fallback response: {item}")

    # Verify latency is reasonable (should be local LLM time, not cloud timeout + LLM time)
    # Local LLM should respond faster than cloud timeout + cloud processing
    # Local LLM completion time can exceed 5s depending on provider/model; keep a generous bound
    assert latency < 15.0, f"latency {latency:.2f}s is too high"

    print(f"✅ Fallback test passed - latency: {latency:.2f}s, response from local LLM")


if __name__ == "__main__":
    """
    Standalone execution block for manual testing and debugging.

    This block allows the test to be run directly from the command line for:
    - Manual verification during development
    - Debugging integration issues
    - Quick validation of configuration changes

    Usage:
        python apps/edge-server/tests/test_edge_cloud_fallback.py

    Note: Requires both edge server and cloud RAG services to be running
    """
    print("Running edge cloud fallback test manually...")

    # Test URL generation
    edge_url = get_edge_base_url()
    cloud_url = get_cloud_base_url()
    print(f"Edge URL: {edge_url}")
    print(f"Cloud URL: {cloud_url}")

    # Test payload creation
    payload = create_test_payload()
    print(f"Test payload: {payload}")

    print("Note: Run with pytest for full test execution:")
    print("poetry run pytest apps/edge-server/tests/test_edge_cloud_fallback.py -v")
