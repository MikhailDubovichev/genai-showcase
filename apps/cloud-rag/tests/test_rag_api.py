"""
Smoke test for cloud RAG API endpoint.

This module contains basic smoke tests for the `/api/rag/answer` endpoint to ensure
it returns valid JSON responses on simple success cases. These tests validate the
end-to-end RAG pipeline functionality without complex assertions, focusing on
basic connectivity and response structure validation.

The tests use dynamic URL configuration by reading from the app's config.json file
or falling back to environment variables, ensuring consistency with the actual
application setup.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

import pytest
import requests


def get_base_url() -> str:
    """
    Get the base URL for the cloud RAG service using dynamic configuration.

    This helper function reads the service port from the application configuration
    to ensure tests use the same port as the running service. It first checks for
    a port setting in the config.json file, then falls back to environment variables,
    and finally uses a default port of 8000.

    The dynamic URL resolution ensures that tests work correctly whether the service
    is running on the default port or a custom port specified in configuration or
    environment variables.

    Returns:
        str: Base URL for the cloud RAG service (e.g., "http://localhost:8000")

    Example:
        >>> get_base_url()
        "http://localhost:8000"
    """
    # Try to read port from config.json
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            port = config.get("port", 8000)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        # Fall back to environment variable or default
        port = int(os.getenv("CLOUD_RAG_PORT", "8000"))

    return f"http://localhost:{port}"


def test_rag_answer_simple_success():
    """
    Test the /api/rag/answer endpoint with a simple success case.

    This smoke test validates that the RAG API endpoint is functioning correctly
    by sending a basic question and verifying that the response contains valid JSON
    with the expected structure. The test focuses on basic connectivity and response
    format rather than detailed content validation.

    The test sends a simple energy-related question and expects a structured response
    containing the answer message, source chunks with similarity scores, and proper
    metadata. This ensures the entire RAG pipeline (retrieval, generation, formatting)
    is working end-to-end.

    The test will pass if:
    - The endpoint returns HTTP 200 status
    - The response is valid JSON
    - The JSON contains required fields with correct types
    - Each content item has proper structure and data types

    This test serves as a quick validation that the RAG service is operational
    and can handle basic requests successfully.
    """
    # Build the full endpoint URL
    base_url = get_base_url()
    endpoint = f"{base_url}/api/rag/answer"

    # Prepare test payload with simple energy question
    payload = {
        "question": "How to reduce energy consumption at home?",
        "interactionId": "test-smoke-123",
        "topK": 3
    }

    # Send POST request to the RAG endpoint
    response = requests.post(endpoint, json=payload, timeout=30)

    # Validate HTTP status
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Validate response is valid JSON
    try:
        result = response.json()
    except json.JSONDecodeError as e:
        pytest.fail(f"Response is not valid JSON: {e}")

    # Validate required top-level keys exist
    required_keys = ["message", "content", "interactionId"]
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"

    # Validate data types of top-level fields
    assert isinstance(result["message"], str), "message should be a string"
    assert isinstance(result["content"], list), "content should be a list"
    assert isinstance(result["interactionId"], str), "interactionId should be a string"

    # Validate content array structure (if not empty)
    if result["content"]:
        for i, item in enumerate(result["content"]):
            assert isinstance(item, dict), f"content[{i}] should be a dict"

            # Each content item should have sourceId, chunk, and score
            required_content_keys = ["sourceId", "chunk", "score"]
            for key in required_content_keys:
                assert key in item, f"content[{i}] missing key: {key}"

            # Validate data types
            assert isinstance(item["sourceId"], str), f"content[{i}].sourceId should be string"
            assert isinstance(item["chunk"], str), f"content[{i}].chunk should be string"
            assert isinstance(item["score"], (int, float)), f"content[{i}].score should be numeric"


# Standalone execution for manual testing
if __name__ == "__main__":
    print("Running smoke test for RAG API...")
    print(f"Using base URL: {get_base_url()}")

    try:
        test_rag_answer_simple_success()
        print("✅ Smoke test PASSED")
        print("RAG API is responding correctly with valid JSON structure")
    except Exception as e:
        print(f"❌ Smoke test FAILED: {e}")
        exit(1)
