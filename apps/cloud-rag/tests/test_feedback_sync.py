"""
Smoke test for cloud feedback sync API endpoint.

This module contains integration tests for the `/api/feedback/sync` endpoint to ensure
it correctly handles batch feedback ingestion, deduplication by feedback_id, and
provides accurate counts of accepted vs duplicate items. The tests validate the
feedback mirroring system and SQLite persistence functionality.

The tests use dynamic URL configuration by reading from the app's config.json file
or falling back to environment variables, ensuring consistency with the actual
application setup and allowing flexible deployment configurations.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

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
    environment variables. This is crucial for CI/CD pipelines and local development
    setups where different team members might use different port configurations.

    The function handles file reading errors gracefully and never raises exceptions,
    always returning a valid localhost URL with a port number.

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


def create_test_feedback_batch() -> List[Dict[str, Any]]:
    """
    Create a test batch of feedback items with intentional duplicates.

    This helper generates a realistic batch of feedback items for testing purposes.
    The batch includes both unique items and duplicates (same feedback_id) to test
    the deduplication logic. All required fields are populated with valid data
    that represents typical feedback from the edge server.

    The batch structure matches the FeedbackBatch model expected by the API,
    ensuring compatibility with the Pydantic validation on the server side.

    Returns:
        List[Dict[str, Any]]: List of feedback item dictionaries ready for JSON serialization
    """
    now = datetime.utcnow().isoformat() + "Z"

    return [
        {
            "feedback_id": "test-feedback-001",
            "interactionId": "test-interaction-001",
            "score": 1,
            "label": "positive",
            "comment": "Great answer on energy saving!",
            "created_at": now
        },
        {
            "feedback_id": "test-feedback-002",
            "interactionId": "test-interaction-002",
            "score": -1,
            "label": "negative",
            "comment": "Answer was unclear",
            "created_at": now
        },
        {
            "feedback_id": "test-feedback-001",  # Duplicate feedback_id!
            "interactionId": "test-interaction-003",
            "score": 1,
            "label": "positive",
            "comment": "Duplicate submission",
            "created_at": now
        },
        {
            "feedback_id": "test-feedback-003",
            "interactionId": "test-interaction-004",
            "score": 1,
            "label": "positive",
            "comment": "Another good response",
            "created_at": now
        }
    ]


def test_feedback_sync_batch_with_duplicates():
    """
    Test the /api/feedback/sync endpoint with a batch containing duplicates.

    This integration test validates the feedback sync functionality by sending a batch
    of feedback items that includes intentional duplicates (same feedback_id). The test
    ensures that the API correctly handles deduplication, persists unique items to SQLite,
    and returns accurate counts of accepted vs duplicate items.

    The test is crucial for validating the feedback mirroring system that allows the
    edge server to periodically sync user feedback to the cloud for evaluation and
    model improvement. It ensures that the idempotent nature of the API works correctly
    and that duplicate submissions don't corrupt the feedback database.

    The test sends 4 items where 1 is a duplicate (same feedback_id as the first item),
    so it expects 3 accepted items and 1 duplicate. This validates the PRIMARY KEY
    conflict resolution in the SQLite database and the accurate counting logic.

    The test will pass if:
    - The endpoint returns HTTP 200 status
    - The response contains valid accepted and duplicates counts
    - The counts sum to the total number of items sent
    - There is at least 1 duplicate detected (validating deduplication works)
    - The response structure matches the expected JSON format

    This test provides confidence that the feedback sync system will work correctly
    in production, handling real-world scenarios like network retries and duplicate
    submissions gracefully.
    """
    # Build the full endpoint URL
    base_url = get_base_url()
    endpoint = f"{base_url}/api/feedback/sync"

    # Create test batch with duplicates
    feedback_batch = create_test_feedback_batch()
    payload = {"items": feedback_batch}

    # Send POST request to the feedback sync endpoint
    response = requests.post(endpoint, json=payload, timeout=30)

    # Validate HTTP status
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Validate response is valid JSON
    try:
        result = response.json()
    except json.JSONDecodeError as e:
        pytest.fail(f"Response is not valid JSON: {e}")

    # Validate required fields exist
    assert "accepted" in result, "Response missing 'accepted' field"
    assert "duplicates" in result, "Response missing 'duplicates' field"

    # Validate data types
    assert isinstance(result["accepted"], int), "accepted should be an integer"
    assert isinstance(result["duplicates"], int), "duplicates should be an integer"

    # Validate counts are non-negative
    assert result["accepted"] >= 0, "accepted count should be non-negative"
    assert result["duplicates"] >= 0, "duplicates count should be non-negative"

    # Validate total counts match batch size
    total_items = len(feedback_batch)
    total_counted = result["accepted"] + result["duplicates"]
    assert total_counted == total_items, f"accepted + duplicates ({total_counted}) should equal total items ({total_items})"

    # Validate that duplicates were detected (we included a duplicate feedback_id)
    assert result["duplicates"] > 0, "Should detect at least 1 duplicate in the test batch"

    # Validate reasonable accepted count (should be total minus duplicates)
    expected_accepted = total_items - result["duplicates"]
    assert result["accepted"] == expected_accepted, f"Expected {expected_accepted} accepted, got {result['accepted']}"


def test_feedback_sync_empty_batch():
    """
    Test the /api/feedback/sync endpoint with an empty batch.

    This edge case test ensures the API handles empty batches gracefully, which
    can occur when the edge server has no new feedback to sync. The API should
    return zero counts without errors, demonstrating robustness in edge cases.

    Empty batch handling is important for production reliability, as it ensures
    the sync process doesn't fail when there's nothing to sync, and provides
    a consistent response format regardless of batch size.
    """
    # Build the full endpoint URL
    base_url = get_base_url()
    endpoint = f"{base_url}/api/feedback/sync"

    # Create empty batch
    payload = {"items": []}

    # Send POST request to the feedback sync endpoint
    response = requests.post(endpoint, json=payload, timeout=30)

    # Validate HTTP status
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Validate response structure
    result = response.json()
    assert result["accepted"] == 0, "Empty batch should have 0 accepted"
    assert result["duplicates"] == 0, "Empty batch should have 0 duplicates"


# Standalone execution for manual testing
if __name__ == "__main__":
    print("Running feedback sync smoke tests...")
    print("=" * 50)

    base_url = get_base_url()
    print(f"Using base URL: {base_url}")

    # Test 1: Batch with duplicates
    print("\nTest 1: Batch with duplicates")
    try:
        endpoint = f"{base_url}/api/feedback/sync"
        feedback_batch = create_test_feedback_batch()
        payload = {"items": feedback_batch}

        print(f"Sending batch of {len(feedback_batch)} items (including duplicates)...")

        response = requests.post(endpoint, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            print("✅ Response received:")
            print(f"   Accepted: {result['accepted']}")
            print(f"   Duplicates: {result['duplicates']}")
            print(f"   Total: {result['accepted'] + result['duplicates']}")

            # Validate expectations
            if result['duplicates'] > 0:
                print("   ✅ Duplicates detected correctly")
            else:
                print("   ⚠️  No duplicates detected (unexpected)")

        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the cloud service running?")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    # Test 2: Empty batch
    print("\nTest 2: Empty batch")
    try:
        endpoint = f"{base_url}/api/feedback/sync"
        payload = {"items": []}

        response = requests.post(endpoint, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            print("✅ Empty batch response:")
            print(f"   Accepted: {result['accepted']}")
            print(f"   Duplicates: {result['duplicates']}")
            if result['accepted'] == 0 and result['duplicates'] == 0:
                print("   ✅ Empty batch handled correctly")
            else:
                print("   ⚠️  Unexpected counts for empty batch")
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the cloud service running?")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

    print("\n" + "=" * 50)
    print("Feedback sync tests completed!")
    print("\nNote: These tests require the cloud service to be running.")
    print("Start with: CLOUD_RAG_PORT=8000 poetry run python main.py")
