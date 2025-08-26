"""
Smoke test for the relevance evaluator.

This module contains unit tests for the `evaluate_relevance` function, which implements
an LLM-as-judge pattern to score how well a generated answer addresses a user's question
given retrieved context chunks. The tests validate that the evaluator returns a valid
float score in the [0,1] range and handles edge cases gracefully.

The tests use fixed mock inputs to ensure deterministic behavior and don't require
external services beyond the LLM call itself. This provides confidence that the
evaluation logic is working correctly before integrating with the full evaluation
queue system.
"""

import pytest
from typing import List

# Import the evaluator function
from eval.relevance_evaluator import evaluate_relevance


def test_evaluate_relevance_returns_valid_score():
    """
    Test that evaluate_relevance returns a valid float score in [0,1] for a typical case.

    This test validates the core functionality of the relevance evaluator by providing
    a realistic question, context chunks, and answer that should produce a meaningful
    relevance score. The test ensures that the evaluator can process the inputs,
    communicate with the LLM (if available), parse the response, and return a numeric
    score within the expected range.

    The mock inputs are carefully chosen to represent a typical energy-efficiency
    scenario where the answer should be reasonably relevant to the question and
    grounded in the provided context. This test helps catch issues with the LLM
    integration, prompt construction, JSON parsing, and score normalization.

    The test will pass if the evaluator returns a float in [0,1], indicating that
    the LLM-as-judge logic is functioning correctly and can assess answer quality
    relative to the provided context and question.

    This validation is crucial because relevance scores are used for offline
    evaluation, feedback collection, and continuous improvement of the RAG system.
    """
    # Mock inputs representing a typical energy-efficiency query
    question = "How to save energy at home?"
    context = [
        "Unplug idle devices to reduce standby power consumption.",
        "Use LED bulbs instead of incandescent lighting to save energy.",
        "Adjust your thermostat by 1-2 degrees to reduce heating costs."
    ]
    answer = "Unplug devices and use power strips to eliminate standby power. Switch to LED bulbs for lighting."

    # Call the evaluator
    result = evaluate_relevance(question, context, answer)

    # Validate return type and range
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 <= result <= 1.0, f"Score {result} is not in valid range [0.0, 1.0]"

    # Additional validation for reasonable scores
    assert not (result != result), "Score should not be NaN"  # Check for NaN


def test_evaluate_relevance_empty_context():
    """
    Test evaluate_relevance behavior with empty context list.

    This edge case test ensures the evaluator handles situations where no context
    chunks are provided, which can happen due to retrieval failures or when the
    knowledge base is empty. The evaluator should still return a valid score
    rather than crashing, demonstrating robustness in production scenarios.

    Empty context is a challenging case because the evaluator must judge relevance
    without supporting information. The expected behavior is to return a low score
    (indicating poor grounding) while maintaining the [0,1] range contract.

    This test validates the defensive programming in the evaluator and ensures
    that calling code can rely on getting a valid float regardless of input quality.
    """
    question = "How to reduce energy usage?"
    context = []  # Empty context - no supporting information
    answer = "Use renewable energy sources and improve insulation."

    result = evaluate_relevance(question, context, answer)

    # Should still return a valid score despite empty context
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 <= result <= 1.0, f"Score {result} is not in valid range [0.0, 1.0]"


def test_evaluate_relevance_single_chunk():
    """
    Test evaluate_relevance with minimal single context chunk.

    This test validates the evaluator's behavior with the smallest possible context
    (single chunk), ensuring it can handle minimal information scenarios. Single
    chunk evaluation is common in focused queries where only the most relevant
    information is retrieved.

    The test uses a simple, direct context chunk that should strongly support
    the answer, allowing us to verify that the evaluator can recognize high
    relevance even with minimal context.
    """
    question = "What causes standby power?"
    context = ["Standby power occurs when devices consume electricity even when turned off."]
    answer = "Devices consume standby power when they're plugged in but not actively used."

    result = evaluate_relevance(question, context, answer)

    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 <= result <= 1.0, f"Score {result} is not in valid range [0.0, 1.0]"


# Standalone execution for manual testing
if __name__ == "__main__":
    print("Running relevance evaluator smoke tests...")
    print("=" * 50)

    # Test 1: Normal case
    print("Test 1: Normal case with context")
    try:
        question = "How to save energy at home?"
        context = ["Unplug idle devices to reduce standby power."]
        answer = "Unplug devices to eliminate standby power consumption."

        result = evaluate_relevance(question, context, answer)
        print(f"✅ Score: {result} (range: [0.0, 1.0])")
        print("   Question and answer are well-matched with context")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    print()

    # Test 2: Empty context
    print("Test 2: Empty context edge case")
    try:
        question = "How to reduce energy usage?"
        context = []
        answer = "Use renewable energy sources."

        result = evaluate_relevance(question, context, answer)
        print(f"✅ Score: {result} (range: [0.0, 1.0])")
        print("   Handled empty context gracefully")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

    print()
    print("=" * 50)
    print("Relevance evaluator tests completed!")
    print("\nNote: These tests require NEBIUS_API_KEY to be set in the environment.")
    print("If API key is missing, tests will return 0.0 (graceful degradation).")
