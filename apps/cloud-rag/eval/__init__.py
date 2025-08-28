"""
Evaluation package for Cloud RAG.

This package exposes evaluation utilities (e.g., relevance scoring) for use in
tests and offline scripts. Keeping an __init__ module ensures Python treats
the directory as a package so imports like `from eval.relevance_evaluator import evaluate_relevance`
work reliably in pytest and runtime contexts.
"""


