"""Utility scripts package for the Cloud RAG service.

This package marker enables running script entry points via Python's
module execution mode (for example: `python -m scripts.seed_index`).
Doing so ensures that absolute imports like `from providers.embeddings
import get_embeddings` resolve reliably because the application root
(`apps/cloud-rag`) is on the import path when executed as a module.
"""
