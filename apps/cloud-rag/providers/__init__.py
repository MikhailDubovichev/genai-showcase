"""
Providers package for Cloud RAG.

This package houses provider-specific client factories such as embeddings and
LLM wrappers. It keeps external integration wiring separate from application
logic so the rest of the codebase can depend on narrow, well-documented
interfaces.
"""


