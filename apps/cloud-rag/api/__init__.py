"""
Cloud RAG API package.

This package will host the FastAPI routers that expose the cloud‑side endpoints
for retrieval‑augmented generation and related operations. In this milestone we
create only the health router to establish a predictable skeleton and to enable
environment readiness checks. Subsequent milestones will add dedicated routers
for the RAG question‑answering flow (e.g., /api/rag/answer) and for feedback
synchronization (e.g., /api/feedback/sync), following the same conventions as
the edge server where practical. Keeping this package separate from the app
factory in main.py promotes clear separation of concerns: routing lives under
api/, while application setup and middleware configuration remain in the
application entrypoint.

The package docstring exists both as documentation and as a reminder that all
routers should include comprehensive docstrings and type hints. This promotes
maintainability and a consistent developer experience across apps within the
monorepo. No public symbols are exported yet; files under this package will be
imported directly by the application as they are implemented.
"""


