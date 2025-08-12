"""
conftest.py – central pytest configuration and test bootstrap ("config test").

Etymology:
- The name "conftest" comes from "config test". In the pytest ecosystem, this file is a conventional
  location for shared test configuration and fixtures. Pytest automatically looks for and imports
  any `conftest.py` files it finds while discovering tests, starting from the tests directory and
  walking upward through parent directories as needed. Because of this auto-discovery behavior,
  no explicit imports are required in individual test modules.

Purpose and behavior:
- Pytest imports this module before it collects any test files. This early import phase allows us to
  prepare the test environment so that subsequent imports and test collection succeed consistently.
  In particular, we:
  1) Extend `sys.path` with the project root directory so absolute-style imports like `from core ...`
     and `from shared ...` resolve without performing an editable install.
  2) Define safe default environment variables required at import time by the application’s
     configuration layer (for example, `NEBIUS_API_KEY`, which the LLM provider reads). Providing
     a default prevents import-time failures during test discovery.

Rationale:
- Centralizing environment setup in `conftest.py` keeps individual test files free of boilerplate and
  enforces consistent, deterministic behavior across local development and CI (Continuous Integration).
  This approach upholds separation of concerns: tests focus on behavior, while environment wiring is
  handled once here.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct imports like `core`, `shared`, etc.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Provide required environment defaults for tests
os.environ.setdefault("NEBIUS_API_KEY", "test-key") 