import sys
from pathlib import Path

# Ensure app root is on sys.path so imports like `from eval.relevance_evaluator` work
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


