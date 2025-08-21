"""
Run a small offline relevance evaluation over a golden dataset (JSONL).

This CLI script reads a dataset file (one JSON object per line; “dataset (JSONL)”
means one JSON object per line), runs the Cloud RAG chain to produce answers for
each question, and then uses an LLM (Large Language Model) as a judge to score
the relevance of the answer with respect to the question and a small set of
provided context chunks. The judge emits a compact JSON (JavaScript Object
Notation) object with a numeric relevance field in [0,1]. We aggregate the
results and print a summary with item count, mean relevance, and JSON-valid
rate. The script keeps dependencies minimal and leverages the project’s existing
providers for embeddings and LLM. It also supports optional best‑effort logging
of relevance scores to LangFuse if configured.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

from config import CONFIG
from providers.nebius_embeddings import get_embeddings
from providers.nebius_llm import get_llm
from rag.chain import run_chain
from eval.relevance_evaluator import evaluate_relevance

try:  # optional, best‑effort scoring to LangFuse
    from providers.langfuse import add_trace_score  # type: ignore
except Exception:  # pragma: no cover - optional
    add_trace_score = None  # type: ignore


logger = logging.getLogger(__name__)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file where each line is a JSON object; skip blanks and invalid lines.

    This function opens the given path and iterates over lines. Blank lines are
    ignored. Each non-blank line is parsed with `json.loads` inside a try/except;
    failures are logged at WARNING level and skipped so a single malformed entry
    does not abort evaluation. The function returns a list of dictionaries ready
    for processing by `run_golden_eval`.

    Args:
        path (str): Filesystem path to the dataset file (JSONL format).

    Returns:
        List[Dict[str, Any]]: A list of parsed JSON objects.
    """
    out: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        logger.warning("dataset not found: %s", path)
        return out
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
            else:
                logger.warning("line %s not an object; skipping", i)
        except Exception as exc:
            logger.warning("failed to parse line %s: %s", i, exc)
    return out


def run_golden_eval(
    dataset_path: str,
    top_k: int = 3,
    log_to_langfuse: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluation over a JSONL dataset and return aggregate metrics.

    The function loads the dataset, prepares embeddings and LLM once, then for each
    item: extracts the question, optional context chunks (used only by the judge),
    and a stable interaction id. It calls the RAG chain to produce an answer JSON,
    validates it, and extracts the `message` field as the final answer text. Then it
    calls `evaluate_relevance(question, context_chunks[:3], answer)` to get a score
    in [0,1]. Scores are aggregated to compute a mean; JSON validity is tracked to
    compute a simple valid-rate. Optionally, if `log_to_langfuse` is true and the
    LangFuse helper is available, the function attempts to record the relevance score
    on the corresponding trace id. Errors are swallowed to keep evaluation robust.

    Args:
        dataset_path (str): Path to the JSONL dataset file to evaluate.
        top_k (int): Retrieval depth used in the RAG chain call (default 3).
        log_to_langfuse (bool): If true and LangFuse is configured, log scores to traces.

    Returns:
        Dict[str, Any]: {"count": N, "mean_relevance": float, "json_valid_rate": float}
    """
    items = read_jsonl(dataset_path)
    if not items:
        return {"count": 0, "mean_relevance": 0.0, "json_valid_rate": 0.0}

    embeddings = get_embeddings()
    llm = get_llm()

    faiss_dir = str((CONFIG.get("paths", {}) or {}).get("faiss_index_dir", "faiss_index"))

    relevances: List[float] = []
    valid_json_count = 0

    for idx, item in enumerate(items):
        try:
            q = str(item.get("question", ""))
            ctx = [str(s) for s in list(item.get("context_chunks", []))][:3]
            # Interaction id: use provided id or a stable fallback
            interaction_id = str(item.get("id") or f"{idx:032x}")

            result_json = run_chain(
                question=q,
                interaction_id=interaction_id,
                top_k=int(top_k),
                faiss_dir=faiss_dir,
                embeddings=embeddings,
                llm=llm,
            )

            try:
                obj = json.loads(result_json)
                valid_json_count += 1
            except Exception:
                obj = {}

            answer_text = str(obj.get("message", "")) if isinstance(obj, dict) else ""
            rel = float(evaluate_relevance(q, ctx, answer_text, llm=llm))
            if rel < 0.0:
                rel = 0.0
            if rel > 1.0:
                rel = 1.0
            relevances.append(rel)

            if log_to_langfuse and add_trace_score is not None:  # type: ignore
                try:
                    add_trace_score(trace_id=interaction_id, name="relevance", value=float(rel), comment="golden-run")  # type: ignore
                except Exception:
                    pass
        except Exception:
            # Swallow per-item failures to keep the run going
            pass

    count = len(items)
    mean_relevance = float(statistics.fmean(relevances)) if relevances else 0.0
    json_valid_rate = float(valid_json_count / count) if count > 0 else 0.0
    # Round for a compact summary
    mean_relevance = round(mean_relevance, 4)
    json_valid_rate = round(json_valid_rate, 4)

    return {
        "count": count,
        "mean_relevance": mean_relevance,
        "json_valid_rate": json_valid_rate,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "apps/cloud-rag/eval/data/golden.jsonl"
    summary = run_golden_eval(dataset_path=dataset, top_k=3, log_to_langfuse=False)
    print(json.dumps(summary, indent=2))


