"""
LLM-as-judge relevance evaluator for Cloud RAG answers (returns a score in [0, 1]).

This module provides a tiny, self-contained evaluator that asks an LLM (Large Language
Model) to judge how well a final answer addresses a user’s question given a small set of
retrieved context chunks. The LLM returns a single JSON (JavaScript Object Notation)
object with a numeric field `relevance`, which we parse, clamp to the unit interval, and
return as a Python float. In this MVP, “relevance” means that the answer directly
addresses the question and is grounded in the provided context (i.e., supported rather
than contradicted). The evaluator is intentionally robust: it never raises, always
returns a float in [0.0, 1.0], and defaults to 0.0 when no model is available or when
parsing fails. The design keeps dependency injection: callers may pass a custom `llm`
instance; otherwise we attempt to load the default via `providers.nebius_llm.get_llm()`.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional
from pathlib import Path

# Attempt to import the default LLM factory. If import fails, we'll handle it at runtime
# by returning 0.0 when no LLM is available.
try:  # pragma: no cover - import-time guard
    from providers.nebius_llm import get_llm  # type: ignore
except Exception:  # pragma: no cover - keep optional
    get_llm = None  # type: ignore


logger = logging.getLogger(__name__)

# TODO: Make the evaluator LLM model configurable via config/config.json (e.g.,
# an eval.llm.model entry), so we can select different models on Nebius for
# evaluation independently of generation models.

# Default on-disk system prompt path for the evaluator model
DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "relevance_evaluator_system_prompt.txt"
)


def clamp_to_unit_interval(value: float) -> float:
    """
    Clamp a numeric value to the closed unit interval [0.0, 1.0].

    This helper normalizes arbitrary numeric inputs by bounding them into the
    inclusive range between zero and one. Clamping ensures that downstream
    consumers receive a stable value even if the evaluator or model returns an
    unexpected number. The function never raises and returns 0.0 if the input
    cannot be converted to a float (e.g., due to a type error).

    Args:
        value (float): Numeric input to clamp.

    Returns:
        float: A number in the interval [0.0, 1.0].
    """
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def extract_relevance(text: str) -> Optional[float]:
    """
    Extract a relevance score from a model response containing a JSON object.

    The function uses best-effort parsing to find a JSON object with a `relevance`
    field in the provided text. It trims code fences if present (for example,
    triple-backtick blocks), looks for the first substring delimited by braces,
    and attempts to load it via `json.loads`. If a numeric `relevance` field is
    present, the value is returned; otherwise `None` is returned to indicate that
    parsing failed. The function never raises and favors robustness over strictness.

    Args:
        text (str): Raw model output that should contain a JSON object.

    Returns:
        Optional[float]: The extracted numeric relevance score, or None on failure.
    """
    if not isinstance(text, str) or not text:
        return None
    s = text.strip()
    # Strip simple Markdown fences if present
    if s.startswith("```"):
        s = s.strip("`").strip()
    # Heuristically find the first JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = s[start : end + 1]
    try:
        obj = json.loads(candidate)
        rel = obj.get("relevance")
        if isinstance(rel, (int, float)):
            return float(rel)
        # Attempt string-to-float if needed
        if isinstance(rel, str):
            return float(rel)
    except Exception:
        return None
    return None


def evaluate_relevance(
    question: str,
    context_chunks: List[str],
    answer: str,
    llm: Optional[object] = None,
    max_context: int = 3,
) -> float:
    """
    Evaluate how well an answer addresses a question using retrieved context; return a score in [0,1].

    This function implements an LLM-as-judge pattern designed to be small, deterministic, and
    robust for MVP purposes. It selects up to `max_context` context chunks, constructs a compact
    judge prompt asking for a single JSON object `{"relevance": <float in [0,1]>}`, invokes the
    provided `llm` (or the default if not provided), and extracts the `relevance` score from the
    model output. The result is clamped to the interval [0.0, 1.0]. On any error (e.g., missing
    model, invocation failure, or parse issues), the function logs a warning and returns 0.0.
    The function never raises exceptions and is safe to use in production control-flow logic.

    Args:
        question (str): The user’s question that the answer attempts to address.
        context_chunks (List[str]): Retrieved context snippets that should ground the answer.
        answer (str): The model-produced final answer to be judged.
        llm (Optional[object]): Optional LLM instance with an `.invoke(...)` method; when None,
            we attempt to load the default via `providers.nebius_llm.get_llm()`.
        max_context (int): Maximum number of context chunks to include in the judge prompt (default 3).

    Returns:
        float: A numeric relevance score in [0.0, 1.0]. Defaults to 0.0 on failure.
    """
    try:
        # Select context
        selected = context_chunks[: max(0, int(max_context))]
        joined_context = "\n---\n".join(str(s) for s in selected)

        system_prompt = _load_system_prompt_safe()

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context (up to {max_context} chunks, separated by ---):\n{joined_context}\n\n"
            f"Answer:\n{answer}\n\n"
            "Return ONLY JSON: {\"relevance\": <float in [0,1]>}"
        )

        # Resolve LLM if not provided
        model = llm
        if model is None:
            if get_llm is None:  # type: ignore
                return 0.0
            try:
                model = get_llm()  # type: ignore
            except Exception:
                return 0.0
            if model is None:
                return 0.0

        # Try invoking in a chat-like style first; fall back to a single string
        output = None
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            output = model.invoke(messages)
        except Exception:
            try:
                output = model.invoke(f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}")
            except Exception:
                return 0.0

        text = getattr(output, "content", output)
        if not isinstance(text, str):
            text = str(text)
        rel = extract_relevance(text)
        if rel is None:
            return 0.0
        return clamp_to_unit_interval(rel)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("relevance evaluation failed: %s", exc)
        return 0.0


def _load_system_prompt_safe(path: Optional[str] = None) -> str:
    """
    Load the evaluator system prompt text from disk; fall back to a minimal inline prompt.

    This helper attempts to read a plain text file containing instructions for the
    evaluator LLM. The intent is to keep prompts configurable and editable without
    changing code. If the file cannot be read (missing or permissions), a compact
    built-in fallback prompt is used. The function never raises and returns a
    string suitable for the LLM system message.

    Args:
        path (Optional[str]): Optional override path to the system prompt file.

    Returns:
        str: The prompt content to use in the evaluator system message.
    """
    prompt_path = Path(path) if path else DEFAULT_PROMPT_PATH
    try:
        return prompt_path.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are a strict evaluator for energy-efficiency answers. Given a question, "
            "retrieved context, and a final answer, return ONLY a JSON object of the form "
            "{\"relevance\": <float in [0,1]>}. The relevance reflects how well the answer "
            "addresses the question and is grounded in the provided context."
        )


