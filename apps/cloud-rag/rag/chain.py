"""
Retrieve‑then‑read chain for the Cloud RAG service (retriever → prompt → LLM → validator).

This module assembles a minimal but production‑minded RAG flow that mirrors the conventions used
by the edge server. The chain performs four sequential stages: (1) retrieval from a FAISS index
using an injected embeddings model, (2) prompt construction that embeds contextual citations and
injects runtime variables such as the interaction identifier and top‑k limit, (3) a call to an
injected Large Language Model that is instructed to return a strict JSON object matching the
EnergyEfficiencyResponse schema, and (4) robust JSON parsing and validation using the shared
Pydantic model. The chain exposes a simple Runnable interface that accepts a mapping with keys
"question", "interaction_id", and "top_k" and returns a JSON string that conforms to the schema.

Design goals:
- Keep dependencies injected (llm, embeddings) so provider wiring can be added later without
  refactoring. The module avoids importing any edge code to preserve clean boundaries.
- Read the system prompt from the app's config directory to keep content separate from logic.
- Validate strictly against the cloud EnergyEfficiencyResponse to guarantee a stable contract
  for the edge app and frontend. Validation errors surface as ValueError with concise messages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError

from ..schemas.energy_efficiency import EnergyEfficiencyResponse

logger = logging.getLogger(__name__)

# Optional imports from LangChain with informative fallbacks for environments
# where LangChain is not yet installed during early milestones.
try:  # pragma: no cover - import guard

    from langchain_core.runnables import Runnable, RunnableLambda  # type: ignore
    from langchain_core.retrievers import BaseRetriever  # type: ignore
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain.schema import Document  # type: ignore

except Exception:  # pragma: no cover - fallback types

    class Runnable:  # minimal protocol‑like stub
        def invoke(self, inputs: Dict[str, Any]) -> Any:  # noqa: D401
            """Stub Runnable; install LangChain to use real Runnable."""

    class RunnableLambda(Runnable):  # type: ignore
        def __init__(self, func):
            self._func = func

        def invoke(self, inputs: Dict[str, Any]) -> Any:
            return self._func(inputs)

    class BaseRetriever:  # minimal stub
        def get_relevant_documents(self, query: str):  # pragma: no cover
            raise NotImplementedError

    FAISS = None  # type: ignore

    class Document:  # minimal stub
        def __init__(self, page_content: str, metadata: Dict[str, Any]):
            self.page_content = page_content
            self.metadata = metadata


def load_system_prompt(prompt_path: str) -> str:
    """
    Load the system prompt template from disk using UTF‑8 encoding.

    The Cloud RAG service keeps its prompt text alongside other configuration assets to separate
    content concerns from application logic. This function reads the prompt file verbatim without
    post‑processing, since formatting and variable placeholders (e.g., {{CONTEXT}}, {{TOP_K}},
    {{INTERACTION_ID}}) must be preserved exactly for the downstream LLM. The caller is expected
    to perform a simple string substitution to inject runtime values immediately before invoking
    the model. Keeping the loader small and side‑effect free helps maintain testability and avoids
    introducing dependencies for templating at this stage.

    Args:
        prompt_path (str): Absolute or relative path to the system prompt file. For this app the
            canonical location is `apps/cloud-rag/config/energy_efficiency_system_prompt.txt`.

    Returns:
        str: The raw prompt contents loaded from the specified file.
    """
    path = Path(prompt_path)
    content = path.read_text(encoding="utf-8")
    return content


def build_retriever(faiss_dir: str, embeddings) -> BaseRetriever:
    """
    Build a FAISS retriever from an on‑disk index using the supplied embeddings model.

    This function loads the vector store in read‑only fashion from the directory created during
    the seeding step (Milestone M2 Step 6). If the directory or serialized index files are not
    present, a clear FileNotFoundError is raised to guide the developer to run seeding first.
    The returned object supports retrieving the top‑k most relevant documents for a query; the
    exact value of k is supplied at runtime and can be adjusted via the retriever's search_kwargs.

    Args:
        faiss_dir (str): Directory path where the FAISS index was persisted.
        embeddings: Embeddings model instance compatible with LangChain's FAISS loader.

    Returns:
        BaseRetriever: A retriever instance. Caller may update `retriever.search_kwargs["k"]` per call.
    """
    index_path = Path(faiss_dir)
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Seed the index first (see M2 Step 6)."
        )

    if FAISS is None:  # pragma: no cover - defensive guard
        raise ImportError("LangChain FAISS not available; install langchain_community to proceed.")

    try:
        # Newer LangChain uses allow_dangerous_deserialization flag for safe loading.
        vectorstore = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    except TypeError:
        # Fallback for older signatures without allow_dangerous_deserialization.
        vectorstore = FAISS.load_local(str(index_path), embeddings)

    retriever: BaseRetriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever


def _format_context_items(docs: Any) -> str:
    """Render retrieved documents into a compact JSON list for the prompt."""
    items = []
    for idx, doc in enumerate(docs):
        # Extract minimal fields with safe fallbacks
        metadata = getattr(doc, "metadata", {}) or {}
        source_id = metadata.get("sourceId") or metadata.get("source", f"source_{idx}")
        score = metadata.get("score", 0.0)
        chunk = getattr(doc, "page_content", "")
        items.append({"sourceId": str(source_id), "chunk": str(chunk), "score": float(score)})
    return json.dumps(items, ensure_ascii=False)


def build_chain(llm, retriever: BaseRetriever, system_prompt: str) -> Runnable:
    """
    Construct a Runnable that executes the full RAG flow and returns validated JSON.

    The returned Runnable expects an input mapping with the keys: "question" (user query string),
    "interaction_id" (string identifier for tracing), and "top_k" (integer for retrieval depth).
    At invocation time the chain updates the retriever's top‑k, fetches matching documents, renders
    the CONTEXT section as a compact JSON list of citations, and substitutes variables into the
    provided system prompt. The LLM is then invoked with the fully rendered prompt. The raw model
    output is parsed as JSON and validated against EnergyEfficiencyResponse. If parsing or validation
    fails, a ValueError is raised with a concise message to aid debugging.

    Args:
        llm: An LLM object supporting `.invoke(prompt: str) -> Any` and returning either a string or
            an object with a `.content` attribute containing the text output.
        retriever (BaseRetriever): A retriever built from the FAISS vector store.
        system_prompt (str): The system prompt template containing {{CONTEXT}}, {{INTERACTION_ID}},
            and {{TOP_K}} placeholders.

    Returns:
        Runnable: A Runnable whose `.invoke({question, interaction_id, top_k})` yields a JSON string
        conforming to EnergyEfficiencyResponse.
    """

    def _execute(inputs: Dict[str, Any]) -> str:
        question = str(inputs.get("question", "")).strip()
        interaction_id = str(inputs.get("interaction_id", "")).strip()
        top_k = int(inputs.get("top_k", 3))

        # Adjust retriever k at runtime and retrieve documents
        if hasattr(retriever, "search_kwargs"):
            retriever.search_kwargs["k"] = top_k  # type: ignore[attr-defined]
        docs = retriever.get_relevant_documents(question)

        # Prepare context JSON for the prompt
        context_json = _format_context_items(docs)

        # Simple, explicit substitution (no external templating)
        rendered = (
            system_prompt
            .replace("{{CONTEXT}}", context_json)
            .replace("{{INTERACTION_ID}}", interaction_id)
            .replace("{{TOP_K}}", str(top_k))
        )

        raw = llm.invoke(rendered)
        text = getattr(raw, "content", raw)
        if not isinstance(text, str):
            text = str(text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Model output was not valid JSON: %s", e)
            raise ValueError("Model output was not valid JSON") from e

        try:
            validated = EnergyEfficiencyResponse(**data)
        except ValidationError as e:  # type: ignore[unreachable]
            logger.error("JSON failed schema validation: %s", e)
            raise ValueError("Output failed EnergyEfficiencyResponse validation") from e

        return validated.model_dump_json()

    return RunnableLambda(_execute)


def run_chain(
    question: str,
    interaction_id: str,
    top_k: int,
    faiss_dir: str,
    embeddings,
    llm,
) -> str:
    """
    High‑level helper to execute the RAG chain and return a validated JSON string.

    This function glues together the discrete steps: it loads the system prompt from the Cloud
    app's config directory, builds a FAISS retriever using the provided embeddings and persisted
    index path, composes the Runnable chain, and then invokes it with the supplied inputs. The
    returned string is guaranteed (by validation) to conform to the EnergyEfficiencyResponse
    schema. Any parsing or validation errors are surfaced as ValueError to the caller.

    Args:
        question (str): The user's question to be answered using retrieval‑augmented generation.
        interaction_id (str): Unique identifier for this interaction for tracing and analytics.
        top_k (int): The number of context chunks to retrieve for grounding the answer.
        faiss_dir (str): Filesystem path where the FAISS index is stored.
        embeddings: Embeddings model instance compatible with the FAISS loader.
        llm: LLM instance supporting `.invoke(str) -> (str|object)` semantics.

    Returns:
        str: A JSON string that validates against EnergyEfficiencyResponse.
    """
    prompt_path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "energy_efficiency_system_prompt.txt"
    )
    system_prompt = load_system_prompt(str(prompt_path))

    retriever = build_retriever(faiss_dir=faiss_dir, embeddings=embeddings)
    chain = build_chain(llm=llm, retriever=retriever, system_prompt=system_prompt)

    inputs = {"question": question, "interaction_id": interaction_id, "top_k": top_k}
    return chain.invoke(inputs)


