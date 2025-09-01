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

M12 Step 1 Update:
- Added BM25 keyword retriever initialization alongside FAISS for future hybrid retrieval.
- BM25 is built from FAISS docstore at chain creation time and stored for efficient access.
- No behavioral changes to retrieval logic yet; FAISS remains the active retriever.
- Guards against docstore access failures with informative logging.

TODO (future retrieval upgrades):
- Add MMR (Maximal Marginal Relevance) to diversify top‑k results without changing the API surface.
- Wire BM25 + FAISS hybrid retrieval and simple re‑ranking (cross‑encoder or LLM‑based),
  keeping the public endpoint and response schema unchanged. These should be toggled via config only.
"""

from __future__ import annotations

import json
import time
import re
import logging
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError
import json as _json

from schemas.energy_efficiency import EnergyEfficiencyResponse
from config import CONFIG

logger = logging.getLogger(__name__)

# Optional imports from LangChain with informative fallbacks for environments
# where LangChain is not yet installed during early milestones.
try:  # pragma: no cover - import guard

    from langchain_core.runnables import Runnable, RunnableLambda  # type: ignore
    from langchain_core.retrievers import BaseRetriever  # type: ignore
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_community.retrievers import BM25Retriever  # type: ignore
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
    BM25Retriever = None  # type: ignore

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


def build_retriever(faiss_dir: str, embeddings) -> tuple[BaseRetriever, Any]:
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

    # Validate manifest if present to catch embedding/shape mismatches early
    manifest_path = index_path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
            # Try to compute current embedding dimension and compare
            try:
                current_dim = len(embeddings.embed_query("probe"))
            except Exception:
                current_dim = len(embeddings.embed_query("test"))
            if int(manifest.get("dimension", current_dim)) != int(current_dim):
                raise RuntimeError(
                    "FAISS index embedding dimension mismatch. Reseed the index with the current embeddings."
                )
        except Exception:
            # Non-fatal: proceed, FAISS will still attempt to load
            pass

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
    return retriever, vectorstore


def build_bm25_retriever_from_vectorstore(vectorstore: Any, keyword_k: int) -> Any:
    """
    Build a BM25 retriever from the FAISS vectorstore's document store.

    This helper function extracts documents from the FAISS vectorstore's docstore to build
    a keyword-based BM25 retriever. This is done early in M12 Step 1 to prepare for hybrid
    retrieval (semantic + keyword) in Step 2, without changing the current retrieval behavior.

    The function accesses the FAISS docstore to get the corpus documents. If the docstore
    is not accessible or empty, it returns None and logs a warning. This approach avoids
    requiring a separate chunks manifest file in this step, while still enabling BM25
    initialization for future hybrid retrieval.

    Args:
        vectorstore: A FAISS vectorstore instance with a docstore attribute containing documents.
        keyword_k (int): The number of top documents the BM25 retriever should return.

    Returns:
        BM25Retriever | None: A configured BM25 retriever if docstore is accessible,
            or None if the docstore cannot be accessed or is empty.

    Note:
        - This function guards against docstore access failures gracefully.
        - The BM25 retriever is built once at startup for efficiency.
        - If docstore access fails, future steps may need a chunks manifest approach.
    """
    if BM25Retriever is None:  # pragma: no cover - defensive guard
        logger.warning("BM25Retriever not available; install langchain_community to enable keyword retrieval.")
        return None

    try:
        # Try to access FAISS docstore to get the corpus
        if not hasattr(vectorstore, 'docstore') or vectorstore.docstore is None:
            logger.warning("FAISS docstore not accessible for BM25 initialization.")
            return None

        # Extract documents from the docstore
        try:
            documents = list(vectorstore.docstore.dict.values())
        except Exception as e:
            logger.warning("Failed to extract documents from FAISS docstore: %s", str(e))
            return None

        if not documents:
            logger.warning("FAISS docstore is empty; cannot build BM25 retriever.")
            return None

        # Build BM25 retriever from the documents
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = keyword_k

        doc_count = len(documents)
        logger.info("BM25 retriever initialized with %d documents (k=%d).", doc_count, keyword_k)
        return bm25_retriever

    except Exception as e:
        logger.warning("BM25 initialization failed: %s", str(e))
        return None


def _format_context_items(docs: Any) -> str:
    """Render retrieved documents into a compact JSON list for the prompt."""
    items = []
    for idx, entry in enumerate(docs):
        # Support both (Document, score) tuples and bare Documents
        if isinstance(entry, tuple) and len(entry) == 2:
            doc, doc_score = entry
            score_val = float(doc_score) if doc_score is not None else 0.0
        else:
            doc = entry
            score_val = 0.0
        metadata = getattr(doc, "metadata", {}) or {}
        source_id = metadata.get("sourceId") or metadata.get("source", f"source_{idx}")
        chunk = getattr(doc, "page_content", "")
        items.append({"sourceId": str(source_id), "chunk": str(chunk), "score": score_val})
    return json.dumps(items, ensure_ascii=False)


def _sanitize_json_text(text: str) -> str:
    """
    Best-effort sanitizer: remove code fences and trim to the first JSON object.
    Keeps the chain lean while handling common formatting wrappers from chat models.
    """
    t = text.strip()
    fence = re.compile(r"^```[a-zA-Z]*\n|\n```$", re.MULTILINE)
    t = fence.sub("\n", t)
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


def build_chain(llm, retriever: BaseRetriever, vectorstore: Any, system_prompt: str) -> Runnable:
    """
    Construct a Runnable that executes the full RAG flow and returns validated JSON.

    The returned Runnable expects an input mapping with the keys: "question" (user query string),
    "interaction_id" (string identifier for tracing), and "top_k" (integer for retrieval depth).
    At invocation time the chain updates the retriever's top‑k, fetches matching documents, renders
    the CONTEXT section as a compact JSON list of citations, and substitutes variables into the
    provided system prompt. The LLM is then invoked with the fully rendered prompt. The raw model
    output is parsed as JSON and validated against EnergyEfficiencyResponse. If parsing or validation
    fails, a ValueError is raised with a concise message to aid debugging.

    This function also initializes a BM25 keyword retriever alongside the FAISS retriever,
    preparing for hybrid retrieval in future steps. The BM25 retriever is built once at
    startup and stored on the chain instance for efficient access.

    Args:
        llm: An LLM object supporting `.invoke(prompt: str) -> Any` and returning either a string or
            an object with a `.content` attribute containing the text output.
        retriever (BaseRetriever): A retriever built from the FAISS vector store.
        vectorstore: The FAISS vectorstore instance for accessing the document corpus.
        system_prompt (str): The system prompt template containing {{CONTEXT}}, {{INTERACTION_ID}},
            and {{TOP_K}} placeholders.

    Returns:
        Runnable: A Runnable whose `.invoke({question, interaction_id, top_k})` yields a JSON string
        conforming to EnergyEfficiencyResponse.
    """

    # Initialize BM25 retriever once at chain creation time
    retrieval_cfg = CONFIG.get("retrieval", {})
    keyword_k = int(retrieval_cfg.get("keyword_k", 6))
    semantic_k_default = int(retrieval_cfg.get("semantic_k", 6))
    fusion_alpha = float((retrieval_cfg.get("fusion", {}) or {}).get("alpha", 0.6))
    mode = str(retrieval_cfg.get("mode", "semantic")).strip().lower()
    keyword_retriever = build_bm25_retriever_from_vectorstore(vectorstore, keyword_k)

    def _get_stable_doc_id(doc: Any, fallback_index: int) -> str:
        """
        Derive a stable document identifier used for fusion across retrievers.

        Prefers `metadata["sourceId"]` when available to ensure consistency with the
        rest of the system. If missing, falls back to a combination of any available
        metadata identifiers or the iteration index to maintain determinism.
        """
        metadata = getattr(doc, "metadata", {}) or {}
        return (
            str(metadata.get("sourceId"))
            or str(metadata.get("source"))
            or str(metadata.get("doc_id"))
            or f"idx_{fallback_index}"
        )

    def _weighted_fuse_by_rank(
        semantic: list[tuple[Any, float]] | list[Any],
        keyword: list[Any] | None,
        alpha: float,
        final_k: int,
    ) -> list[tuple[Any, float]]:
        """
        Fuse semantic and keyword results using rank-normalized weighted scores.

        Rank-based normalization assigns each result a score of 1/(rank+1), which
        keeps scales comparable across heterogeneous retrieval systems (e.g., FAISS
        similarity vs BM25 keyword ranks). Weighted fusion then computes:
        fused = alpha * semantic_norm + (1 - alpha) * keyword_norm.

        Missing sides default to 0.0. The function returns the top `final_k` pairs
        of (Document, fused_score) sorted by fused score descending, using the
        semantic side as the source of canonical Document objects when available,
        and falling back to keyword docs if a doc appears only on the keyword side.
        """
        # Build rank maps
        semantic_rank: dict[str, tuple[int, Any]] = {}
        for rank, item in enumerate(semantic):
            if isinstance(item, tuple) and len(item) == 2:
                doc = item[0]
            else:
                doc = item
            doc_id = _get_stable_doc_id(doc, rank)
            semantic_rank[doc_id] = (rank, doc)

        keyword_rank: dict[str, tuple[int, Any]] = {}
        if keyword:
            for rank, doc in enumerate(keyword):
                doc_id = _get_stable_doc_id(doc, rank)
                keyword_rank[doc_id] = (rank, doc)

        # Union of ids
        all_ids = set(semantic_rank.keys()) | set(keyword_rank.keys())

        fused: list[tuple[Any, float]] = []
        for doc_id in all_ids:
            sem_rank_doc = semantic_rank.get(doc_id)
            key_rank_doc = keyword_rank.get(doc_id)

            sem_norm = 1.0 / (sem_rank_doc[0] + 1) if sem_rank_doc else 0.0
            key_norm = 1.0 / (key_rank_doc[0] + 1) if key_rank_doc else 0.0
            fused_score = alpha * sem_norm + (1.0 - alpha) * key_norm

            # Prefer the semantic Document object when available for downstream formatting
            doc_obj = sem_rank_doc[1] if sem_rank_doc else key_rank_doc[1]  # type: ignore[index]
            fused.append((doc_obj, fused_score))

        fused.sort(key=lambda x: float(x[1]), reverse=True)
        return fused[:final_k]

    def llm_judge_rerank(question: str, docs_list: list[Any], cfg: dict) -> list[tuple[Any, float]]:
        """
        Score and reorder retrieved candidates using a single LLM-as-judge call.

        This helper performs a lightweight, deterministic reranking step after we have
        collected the top candidates (either from semantic-only retrieval or from the
        hybrid fusion path). It constructs a compact prompt that contains the user
        question and a JSON array of candidate objects. Each candidate includes a stable
        identifier (`id`) and a short text preview taken from the candidate document’s
        `page_content`. The provider’s chat model is asked to return ONLY a strict JSON
        array of objects in the shape `{id, score}`, where `score` is a relevance value
        in the closed interval [0.0, 1.0]. The function parses this output, clamps any
        numeric irregularities into [0, 1], and returns the input candidates ordered by
        descending score, as pairs `(Document, float)`.

        The preview length is configurable via `cfg["preview_chars"]` (default 600),
        allowing operators to trade off latency and context sufficiency. If the model
        invocation fails or the response is not valid JSON, the function logs a concise
        INFO message and returns the original candidate order with zero scores rather
        than raising. This keeps the reranker safe to enable without impacting the API
        surface or error behavior of the main chain.

        Args:
            question (str): The user’s question that candidates must be relevant to.
            docs_list (list[Any]): A list of candidate documents (or `(Document, score)`
                tuples). The function will read each item’s `page_content` and metadata
                to build a minimal preview and stable identifier.
            cfg (dict): Rerank configuration mapping. Expected keys include:
                - "timeout_ms" (int): Soft timeout for logging elapsed durations.
                - "preview_chars" (int): Maximum characters from `page_content` to include
                  per candidate preview (default 600).

        Returns:
            list[tuple[Any, float]]: Candidates paired with rerank scores in [0, 1],
            sorted descending by score. On failure, original order with zero scores.
        """
        timeout_ms = int(cfg.get("timeout_ms", 3500))
        preview_chars = int(cfg.get("preview_chars", 600))
        # Prepare candidates (doc, id, preview)
        prepared: list[tuple[Any, str, str]] = []
        for idx, item in enumerate(docs_list):
            doc = item[0] if isinstance(item, tuple) and len(item) == 2 else item
            doc_id = _get_stable_doc_id(doc, idx)
            preview = str(getattr(doc, "page_content", ""))[:preview_chars]
            prepared.append((doc, doc_id, preview))
        # Build prompt with all candidates in one request
        system_msg = (
            "You are a ranking assistant. Score each candidate's relevance to the user "
            "question from 0.0 to 1.0. Return ONLY a JSON array of objects with fields "
            "{id, score}. No prose, no extra keys."
        )
        candidates_payload = [
            {"id": doc_id, "preview": preview} for (_doc, doc_id, preview) in prepared
        ]
        user_msg = (
            "Question:\n" + question + "\n\nCandidates (JSON array):\n" + json.dumps(candidates_payload, ensure_ascii=False)
        )

        t0 = time.time()
        try:
            out = llm.invoke([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])  # type: ignore[var-annotated]
        except Exception:
            try:
                out = llm.invoke(f"SYSTEM:\n{system_msg}\n\nUSER:\n{user_msg}")  # type: ignore[var-annotated]
            except Exception:
                out = None
        elapsed_ms = int((time.time() - t0) * 1000.0)
        if elapsed_ms > timeout_ms:
            logger.info("LLM rerank call exceeded timeout_ms (elapsed=%dms)", elapsed_ms)

        text = getattr(out, "content", out)
        if not isinstance(text, str):
            text = str(text)

        # Parse once; on failure, skip rerank
        try:
            arr = json.loads(text)
        except Exception:
            logger.info("Rerank JSON parse failed; keeping original order for %d candidates.", len(prepared))
            return [
                (item[0] if isinstance(item, tuple) and len(item) == 2 else item, 0.0)
                for item in docs_list
            ]

        tmp: dict[str, float] = {}
        if isinstance(arr, list):
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                did = obj.get("id")
                sc = obj.get("score")
                if did is None or sc is None:
                    continue
                try:
                    val = float(sc)
                except Exception:
                    continue
                tmp[str(did)] = val

        # Normalize into [0,1]
        if tmp and max(tmp.values()) > 1.0 and max(tmp.values()) <= 10.0:
            for k in list(tmp.keys()):
                tmp[k] = float(tmp[k]) / 10.0
        for k in list(tmp.keys()):
            v = tmp[k]
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            tmp[k] = v

        ranked: list[tuple[Any, float]] = []
        for idx, item in enumerate(docs_list):
            doc = item[0] if isinstance(item, tuple) and len(item) == 2 else item
            did = _get_stable_doc_id(doc, idx)
            ranked.append((doc, float(tmp.get(did, 0.0))))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _execute(inputs: Dict[str, Any]) -> str:
        question = str(inputs.get("question", "")).strip()
        interaction_id = str(inputs.get("interaction_id", "")).strip()
        top_k = int(inputs.get("top_k", 3))

        # Retrieval mode selection: semantic-only or hybrid
        if hasattr(retriever, "search_kwargs"):
            retriever.search_kwargs["k"] = top_k  # type: ignore[attr-defined]

        retrieval_mode = mode
        sem_k = int(semantic_k_default)
        key_k = int(keyword_k)
        alpha = float(fusion_alpha)

        # Semantic retrieval
        try:
            semantic_results = vectorstore.similarity_search_with_score(question, k=sem_k)
        except Exception:
            try:
                semantic_results = vectorstore.similarity_search_with_relevance_scores(question, k=sem_k)
            except Exception:
                # Fallback to retriever without scores
                semantic_results = retriever.invoke(question)

        final_top_k = int((CONFIG.get("retrieval", {}) or {}).get("default_top_k", 3))

        if retrieval_mode == "hybrid":
            keyword_results = None
            if keyword_retriever is not None:
                try:
                    keyword_results = keyword_retriever.invoke(question)[:key_k]
                except Exception:
                    keyword_results = None

            if keyword_results is None:
                logger.info(
                    "Retrieval mode=hybrid but BM25 unavailable; falling back to semantic-only."
                )
                fused_docs = [
                    (item[0], 0.0) if isinstance(item, tuple) else (item, 0.0)
                    for item in (semantic_results if isinstance(semantic_results, list) else [])
                ]
            else:
                fused_docs = _weighted_fuse_by_rank(
                    semantic=semantic_results,
                    keyword=keyword_results,
                    alpha=alpha,
                    final_k=final_top_k,
                )

            # Logging
            logger.info(
                "Retrieval mode=%s | semantic_k=%d keyword_k=%d final_top_k=%d",
                retrieval_mode,
                sem_k,
                key_k,
                final_top_k,
            )
            top_preview = [
                (
                    _get_stable_doc_id(d if not isinstance(d, tuple) else d[0], i),
                    float(s if not isinstance(d, tuple) else d[1] if i < len(fused_docs) else 0.0),
                )
                for i, (d, s) in enumerate(fused_docs[:3])
            ]
            logger.info("Top fused (doc_id, score): %s", top_preview)

            docs = fused_docs
        else:
            # semantic-only path uses semantic results as-is, with scores if available
            docs = semantic_results[:final_top_k] if isinstance(semantic_results, list) else semantic_results

        # Optional LLM-as-judge rerank stage
        rerank_cfg = (CONFIG.get("rerank", {}) or {})
        if bool(rerank_cfg.get("enabled", False)):
            # Assemble candidate list limited to top_n
            top_n = int(rerank_cfg.get("top_n", 10))
            candidates = list(docs)[:top_n]
            logger.info(
                "Rerank enabled: top_n=%d batch_size=%d",
                top_n,
                int(rerank_cfg.get("batch_size", 8)),
            )
            ranked = llm_judge_rerank(question=question, docs_list=candidates, cfg=rerank_cfg)
            # Keep only final_top_k after rerank
            docs = ranked[:final_top_k]
            top_prev = []
            for i, (d, s) in enumerate(docs[:3]):
                top_prev.append((_get_stable_doc_id(d, i), float(s)))
            logger.info("Top reranked (doc_id, score): %s", top_prev)

        # Prepare context JSON for the prompt
        context_json = _format_context_items(docs)

        # Decide whether to allow general-knowledge fallback when retrieval is weak
        retrieval_cfg = (CONFIG.get("retrieval", {}) or {})
        allow_general = bool(retrieval_cfg.get("allow_general_knowledge", False))
        no_context = False
        try:
            parsed_items = json.loads(context_json)
            no_context = not isinstance(parsed_items, list) or len(parsed_items) == 0
        except Exception:
            no_context = True

        # Simple, explicit substitution (no external templating)
        fallback_text = (
            "If context is missing or insufficient, you may answer briefly based on household "
            "best practices; return an empty content array."
            if allow_general
            else "If context is missing or insufficient, say so briefly and return an empty content array."
        )

        rendered = (
            system_prompt
            .replace("{{CONTEXT}}", context_json)
            .replace("{{INTERACTION_ID}}", interaction_id)
            .replace("{{TOP_K}}", str(top_k))
            .replace("{{QUESTION}}", question)
            .replace("{{FALLBACK_POLICY}}", fallback_text)
        )

        # Prefer chat-style invocation with explicit system+user messages to improve compliance
        if allow_general and no_context:
            guidance = (
                "If context is missing or insufficient, you MAY answer briefly based on general household "
                "energy-efficiency best practices. Set content to an empty array. Return ONLY JSON."
            )
        else:
            guidance = "Return ONLY one valid JSON object matching the schema."

        messages = [
            {"role": "system", "content": rendered},
            {"role": "user", "content": guidance},
        ]
        raw = llm.invoke(messages)
        text = getattr(raw, "content", raw)
        if not isinstance(text, str):
            text = str(text)

        try:
            data = json.loads(_sanitize_json_text(text))
        except json.JSONDecodeError as e:
            # One cheap retry with an explicit JSON-only reminder
            logger.error("Model output was not valid JSON: %s", e)
            retry_messages = [
                {"role": "system", "content": rendered},
                {"role": "user", "content": "Return ONLY one valid JSON object that matches the schema. No extra text."},
            ]
            raw_retry = llm.invoke(retry_messages)
            text_retry = getattr(raw_retry, "content", raw_retry)
            if not isinstance(text_retry, str):
                text_retry = str(text_retry)
            try:
                data = json.loads(_sanitize_json_text(text_retry))
            except json.JSONDecodeError as e2:
                # Final minimal fallback: attempt balanced braces extraction
                start = text_retry.find("{")
                end = text_retry.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = text_retry[start : end + 1]
                    try:
                        data = json.loads(candidate)
                    except Exception as e3:
                        raise ValueError("Model output was not valid JSON") from e3
                else:
                    raise ValueError("Model output was not valid JSON") from e2

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

    retriever, vectorstore = build_retriever(faiss_dir=faiss_dir, embeddings=embeddings)
    chain = build_chain(llm=llm, retriever=retriever, vectorstore=vectorstore, system_prompt=system_prompt)

    inputs = {"question": question, "interaction_id": interaction_id, "top_k": top_k}
    return chain.invoke(inputs)


