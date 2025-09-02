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
"""

from __future__ import annotations

import json
import time
import re
import logging
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError

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


# Configuration Management
# -----------------------------------------------------------------------------
# Centralized configuration handling with clear data structures.

class ChainConfig:
    """
    Immutable configuration object that freezes all config values at build time.
    
    This eliminates repeated CONFIG reads and makes dependencies explicit.
    """
    
    def __init__(
        self,
        # Retrieval config
        keyword_k: int,
        semantic_k: int,
        fusion_alpha: float,
        mode: str,
        final_top_k: int,
        allow_general: bool,
        # Rerank config
        rerank_enabled: bool,
        rerank_top_n: int,
        rerank_timeout_ms: int,
        rerank_preview_chars: int,
        rerank_batch_size: int,
    ):
        """
        Initialize chain configuration with all necessary parameters.
        
        Args:
            keyword_k: Number of documents to retrieve via BM25.
            semantic_k: Number of documents to retrieve via FAISS.
            fusion_alpha: Weight for semantic scores in hybrid fusion (0.0 to 1.0).
            mode: Retrieval mode ("semantic" or "hybrid").
            final_top_k: Final number of documents to use for context.
            allow_general: Whether to allow general knowledge fallback.
            rerank_enabled: Whether LLM-as-judge reranking is enabled.
            rerank_top_n: Number of candidates to consider for reranking.
            rerank_timeout_ms: Timeout for reranking LLM calls.
            rerank_preview_chars: Number of characters to include in rerank previews.
            rerank_batch_size: Batch size for reranking (currently unused but kept for compatibility).
        """
        self.keyword_k = keyword_k
        self.semantic_k = semantic_k
        self.fusion_alpha = fusion_alpha
        self.mode = mode
        self.final_top_k = final_top_k
        self.allow_general = allow_general
        self.rerank_enabled = rerank_enabled
        self.rerank_top_n = rerank_top_n
        self.rerank_timeout_ms = rerank_timeout_ms
        self.rerank_preview_chars = rerank_preview_chars
        self.rerank_batch_size = rerank_batch_size

    @classmethod
    def from_global_config(cls) -> "ChainConfig":
        """
        Create a ChainConfig instance from the global CONFIG dictionary.
        
        This method reads CONFIG once and freezes all values, preventing
        repeated dictionary lookups during chain execution.
        
        Returns:
            ChainConfig: Immutable configuration object.
        """
        retrieval_cfg = CONFIG.get("retrieval", {})
        rerank_cfg = CONFIG.get("rerank", {})
        
        return cls(
            # Retrieval configuration
            keyword_k=int(retrieval_cfg.get("keyword_k", 6)),
            semantic_k=int(retrieval_cfg.get("semantic_k", 6)),
            fusion_alpha=float((retrieval_cfg.get("fusion", {}) or {}).get("alpha", 0.6)),
            mode=str(retrieval_cfg.get("mode", "semantic")).strip().lower(),
            final_top_k=int(retrieval_cfg.get("default_top_k", 3)),
            allow_general=bool(retrieval_cfg.get("allow_general_knowledge", False)),
            # Rerank configuration
            rerank_enabled=bool(rerank_cfg.get("enabled", False)),
            rerank_top_n=int(rerank_cfg.get("top_n", 10)),
            rerank_timeout_ms=int(rerank_cfg.get("timeout_ms", 3500)),
            rerank_preview_chars=int(rerank_cfg.get("preview_chars", 600)),
            rerank_batch_size=int(rerank_cfg.get("batch_size", 8)),
        )


# Core Chain Functions
# -----------------------------------------------------------------------------
# The main entry points and core execution flow. These functions handle the overall
# RAG pipeline: retrieval → fusion → rerank → LLM → validate.

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


def build_chain(llm, retriever: BaseRetriever, vectorstore: Any, system_prompt: str) -> Runnable:
    """
    Construct a Runnable that executes the full RAG flow and returns validated JSON.

    This function is now clean and focused: it freezes configuration, initializes BM25,
    and delegates actual execution to the modular pipeline functions.

    Args:
        llm: An LLM object supporting `.invoke(prompt: str) -> Any`.
        retriever (BaseRetriever): A retriever built from the FAISS vector store.
        vectorstore: The FAISS vectorstore instance for accessing the document corpus.
        system_prompt (str): The system prompt template with placeholders.

    Returns:
        Runnable: A Runnable whose `.invoke({question, interaction_id, top_k})` yields JSON.
    """
    # Freeze config snapshot at build time
    config = ChainConfig.from_global_config()
    # Emit one concise line to confirm the frozen configuration
    try:
        logger.info(
            "RAG chain configured: mode=%s semantic_k=%d keyword_k=%d final_top_k=%d rerank_enabled=%s",
            config.mode, config.semantic_k, config.keyword_k, config.final_top_k, str(config.rerank_enabled),
        )
    except Exception:
        pass
    
    # Initialize BM25 retriever once at chain creation time
    keyword_retriever = build_bm25_retriever_from_vectorstore(vectorstore, config.keyword_k)
    
    def _execute(inputs: Dict[str, Any]) -> str:
        return execute_rag_pipeline(
            inputs=inputs,
            llm=llm,
            retriever=retriever,
            vectorstore=vectorstore,
            keyword_retriever=keyword_retriever,
            system_prompt=system_prompt,
            config=config
        )

    return RunnableLambda(_execute)


# RAG Pipeline Execution
# -----------------------------------------------------------------------------
# Core pipeline logic separated into focused, testable functions.

def execute_rag_pipeline(
    inputs: Dict[str, Any],
    llm,
    retriever: BaseRetriever,
    vectorstore: Any,
    keyword_retriever: Any,
    system_prompt: str,
    config: ChainConfig
) -> str:
    """
    Execute the complete RAG pipeline: retrieval → fusion → rerank → LLM → validate.
    This is the main orchestrator function that coordinates all pipeline stages.
    
    Args:
        inputs: Input dictionary with question, interaction_id, top_k.
        llm: LLM instance for generation and reranking.
        retriever: FAISS retriever instance.
        vectorstore: FAISS vectorstore for direct queries.
        keyword_retriever: BM25 retriever instance (may be None).
        system_prompt: Template string with placeholders.
        config: Frozen configuration object.
        
    Returns:
        str: Validated JSON string conforming to EnergyEfficiencyResponse.
    """
    question = str(inputs.get("question", "")).strip()
    interaction_id = str(inputs.get("interaction_id", "")).strip()
    top_k = int(inputs.get("top_k", 3))

    # Update retriever's search kwargs if needed
    if hasattr(retriever, "search_kwargs"):
        retriever.search_kwargs["k"] = top_k  # type: ignore[attr-defined]

    # Step 1: Retrieve documents
    docs = retrieve_documents(
        question=question,
        vectorstore=vectorstore,
        keyword_retriever=keyword_retriever,
        config=config
    )

    # Step 2: Optional reranking
    if config.rerank_enabled:
        docs = rerank_documents(
            question=question,
            docs=docs,
            llm=llm,
            config=config
        )

    # Step 3: Generate and validate response
    return generate_response(
        question=question,
        interaction_id=interaction_id,
        top_k=top_k,
        docs=docs,
        llm=llm,
        system_prompt=system_prompt,
        config=config
    )


def retrieve_documents(
    question: str,
    vectorstore: Any,
    keyword_retriever: Any,
    config: ChainConfig
) -> list[tuple[Any, float]]:
    """
    Retrieve and optionally fuse semantic and keyword search results.
    
    This function handles both semantic-only and hybrid retrieval modes.
    All results are normalized to (Document, score) tuples for consistency.
    
    Args:
        question: User's question for retrieval.
        vectorstore: FAISS vectorstore instance.
        keyword_retriever: BM25 retriever instance (may be None).
        config: Configuration object with retrieval settings.
        
    Returns:
        List of (Document, score) tuples, limited to final_top_k.
    """
    # Semantic retrieval with single fallback path
    try:
        semantic_raw = vectorstore.similarity_search_with_score(question, k=config.semantic_k)
    except Exception as e:
        logger.warning("Semantic search failed: %s", str(e))
        semantic_raw = []

    semantic_normalized = normalize_to_doc_score_pairs(semantic_raw)

    if config.mode == "hybrid":
        # Keyword retrieval
        keyword_normalized: list[tuple[Any, float]] = []
        if keyword_retriever is not None:
            try:
                keyword_raw = keyword_retriever.invoke(question)[:config.keyword_k]
                keyword_normalized = normalize_to_doc_score_pairs(keyword_raw)
            except Exception:
                keyword_normalized = []

        if not keyword_normalized:
            logger.info("Retrieval mode=hybrid but BM25 unavailable; falling back to semantic-only.")
            docs = semantic_normalized[:config.final_top_k]
        else:
            # Weighted fusion using rank normalization
            docs = weighted_fuse_by_rank(
                semantic=semantic_normalized,
                keyword=keyword_normalized,
                alpha=config.fusion_alpha,
                final_k=config.final_top_k,
            )

        # Logging for hybrid mode
        logger.info(
            "Retrieval mode=%s | semantic_k=%d keyword_k=%d final_top_k=%d",
            config.mode, config.semantic_k, config.keyword_k, config.final_top_k,
        )
        top_preview = [
            (get_stable_doc_id(doc, i), float(score))
            for i, (doc, score) in enumerate(docs[:3])
        ]
        logger.info("Top fused (doc_id, score): %s", top_preview)
    else:
        # Semantic-only path
        docs = semantic_normalized[:config.final_top_k]
        try:
            logger.info(
                "Retrieval mode=%s | semantic_k=%d final_top_k=%d",
                config.mode, config.semantic_k, config.final_top_k,
            )
        except Exception:
            pass

    return docs


def rerank_documents(
    question: str,
    docs: list[tuple[Any, float]],
    llm,
    config: ChainConfig
) -> list[tuple[Any, float]]:
    """
    Rerank documents using LLM-as-judge scoring.
    
    This function is now simple and focused on just the reranking logic.
    
    Args:
        question: User's question for relevance scoring.
        docs: List of (Document, score) tuples to rerank.
        llm: LLM instance for scoring.
        config: Configuration object with rerank settings.
        
    Returns:
        Reranked list of (Document, score) tuples, limited to final_top_k.
    """
    candidates = docs[:config.rerank_top_n]
    logger.info("Rerank enabled: top_n=%d batch_size=%d", config.rerank_top_n, config.rerank_batch_size)
    
    reranked = llm_judge_rerank(
        question=question,
        docs_list=candidates,
        llm=llm,
        timeout_ms=config.rerank_timeout_ms,
        preview_chars=config.rerank_preview_chars
    )
    docs = reranked[:config.final_top_k]
    
    # Log top reranked results
    top_prev = [
        (get_stable_doc_id(doc, i), float(score))
        for i, (doc, score) in enumerate(docs[:3])
    ]
    logger.info("Top reranked (doc_id, score): %s", top_prev)
    
    return docs


def generate_response(
    question: str,
    interaction_id: str,
    top_k: int,
    docs: list[tuple[Any, float]],
    llm,
    system_prompt: str,
    config: ChainConfig
) -> str:
    """
    Generate and validate the final LLM response.
    
    This function handles context preparation, template rendering,
    LLM invocation, and JSON validation.
    
    Args:
        question: User's question.
        interaction_id: Unique interaction identifier.
        top_k: Number of documents requested.
        docs: List of (Document, score) tuples for context.
        llm: LLM instance for generation.
        system_prompt: Template string with placeholders.
        config: Configuration object.
        
    Returns:
        Validated JSON string conforming to EnergyEfficiencyResponse.
    """
    # Context preparation
    context_json = format_context_items(docs)
    
    # Check for missing context
    no_context = False
    try:
        parsed_items = json.loads(context_json)
        no_context = not isinstance(parsed_items, list) or len(parsed_items) == 0
    except Exception:
        no_context = True

    # Template rendering using frozen config
    fallback_text = (
        "If context is missing or insufficient, you may answer briefly based on household "
        "best practices; return an empty content array."
        if config.allow_general
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

    # Guidance message based on context availability
    if config.allow_general and no_context:
        guidance = (
            "If context is missing or insufficient, you MAY answer briefly based on general household "
            "energy-efficiency best practices. Set content to an empty array. Return ONLY JSON."
        )
    else:
        guidance = "Return ONLY one valid JSON object matching the schema."

    # LLM invocation
    messages = [
        {"role": "system", "content": rendered},
        {"role": "user", "content": guidance},
    ]
    raw = llm.invoke(messages)
    text = getattr(raw, "content", raw)
    if not isinstance(text, str):
        text = str(text)

    # JSON parsing and validation
    try:
        data = json.loads(sanitize_json_text(text))
    except json.JSONDecodeError as e:
        logger.error("LLM output was not valid JSON: %s", e)
        raise ValueError("LLM output was not valid JSON") from e

    try:
        validated = EnergyEfficiencyResponse(**data)
    except ValidationError as e:
        logger.error("JSON failed schema validation: %s", e)
        raise ValueError("Output failed EnergyEfficiencyResponse validation") from e

    return validated.model_dump_json()


# Retrieval Functions
# -----------------------------------------------------------------------------
# Functions for document retrieval, including FAISS setup, BM25 initialization,
# hybrid fusion, and LLM-based reranking.

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
        tuple[BaseRetriever, Any]: A retriever instance and the vectorstore.
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
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def _load_chunks_jsonl(path: Path) -> list[dict]:
    """
    Stream read chunks from the canonical chunks.jsonl file with validation.

    This helper loads the complete set of processed chunks from the JSONL export,
    performing minimal validation to ensure each chunk has the required "text" field
    for BM25 corpus creation. Invalid entries are logged and skipped to maintain
    robustness when initializing the keyword retriever from the portable truth.

    Args:
        path (Path): Path to the chunks.jsonl file.

    Returns:
        list[dict]: List of validated chunk dictionaries with "text" field present.
    """
    chunks = []
    
    if not path.exists():
        logger.debug("chunks.jsonl not found at %s", path)
        return chunks
    
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    chunk = json.loads(line)
                    
                    # Validate that "text" field is present for BM25 corpus
                    if "text" in chunk:
                        chunks.append(chunk)
                    else:
                        logger.debug("Chunk at line %d missing 'text' field", line_num)
                
                except json.JSONDecodeError:
                    logger.debug("Invalid JSON at line %d in chunks.jsonl", line_num)
                    continue
    
    except Exception as exc:
        logger.warning("Failed to read chunks.jsonl from %s: %s", path, exc)
        return []
    
    logger.debug("Loaded %d chunks from %s", len(chunks), path)
    return chunks


def _bm25_corpus_from_chunks(chunks: list[dict]) -> list[str]:
    """
    Extract plain text corpus from chunks for BM25 retriever initialization.

    This helper prepares the text corpus that enables BM25 initialization directly
    from chunks.jsonl rather than depending on FAISS docstore structure. This
    decouples lexical indexing from vector indexing, providing a more portable
    and reliable approach to keyword retrieval initialization.

    Args:
        chunks (list[dict]): Chunk dictionaries from chunks.jsonl.

    Returns:
        list[str]: Plain text list suitable for BM25Retriever.from_texts().
    """
    return [chunk.get("text", "") for chunk in chunks if chunk.get("text", "").strip()]


def _build_bm25_from_chunks_jsonl(chunks_path: Path, k: int) -> Any:
    """
    Build BM25 retriever from chunks.jsonl if available, returning None otherwise.

    This factory function implements the preferred BM25 initialization path that
    uses the canonical chunks.jsonl as the source of truth. When chunks.jsonl
    exists, this approach is more reliable than extracting documents from FAISS
    docstore, as it avoids dependency on internal docstore implementation details.

    Args:
        chunks_path (Path): Path to the chunks.jsonl file.
        k (int): Number of documents the BM25 retriever should return.

    Returns:
        BM25Retriever | None: Configured BM25 retriever if chunks.jsonl exists,
            None if the file is missing or empty.
    """
    if not chunks_path.exists():
        return None
    
    chunks = _load_chunks_jsonl(chunks_path)
    if not chunks:
        logger.warning("chunks.jsonl exists but contains no valid chunks")
        return None
    
    corpus = _bm25_corpus_from_chunks(chunks)
    if not corpus:
        logger.warning("No text content found in chunks for BM25 corpus")
        return None
    
    if BM25Retriever is None:  # pragma: no cover - defensive guard
        logger.warning("BM25Retriever not available; install langchain_community to enable keyword retrieval.")
        return None
    
    try:
        bm25_retriever = BM25Retriever.from_texts(corpus)
        bm25_retriever.k = k
        logger.info("BM25 initialized from chunks.jsonl with %d texts (k=%d)", len(corpus), k)
        return bm25_retriever
    except Exception as exc:
        logger.warning("Failed to build BM25 from chunks.jsonl: %s", exc)
        return None


def build_bm25_retriever_from_vectorstore(vectorstore: Any, keyword_k: int) -> Any:
    """
    Build a BM25 retriever, preferring chunks.jsonl over FAISS docstore extraction.

    This function implements the M13 Step 3 enhancement: it first attempts to initialize
    BM25 from the canonical chunks.jsonl file (preferred approach) and falls back to
    extracting documents from the FAISS vectorstore's docstore only if chunks.jsonl
    is not available. This decouples BM25 from FAISS internals for better reliability.

    Args:
        vectorstore: A FAISS vectorstore instance with a docstore attribute containing documents.
        keyword_k (int): The number of top documents the BM25 retriever should return.

    Returns:
        BM25Retriever | None: A configured BM25 retriever from chunks.jsonl or docstore,
            or None if neither source is accessible or contains valid data.
    """
    if BM25Retriever is None:  # pragma: no cover - defensive guard
        logger.warning("BM25Retriever not available; install langchain_community to enable keyword retrieval.")
        return None

    # Step 1: Try to resolve chunks.jsonl path
    chunks_jsonl_path = None
    try:
        # Prefer the configured FAISS index dir if present in CONFIG
        faiss_index_dir = CONFIG.get("paths", {}).get("faiss_index_dir")
        if faiss_index_dir:
            chunks_jsonl_path = Path(faiss_index_dir) / "chunks.jsonl"
        else:
            # Default: compute relative to chain.py location
            chunks_jsonl_path = Path(__file__).resolve().parents[1] / "faiss_index" / "chunks.jsonl"
    except Exception:
        # Fallback path
        chunks_jsonl_path = Path(__file__).resolve().parents[1] / "faiss_index" / "chunks.jsonl"

    # Step 2: Try BM25 from chunks.jsonl (preferred)
    bm25_retriever = _build_bm25_from_chunks_jsonl(chunks_jsonl_path, keyword_k)
    if bm25_retriever is not None:
        return bm25_retriever

    # Step 3: Fall back to docstore extraction (legacy)
    logger.info("chunks.jsonl not found; falling back to docstore BM25")
    
    try:
        # Try to access FAISS docstore to get the corpus
        if not hasattr(vectorstore, 'docstore') or vectorstore.docstore is None:
            logger.warning("FAISS docstore not accessible for BM25 initialization.")
            return None

        # Extract documents from the docstore (resilient to different backends)
        try:
            ds = vectorstore.docstore
            documents = None
            # Prefer public mapping if available
            if hasattr(ds, 'dict') and isinstance(getattr(ds, 'dict'), dict):
                documents = list(getattr(ds, 'dict').values())
            # Fallback to private storage used by InMemoryDocstore
            elif hasattr(ds, '_dict') and isinstance(getattr(ds, '_dict'), dict):
                documents = list(getattr(ds, '_dict').values())
            # Final fallback: mapping-like interface exposing values()
            elif hasattr(ds, 'values'):
                try:
                    documents = list(ds.values())  # type: ignore[call-arg]
                except Exception:
                    documents = None
            if documents is None:
                raise AttributeError("Unsupported docstore interface for extracting values")
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
        logger.info("BM25 initialized from docstore with %d documents (k=%d)", doc_count, keyword_k)
        return bm25_retriever

    except Exception as e:
        logger.warning("BM25 initialization failed: %s", str(e))
        return None


def normalize_to_doc_score_pairs(results: Any) -> list[tuple[Any, float]]:
    """
    Normalize retrieval results to consistent (Document, score) tuples.
    
    Handles both FAISS similarity_search_with_score format and plain document lists.
    
    Args:
        results: Raw retrieval results in various formats.
        
    Returns:
        List of (Document, score) tuples with normalized scores.
    """
    if not results:
        return []
    
    normalized = []
    for item in results:
        if isinstance(item, tuple) and len(item) == 2:
            doc, score = item
            normalized.append((doc, float(score) if score is not None else 0.0))
        else:
            # Plain document, assign zero score
            normalized.append((item, 0.0))
    
    return normalized


def weighted_fuse_by_rank(
    semantic: list[tuple[Any, float]],
    keyword: list[tuple[Any, float]],
    alpha: float,
    final_k: int,
) -> list[tuple[Any, float]]:
    """
    Fuse semantic and keyword results using rank-normalized weighted scores.

    Rank-based normalization assigns each result a score of 1/(rank+1), which
    keeps scales comparable across heterogeneous retrieval systems (e.g., FAISS
    similarity vs BM25 keyword ranks). Weighted fusion then computes:
    fused = alpha * semantic_norm + (1 - alpha) * keyword_norm.

    Args:
        semantic: List of (Document, score) tuples from semantic retrieval.
        keyword: List of (Document, score) tuples from keyword retrieval.
        alpha: Weight for semantic scores (0.0 to 1.0).
        final_k: Number of top results to return.
        
    Returns:
        List of (Document, fused_score) tuples sorted by score descending.
    """
    # Build rank maps using stable document IDs
    semantic_rank: dict[str, tuple[int, Any]] = {}
    for rank, (doc, _score) in enumerate(semantic):
        doc_id = get_stable_doc_id(doc, rank)
        semantic_rank[doc_id] = (rank, doc)

    keyword_rank: dict[str, tuple[int, Any]] = {}
    for rank, (doc, _score) in enumerate(keyword):
        doc_id = get_stable_doc_id(doc, rank)
        keyword_rank[doc_id] = (rank, doc)

    # Union of document IDs
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


def llm_judge_rerank(
    question: str, 
    docs_list: list[tuple[Any, float]], 
    llm,
    timeout_ms: int,
    preview_chars: int
) -> list[tuple[Any, float]]:
    """
    Score and reorder retrieved candidates using a single LLM-as-judge call.

    This helper performs a lightweight, deterministic reranking step after we have
    collected the top candidates (either from semantic-only retrieval or from the
    hybrid fusion path). It constructs a compact prompt that contains the user
    question and a JSON array of candidate objects. Each candidate includes a stable
    identifier (`id`) and a short text preview taken from the candidate document's
    `page_content`. The provider's chat model is asked to return ONLY a strict JSON
    array of objects in the shape `{id, score}`, where `score` is a relevance value
    in the closed interval [0.0, 1.0].

    Args:
        question: The user's question that candidates must be relevant to.
        docs_list: List of (Document, score) tuples.
        llm: LLM instance for scoring.
        timeout_ms: Soft timeout for logging elapsed durations.
        preview_chars: Maximum characters from `page_content` to include per preview.

    Returns:
        List of (Document, rerank_score) tuples sorted by score descending.
        On failure, returns original order with zero scores.
    """
    # Prepare candidates (doc, id, preview) using stable ID helper
    prepared: list[tuple[Any, str, str]] = []
    for idx, (doc, _score) in enumerate(docs_list):
        doc_id = get_stable_doc_id(doc, idx)
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
        return [(doc, 0.0) for doc, _score in docs_list]

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
    for idx, (doc, _score) in enumerate(docs_list):
        did = get_stable_doc_id(doc, idx)
        ranked.append((doc, float(tmp.get(did, 0.0))))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# Utility Functions
# -----------------------------------------------------------------------------
# Helper functions for document ID generation, JSON formatting, and prompt loading.

def load_system_prompt(prompt_path: str) -> str:
    """
    Load the system prompt template from disk using UTF‑8 encoding.

    The Cloud RAG service keeps its prompt text alongside other configuration assets to separate
    content concerns from application logic. This function reads the prompt file verbatim without
    post‑processing, since formatting and variable placeholders (e.g., {{CONTEXT}}, {{TOP_K}},
    {{INTERACTION_ID}}) must be preserved exactly for the downstream LLM.

    Args:
        prompt_path: Absolute or relative path to the system prompt file.

    Returns:
        str: The raw prompt contents loaded from the specified file.
    """
    path = Path(prompt_path)
    content = path.read_text(encoding="utf-8")
    return content


def format_context_items(docs: list[tuple[Any, float]]) -> str:
    """
    Render retrieved documents into a compact JSON list for the prompt.
    
    Args:
        docs: List of (Document, score) tuples.
        
    Returns:
        JSON string representation of context items.
    """
    items = []
    for idx, (doc, score) in enumerate(docs):
        metadata = getattr(doc, "metadata", {}) or {}
        source_id = metadata.get("sourceId") or metadata.get("source", f"source_{idx}")
        chunk = getattr(doc, "page_content", "")
        items.append({"sourceId": str(source_id), "chunk": str(chunk), "score": float(score)})
    return json.dumps(items, ensure_ascii=False)


def sanitize_json_text(text: str) -> str:
    """
    Best-effort sanitizer: remove code fences and trim to the first JSON object.
    
    Keeps the chain lean while handling common formatting wrappers from chat models.
    
    Args:
        text: Raw LLM output text.
        
    Returns:
        Sanitized JSON string.
    """
    t = text.strip()
    fence = re.compile(r"^```[a-zA-Z]*\n|\n```$", re.MULTILINE)
    t = fence.sub("\n", t)
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


def get_stable_doc_id(doc: Any, fallback_index: int) -> str:
    """
    Derive a stable document identifier used for fusion across retrievers.

    Prefers `metadata["sourceId"]` when available to ensure consistency with the
    rest of the system. If missing, falls back to a combination of any available
    metadata identifiers or the iteration index to maintain determinism.
    
    Args:
        doc: Document object with metadata.
        fallback_index: Index to use if no ID found in metadata.
        
    Returns:
        Stable string identifier for the document.
    """
    metadata = getattr(doc, "metadata", {}) or {}
    return (
        str(metadata.get("sourceId"))
        or str(metadata.get("source"))
        or str(metadata.get("doc_id"))
        or f"idx_{fallback_index}"
    )