"""
Seed utilities for the Cloud RAG service: unified document ingestion, chunking, export, and FAISS build.

This script serves two closely related purposes depending on milestone needs:

1) Legacy text seeding (M2 flow): Walk a directory of plain‑text snippets (e.g., .txt and .md),
   chunk each file into overlapping windows to preserve local context, embed the chunks via the
   configured provider (Nebius/OpenAI) through LangChain integrations, and build a FAISS vector
   index saved to disk. The Cloud RAG chain later loads this FAISS index for semantic retrieval.

2) PDF ingestion to portable chunks.jsonl (M13 Step 1): Scan a fixed input directory for PDF files,
   load with a robust loader strategy (prefer PyMuPDFLoader; fall back to PyPDFLoader), apply a
   heading‑aware → sentence‑window policy (heading path is kept when available; otherwise we use a
   simple 10‑sentence window with 2‑sentence overlap), normalize whitespace, and write one JSON
   object per line to `faiss_index/chunks.jsonl`. This JSONL becomes a canonical, provider‑agnostic
   artifact for downstream indexing (FAISS) and lexical initialization (BM25) without re‑parsing PDFs.

The module is provider‑neutral. Earlier milestones defaulted to Nebius, but current code allows
switching Nebius↔OpenAI via app configuration. For text seeding, embeddings are created and FAISS is
persisted immediately. For PDF ingestion, we only export chunks in this step; later steps build
indexes from `chunks.jsonl` to support idempotency and clean separation of ingestion from indexing.

CLI usage (macOS zsh examples)

A) Text seeding → FAISS (legacy path; parameters are optional and used only by the text flow):
    export NEBIUS_API_KEY=...  # or OPENAI_API_KEY=...
    poetry run python -m scripts.seed_index \
        --data-dir rag/data/seed \
        --index-dir faiss_index

B) PDF ingestion → chunks.jsonl (M13 Step 1 default when run as a module):
    # Uses fixed input dir: apps/cloud-rag/rag/data/seed, and writes faiss_index/chunks.jsonl
    poetry run python -m scripts.seed_index

Logging
- Reports discovered files/PDFs, number of chunks produced, and output paths
- Fails fast with actionable messages if FAISS integration or loaders are missing (text flow)
- For PDF flow, missing loaders trigger graceful fallback and concise warnings

Future improvements (tracked in TASKS.md)
- Idempotent manifest with content hashes and splitter/embedding configs
- Rebuild FAISS and initialize BM25 directly from chunks.jsonl (no PDF re‑parse on every run)
- Optional recursive directory scanning and additional file formats
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any

# Ensure app root is on sys.path so `providers` can be imported when running this
# file directly (e.g., via Cursor's "Run Python File" or python path/to/script.py)
_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from providers import get_embeddings
import json
from datetime import datetime, timezone
import re
import hashlib

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# M13 Step 1: Unified document ingestion → sentence-window chunking → chunks.jsonl export
# -----------------------------------------------------------------------------
SENT_WINDOW_SIZE = 10  # 8–12 sentences per chunk
SENT_WINDOW_OVERLAP = 2  # 2-sentence overlap


def _list_all_files(root: Path) -> List[Path]:
    """
    Enumerate all supported files (PDF, txt, md) in the seed directory for unified processing.

    This function inspects the immediate children of the provided directory and returns all
    paths whose extension is ".pdf", ".txt", or ".md" (case‑insensitive). The scan is
    intentionally non‑recursive to keep behavior deterministic and easy to reason about.

    Args:
        root (Path): The directory under which source files are searched.

    Returns:
        List[Path]: A list of file paths pointing to supported source files.
    """
    supported_exts = {".pdf", ".txt", ".md"}
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]


def _load_document_content(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load a single file into structured document records with unified metadata.

    This function handles different file types and returns a consistent structure:
    - PDFs: Use PyMuPDFLoader (fallback PyPDFLoader), one record per page
    - Text/Markdown: Read as UTF-8, single record with full content

    Each record contains:
    - content: The text content
    - source_path: File path
    - source_type: File extension without dot (pdf, txt, md)
    - page: Page number for PDFs, None for text files
    - heading_path: Extracted headings (empty list if not available)

    Args:
        file_path (Path): The path of the file to load.

    Returns:
        List[Dict[str, Any]]: A list of document records with unified metadata.
    """
    records = []
    source_type = file_path.suffix.lower().lstrip(".")
    
    if source_type == "pdf":
        # Load PDF documents
        try:
            from langchain_community.document_loaders import PyMuPDFLoader  # type: ignore
            loader = PyMuPDFLoader(str(file_path))
            docs = loader.load()
        except Exception:
            try:
                from langchain_community.document_loaders import PyPDFLoader  # type: ignore
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            except Exception as exc:
                logger.warning("Failed to load PDF %s: %s", file_path, exc)
                return []
        
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            page = metadata.get("page") or metadata.get("page_number") or None
            heading_path = metadata.get("heading_path") or []
            if not isinstance(heading_path, list):
                heading_path = []
            
            records.append({
                "content": getattr(doc, "page_content", ""),
                "source_path": str(file_path),
                "source_type": source_type,
                "page": int(page) if isinstance(page, int) else page,
                "heading_path": heading_path,
            })
    
    elif source_type in {"txt", "md"}:
        # Load text/markdown files
        try:
            content = file_path.read_text(encoding="utf-8")
            records.append({
                "content": content,
                "source_path": str(file_path),
                "source_type": source_type,
                "page": None,
                "heading_path": [],
            })
        except Exception as exc:
            logger.warning("Failed to read text file %s: %s", file_path, exc)
    
    return records


def _sentence_tokenize(text: str) -> List[str]:
    """
    Split raw text into sentences using a lightweight regular‑expression heuristic.

    This tokenizer normalizes whitespace first (collapsing repeated spaces and trimming ends) and
    then splits on sentence‑final punctuation (period, question mark, exclamation) followed by
    whitespace. The approach is intentionally dependency‑light to avoid heavyweight NLP libraries
    at this stage, yet robust enough for typical energy‑efficiency documents. The output is used
    by the sentence‑window chunker to build stable, human‑readable chunks that respect sentence
    boundaries and reduce mid‑sentence cuts that harm retrieval and reranking quality.

    Args:
        text (str): Raw text extracted from a document or an earlier preprocessing step.

    Returns:
        List[str]: A list of sentence strings with whitespace normalized and empties removed.
    """
    if not text:
        return []
    # Normalize whitespace first
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return []
    # Split while keeping punctuation with the sentence
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [s.strip() for s in parts if s.strip()]


def _chunk_sentences(sentences: List[str], size: int, overlap: int) -> List[str]:
    """
    Create overlapping sentence windows to preserve local context during chunking.

    Given a list of sentences, this function forms chunks by taking `size` sentences per chunk and
    advancing by `size - overlap` sentences each step. Overlap (e.g., two sentences) provides local
    continuity across adjacent chunks, improving downstream retrieval where tight sentence borders
    could otherwise clip critical context. The chunker returns a list of strings where each string
    concatenates the corresponding window of sentences with normalized spacing preserved.

    Args:
        sentences (List[str]): The sentence‑tokenized input.
        size (int): Number of sentences per chunk window.
        overlap (int): Number of sentences to overlap between consecutive windows.

    Returns:
        List[str]: Overlapping sentence‑window strings suitable for export or embedding.
    """
    if size <= 0:
        return [" ".join(sentences)] if sentences else []
    chunks: List[str] = []
    step = max(1, size - max(0, overlap))
    i = 0
    while i < len(sentences):
        window = sentences[i : i + size]
        if not window:
            break
        chunks.append(" ".join(window).strip())
        i += step
    return chunks


def _normalize_text(text: str) -> str:
    """
    Normalize whitespace in a chunk to produce stable, comparable text payloads.

    Exported chunks should be consistent across platforms and minor variations in loaders. This
    function collapses repeated whitespace, converts all internal runs to single spaces, and trims
    leading and trailing spaces. The normalized text becomes the basis for computing a stable
    SHA‑256 hash and for consistent JSONL output. Normalization also helps deduplicate near‑identical
    chunks generated from slightly different sentence segmentation or line‑break conventions.

    Args:
        text (str): The raw chunk text as produced by windowing.

    Returns:
        str: A whitespace‑normalized, trimmed text string ready for hashing and export.
    """
    return re.sub(r"\s+", " ", str(text)).strip()


def _stable_doc_id_from_stem(stem: str) -> str:
    """
    Generate a stable `doc_id` from a filename stem by normalizing characters.

    The function lowercases the given stem, replaces any non‑alphanumeric characters with
    underscores, collapses multiple underscores, and strips leading/trailing underscores. This
    produces a deterministic identifier suitable for building chunk IDs of the form
    `f"{doc_id}#{chunk_index}"`. Keeping `doc_id` stable ensures chunk IDs remain consistent across
    runs and environments, which is critical for downstream fusion, reranking, and trace linking.

    Args:
        stem (str): Filename stem (without extension) to convert into a `doc_id`.

    Returns:
        str: A normalized identifier comprised of lowercase letters, digits, and underscores.
    """
    s = stem.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "doc"


def _build_unified_chunks(input_dir: Path) -> List[dict]:
    """
    Unified document ingestion and chunking pipeline for all supported file types.

    This function implements the M13 Step 1 unified approach:
    1. Discover all supported files (PDF, txt, md) in the input directory
    2. Load each file into structured document records
    3. Apply consistent sentence-window chunking to all document content
    4. Build portable chunk dictionaries with stable IDs and metadata

    The pipeline ensures all file types use the same chunking logic (sentence-window with
    10 sentences per chunk, 2 sentences overlap) for consistency in downstream retrieval
    and reranking performance.

    Args:
        input_dir (Path): The directory containing source files to ingest.

    Returns:
        List[dict]: A list of chunk dictionaries ready for JSONL export and indexing.
    """
    chunks_out: List[dict] = []
    files = _list_all_files(input_dir)
    
    logger.info("Discovered %d files for unified ingestion", len(files))
    
    # Step 1: Load all documents with unified metadata
    all_documents = []
    for file_path in files:
        records = _load_document_content(file_path)
        all_documents.extend(records)
    
    logger.info("Loaded %d document records from %d files", len(all_documents), len(files))
    
    # Step 2: Apply consistent sentence-window chunking to all documents
    total_chunks = 0
    for doc_record in all_documents:
        content = doc_record["content"]
        if not content.strip():
            continue
            
        # Generate stable doc_id from filename
        file_path = Path(doc_record["source_path"])
        doc_id = _stable_doc_id_from_stem(file_path.stem)
        
        # Apply sentence-window chunking
        sentences = _sentence_tokenize(content)
        windows = _chunk_sentences(sentences, SENT_WINDOW_SIZE, SENT_WINDOW_OVERLAP)
        
        chunk_index = 0
        for window in windows:
            text_norm = _normalize_text(window)
            if not text_norm:
                continue
                
            chunk_id = f"{doc_id}#{chunk_index}"
            h = hashlib.sha256(text_norm.encode("utf-8")).hexdigest()
            
            chunks_out.append({
                "id": chunk_id,
                "doc_id": doc_id,
                "source_path": doc_record["source_path"],
                "source_type": doc_record["source_type"],
                "page": doc_record["page"],
                "heading_path": doc_record["heading_path"],
                "text": text_norm,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "hash": h,
            })
            chunk_index += 1
        
        if chunk_index > 0:
            total_chunks += chunk_index
    
    logger.info("Generated %d chunks using unified sentence-window chunking", len(chunks_out))
    return chunks_out


def _write_chunks_jsonl(out_path: Path, chunks: List[dict]) -> None:
    """
    Write chunk records to a JSON Lines file that downstream jobs can stream efficiently.

    The JSONL format emits exactly one JSON object per line and is easy to process from Python,
    shell tools, or data pipelines. This function rewrites the output file on each run to provide
    a clean snapshot of the current ingestion. It ensures the parent directory exists, serializes
    each chunk with `ensure_ascii=False` to preserve characters, and logs a concise summary with
    the total number of chunks and file path. Later steps will use this file to build FAISS and
    initialize BM25 without needing to re‑parse source files.

    Args:
        out_path (Path): The destination path for `chunks.jsonl`.
        chunks (List[dict]): The in‑memory list of chunk dictionaries to serialize.

    Returns:
        None: This function performs I/O and logs progress but returns no value.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in chunks:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
    logger.info("Wrote %d chunks to %s", len(chunks), out_path)


def _build_documents_from_chunks(chunks: List[dict], source_prefix: str = ""):
    """
    Convert chunk records into LangChain Documents for FAISS index building.

    This helper bridges the unified chunking pipeline to the legacy FAISS building path.
    It takes chunk dictionaries (from the JSONL export) and converts them to LangChain
    Document objects with the expected metadata format for the existing index builder.

    Args:
        chunks (List[dict]): Chunk dictionaries from unified ingestion.
        source_prefix (str): Optional prefix for sourceId metadata.

    Yields:
        Document: LangChain Document objects ready for FAISS embedding and indexing.
    """
    # Import Document with compatibility
    try:
        from langchain_core.documents import Document  # type: ignore
    except Exception:
        try:
            from langchain.schema import Document  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing LangChain dependency. Install it via: poetry add langchain langchain-community"
            ) from exc
    
    for chunk in chunks:
        source_id = f"{source_prefix}{chunk['id']}"
        yield Document(
            page_content=chunk["text"],
            metadata={"sourceId": source_id, "score": 0.0}
        )


def seed_index(
    data_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    source_prefix: str,
) -> None:
    """
    Seed the FAISS index using the unified document ingestion and chunking pipeline.

    This function implements the legacy FAISS seeding path using the new unified approach.
    It processes all supported file types (PDF, txt, md) with consistent sentence-window
    chunking, then builds and persists a FAISS index. The chunk_size and chunk_overlap
    parameters are preserved for API compatibility but the actual chunking uses the
    sentence-window approach for consistency with the M13 pipeline.

    Args:
        data_dir (Path): Directory containing source files (PDF, txt, md).
        index_dir (Path): Output directory for FAISS index files.
        chunk_size (int): Legacy parameter (preserved for API compatibility).
        chunk_overlap (int): Legacy parameter (preserved for API compatibility).
        source_prefix (str): Prefix used to build stable sourceId values.
    """
    logger.info("Starting unified FAISS seeding from %s", data_dir)
    
    # Step 1: Build chunks using unified pipeline
    chunks = _build_unified_chunks(data_dir)
    if not chunks:
        logger.warning("No chunks generated from %s", data_dir)
        return
    
    # Step 2: Convert chunks to LangChain Documents
    docs = list(_build_documents_from_chunks(chunks, source_prefix))
    logger.info("Prepared %d documents for embedding", len(docs))
    
    # Step 3: Build FAISS index
    try:
        from config import CONFIG  # local import to avoid circulars at module import
    except Exception:  # pragma: no cover - defensive
        CONFIG = {}
    embeddings = get_embeddings(CONFIG)
    logger.info("Using embeddings provider from CONFIG")

    # Determine embedding dimension for manifest
    try:
        dim = len(embeddings.embed_query("probe"))
    except Exception:
        # Fallback probe text
        dim = len(embeddings.embed_query("test"))

    try:
        from langchain_community.vectorstores import FAISS  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing FAISS integration. Install it via: pip install langchain-community faiss-cpu"
        ) from exc

    vectorstore = FAISS.from_documents(docs, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    try:
        vectorstore.save_local(str(index_dir), allow_dangerous_serialization=True)
    except TypeError:
        vectorstore.save_local(str(index_dir))

    logger.info("Saved FAISS index to %s", index_dir)

    # Write a manifest with embedding metadata to assist validation at load time
    manifest = {
        "model": os.environ.get("EMBEDDINGS_MODEL") or "",
        "config_model": (os.environ.get("EMBEDDINGS_MODEL") or ""),
        "dimension": dim,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "seeded_at": datetime.now(timezone.utc).isoformat(),
    }
    # If available, prefer configured model name from CONFIG
    try:
        from config import CONFIG  # type: ignore
        manifest["config_model"] = (CONFIG.get("embeddings", {}) or {}).get("name", "")
    except Exception:
        pass
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote FAISS manifest to %s", index_dir / "manifest.json")


def main() -> None:
    """
    Entry point for M13 Step 1: unified document ingestion and chunks.jsonl export.

    This command implements the streamlined pipeline:
    1. Load documents from all supported file types (PDF, txt, md)
    2. Apply consistent sentence-window chunking to all content
    3. Write unified chunks.jsonl with stable IDs and provenance metadata
    4. Log progress and summary statistics

    The approach ensures all file types use the same chunking logic for downstream
    consistency in retrieval and reranking performance.

    Returns:
        None: Coordinates unified ingestion and logs progress and output location.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app_root = Path(__file__).resolve().parents[1]
    input_dir = (app_root / "rag" / "data" / "seed").resolve()
    output_jsonl = (app_root / "faiss_index" / "chunks.jsonl").resolve()

    if not input_dir.exists():
        logger.info("Seed directory %s does not exist. Creating it.", input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Place source files (.pdf, .txt, .md) under %s and rerun seeding.", input_dir)
        # Continue; will produce zero chunks.

    # Unified pipeline: documents → chunks → jsonl
    chunks = _build_unified_chunks(input_dir)
    _write_chunks_jsonl(output_jsonl, chunks)
    
    logger.info("Unified ingestion complete. Total chunks: %d", len(chunks))


if __name__ == "__main__":
    # Support both main() and legacy CLI args
    if len(sys.argv) > 1:
        # Legacy CLI mode for backward compatibility
        parser = argparse.ArgumentParser(description="Seed FAISS index from documents")
        parser.add_argument("--data-dir", type=Path, default="rag/data/seed")
        parser.add_argument("--index-dir", type=Path, default="faiss_index")
        parser.add_argument("--chunk-size", type=int, default=1000)
        parser.add_argument("--chunk-overlap", type=int, default=150)
        parser.add_argument("--source-prefix", type=str, default="")
        
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        seed_index(
            args.data_dir,
            args.index_dir,
            args.chunk_size,
            args.chunk_overlap,
            args.source_prefix,
        )
    else:
        # M13 Step 1 mode (default)
        main()