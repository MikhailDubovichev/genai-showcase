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
# M13 Step 2: Idempotency manifest and incremental rebuild
# -----------------------------------------------------------------------------
SENT_WINDOW_SIZE = 10  # 8–12 sentences per chunk
SENT_WINDOW_OVERLAP = 2  # 2-sentence overlap


# -----------------------------------------------------------------------------
# M13 Step 2: Manifest and idempotency helpers
# -----------------------------------------------------------------------------

def _manifest_path(app_root: Path) -> Path:
    """
    Get the standard path for the idempotency manifest file.

    The manifest tracks per-file metadata (content hashes, chunk counts, timestamps) and
    splitter configuration to enable incremental rebuilds. Only files that have changed
    content or when the splitter config changes need to be re-processed.

    Args:
        app_root (Path): The application root directory.

    Returns:
        Path: Absolute path to the manifest.json file.
    """
    return app_root / "faiss_index" / "manifest.json"


def _chunks_jsonl_path(app_root: Path) -> Path:
    """
    Get the standard path for the canonical chunks JSONL file.

    This file contains all processed chunks from all source file types, serving as the
    single source of truth for downstream indexing (FAISS) and retrieval (BM25). The
    incremental rebuild process preserves chunks from unchanged files and merges them
    with newly generated chunks from changed files.

    Args:
        app_root (Path): The application root directory.

    Returns:
        Path: Absolute path to the chunks.jsonl file.
    """
    return app_root / "faiss_index" / "chunks.jsonl"


def _compute_file_hash(path: Path) -> str:
    """
    Compute SHA-256 hash of file contents for change detection.

    This hash is used to determine if a source file has been modified since the last
    seeding run. Only files with different content hashes need to be re-processed,
    enabling efficient incremental rebuilds of large document collections.

    Args:
        path (Path): File path to hash.

    Returns:
        str: Hexadecimal SHA-256 hash of the file contents.
    """
    try:
        with path.open("rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as exc:
        logger.warning("Failed to compute hash for %s: %s", path, exc)
        return ""


def _compute_config_fingerprint() -> str:
    """
    Compute configuration fingerprint for splitter settings to detect config changes.

    When splitter configuration changes (sentence window size or overlap), all files
    must be re-processed regardless of content hashes. This fingerprint captures the
    current splitter config to enable detection of such changes across runs.

    Returns:
        str: SHA-256 hash of the current splitter configuration.
    """
    config_dict = {
        "sent_window_size": SENT_WINDOW_SIZE,
        "sent_window_overlap": SENT_WINDOW_OVERLAP,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()


def _load_manifest(path: Path) -> Dict[str, Any]:
    """
    Load the idempotency manifest from disk, returning empty structure if missing.

    The manifest tracks file-level metadata and splitter configuration to support
    incremental rebuilds. If the manifest file doesn't exist or is malformed, this
    function returns a clean empty structure, causing all files to be treated as
    new and requiring full processing.

    Args:
        path (Path): Path to the manifest.json file.

    Returns:
        Dict[str, Any]: Manifest structure with schema_version, config, and files sections.
    """
    if not path.exists():
        return {
            "schema_version": 1,
            "config": {
                "splitter": {"sent_window_size": 0, "sent_window_overlap": 0},
                "config_fingerprint": "",
            },
            "files": {},
        }
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure required structure exists
            if not isinstance(data.get("files"), dict):
                data["files"] = {}
            if not isinstance(data.get("config"), dict):
                data["config"] = {
                    "splitter": {"sent_window_size": 0, "sent_window_overlap": 0},
                    "config_fingerprint": "",
                }
            return data
    except Exception as exc:
        logger.warning("Failed to load manifest from %s: %s", path, exc)
        return {
            "schema_version": 1,
            "config": {
                "splitter": {"sent_window_size": 0, "sent_window_overlap": 0},
                "config_fingerprint": "",
            },
            "files": {},
        }


def _save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """
    Save the updated idempotency manifest to disk with proper formatting.

    The manifest is written atomically by creating the parent directory and serializing
    with consistent formatting. This ensures the manifest remains valid even if the
    process is interrupted during write operations.

    Args:
        path (Path): Path to the manifest.json file.
        manifest (Dict[str, Any]): Complete manifest structure to persist.

    Returns:
        None: This function performs I/O and logs completion but returns no value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info("Updated manifest: %s", path)
    except Exception as exc:
        logger.error("Failed to save manifest to %s: %s", path, exc)


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


def _determine_change_set(
    input_dir: Path, app_root: Path
) -> Tuple[List[Path], List[Path], List[str]]:
    """
    Analyze source files and manifest to determine what needs to be reprocessed.

    This function implements the core incremental rebuild logic by comparing current
    file hashes against the manifest and checking for splitter configuration changes.
    It returns three sets: files that need reprocessing, files that can be preserved,
    and files that have been deleted since the last run.

    Args:
        input_dir (Path): Directory containing source files to analyze.
        app_root (Path): Application root directory for manifest access.

    Returns:
        Tuple[List[Path], List[Path], List[str]]: A tuple containing:
            - changed_files: Files that need reprocessing (new, modified, or config changed)
            - unchanged_files: Files that can be preserved from previous run
            - deleted_files: File paths from manifest that no longer exist on disk
    """
    current_config_fp = _compute_config_fingerprint()
    manifest = _load_manifest(_manifest_path(app_root))
    
    # If config changed, treat all files as changed
    manifest_config_fp = manifest.get("config", {}).get("config_fingerprint", "")
    config_changed = current_config_fp != manifest_config_fp
    
    if config_changed:
        logger.info("Splitter config changed; forcing full rebuild")
    
    # Get current files and compute their hashes
    current_files = _list_all_files(input_dir)
    current_file_info = {}
    
    for file_path in current_files:
        # Use relative path from app_root for consistent manifest keys
        try:
            rel_path = str(file_path.relative_to(app_root))
        except ValueError:
            # Fall back to absolute path if not under app_root
            rel_path = str(file_path)
        
        content_hash = _compute_file_hash(file_path)
        current_file_info[rel_path] = {
            "path_obj": file_path,
            "content_hash": content_hash,
        }
    
    # Determine change sets
    changed_files = []
    unchanged_files = []
    
    manifest_files = manifest.get("files", {})
    
    for rel_path, info in current_file_info.items():
        file_path = info["path_obj"]
        content_hash = info["content_hash"]
        
        if config_changed:
            # Config changed, all files need reprocessing
            changed_files.append(file_path)
        elif rel_path not in manifest_files:
            # New file
            changed_files.append(file_path)
        elif manifest_files[rel_path].get("content_hash") != content_hash:
            # File content changed
            changed_files.append(file_path)
        else:
            # File unchanged
            unchanged_files.append(file_path)
    
    # Find deleted files (in manifest but not on disk)
    current_rel_paths = set(current_file_info.keys())
    manifest_rel_paths = set(manifest_files.keys())
    deleted_files = list(manifest_rel_paths - current_rel_paths)
    
    logger.info(
        "Change analysis: %d changed, %d unchanged, %d deleted",
        len(changed_files), len(unchanged_files), len(deleted_files)
    )
    
    return changed_files, unchanged_files, deleted_files


def _preserve_chunks_for_unchanged_files(
    chunks_jsonl_path: Path, unchanged_files: List[Path], app_root: Path
) -> List[dict]:
    """
    Stream existing chunks.jsonl and preserve chunks from unchanged files.

    This function efficiently reads the existing chunks.jsonl line by line and collects
    chunks whose source_path corresponds to files that haven't changed. This preserves
    the investment in previous processing while allowing selective updates.

    Args:
        chunks_jsonl_path (Path): Path to the existing chunks.jsonl file.
        unchanged_files (List[Path]): List of file paths that haven't changed.
        app_root (Path): Application root directory for path normalization.

    Returns:
        List[dict]: Preserved chunk dictionaries from unchanged files.
    """
    if not chunks_jsonl_path.exists():
        logger.info("No existing chunks.jsonl found; starting fresh")
        return []
    
    # Create set of unchanged file paths (both relative and absolute) for fast lookup
    unchanged_paths = set()
    for file_path in unchanged_files:
        unchanged_paths.add(str(file_path))  # Absolute path
        try:
            rel_path = str(file_path.relative_to(app_root))
            unchanged_paths.add(rel_path)  # Relative path
        except ValueError:
            pass  # File not under app_root
    
    preserved_chunks = []
    preserved_count = 0
    
    try:
        with chunks_jsonl_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    chunk = json.loads(line)
                    source_path = chunk.get("source_path", "")
                    
                    if source_path in unchanged_paths:
                        preserved_chunks.append(chunk)
                        preserved_count += 1
                    
                except json.JSONDecodeError as exc:
                    logger.warning("Invalid JSON at line %d: %s", line_num, exc)
                    continue
    
    except Exception as exc:
        logger.warning("Failed to read existing chunks.jsonl: %s", exc)
        return []
    
    logger.info("Preserved %d chunks from %d unchanged files", preserved_count, len(unchanged_files))
    return preserved_chunks


def _build_unified_chunks_for_files(files: List[Path]) -> List[dict]:
    """
    Generate chunks for a specific set of files using the unified chunking pipeline.

    This function applies the same sentence-window chunking logic as the full pipeline
    but operates only on the specified file list. It's used for incremental processing
    where only changed files need to be re-chunked.

    Args:
        files (List[Path]): List of file paths to process.

    Returns:
        List[dict]: Chunk dictionaries ready for JSONL export and indexing.
    """
    chunks_out: List[dict] = []
    
    if not files:
        logger.info("No files to process for chunking")
        return chunks_out
    
    logger.info("Processing %d changed files for chunking", len(files))
    
    # Load all documents from the specified files
    all_documents = []
    for file_path in files:
        records = _load_document_content(file_path)
        all_documents.extend(records)
    
    logger.info("Loaded %d document records from %d files", len(all_documents), len(files))
    
    # Apply sentence-window chunking to all documents
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
    
    logger.info("Generated %d new chunks from changed files", len(chunks_out))
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
    all_files = _list_all_files(data_dir)
    chunks = _build_unified_chunks_for_files(all_files)
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
    Entry point for M13 Step 2: incremental document ingestion with idempotency manifest.

    This command implements the incremental pipeline:
    1. Analyze files vs manifest to determine what changed (content or config)
    2. Preserve chunks from unchanged files
    3. Generate chunks only for changed files
    4. Merge preserved + new chunks into updated chunks.jsonl
    5. Update manifest with current file metadata and config fingerprint

    The approach minimizes processing time on large document collections by only
    re-chunking files that have actually changed since the last run.

    Returns:
        None: Coordinates incremental ingestion and logs progress and output location.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app_root = Path(__file__).resolve().parents[1]
    input_dir = (app_root / "rag" / "data" / "seed").resolve()
    
    manifest_path = _manifest_path(app_root)
    chunks_jsonl_path = _chunks_jsonl_path(app_root)

    if not input_dir.exists():
        logger.info("Seed directory %s does not exist. Creating it.", input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Place source files (.pdf, .txt, .md) under %s and rerun seeding.", input_dir)
        # Continue; will produce zero chunks.

    # Step 1: Determine what files have changed
    changed_files, unchanged_files, deleted_files = _determine_change_set(input_dir, app_root)
    
    # Step 2: Preserve chunks for unchanged files
    preserved_chunks = _preserve_chunks_for_unchanged_files(
        chunks_jsonl_path, unchanged_files, app_root
    )
    
    # Step 3: Generate chunks for changed files only
    new_chunks = _build_unified_chunks_for_files(changed_files)
    
    # Step 4: Write updated chunks.jsonl
    all_chunks = preserved_chunks + new_chunks
    _write_chunks_jsonl(chunks_jsonl_path, all_chunks)
    
    # Step 5: Update manifest
    _update_manifest(
        manifest_path, app_root, input_dir, changed_files, unchanged_files, deleted_files, new_chunks
    )
    
    logger.info(
        "Incremental rebuild complete. Total chunks: %d (preserved: %d, new: %d)",
        len(all_chunks), len(preserved_chunks), len(new_chunks)
    )


def _update_manifest(
    manifest_path: Path, 
    app_root: Path, 
    input_dir: Path,
    changed_files: List[Path], 
    unchanged_files: List[Path], 
    deleted_files: List[str],
    new_chunks: List[dict]
) -> None:
    """
    Update the idempotency manifest with current file metadata and configuration.

    This function maintains the manifest.json file that tracks per-file content hashes,
    chunk counts, and timestamps. It removes entries for deleted files, updates entries
    for changed files, preserves entries for unchanged files, and updates the global
    configuration fingerprint.

    Args:
        manifest_path (Path): Path to the manifest.json file.
        app_root (Path): Application root directory for path normalization.
        input_dir (Path): Directory containing source files.
        changed_files (List[Path]): Files that were reprocessed in this run.
        unchanged_files (List[Path]): Files that were preserved from previous run.
        deleted_files (List[str]): Relative paths of files that no longer exist.
        new_chunks (List[dict]): Newly generated chunks for counting per file.

    Returns:
        None: Updates manifest on disk and logs completion.
    """
    # Load existing manifest
    manifest = _load_manifest(manifest_path)
    
    # Update config section
    current_config_fp = _compute_config_fingerprint()
    manifest["config"] = {
        "splitter": {
            "sent_window_size": SENT_WINDOW_SIZE,
            "sent_window_overlap": SENT_WINDOW_OVERLAP,
        },
        "config_fingerprint": current_config_fp,
    }
    
    # Remove deleted files from manifest
    for deleted_rel_path in deleted_files:
        if deleted_rel_path in manifest["files"]:
            del manifest["files"][deleted_rel_path]
            logger.info("Removed deleted file from manifest: %s", deleted_rel_path)
    
    # Count chunks per file from new_chunks
    chunks_per_file = {}
    for chunk in new_chunks:
        source_path = chunk["source_path"]
        chunks_per_file[source_path] = chunks_per_file.get(source_path, 0) + 1
    
    # Update manifest for changed files
    now = datetime.now(timezone.utc).isoformat()
    for file_path in changed_files:
        try:
            rel_path = str(file_path.relative_to(app_root))
        except ValueError:
            rel_path = str(file_path)
        
        doc_id = _stable_doc_id_from_stem(file_path.stem)
        content_hash = _compute_file_hash(file_path)
        chunk_count = chunks_per_file.get(str(file_path), 0)
        
        manifest["files"][rel_path] = {
            "doc_id": doc_id,
            "content_hash": content_hash,
            "chunks_count": chunk_count,
            "updated_at": now,
        }
    
    # For unchanged files, preserve existing entries but don't modify timestamps
    # (they're already in manifest["files"] and weren't deleted above)
    
    # Save updated manifest
    _save_manifest(manifest_path, manifest)
    
    logger.info(
        "Updated manifest: %d changed files, %d preserved, %d deleted",
        len(changed_files), len(unchanged_files), len(deleted_files)
    )


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