"""
Seed a FAISS index for the Cloud RAG service using Nebius embeddings.

This script walks a directory of plain-text snippets (e.g., .txt and .md),
chunks each file into overlapping windows to preserve local context, embeds the
chunks with NebiusEmbeddings via the official LangChain integration, and then
builds a FAISS vector index that is saved to disk. The index can later be
loaded by the Cloud RAG retrieval chain to answer questions grounded in the
seeded content. The workflow mirrors common RAG practices while keeping the
implementation intentionally small and dependency-light so it is easy to run
locally during development.

CLI usage (macOS zsh example):
    export NEBIUS_API_KEY=...; \
    poetry run python -m scripts.seed_index \
        --data-dir rag/data/seed \
        --index-dir faiss_index

The script emits structured log messages covering the files discovered, the
number of chunks produced, the embedding model used, and the final index path.
It fails fast with actionable errors if the Nebius API key is missing or if the
FAISS integration is unavailable, guiding the developer to install any missing
packages or to provide required environment variables.

TODO: (future improvements):
1) Incremental seeding with change detection
   - Maintain a manifest of processed files with a strong content hash (e.g.,
     SHA‑256). On each run, compute hashes for files under `rag/data/seed/` and
     only (re)embed those that are new or changed. Skip unchanged files and
     merge new vectors into FAISS (or rebuild when the embedding model changes).
   - Store: { path, sha256, chunk_size, chunk_overlap, embedding_model, seeded_at }.

2) PDF ingestion with paragraph‑aware chunking
   - Add support for PDFs via LangChain loaders (e.g., PyPDFDirectoryLoader or
     UnstructuredPDFLoader). Use a paragraph‑based splitter that honors blank
     lines (chunk between two empty lines above/below). LangChain’s text
     splitters can be configured with separators such as ["\n\n", "\n", " ", ""].
   - Preserve metadata like page number and file name so the retriever can
     surface meaningful citations (sourceId could include page markers).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Iterable, List, Tuple

# Ensure app root is on sys.path so `providers` can be imported when running this
# file directly (e.g., via Cursor's "Run Python File" or python path/to/script.py)
_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from providers import get_embeddings
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _read_text_files(root: Path) -> List[Tuple[str, str]]:
    """
    Recursively read .txt and .md files from a directory tree.

    Args:
        root (Path): Directory to scan for text files.

    Returns:
        List[Tuple[str, str]]: List of (path_str, file_text) pairs.
    """
    results: List[Tuple[str, str]] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            try:
                text = path.read_text(encoding="utf-8")
                results.append((str(path), text))
            except Exception as exc:  # pragma: no cover - rare I/O failures
                logger.warning("Failed to read %s: %s", path, exc)
    return results


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Chunk input text into overlapping windows, attempting to split on whitespace.

    The algorithm walks the text in steps of (chunk_size - chunk_overlap), and for
    each window it expands forward to the next whitespace boundary where feasible.
    This keeps words intact while still enforcing approximate chunk sizes. The
    function returns a list of substrings ready to be embedded.

    Args:
        text (str): The raw file contents to be chunked.
        chunk_size (int): Target size in characters for each chunk.
        chunk_overlap (int): Overlap in characters between consecutive chunks.

    Returns:
        List[str]: Overlapping chunks of text, preserving word boundaries where possible.
    """
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - max(0, chunk_overlap))
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + chunk_size)
        # Expand end to the right until whitespace to avoid cutting words, within a small limit
        if end < n:
            j = end
            limit = min(n, end + 50)
            while j < limit and not text[j].isspace():
                j += 1
            if j > end:
                end = j
        chunks.append(text[i:end].strip())
        i += step
    return [c for c in chunks if c]


def _build_documents(
    pairs: List[Tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
    source_prefix: str,
):
    """
    Convert (path, text) pairs into LangChain Documents with metadata for source and scores.

    Each chunk receives a deterministic source identifier that encodes the original file stem and
    chunk index. This metadata is used later by the chain to populate citation entries. The
    function yields documents to avoid holding too much in memory when seeding large corpora.

    Yields:
        Document: A LangChain Document with `page_content` set to a chunk and metadata including
        `sourceId` and default `score` (0.0 during seeding; retriever will supply scores at query time).
    """
    # Newer LangChain exposes Document under langchain_core; keep a fallback for older layouts.
    try:
        from langchain_core.documents import Document  # type: ignore
    except Exception:
        try:
            from langchain.schema import Document  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency not installed
            raise RuntimeError(
                "Missing LangChain dependency. Install it via: "
                "poetry add langchain langchain-community"
            ) from exc

    for path_str, text in pairs:
        stem = Path(path_str).stem
        chunks = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk in enumerate(chunks):
            source_id = f"{source_prefix}{stem}#{idx}"
            yield Document(page_content=chunk, metadata={"sourceId": source_id, "score": 0.0})


def seed_index(
    data_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    source_prefix: str,
) -> None:
    """
    Seed the FAISS index with documents embedded by NebiusEmbeddings.

    This function orchestrates the seeding flow end-to-end: it enumerates source files in the data
    directory, chunks them using the overlapping window strategy, builds LangChain Document objects
    with source metadata, obtains Nebius embeddings via the providers module, and finally constructs
    a FAISS index which is saved to the specified directory. It logs key progress points to assist
    with troubleshooting and to provide basic observability during development runs.

    Args:
        data_dir (Path): Directory tree containing .txt/.md snippets to index.
        index_dir (Path): Target directory to write the FAISS index files.
        chunk_size (int): Target character length of each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
        source_prefix (str): Prefix used to build stable `sourceId` values.
    """
    files = _read_text_files(data_dir)
    logger.info("Discovered %d source files under %s", len(files), data_dir)

    docs = list(_build_documents(files, chunk_size, chunk_overlap, source_prefix))
    logger.info("Prepared %d chunks for embedding", len(docs))

    # Use provider factory to resolve embeddings by CONFIG
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
    Parse CLI arguments and seed the FAISS index accordingly.

    This entry point wires together the argument parser with the seeding function. It supports
    optional parameters for the data directory, index directory, chunk sizing, and the prefix used
    to construct stable source identifiers. Errors surfaced from missing dependencies or missing
    credentials are allowed to terminate the process with clear, actionable messages.
    """
    parser = argparse.ArgumentParser(description="Seed FAISS index for Cloud RAG")
    parser.add_argument(
        "--data-dir",
        default="rag/data/seed",
        help="Directory containing .txt/.md snippets (default: rag/data/seed)",
    )
    parser.add_argument(
        "--index-dir",
        default="faiss_index",
        help="Directory to write FAISS index (default: faiss_index)",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size (characters)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Overlap (characters)")
    parser.add_argument(
        "--source-prefix",
        default="doc_",
        help="Prefix for sourceId metadata",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Resolve paths relative to the cloud app root for robust execution via `-m`.
    app_root = Path(__file__).resolve().parents[1]
    data_dir = (app_root / args.data_dir).resolve()
    index_dir = (app_root / args.index_dir).resolve()

    if not data_dir.exists():
        logger.info("Seed directory %s does not exist. Creating it.", data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Place one or more .txt/.md files under %s and rerun seeding.", data_dir
        )
        # Continue execution; _read_text_files will simply find zero files and report.

    seed_index(
        data_dir=data_dir,
        index_dir=index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        source_prefix=args.source_prefix,
    )


if __name__ == "__main__":
    main()


