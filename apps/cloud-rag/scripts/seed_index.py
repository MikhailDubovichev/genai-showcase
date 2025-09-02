"""
Seed utilities for the Cloud RAG service: text/PDF ingestion, chunking, export, and FAISS build.

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
from typing import Iterable, List, Tuple

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
# M13 Step 1: PDF ingestion → sentence-window chunking → chunks.jsonl export
# -----------------------------------------------------------------------------
SENT_WINDOW_SIZE = 10  # 8–12 sentences per chunk
SENT_WINDOW_OVERLAP = 2  # 2-sentence overlap


def _list_pdf_files(root: Path) -> List[Path]:
    """
    Enumerate PDF files in the configured seed directory using a simple, predictable policy.

    This function inspects the immediate children of the provided directory and returns all
    paths whose extension is ".pdf" (case‑insensitive). The scan is intentionally non‑recursive
    to keep behavior deterministic and easy to reason about in local development and CI runs.
    A non‑recursive design also reduces surprises from deeply nested folders or accidentally
    large corpora. Later milestones may introduce recursive discovery or CLI flags; for the
    current step we keep the surface minimal and opinionated to ensure repeatability.

    Args:
        root (Path): The directory under which PDF files are searched.

    Returns:
        List[Path]: A list of absolute or relative file system paths pointing to PDF files.
    """
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]


def _load_pdf_documents(pdf_path: Path) -> List[object]:
    """
    Load a PDF into LangChain Documents, preferring layout‑aware PyMuPDF, with a robust fallback.

    The loader selection prioritizes `PyMuPDFLoader` for richer layout handling and improved text
    extraction on many documents. If that import or load fails for any reason (missing optional
    dependency, unsupported file, or platform differences), the function falls back to
    `PyPDFLoader`. Both loaders return page‑level `Document` objects with `page_content` and
    `metadata`. We pass through whatever metadata is available (e.g., page numbers), which is
    later used to populate the `page` field in exported JSONL chunks. Failures are logged and an
    empty list is returned to keep the ingestion run resilient and observable without crashing.

    Args:
        pdf_path (Path): The path of the PDF file to load and parse into `Document` objects.

    Returns:
        List[Document]: A list of LangChain `Document` objects, typically one per page, or an empty
        list if loading failed. Each document includes `page_content` and a `metadata` mapping.
    """
    try:
        from langchain_community.document_loaders import PyMuPDFLoader  # type: ignore
        loader = PyMuPDFLoader(str(pdf_path))
        return loader.load()
    except Exception:
        try:
            from langchain_community.document_loaders import PyPDFLoader  # type: ignore
            loader = PyPDFLoader(str(pdf_path))
            return loader.load()
        except Exception as exc:
            logger.warning("Failed to load PDF %s: %s", pdf_path, exc)
            return []


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
        text (str): Raw text extracted from a PDF page or an earlier preprocessing step.

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


def _build_pdf_chunks(input_dir: Path) -> List[dict]:
    """
    Ingest PDFs from the seed directory and produce JSONL‑ready chunk dictionaries.

    This routine implements the M13 Step 1 policy: it enumerates `.pdf` files, loads them into
    page‑level LangChain `Document` objects, and transforms each page into overlapping
    sentence‑window chunks (10 sentences with 2 sentences overlap by default). For every chunk it
    builds a portable record with a stable `id`, `doc_id`, `source_path`, optional `page`, optional
    `heading_path`, the normalized `text`, a `created_at` timestamp in ISO 8601 format, and a SHA‑256
    `hash` of the normalized text. The output list is later written to `faiss_index/chunks.jsonl` and
    serves as the canonical source for downstream indexing and retrieval steps.

    Args:
        input_dir (Path): The directory containing source PDF files to ingest.

    Returns:
        List[dict]: A list of chunk dictionaries ready for JSONL export.
    """
    chunks_out: List[dict] = []
    pdfs = _list_pdf_files(input_dir)
    total_docs = 0
    for pdf in pdfs:
        docs = _load_pdf_documents(pdf)
        if not docs:
            continue
        total_docs += 1
        doc_id = _stable_doc_id_from_stem(pdf.stem)
        chunk_index = 0
        for d in docs:
            page_content = getattr(d, "page_content", "")
            metadata = getattr(d, "metadata", {}) or {}
            # page could be under different keys depending on loader
            page = metadata.get("page") or metadata.get("page_number") or None
            heading_path = metadata.get("heading_path") or []
            if not isinstance(heading_path, list):
                heading_path = []

            sentences = _sentence_tokenize(page_content)
            windows = _chunk_sentences(sentences, SENT_WINDOW_SIZE, SENT_WINDOW_OVERLAP)
            for w in windows:
                text_norm = _normalize_text(w)
                if not text_norm:
                    continue
                chunk_id = f"{doc_id}#{chunk_index}"
                h = hashlib.sha256(text_norm.encode("utf-8")).hexdigest()
                chunks_out.append(
                    {
                        "id": chunk_id,
                        "doc_id": doc_id,
                        "source_path": str(pdf),
                        "page": int(page) if isinstance(page, int) else page,
                        "heading_path": heading_path,
                        "text": text_norm,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "hash": h,
                    }
                )
                chunk_index += 1
    logger.info("Processed %d PDFs; produced %d chunks", total_docs, len(chunks_out))
    return chunks_out


def _write_chunks_jsonl(out_path: Path, chunks: List[dict]) -> None:
    """
    Write chunk records to a JSON Lines file that downstream jobs can stream efficiently.

    The JSONL format emits exactly one JSON object per line and is easy to process from Python,
    shell tools, or data pipelines. This function rewrites the output file on each run to provide
    a clean snapshot of the current ingestion. It ensures the parent directory exists, serializes
    each chunk with `ensure_ascii=False` to preserve characters, and logs a concise summary with
    the total number of chunks and file path. Later steps will use this file to build FAISS and
    initialize BM25 without needing to re‑parse PDFs.

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


# -----------------------------------------------------------------------------
# Legacy text seeding helpers (kept for reference; not used in M13 Step 1 run)
# -----------------------------------------------------------------------------

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
    Entry point for M13 Step 1: ingest PDFs and export a canonical chunks.jsonl snapshot.

    This command scans the fixed input directory `apps/cloud-rag/rag/data/seed` for `.pdf` files,
    loads each with a robust loader strategy (try PyMuPDF, then fall back to PyPDF), and converts
    pages into overlapping sentence‑window chunks. It then writes the consolidated list of chunks to
    `apps/cloud-rag/faiss_index/chunks.jsonl`, overwriting any previous file to provide a single,
    authoritative snapshot. The output includes stable identifiers, basic provenance metadata, and a
    SHA‑256 hash of each normalized chunk. Logging reports how many PDFs were processed, how many
    chunks were written, and the destination path so developers can quickly verify ingestion.

    Returns:
        None: This function coordinates the ingestion and export flow and logs progress.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app_root = Path(__file__).resolve().parents[1]
    input_dir = (app_root / "rag" / "data" / "seed").resolve()
    output_jsonl = (app_root / "faiss_index" / "chunks.jsonl").resolve()

    if not input_dir.exists():
        logger.info("Seed directory %s does not exist. Creating it.", input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Place one or more .pdf files under %s and rerun seeding.", input_dir)
        # Continue; will produce zero chunks.

    chunks = _build_pdf_chunks(input_dir)
    _write_chunks_jsonl(output_jsonl, chunks)
    logger.info("PDF ingestion complete.")


if __name__ == "__main__":
    main()


