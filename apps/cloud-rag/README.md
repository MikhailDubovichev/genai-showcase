# Cloud RAG Service

This service is the cloud-side Retrieval-Augmented Generation (RAG) API that complements the edge server. It retrieves context from a FAISS vector index, renders a strict JSON prompt, calls a Nebius-backed LLM, and validates the response against a shared schema so the edge can consume it safely.

## Environment Variables

The cloud RAG service requires the following environment variables:

### Required
- **`NEBIUS_API_KEY`**: Your Nebius API key for LLM and embeddings access

### Optional (for observability)
- **`LANGFUSE_PUBLIC_KEY`**: Public key for LangFuse tracing
- **`LANGFUSE_SECRET_KEY`**: Secret key for LangFuse tracing and evaluation

### Setting Environment Variables
```bash
# Option 1: From repo root .env file (recommended)
set -a; source .env; set +a

# Option 2: Export explicitly
export NEBIUS_API_KEY="<your_nebius_key>"
export LANGFUSE_PUBLIC_KEY="<your_langfuse_public_key>"
export LANGFUSE_SECRET_KEY="<your_langfuse_secret_key>"
```

## Quick start (macOS, Poetry)

1) Install dependencies (package mode)
```
cd apps/cloud-rag
poetry install
```

2) Provide credentials via environment
- Put your key in the repo root `.env` or export it in the current shell:
```
# from repo root (preferred)
set -a; source .env; set +a
# or explicitly
export NEBIUS_API_KEY="<your_key>"
```

3) Seed the FAISS index with a tiny snippet
```
cd apps/cloud-rag
cat > rag/data/seed/standby_power.txt <<'TXT'
Standby power (vampire power) can be reduced by unplugging idle chargers and using advanced power strips.
TXT
poetry run python -m scripts.seed_index
```

4) Run the server
```
# Port is dynamic; set CLOUD_RAG_PORT or use Compose (recommended)
CLOUD_RAG_PORT=8000 poetry run python main.py
```

## Endpoints
- GET /health → simple healthcheck `{status:"ok", timestamp:"..."}`
- POST /api/rag/answer → returns a strict JSON object validating to `EnergyEfficiencyResponse`

Example query:
```
curl -s -X POST http://localhost:8000/api/rag/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question":"How to reduce standby power at home?",
    "interactionId":"uuid-123",
    "topK":3
  }' | jq .
```

## Configuration
- `apps/cloud-rag/config/config.json` controls model names and paths.
- `NEBIUS_API_KEY` must be present in the environment.
- FAISS index path defaults to `faiss_index` (relative to this folder).
- A `faiss_index/manifest.json` is written during seeding with embedding model and dimension. If you change the embedding model, delete/reseed the index.
 - Fallback behavior:
   - `retrieval.allow_general_knowledge: true|false` — when true and no chunks are retrieved, the service returns a brief, general best‑practice answer with `content: []`.

### Provider switching (Nebius ↔ OpenAI)

Cloud supports multiple providers via full-file templates and a simple switcher.

1) Switch provider (copies template → active config):
```
python scripts/switch_provider.py nebius
python scripts/switch_provider.py openai
```

2) Set environment variables (already supported):
- Nebius: `NEBIUS_API_KEY`
- OpenAI: `OPENAI_API_KEY`

3) Restart the server to apply changes:
```
CLOUD_RAG_PORT=8000 poetry run python main.py
```

4) Verify selection at startup (and LangFuse traces, if enabled).

Notes:
- Templates live under `config/templates/` as `config.nebius.json` and `config.openai.json`.
- If you switch embeddings provider, re-seed FAISS (`rm -rf faiss_index && poetry run python -m scripts.seed_index`).

## Retrieval and Rerank (M12)

The retrieval pipeline now supports two modes and an optional rerank stage:

- Semantic only: FAISS similarity search
- Hybrid: FAISS (semantic) + BM25 (keyword) with weighted fusion (alpha)
- Optional rerank: LLM-as-judge, scores candidates in [0,1] and reorders

Minimal config examples (apps/cloud-rag/config/config.json):

Semantic only (no rerank):

```json
{
  "retrieval": {
    "mode": "semantic",
    "semantic_k": 6,
    "default_top_k": 3,
    "allow_general_knowledge": true
  },
  "rerank": { "enabled": false }
}
```

Hybrid + rerank:

```json
{
  "retrieval": {
    "mode": "hybrid",
    "semantic_k": 6,
    "keyword_k": 6,
    "default_top_k": 3,
    "fusion": { "alpha": 0.6 },
    "allow_general_knowledge": true
  },
  "rerank": {
    "enabled": true,
    "top_n": 10,
    "model": "gpt-4o-mini",
    "timeout_ms": 3500,
    "batch_size": 8,
    "preview_chars": 600
  }
}
```

Notes:
- BM25 requires the package `rank-bm25`. Install in cloud-rag: `poetry add rank-bm25` (inside `apps/cloud-rag`).
- The chain freezes config at startup. After editing config.json, restart the server.
- Expected INFO logs (hybrid):
  - `Retrieval mode=hybrid | semantic_k=... keyword_k=... final_top_k=...`
  - `Top fused (doc_id, score): [...]`
  - If rerank is enabled: `Rerank enabled: top_n=...` and `Top reranked (doc_id, score): [...]`

## Document Ingestion (M13)

The cloud RAG service supports unified ingestion of multiple document types with incremental rebuilds and robust chunking. This section covers the complete M13 pipeline for processing PDFs, text files, and markdown.

### Supported File Types

Drop source files in `apps/cloud-rag/rag/data/seed/` (non-recursive scan):
- **PDFs (`.pdf`)**: Preferred loader is `PyMuPDFLoader`, falls back to `PyPDFLoader` if unavailable
- **Text files (`.txt`)**: Plain UTF-8 text files
- **Markdown files (`.md`)**: Markdown content (processed as plain text)

### Chunking Policy

All file types use the same sentence-window chunking strategy for consistency:

**Sentence-window chunking**:
- Window size: 10 sentences per chunk
- Overlap: 2 sentences between consecutive chunks
- Sentence detection: Lightweight regex-based splitter (handles `.`, `!`, `?`)
- Whitespace normalization: Collapses multiple spaces, trims ends
- Stable chunk IDs: Format `doc_id#chunk_index` (e.g., `energy_guide#0`)

**Per-type processing**:
- **PDFs**: Process page-by-page, extract text content, preserve page numbers in metadata
- **Text/Markdown**: Read entire file as single content block, process as one document

### chunks.jsonl: The Portable Truth

The ingestion pipeline produces a canonical `faiss_index/chunks.jsonl` file that serves as the single source of truth for all downstream processing:

**Schema** (one JSON object per line):
```json
{
  "id": "energy_guide#0",
  "doc_id": "energy_guide",
  "source_path": "/path/to/energy_guide.pdf",
  "source_type": "pdf",
  "page": 1,
  "heading_path": [],
  "text": "Normalized chunk text...",
  "created_at": "2024-01-15T10:30:00Z",
  "hash": "sha256_hex_digest"
}
```

**Benefits of chunks.jsonl as portable truth**:
- **Multi-source ready**: All file types (PDF, txt, md) produce the same schema
- **Provider-agnostic**: No embeddings or model dependencies in the raw chunks
- **Streaming friendly**: JSONL format enables efficient line-by-line processing
- **Metadata rich**: Preserves source attribution, page numbers, timestamps, and content hashes
- **Version control**: Can be committed for reproducibility
- **Debuggable**: Easy to inspect chunks without parsing source files

### Idempotency Rules

The ingestion system tracks file changes via content hashes and splitter configuration:

**Change detection**:
- File hash: SHA-256 of file bytes
- Config fingerprint: SHA-256 of sentence window parameters
- Manifest: `faiss_index/manifest.json` tracks per-file metadata

**Incremental rebuild logic**:
1. **New files**: Detected by absence in manifest → process fully
2. **Changed files**: Content hash differs → re-chunk and merge
3. **Unchanged files**: Preserve existing chunks from `chunks.jsonl`
4. **Deleted files**: Remove from manifest and `chunks.jsonl`
5. **Config changes**: Force full rebuild (e.g., window size changed)

**Manifest structure**:
```json
{
  "schema_version": 1,
  "config": {
    "splitter": { "sent_window_size": 10, "sent_window_overlap": 2 },
    "config_fingerprint": "sha256_hash"
  },
  "files": {
    "rag/data/seed/energy_guide.pdf": {
      "doc_id": "energy_guide",
      "content_hash": "sha256_hash",
      "chunks_count": 12,
      "updated_at": "2024-01-15T10:30:00Z"
    }
  },
  "faiss": {
    "vectors_count": 120,
    "embedding_dim": 1536,
    "model_from_config": "text-embedding-3-small",
    "built_at": "2024-01-15T10:30:05Z",
    "config_fingerprint": "sha256_hash"
  }
}
```

### How to Force Rebuild

To force a complete rebuild (delete all cached chunks and rebuild FAISS):

```bash
# From apps/cloud-rag directory
rm -rf faiss_index/
poetry run python -m scripts.seed_index
```

This removes:
- `faiss_index/chunks.jsonl` (canonical chunks)
- `faiss_index/manifest.json` (change tracking)
- `faiss_index/index.faiss` (FAISS vectors)
- All other FAISS metadata files

### Usage Examples

**Basic ingestion**:
```bash
# Add files to rag/data/seed/ then run
poetry run python -m scripts.seed_index
```

**Monitor incremental rebuilds**:
```bash
# Logs show change analysis and processing counts
# Look for: "Change analysis: X changed, Y unchanged, Z deleted"
# And: "Incremental rebuild complete. Total chunks: N (preserved: P, new: Q)"
```

**Check ingestion results**:
```bash
# Count chunks
wc -l faiss_index/chunks.jsonl

# Inspect first few chunks
head -3 faiss_index/chunks.jsonl | jq .

# View manifest summary
cat faiss_index/manifest.json | jq '.files | keys'
```

## Project structure (cloud)
```
api/            # FastAPI routers (health, rag, feedback)
config/         # CONFIG/ENV loader and system prompt
providers/      # Nebius embeddings & LLM providers (LangChain integrations)
rag/            # Chain (retriever → prompt → LLM → validator) and seed data
schemas/        # Pydantic response schema matching the edge contract
scripts/        # CLI utilities (e.g., seed_index.py)
services/       # SQLite stores (feedback, eval queue) and processors
eval/           # Evaluator and golden-run CLI
```

## Running with Compose

For a complete development environment, use Docker Compose from the project root:

```bash
# From project root
cd infra/compose

# Generate .env file from app configs (dynamic ports)
python generate_env.py

# Start the full stack (edge + cloud + gradio UIs)
docker compose up --build
```

The cloud RAG service will be available (default `http://localhost:8000`, port may differ if overridden) with all dependencies properly configured. Compose waits for service health before dependent UIs start.

## Seeding FAISS

### Via Docker Compose (Recommended)
```bash
# One-time setup for vector index
docker compose --profile seed run --rm seed-index
```

### Via Poetry (Alternative)
```bash
# Traditional method
cd apps/cloud-rag
poetry run python scripts/seed_index.py
```

### What seeding does:
- Builds FAISS vector index from text snippets in `rag/data/seed/`
- Creates embeddings using Nebius API
- Saves index files to `faiss_index/` directory
- Generates `manifest.json` with metadata
- **Idempotent**: Safe to run multiple times

### Seed Data
Add text files to `apps/cloud-rag/rag/data/seed/`:
```
standby_power.txt
energy_monitoring.txt
smart_devices.txt
```

## Gradio Integration

The cloud RAG service integrates with the Gradio RAG Explorer UI:

### RAG Explorer Features
- **Question input**: Send natural language questions to `/api/rag/answer`
- **Retrieved chunks**: View the top-K most relevant document snippets
- **Similarity scores**: See relevance scores for each retrieved chunk
- **Raw JSON**: Inspect the complete API response
- **Latency monitoring**: Track response times

### API Usage
```bash
# The RAG Explorer calls this endpoint
curl -X POST http://localhost:8000/api/rag/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How to reduce standby power?",
    "interactionId": "uuid-123",
    "topK": 3
  }'
```

### Response Format
```json
{
  "message": "Reduce standby power by unplugging idle devices...",
  "interactionId": "uuid-123",
  "type": "text",
  "content": [
    {
      "sourceId": "standby_power.txt#0",
      "chunk": "Standby power can be reduced by...",
      "score": 0.87
    }
  ]
}
```

## Tests (M9, smoke level)
- RAG JSON: `poetry run pytest apps/cloud-rag/tests/test_rag_api.py -v`
- Evaluator range: `poetry run pytest apps/cloud-rag/tests/test_relevance_evaluator.py -v`
- Feedback upsert/duplicates: `poetry run pytest apps/cloud-rag/tests/test_feedback_sync.py -v`

## Notes
- This service mirrors edge conventions (verbose docstrings, clear contracts).
- Responses must validate to the same `EnergyEfficiencyResponse` as the edge expects.
