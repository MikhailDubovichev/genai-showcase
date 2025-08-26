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
