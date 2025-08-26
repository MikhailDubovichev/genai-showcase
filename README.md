# GenAI Showcase — Edge Server + Cloud RAG Monorepo

This monorepo hosts two complementary services (edge + cloud) and demo UIs:

- Edge server: a lightweight FastAPI app for on‑device classification and pipelines (smart device control, energy‑efficiency). Stable and feature‑flagged to call cloud first with graceful fallback.
- Cloud RAG service: FastAPI + LangChain + FAISS + SQLite + LangFuse. Answers energy‑efficiency questions with strict JSON, persists feedback, and runs an offline evaluator.
- Gradio demos: implemented minimal UIs for chat (edge) and a RAG explorer (cloud).

High‑level goals:
- Keep the edge device small, reliable, and container‑ready.
- Add a separate cloud pipeline to showcase RAG and evaluation while leaving the edge server stable.
- Use LangFuse for observability (traces/metrics) and an LLM‑as‑judge evaluator (relevance).

## Monorepo layout (current state)

```
apps/
  edge-server/           # Current edge FastAPI app (moved here from root)
    api/                 # Context, prompt, feedback endpoints
    core/                # Orchestrator and classifier
    pipelines/           # Device control and energy efficiency pipelines
    llm_cloud/           # OpenAI-compatible client and tool system
    provider_api/        # Integrator abstraction + mock provider
    services/            # History, feedback sync service, APScheduler wiring
    monitoring/          # Prometheus metrics helpers
    shared/              # Pydantic models and utilities (incl. cloud HTTP clients)
    pyproject.toml       # Python project for edge
    README.md            # Edge-specific docs

  cloud-rag/             # Cloud service: FastAPI + LangChain + FAISS + SQLite + LangFuse
    api/                 # /health, /api/rag/answer, /api/feedback/sync
    config/              # CONFIG/ENV loader, system prompts, config.json
    providers/           # Nebius embeddings & chat LLM, LangFuse helpers
    rag/                 # Chain (retriever → prompt → LLM → validator) and seed data
    schemas/             # Pydantic response schema (matches edge contract)
    scripts/             # Seeding CLI (build/persist FAISS)
    services/            # SQLite stores (feedback, eval queue), processors
    eval/                # LLM‑as‑judge evaluator and golden‑run CLI
    faiss_index/         # Persisted FAISS index (created by seeding)

  gradio/                # Chat and RAG explorer UIs (M7 done)

infra/
  compose/               # Docker Compose for local dev (edge + cloud + gradio); dynamic ports; healthchecks

PRD.md                   # Product Requirements Document (kept up-to-date)
.cursor/
  .cursorrules           # Development rules used by Cursor
  TASKS.md               # Single source of truth for milestones/tasks
.cursorignore            # Active ignore rules for Cursor context
```

Status:
- Edge server is stable; it exposes `/api/prompt`, `/api/context`, `/api/feedback/*`, and `/metrics`.
- Cloud RAG is implemented and integrated with the edge via a feature flag and a shared HTTP client.
- Gradio UIs are implemented and integrated with both services.
- Docker Compose infrastructure is ready for the full stack with persistent volumes.

Delivered milestones (M1–M9):
- M1: Monorepo created; existing repo moved into `apps/edge-server/`; `.cursorignore` added.
- M2: Cloud RAG MVP (FastAPI skeleton, config/env loader, strict JSON schema/prompt, RAG chain, FAISS seeding script, `/api/rag/answer`).
- M3: Minimal LangFuse (client init on startup; per‑request trace with `interactionId`; basic metadata).
- M4: Edge integration (feature flag `features.energy_efficiency_rag_enabled`; shared HTTP client; cloud‑first with local fallback).
- M5: Feedback sync (edge daily APScheduler job; cloud `/api/feedback/sync` dedupes to SQLite; attaches `user_feedback` score in LangFuse).
- M6: Minimal evaluator (LLM‑as‑judge relevance; enqueue artifacts post‑answer; offline processor computes scores and logs to LangFuse; golden‑run CLI).
- M7: Gradio UIs (Chat UI for edge server; RAG Explorer for cloud service with chunk visualization).
- M8: Compose infrastructure (docker-compose.yml with services, volumes, and dynamic config; seeding script integration; comprehensive documentation).
 - M9: Smoke tests (cloud RAG JSON, evaluator range, feedback upsert, edge flag-off local path, cloud-timeout fallback).

## Quick Start (Monorepo)

### Prerequisites
- **Docker Desktop** installed and running
- **`.env` file** in project root with required environment variables:
  - `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (for tracing and evaluation)
  - `NEBIUS_API_KEY` (for LLM and embeddings)

### Getting Started
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd genai-showcase
   ```

2. **Set up environment**
   ```bash
   # Create .env file with your API keys
   cp .env.example .env  # If example exists, or create manually
   # Edit .env with your actual keys
   ```

3. **Seed the FAISS index** (one-time setup)
   ```bash
   cd infra/compose
   python generate_env.py  # creates dynamic ports env (.env) from configs
   docker compose --profile seed run --rm seed-index
   ```

4. **Start the full stack**
   ```bash
   docker compose up --build
   ```

5. **Access the services** (ports may differ if overridden by .env)
   - Edge server: http://localhost:8080/docs
   - Cloud RAG: http://localhost:8000/health
   - Gradio Chat: http://localhost:7860
   - RAG Explorer: http://localhost:7861

## Services Overview

### Edge Server (Port 8080)
- **Purpose**: Lightweight FastAPI app for on-device classification and pipelines
- **Features**: Smart device control, energy efficiency advice, conversation history
- **Integration**: Calls cloud RAG service when feature flag is enabled
- **Fallback**: Gracefully falls back to local LLM if cloud is unavailable

### Cloud RAG Service (Port 8000)
- **Purpose**: Retrieval-Augmented Generation with vector similarity search
- **Features**: FAISS-based vector search, Nebius LLM integration, feedback evaluation
- **Data**: Persists feedback to SQLite, runs offline evaluator for relevance scoring

### Gradio UIs
- **Chat UI (Port 7860)**: Interface for the edge server with latency monitoring
- **RAG Explorer (Port 7861)**: Interface for the cloud RAG service showing retrieved chunks

## How to run (macOS, Poetry, absolute paths)

Edge server:

1) Install dependencies

```
poetry -C /Users/mikhaildubovichev/Projects/genai-showcase/apps/edge-server install
```

2) Run the server

```
poetry -C /Users/mikhaildubovichev/Projects/genai-showcase/apps/edge-server run python main.py
```

Cloud RAG service:

1) Install dependencies

```
poetry -C /Users/mikhaildubovichev/Projects/genai-showcase/apps/cloud-rag install
```

2) Provide environment (at minimum `NEBIUS_API_KEY`; optionally `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`)

```
export NEBIUS_API_KEY="<your_key>"
export LANGFUSE_PUBLIC_KEY="<optional_public>"
export LANGFUSE_SECRET_KEY="<optional_secret>"
```

3) Seed FAISS (place seed snippets under `apps/cloud-rag/rag/data/seed/` first)

```
poetry -C /Users/mikhaildubovichev/Projects/genai-showcase/apps/cloud-rag run \
  python -m scripts.seed_index
```

4) Run the server (port configurable via `CLOUD_RAG_PORT`)

```
CLOUD_RAG_PORT=8000 poetry -C /Users/mikhaildubovichev/Projects/genai-showcase/apps/cloud-rag run \
  python main.py
```

Health and example requests:

```
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/api/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"How to reduce standby power?","interactionId":"uuid-123","topK":3}'
```

Edge ↔ Cloud configuration:

- Toggle cloud‑first mode in `apps/edge-server/config/config.json`:
  - `features.energy_efficiency_rag_enabled: true|false`
  - `cloud_rag.base_url: "http://localhost:8000"`
  
- When using Compose, healthchecks use dynamic ports from `.env` (generated by `infra/compose/generate_env.py`).

Feedback sync (M5):

- Edge schedules a daily job (02:00) using APScheduler to POST new feedback to the cloud.
- Cloud persists feedback to SQLite at `apps/cloud-rag/data/db.sqlite` and attaches `user_feedback` scores to LangFuse traces.

Evaluator (M6):

- Cloud enqueues evaluation artifacts after `/api/rag/answer`; run the offline processor to score relevance:

```
poetry -C /Users/mikhaildubovichev/Projects/genai-showcase/apps/cloud-rag run \
  python -m services.eval_queue_processor
```

- Golden dataset run (prints aggregate stats):

```
poetry -C /Users/mikhaildubovichev/Projects/genai-showcase/apps/cloud-rag run \
  python -m eval.run_eval /Users/mikhaildubovichev/Projects/genai-showcase/apps/cloud-rag/eval/data/golden.jsonl
```

## Tests (M9, smoke level)

Run selected tests (ensure services are running where needed):

- Cloud RAG JSON:
```
poetry run pytest apps/cloud-rag/tests/test_rag_api.py -v
```

- Evaluator range:
```
poetry run pytest apps/cloud-rag/tests/test_relevance_evaluator.py -v
```

- Feedback upsert/duplicate counts:
```
poetry run pytest apps/cloud-rag/tests/test_feedback_sync.py -v
```

- Edge local path (flag off):
```
poetry run pytest apps/edge-server/tests/test_edge_flag_off.py -v
```

- Edge cloud timeout fallback (flag on + short timeout):
```
poetry run pytest apps/edge-server/tests/test_edge_cloud_fallback.py -v
```

## Feature Toggle

### Cloud-First RAG Mode
Control whether the edge server uses cloud RAG or local LLM for energy efficiency questions:

**Enable cloud-first mode:**
```json
// apps/edge-server/config/config.json
{
  "features": {
    "energy_efficiency_rag_enabled": true
  },
  "cloud_rag": {
    "base_url": "http://localhost:8000"
  }
}
```

**Disable for local-only mode:**
```json
{
  "features": {
    "energy_efficiency_rag_enabled": false
  }
}
```

**Behavior:**
- **Enabled**: Edge calls cloud RAG first, falls back to local LLM on error/timeout
- **Disabled**: Edge uses local LLM only (default behavior)

## Privacy Note

### Feedback Sync and Data Handling
- **Edge server** collects conversation feedback and sends it to cloud daily via APScheduler
- **Anonymization**: Only `interactionId` (UUID), question content, and feedback scores are sent
- **No personal data**: No user emails, IP addresses, or personal identifiers are transmitted
- **Storage**: Feedback stored in cloud SQLite database for evaluation and model improvement
- **LangFuse integration**: Feedback scores attached to traces for performance monitoring
- **Local control**: Edge server controls what feedback is sent; can be disabled by removing API keys

## Developer workflow (Cursor)

- Rules: follow `.cursor/.cursorrules` (never touch `llm_cloud/provider.py` without approval; minimal edits; preserve endpoints/pipelines).
- Tasks: use `.cursor/TASKS.md` as the execution plan; work milestone‑by‑milestone.
- Context: switch `.cursorignore` presets under `.cursor/presets/` to focus on edge vs cloud.
- Reference: keep `PRD.md` in context; it states contracts and acceptance criteria.

## References

- Edge server details: `apps/edge-server/README.md`
- Cloud RAG details: `apps/cloud-rag/README.md`
- Product requirements: `PRD.md`
- Tasks/milestones: `.cursor/TASKS.md`

## Notes

For edge‑server technical details (architecture, endpoints, directory breakdown, getting started, tests), see `apps/edge-server/README.md`.