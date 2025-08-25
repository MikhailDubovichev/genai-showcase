# GenAI Showcase — Edge Server + Cloud RAG Monorepo

This monorepo hosts two complementary services (edge + cloud) and planned demo UIs:

- Edge server: a lightweight FastAPI app for on‑device classification and pipelines (smart device control, energy‑efficiency). Stable and feature‑flagged to call cloud first with graceful fallback.
- Cloud RAG service: FastAPI + LangChain + FAISS + SQLite + LangFuse. Answers energy‑efficiency questions with strict JSON, persists feedback, and runs an offline evaluator.
- Gradio demos: planned minimal UIs for chat (edge) and a RAG explorer (cloud).

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

  gradio/                # Planned: minimal UIs for chat (edge) and RAG explorer (cloud)
    (to be implemented per .cursor/TASKS.md M7)

infra/
  compose/               # Planned: docker-compose for local dev (edge + cloud + gradio)

PRD.md                   # Product Requirements Document (kept up-to-date)
.cursor/
  .cursorrules           # Development rules used by Cursor
  TASKS.md               # Single source of truth for milestones/tasks
.cursorignore            # Active ignore rules for Cursor context
```

Status:
- Edge server is stable; it exposes `/api/prompt`, `/api/context`, `/api/feedback/*`, and `/metrics`.
- Cloud RAG is implemented and integrated with the edge via a feature flag and a shared HTTP client.
- Gradio UIs and compose files will be added in later milestones.

Delivered milestones (M1–M6):
- M1: Monorepo created; existing repo moved into `apps/edge-server/`; `.cursorignore` added.
- M2: Cloud RAG MVP (FastAPI skeleton, config/env loader, strict JSON schema/prompt, RAG chain, FAISS seeding script, `/api/rag/answer`).
- M3: Minimal LangFuse (client init on startup; per‑request trace with `interactionId`; basic metadata).
- M4: Edge integration (feature flag `features.energy_efficiency_rag_enabled`; shared HTTP client; cloud‑first with local fallback).
- M5: Feedback sync (edge daily APScheduler job; cloud `/api/feedback/sync` dedupes to SQLite; attaches `user_feedback` score in LangFuse).
- M6: Minimal evaluator (LLM‑as‑judge relevance; enqueue artifacts post‑answer; offline processor computes scores and logs to LangFuse; golden‑run CLI).

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