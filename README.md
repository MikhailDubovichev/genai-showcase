# GenAI Showcase — Edge Server + Cloud RAG Monorepo

This monorepo contains two complementary services and small demo UIs:

- Edge server: a lightweight FastAPI app designed for Raspberry Pi that classifies user messages and routes them to pipelines for smart device control and energy‑efficiency advice.
- Cloud RAG service (planned/in progress): a FastAPI + LangChain service that answers energy‑efficiency questions grounded in a FAISS vector store, logs traces to LangFuse, and mirrors user feedback in SQLite.
- Gradio demos (planned/in progress): minimal UIs for chat (edge) and a RAG explorer (cloud).

High‑level goals:
- Keep the edge device small, reliable, and container‑ready.
- Add a separate cloud pipeline to showcase RAG and evaluation while leaving the edge server stable.
- Use LangFuse for minimal observability and a single evaluator (relevance) at first.

## Monorepo layout (current state and intentions)

```
apps/
  edge-server/           # Current edge FastAPI app (moved here from root)
    api/                 # Context, prompt, feedback endpoints
    core/                # Orchestrator and classifier
    pipelines/           # Device control and energy efficiency pipelines
    llm_cloud/           # OpenAI-compatible client and tool system
    provider_api/        # Integrator abstraction + mock provider
    services/            # History and feedback managers
    monitoring/          # Prometheus metrics helpers
    shared/              # Pydantic models and utilities
    pyproject.toml       # Python project for edge
    README.md            # Edge-specific docs

  cloud-rag/             # Planned: FastAPI + LangChain + FAISS + LangFuse + SQLite
    (to be implemented per .cursor/TASKS.md M2–M6)

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
- Edge server is production‑ready for demos and remains stable; it exposes `/api/prompt`, `/api/context`, `/api/feedback/*`, and `/metrics`.
- Cloud RAG, Gradio UIs, and compose files will be added incrementally according to PRD/TASKS.

Roadmap (summary):
- M1: Monorepo skeleton + move edge into `apps/edge-server/` (done/ongoing)
- M2–M3: Cloud RAG MVP + LangFuse traces (minimal fields)
- M4: Edge integration via feature flag with safe fallback
- M5: Daily feedback sync to cloud (SQLite mirror) + tag LangFuse traces
- M6: Minimal evaluator (relevance) in cloud
- M7: Gradio chat + RAG explorer
- M8–M9: Compose + smoke tests

## Developer workflow (Cursor)

- Rules: follow `.cursor/.cursorrules` (never touch `llm_cloud/provider.py` without approval; minimal edits; preserve endpoints/pipelines).
- Tasks: use `.cursor/TASKS.md` as the execution plan; work milestone‑by‑milestone.
- Context: switch `.cursorignore` presets under `.cursor/presets/` to focus on edge vs cloud.
- Reference: keep `PRD.md` in context; it states contracts and acceptance criteria.

## References

- Edge server details: `apps/edge-server/README.md`
- Cloud RAG details: `apps/cloud-rag/README.md` (planned)
- Product requirements: `PRD.md`
- Tasks/milestones: `.cursor/TASKS.md`

## Notes

For edge‑server technical details (architecture, endpoints, directory breakdown, getting started, tests), see `apps/edge-server/README.md`.