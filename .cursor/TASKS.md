# TASKS.md — Milestone plan (M1–M9)

Develop strictly in accordance with these tasks (see `.cursor/.cursorrules`). Keep edits minimal and focused.

## M1 — Monorepo and basics
- [X] Create structure: `apps/edge-server/`, `apps/cloud-rag/`, `apps/gradio/`, `infra/compose/`
- [X] Move current repo into `apps/edge-server/` (no logic changes)
- [X] Add `.cursorignore` to limit context per app when working in Cursor
- [X] Update root `apps/edge-server/README.md` with monorepo map and how to run edge locally

## M2 — Cloud RAG MVP
- [X] Cloud FastAPI skeleton: `main.py`, `/health`, route mounting
- [X] Config/env loader for `LANGFUSE_*` and service port
- [X] Response schema file mirroring `EnergyEfficiencyResponse`
- [X] Prompts: `prompts/answer_system.txt` (strict JSON)
- [X] Chain: `rag/chain.py` (embedder + FAISS retriever + prompt + LLM + validator)
- [X] Seed: `rag/data/seed/` snippets + `scripts/seed_index.py` → persist to `faiss_index/`
- [X] Endpoint: `POST /api/rag/answer` returns validated JSON

## M3 — LangFuse (cloud-only minimal)
- [X] Initialize LangFuse client from env at startup
- [X] Create trace per `/api/rag/answer` using `interactionId`
- [X] Log fields: `latency_ms`, `model`, `tokens_prompt`, `tokens_completion`, `json_valid`, `retrieved_k`, optional `error_type`

## M4 — Edge integration (cloud-first with fallback)
- [X] Add to edge `config.json`: `features.energy_efficiency_rag_enabled`, `cloud_rag.base_url`
- [X] Add `shared/rag_client.py` (HTTP POST to cloud with short timeout)
- [X] Edit `pipelines/energy_efficiency/pipeline_energy_efficiency.py`: try cloud when flag on; fallback to local LLM on error/timeout
- [X] Ensure history/logging unchanged; responses still match `EnergyEfficiencyResponse`

## M5 — Feedback sync (daily batch)
- [ ] Edge: `services/feedback_sync.py` reads local feedback, filters by `last_synced_at` (`user_data/feedback_sync_state.json`), POSTs batch, updates state
- [ ] Scheduler: add APScheduler to edge startup to run daily
- [ ] Cloud: `POST /api/feedback/sync` dedupes by `feedback_id`, persists to SQLite (`data/db.sqlite`), returns `{accepted, duplicates}`
- [ ] Cloud: attach LangFuse `user_feedback` score (+1/-1) to the trace by `interactionId`

## M6 — Minimal evaluator (single score)
- [ ] `eval/relevance_evaluator.py`: LLM-as-judge relevance ∈ [0,1]
- [ ] After `/api/rag/answer`, compute and log `score(name='relevance')` in LangFuse
- [ ] (Optional) `eval/data/golden.jsonl` (~20 Qs) + `eval/run_eval.py` to compute avg relevance & JSON-valid rate

## M7 — Gradio UIs (tiny demos)
- [ ] Chat UI (`apps/gradio/chat/app.py`): textbox → Edge `/api/prompt`; show JSON + latency (config `EDGE_API_BASE_URL`)
- [ ] RAG Explorer (`apps/gradio/rag_explorer/app.py`): textbox → Cloud `/api/rag/answer`; show retrieved chunks and final JSON (config `CLOUD_RAG_BASE_URL`)

## M8 — Compose and DX
- [ ] `infra/compose/docker-compose.yml`: services for edge, cloud, gradio UIs; volumes for `faiss_index/` and `data/db.sqlite`
- [ ] Script wiring for `cloud-rag/scripts/seed_index.py`
- [ ] README: env vars (`LANGFUSE_*`), compose run, seeding FAISS, starting Gradio, feature flag toggle, feedback sync privacy note

## M9 — Tests (smoke-level)
- [ ] Cloud: `/api/rag/answer` returns valid JSON; simple success case
- [ ] Cloud: relevance evaluator returns float in [0,1]
- [ ] Cloud: `/api/feedback/sync` upserts and reports accepted/duplicates
- [ ] Edge: flag off → local LLM path
- [ ] Edge: flag on → cloud path; simulated timeout → fallback path
