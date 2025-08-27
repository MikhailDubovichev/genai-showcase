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
- [X] Edge: feedback sync service (`services/feedback_sync.py`) with stdlib I/O; extract HTTP provider to `shared/feedback_client.py`; reads local feedback, filters by `last_synced_at` (`user_data/feedback_sync_state.json`), POSTs batch, updates state
- [X] Scheduler: add APScheduler to edge startup to run daily
- [X] Cloud: `POST /api/feedback/sync` dedupes by `feedback_id`, persists to SQLite (`data/db.sqlite`), returns `{accepted, duplicates}`
- [X] Cloud: attach LangFuse `user_feedback` score (+1/-1) to the trace by `interactionId`

## M6 — Minimal evaluator (single score)
- [X] `eval/relevance_evaluator.py`: LLM-as-judge relevance ∈ [0,1]
- [X] After `/api/rag/answer`, enqueue eval artifacts to `eval_queue` (no latency in request path)
- [X] Process `eval_queue`: compute relevance offline, log `score(name='relevance')` to LangFuse, mark processed
- [X] Golden-run: `eval/run_eval.py` reads `eval/data/golden.jsonl`, runs RAG, evaluates relevance, prints aggregate stats

## M7 — Gradio UIs (tiny demos)
- [X] Chat UI (`apps/gradio/chat/app.py`): textbox → Edge `/api/prompt`; show JSON + latency (config `EDGE_API_BASE_URL`)
- [X] RAG Explorer (`apps/gradio/rag_explorer/app.py`): textbox → Cloud `/api/rag/answer`; show retrieved chunks and final JSON (config `CLOUD_RAG_BASE_URL`)

## M8 — Compose and DX
- [X] `infra/compose/docker-compose.yml`: services for edge, cloud, gradio UIs; volumes for `faiss_index/` and `data/db.sqlite`
- [X] Script wiring for `cloud-rag/scripts/seed_index.py`
- [X] README: env vars (`LANGFUSE_*`), compose run, seeding FAISS, starting Gradio, feature flag toggle, feedback sync privacy note
- [X] Healthchecks: wait for cloud before RAG explorer

## M9 — Tests (smoke-level)
- [X] Cloud: `/api/rag/answer` returns valid JSON; simple success case
- [X] Cloud: relevance evaluator returns float in [0,1]
- [X] Cloud: `/api/feedback/sync` upserts and reports accepted/duplicates
- [X] Edge: flag off → local LLM path
- [X] Edge: flag on → cloud path; simulated timeout → fallback path

## M10 — Prompt improvements (classifier + energy efficiency)
- [ ] Classifier prompt: upgrade `apps/edge-server/config/classification_system_prompt.txt` with clearer instructions, more examples, ambiguity handling, and domain cues
- [ ] Energy efficiency prompt: refine `apps/cloud-rag/config/energy_efficiency_system_prompt.txt` (clarify JSON schema, safety disclaimers, deflection rules, concise style)
- [ ] Small labeled sets: add `apps/edge-server/data/classifier_samples.jsonl` and `apps/cloud-rag/eval/data/ee_prompt_examples.jsonl`
- [ ] Local eval scripts: quick accuracy script for classifier; JSON-adherence checker for energy prompt
- [ ] Docs: update READMEs with guidance and acceptance checks

## M11 — Multi-provider (Nebius/OpenAI) switch
- [ ] Config: add provider toggles in `apps/cloud-rag/config/config.json` (e.g., `llm.provider`, `embeddings.provider`), and read API keys from `.env` (`NEBIUS_API_KEY`, `OPENAI_API_KEY`)
- [ ] Providers: add `providers/openai_llm.py` and `providers/openai_embeddings.py` (LangChain integrations) mirroring Nebius factories
- [ ] Factory switch: update provider factories to select by `CONFIG` with graceful error if key missing
- [ ] README: document how to switch providers; examples for both
- [ ] Smoke tests: minimal calls per provider (skip if key missing) to ensure initialization works

## M12 — Hybrid retrieval with re-ranking (API unchanged)
- [ ] Add BM25 keyword retriever (langchain-community) alongside FAISS semantic retriever
- [ ] Config toggles: `retrieval.mode = semantic|hybrid`, `rerank.enabled = true|false`, `rerank.max_items`
- [ ] Reranking (optional): LLM-based re-ranker using existing `evaluate_relevance` on top-N combined candidates
- [ ] Chain update: choose path by config; keep response schema stable; surface retrieval metadata in LangFuse traces
- [ ] Tests/Docs: smoke tests for semantic vs hybrid; README notes on trade-offs and latency

## M13 — PDF ingestion for seeding
- [ ] Add PDF loader in `apps/cloud-rag/scripts/seed_index.py` (e.g., PyPDFLoader); configurable chunking by headings/blank lines
- [ ] Export chunks: write `faiss_index/chunks.jsonl` during seeding for BM25/hybrid reuse; update `manifest.json`
- [ ] CLI options: allow `--input-dir` and include `.pdf` and `.txt`; keep idempotency (skip unchanged)
- [ ] Tests: seed with a small sample PDF; ensure FAISS builds and chunks export exists
- [ ] Docs: update cloud README with PDF instructions and caveats

## M14 — Golden eval dataset expansion
- [ ] Expand `apps/cloud-rag/eval/data/golden.jsonl` to 20–50 diverse energy-efficiency questions with context hints
- [ ] Add rubric notes per item (what constitutes a good answer); optional difficulty tags
- [ ] Enhance `eval/run_eval.py` to print mean/median/stddev, invalid_json_rate, and simple histograms
- [ ] Docs: describe dataset scope, how to contribute new items, and how to interpret metrics
