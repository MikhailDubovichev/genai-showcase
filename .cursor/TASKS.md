# TASKS.md — Milestone plan (M1–M9)

Develop strictly in accordance with these tasks (see `.cursor/.cursorrules`). Keep edits minimal and focused.

## M1 — Monorepo and basics
- [X] Step 1: Create structure: `apps/edge-server/`, `apps/cloud-rag/`, `apps/gradio/`, `infra/compose/`
- [X] Step 2: Move current repo into `apps/edge-server/` (no logic changes)
- [X] Step 3: Add `.cursorignore` to limit context per app when working in Cursor
- [X] Step 4: Update root `apps/edge-server/README.md` with monorepo map and how to run edge locally

## M2 — Cloud RAG MVP
- [X] Step 1: Cloud FastAPI skeleton: `main.py`, `/health`, route mounting
- [X] Step 2: Config/env loader for `LANGFUSE_*` and service port
- [X] Step 3: Response schema file mirroring `EnergyEfficiencyResponse`
- [X] Step 4: Prompts: `prompts/answer_system.txt` (strict JSON)
- [X] Step 5: Chain: `rag/chain.py` (embedder + FAISS retriever + prompt + LLM + validator)
- [X] Step 6: Seed: `rag/data/seed/` snippets + `scripts/seed_index.py` → persist to `faiss_index/`
- [X] Step 7: Endpoint: `POST /api/rag/answer` returns validated JSON

## M3 — LangFuse (cloud-only minimal)
- [X] Step 1: Initialize LangFuse client from env at startup
- [X] Step 2: Create trace per `/api/rag/answer` using `interactionId`
- [X] Step 3: Log fields: `latency_ms`, `model`, `tokens_prompt`, `tokens_completion`, `json_valid`, `retrieved_k`, optional `error_type`

## M4 — Edge integration (cloud-first with fallback)
- [X] Step 1: Add to edge `config.json`: `features.energy_efficiency_rag_enabled`, `cloud_rag.base_url`
- [X] Step 2: Add `shared/rag_client.py` (HTTP POST to cloud with short timeout)
- [X] Step 3: Edit `pipelines/energy_efficiency/pipeline_energy_efficiency.py`: try cloud when flag on; fallback to local LLM on error/timeout
- [X] Step 4: Ensure history/logging unchanged; responses still match `EnergyEfficiencyResponse`

## M5 — Feedback sync (daily batch)
- [X] Step 1: Edge: feedback sync service (`services/feedback_sync.py`) with stdlib I/O; extract HTTP provider to `shared/feedback_client.py`; reads local feedback, filters by `last_synced_at` (`user_data/feedback_sync_state.json`), POSTs batch, updates state
- [X] Step 2: Scheduler: add APScheduler to edge startup to run daily
- [X] Step 3: Cloud: `POST /api/feedback/sync` dedupes by `feedback_id`, persists to SQLite (`data/db.sqlite`), returns `{accepted, duplicates}`
- [X] Step 4: Cloud: attach LangFuse `user_feedback` score (+1/-1) to the trace by `interactionId`

## M6 — Minimal evaluator (single score)
- [X] Step 1: `eval/relevance_evaluator.py`: LLM-as-judge relevance ∈ [0,1]
- [X] Step 2: After `/api/rag/answer`, enqueue eval artifacts to `eval_queue` (no latency in request path)
- [X] Step 3: Process `eval_queue`: compute relevance offline, log `score(name='relevance')` to LangFuse, mark processed
- [X] Step 4: Golden-run: `eval/run_eval.py` reads `eval/data/golden.jsonl`, runs RAG, evaluates relevance, prints aggregate stats

## M7 — Gradio UIs (tiny demos)
- [X] Step 1: Chat UI (`apps/gradio/chat/app.py`): textbox → Edge `/api/prompt`; show JSON + latency (config `EDGE_API_BASE_URL`)
- [X] Step 2: RAG Explorer (`apps/gradio/rag_explorer/app.py`): textbox → Cloud `/api/rag/answer`; show retrieved chunks and final JSON (config `CLOUD_RAG_BASE_URL`)

## M8 — Compose and DX
- [X] Step 1: `infra/compose/docker-compose.yml`: services for edge, cloud, gradio UIs; volumes for `faiss_index/` and `data/db.sqlite`
- [X] Step 2: Script wiring for `cloud-rag/scripts/seed_index.py`
- [X] Step 3: README: env vars (`LANGFUSE_*`), compose run, seeding FAISS, starting Gradio, feature flag toggle, feedback sync privacy note
- [X] Step 4: Healthchecks: wait for cloud before RAG explorer

## M9 — Tests (smoke-level)
- [X] Step 1: Cloud: `/api/rag/answer` returns valid JSON; simple success case
- [X] Step 2: Cloud: relevance evaluator returns float in [0,1]
- [X] Step 3: Cloud: `/api/feedback/sync` upserts and reports accepted/duplicates
- [X] Step 4: Edge: flag off → local LLM path
- [X] Step 5: Edge: flag on → cloud path; simulated timeout → fallback path

## M10 — Prompt improvements (classifier + energy efficiency)
- [X] Step 1: Classifier prompt: upgrade `apps/edge-server/config/classification_system_prompt.txt` with clearer instructions, more examples, ambiguity handling, and domain cues
- [X] Step 2: Energy efficiency prompt: refine `apps/cloud-rag/config/energy_efficiency_system_prompt.txt` (clarify JSON schema, safety disclaimers, deflection rules, concise style)
- [X] Step 3: Docs: update READMEs with guidance and acceptance checks

## M11 — Multi-provider (Nebius/OpenAI) switch
- [ ] Step 1 (cloud config only): add provider toggles in `apps/cloud-rag/config/config.json` (e.g., `llm.provider`, `embeddings.provider`); validate env keys at startup (`NEBIUS_API_KEY` or `OPENAI_API_KEY`) — no provider code yet
- [ ] Step 2 (cloud providers): add `apps/cloud-rag/providers/openai_llm.py` and `apps/cloud-rag/providers/openai_embeddings.py`; update cloud provider factory to switch by `CONFIG` with clear errors if keys missing
- [ ] Step 3 (edge config only): add provider toggle(s) in `apps/edge-server/config/config.json` (global `llm.provider` with optional per-model override is acceptable); validate env at startup in `llm_cloud/provider.py` — no wiring changes yet
- [ ] Step 4 (edge providers wiring): update `apps/edge-server/llm_cloud/provider.py` to select Nebius/OpenAI based on config; ensure classification, device_control, and energy_efficiency models use the selected provider without changing public APIs
- [ ] Step 5 (docs): update both READMEs with provider selection examples and required env vars for each provider
- [ ] Step 6 (smoke tests): minimal provider init tests for cloud and edge (skip if corresponding API key not present); keep existing behavior defaulting to Nebius

## M12 — Hybrid retrieval with re-ranking (API unchanged)
- [ ] Step 1: Add BM25 keyword retriever (langchain-community) alongside FAISS semantic retriever
- [ ] Step 2: Config toggles: `retrieval.mode = semantic|hybrid`, `rerank.enabled = true|false`, `rerank.max_items`
- [ ] Step 3: Reranking (optional): LLM-based re-ranker using existing `evaluate_relevance` on top-N combined candidates
- [ ] Step 4: Chain update: choose path by config; keep response schema stable; surface retrieval metadata in LangFuse traces
- [ ] Step 5: Tests/Docs: smoke tests for semantic vs hybrid; README notes on trade-offs and latency

## M13 — PDF ingestion for seeding
- [ ] Step 1: Add PDF loader in `apps/cloud-rag/scripts/seed_index.py` (e.g., PyPDFLoader); configurable chunking by headings/blank lines
- [ ] Step 2: Export chunks: write `faiss_index/chunks.jsonl` during seeding for BM25/hybrid reuse; update `manifest.json`
- [ ] Step 3: CLI options: allow `--input-dir` and include `.pdf` and `.txt`; idempotent vectorization using per-file content-hash manifest (only process new/changed files; skip unchanged)
- [ ] Step 4: Tests: seed with a small sample PDF; ensure FAISS builds and chunks export exists
- [ ] Step 5: Docs: update cloud README with PDF instructions and caveats

## M14 — Datasets and evaluation scripts
- [ ] Step 1: Small labeled sets: add `apps/edge-server/data/classifier_samples.jsonl` and `apps/cloud-rag/eval/data/ee_prompt_examples.jsonl`
- [ ] Step 2: Local eval scripts: quick accuracy script for classifier; JSON-adherence checker for energy prompt
- [ ] Step 3: Expand `apps/cloud-rag/eval/data/golden.jsonl` to 20–50 diverse energy-efficiency questions with context hints
- [ ] Step 4: Add rubric notes per item (what constitutes a good answer); optional difficulty tags
- [ ] Step 5: Enhance `eval/run_eval.py` to print mean/median/stddev, invalid_json_rate, and simple histograms
- [ ] Step 6: Docs: describe dataset scope, how to contribute new items, and how to interpret metrics
