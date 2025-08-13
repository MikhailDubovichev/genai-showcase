## Product Requirements Document (PRD) — GenAI Showcase (Edge + Cloud RAG, LangFuse-only)

### 1. Overview
- **Goal**: Showcase edge-device orchestration and cloud Retrieval-Augmented Generation (RAG) with minimal evaluation and monitoring using LangFuse (LangFuse comes from “language” + “fuse”; a platform to fuse traces, datasets, and scores).
 - **Architecture**: Two services plus demo UIs in one monorepo:
  - `apps/edge-server/` (existing FastAPI on Raspberry Pi; device control + energy-efficiency pipeline).
  - `apps/cloud-rag/` (FastAPI + LangChain RAG + FAISS; SQLite for feedback mirror; LangFuse for traces).
  - `apps/gradio/` (Gradio demos: chat UI for edge; RAG explorer for cloud)
- **Principles**: Minimal changes, safe fallbacks, privacy-first (no raw emails; only `user_hash`), clear separation.

### 2. Objectives and Success Metrics
- **Objectives**
  - Add cloud RAG answering for energy-efficiency queries.
  - Log minimal telemetry to LangFuse.
  - Sync user feedback from edge to cloud daily and attach to traces.
  - Provide one simple evaluator: “relevance”.
- **Success metrics**
  - ≥ 95% responses JSON-parse into `EnergyEfficiencyResponse`.
  - Cloud RAG p95 latency ≤ 2.5s (local/dev).
  - 100% daily feedback sync success (with retries).
  - LangFuse traces include latency, tokens, json_valid, retrieved_k, relevance, and user_feedback when present.

### 3. Scope
- **In scope**
  - Cloud: FAISS-backed RAG (FAISS comes from “Facebook AI Similarity Search”), LangChain chain, `/api/rag/answer`, `/api/feedback/sync`, SQLite feedback mirror, LangFuse logging, single evaluator (relevance).
  - Edge: Feature-flagged call to cloud RAG with fallback; daily feedback sync job.
  - Gradio: Minimal UIs for demoability — chat against edge; RAG explorer against cloud.
- **Out of scope (for v0)**
  - Evidently AI, drift detection, Grafana dashboards.
  - Multi-evaluator suites (faithfulness, readability, etc.).
  - Multi-tenant auth/RBAC.
  - Full document ingestion pipelines (seed with local snippets only).

### 4. Users and Use Cases
- **Users**
  - Developer/demo operator evaluating skills in RAG and evaluation.
- **Primary flows**
  - User asks energy-efficiency question → edge routes to cloud RAG → answer returns JSON → trace appears in LangFuse with minimal metrics.
  - User submits feedback in app → stored on edge → daily batch sync to cloud → attached to LangFuse trace as score.

### 5. System Architecture
- **Edge-server**
  - FastAPI endpoints (`/api/prompt`, `/api/context`, feedback endpoints), pipelines, tools for device control.
  - Feature flag `features.energy_efficiency_rag_enabled` toggles cloud-first behavior.
- **Cloud-rag**
  - FastAPI service with LangChain pipeline and FAISS index on disk.
  - LangFuse client initialized via env vars.
  - SQLite `data/db.sqlite` for mirrored feedback (SQLite is public domain and free).
- **Data flow**
  - Prompt → Edge classifier → Energy pipeline → Cloud RAG (if flag on) → Response → Edge to client.
  - Feedback → Edge file → Daily sync → Cloud SQLite → LangFuse score on trace.
 - **Gradio UIs**
   - Chat UI → calls Edge `/api/prompt` with `user_email`; displays JSON and latency.
   - RAG Explorer → calls Cloud `/api/rag/answer`; shows retrieved chunks/citations and relevance score.

### 6. Functional Requirements
- **FR-1** Cloud RAG Answer
  - Endpoint: `POST /api/rag/answer`
  - Input: `question`, `interactionId`, optional `topK`
  - Output: Valid `EnergyEfficiencyResponse` JSON
  - Logs trace and minimal metrics to LangFuse.
- **FR-2** Edge Integration
  - If flag enabled, call cloud RAG; on timeout/error fallback to current LLM-only path.
  - No changes to device-control pipeline.
- **FR-3** Feedback Sync (daily)
  - Edge job reads new feedback since last sync, POSTs batch to cloud, updates sync state.
  - Cloud dedupes by `feedback_id`, persists to SQLite, adds LangFuse score `user_feedback` (+1/-1).
- **FR-4** Minimal Evaluation
  - Cloud computes one “relevance” score per answer and logs to LangFuse.
- **FR-5** Gradio Interfaces
  - Chat UI: simple conversation window bound to Edge `/api/prompt`; display raw JSON and response time.
  - RAG Explorer: input box to query Cloud RAG; visualize top-k retrieved chunks (sourceId, score, snippet) and final JSON answer.

### 7. API Contracts
- **Cloud RAG request**
```json
{
  "question": "How to reduce standby power?",
  "interactionId": "uuid-123",
  "topK": 4
}
```

- **Cloud RAG response** (must conform to `EnergyEfficiencyResponse`)
```json
{
  "message": "Here are a few grounded tips ...",
  "interactionId": "uuid-123",
  "type": "text",
  "content": [
    {"sourceId":"doc_1","chunk":"Advanced power strips ...","score":0.78}
  ]
}
```

- **Feedback sync** (edge → cloud)
```json
{
  "synced_at": "2025-08-13T12:00:00Z",
  "feedback": [
    {
      "feedback_id": "uuid-fb1",
      "interaction_id": "uuid-123",
      "type": "positive",
      "timestamp": "2025-08-13T11:58:21Z",
      "user_hash": "a1b2c3d4...",
      "location_id": "loc-123",
      "message_preview": "Turn off living room lights...",
      "response_preview": "Ok, turning off..."
    }
  ]
}
```

- **Feedback sync response**
```json
{"accepted": 1, "duplicates": 0}
```

### 8. Data Storage
- **Edge**: Conversation history, active/archived sessions, feedback JSON files, device context.
- **Cloud**
  - FAISS index directory (persisted volume).
  - SQLite `data/db.sqlite` (feedback mirror).
  - LangFuse (managed externally): traces and scores.
- **PII**: Cloud receives `user_hash` only, not raw emails. Previews can be omitted or truncated.

### 9. Monitoring and Evaluation (Minimal)
- **LangFuse trace fields**
  - `latency_ms`
  - `model`
  - `tokens_prompt`
  - `tokens_completion`
  - `json_valid` (bool)
  - `retrieved_k` (int)
  - `error_type` (optional)
  - `relevance` (score 0–1)
  - `user_feedback` (+1/-1 when synced)
- **Acceptance thresholds (soft)**
  - JSON-valid rate ≥ 95%
  - p95 latency ≤ 2.5s (dev)
  - Evaluator present for each cloud answer

### 10. Non-Functional Requirements
- **Performance**: RAG request finishes ≤ 2.5s p95 on dev hardware.
- **Reliability**: Edge must return an answer even if cloud fails (fallback).
- **Security**: No raw `user_email` to cloud; transport over HTTP in dev, can switch to HTTPS in prod.
- **Portability**: Dockerized; compose for local; independent deploys.

### 11. Configuration
- **Edge `config.json`**
  - `features.energy_efficiency_rag_enabled: true`
  - `cloud_rag.base_url: "http://cloud-rag:8000"`
- **Env vars (both apps)**
  - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
  
 - **Gradio config**
   - `EDGE_API_BASE_URL` for chat UI (e.g., `http://edge-server:8080`)
   - `CLOUD_RAG_BASE_URL` for RAG explorer (e.g., `http://cloud-rag:8000`)

### 12. Monorepo Layout
- `apps/edge-server/` (current code; minimal edits)
- `apps/cloud-rag/` (new service)
 - `apps/gradio/chat/` (chat UI → Edge)
 - `apps/gradio/rag_explorer/` (RAG explorer → Cloud)
- `infra/compose/docker-compose.yml` (local run)
- `.cursorignore` (exclude the other app when focusing in Cursor)

### 13. Milestones
- **M1: Repo restructure**
  - Move current code to `apps/edge-server/`
  - Compose skeleton for both services
- **M2: Cloud RAG MVP**
  - `/api/rag/answer` + FAISS seed + LangFuse logging (latency/tokens/json_valid/retrieved_k)
- **M3: Edge integration**
  - Feature-flagged call + fallback
- **M4: Feedback sync**
  - Edge daily job + cloud `/api/feedback/sync` + SQLite + LangFuse `user_feedback`
- **M5: Minimal evaluator**
  - Relevance score per answer → LangFuse
- Optional: Tiny golden set (20 Qs) and a simple eval runner
 - **M6: Gradio UIs**
   - Chat UI wired to Edge `/api/prompt`
   - RAG Explorer wired to Cloud `/api/rag/answer`

### 14. Risks and Mitigations
- **Cloud downtime**: Edge fallback preserves UX.
- **Schema drift**: Validate JSON with `EnergyEfficiencyResponse` before returning; log `json_valid=false` in LangFuse.
- **Feedback duplication**: Deduplicate by `feedback_id` and return duplicates count.

### 15. Out of Scope (v0)
- Evidently AI, drift analytics, Grafana.
- Multi-evaluator suites beyond relevance.
- Complex ingestion; start with local seed docs for FAISS.

### 16. Acceptance Criteria
- Edge returns valid JSON both with flag on/off; on failures, fallback works.
- Cloud logs traces with minimal metrics and relevance score to LangFuse.
- Daily feedback sync stores records in SQLite and tags related traces with `user_feedback`.
- Documentation explains setup, LangFuse env, and how to run both apps via docker-compose.
 - Gradio chat and RAG explorer function end-to-end against Edge and Cloud respectively.
