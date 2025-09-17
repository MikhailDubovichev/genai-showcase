# GenAI Showcase — Edge Server + Cloud RAG Monorepo

## Executive Summary

I created this monorepo to showcase the technical skills and architectural decisions required to build robust and scalable AI systems.

The project was inspired by a Proof of concept (POC) of a simple AI assistant I built for a client's smart home management system. The initial idea was to create an assistant that could execute a limited set of commands in natural language (e.g., “Switch off the light in the kitchen,” “Set the thermostat to 21 degrees”). A key constraint was that the assistant had to run on an edge device (a Raspberry Pi) alongside other services, requiring its Docker container to be as lean as possible. As a result, I avoided heavy abstractions like LangChain and instead coded a custom, low-level tool-management system.

To demonstrate my RAG skills and experience with frameworks like LangChain, I decided to add question-answering functionality to the project. What started as a simple "add-on" quickly evolved into a complete refactoring of the original codebase, resulting in the comprehensive system presented here. Key reasons for the complete refactoring was that I had to add logic for routing user requests (and make it scalable, ready to extend to a vider range of user questions), I had to host RAG index in the cloud, not on the edge and I needed to add other features required for production grade systems, like monitoring capabilities.

This monorepo's AI assistant is designed to do two things:
1.  Execute device control commands.
2.  Answer questions related to energy efficiency (e.g., “How can I save on my electricity bills?”).

It gracefully rejects any out-of-scope questions (on weather, politics, etc.).

The system consists of two FastAPI servers (**edge-server** and **cloud-rag**) and corresponding Gradio interfaces to simplify the testing experience. The lean **edge-server** acts as the brain of the operation, featuring an LLM-based classifier that intelligently routes user requests:
*   **Simple device commands** are handled locally and instantly.
*   **Complex energy-efficiency questions** are forwarded to the dedicated Cloud RAG service.

Beyond the core agentic logic, this project includes all the functionality required for a production-grade system: idempotent document indexing (for PDFs, markdown, and text files), user feedback collection, integration with monitoring tools (LangFuse), seamless switching between LLM providers (OpenAI vs. Nebius - a platform hosting almost all the open-sourced LLMs, like DeepSeek, Llama, Qwen), and more.

**Note on RAG Knowledge Base:** TODO: The index used in the RAG pipeline was built on a small set of documents for demonstration purposes. To make the system truly useful, the knowledge base would need to be expanded with more comprehensive materials.

A non-comprehensive list of the showcased capabilities is detailed below.

P.S. To showcase my technical capabilities in **data analysis / classical machine learning (ML)** I also pushed to my GitHub **Metallurgy-Flotation-Plant-Optimization/** repo, with model predicting silica content in iron ore concentrate. I built this model in Google Colab in 2024 as part of my ML self-stydies, using Kaggle dataset and one A100 GPU.

---

## Key Technical Capabilities Showcased

*   **GenAI & LLM Orchestration**
    *   **Agentic Routing:** An LLM-based classifier on the edge server intelligently routes user requests to the correct tool or service.
    *   **Low-Level Tool Management:** The edge server features a custom, lightweight framework for tool registration and execution, demonstrating core agentic principles without heavy abstractions.
    *   **Advanced RAG Pipeline:** The cloud service implements a state-of-the-art RAG pipeline with:
        *   **Hybrid Retrieval:** Combines **FAISS** (semantic) and **BM25** (keyword) search for comprehensive context gathering.
        *   **LLM-as-Judge Reranking:** A secondary LLM scores and re-orders retrieved chunks to ensure only the most relevant context is used for generation.
    *   **Strategic LangChain Usage:** Judicious use of **LangChain** in the cloud service to orchestrate the RAG chain, demonstrating the ability to leverage frameworks where they provide the most value.
    *   **Multi-Provider Abstraction:** A provider-agnostic design that allows seamlessly switching between **OpenAI** and **Nebius** (hosting various open-source models) via a simple script.

*   **Production-Grade Engineering & Architecture**
    *   **Edge-Cloud System Design:** A robust architecture featuring a lean, responsive edge server and a powerful, scalable cloud service.
    *   **Idempotent Data Pipelines:** The data ingestion script for the RAG service is idempotent, using a manifest and content hashing to process only new or changed files—a critical feature for reliable data systems.
    *   **Comprehensive Monitoring & Evaluation:**
        *   **Observability:** Deep integration with **LangFuse** for tracing, latency monitoring, and token usage tracking.
        *   **Offline Quality Evaluation:** An LLM-as-judge relevance evaluator that runs on a queue to score RAG quality without impacting request latency.
    *   **Robust Configuration Management:** A centralized system that loads configuration from defaults, JSON files, and environment variables.
    *   **API Development & Validation:** Clean, documented **FastAPI** endpoints with strict JSON schema enforcement using **Pydantic**.

*   **Core Technologies & DevOps**
    *   **Containerization:** Full-stack **Docker Compose** setup for reproducible development environments and production-readiness.
    *   **Dependency Management:** Python environments are managed cleanly with **Poetry**.
    *   **Database & Scheduling:** Use of **SQLite** for persistent feedback storage and **APScheduler** for running scheduled background tasks.
    *   **UI Prototyping:** Simple and effective testing interfaces built with **Gradio**.

---

## Monorepo Layout

```
apps/
  edge-server/           # Edge FastAPI app
    api/                 # Context, prompt, feedback endpoints
    core/                # Orchestrator and classifier
    ...
    scripts/             # Provider switching script
    pyproject.toml       # Python project for edge
    README.md            # Edge-specific docs

  cloud-rag/             # Cloud service: FastAPI + LangChain + RAG
    api/                 # /health, /api/rag/answer, /api/feedback/sync
    ...
    scripts/             # Seeding CLI and provider switching
    faiss_index/         # Persisted FAISS index, chunks, and manifest
    pyproject.toml       # Python project for cloud

  gradio/                # Chat and RAG explorer UIs

infra/
  compose/               # Docker Compose for local dev

PRD.md                   # Product Requirements Document
.cursor/
  TASKS.md               # Milestones and tasks
...
```

### Delivered Milestones
- **M1-M9**: Foundational setup including the monorepo structure, Cloud RAG MVP, LangFuse integration, edge-to-cloud communication with fallback, feedback sync, an offline relevance evaluator, Gradio UIs, Docker Compose setup, and smoke tests.
- **M10**: **Prompt Engineering**: Enhanced the core system prompts for classification and energy efficiency to improve clarity, add domain-specific examples, and handle ambiguity, ensuring more reliable and accurate LLM responses.
- **M11**: **Multi-Provider Support**: Refactored both services to support **Nebius and OpenAI** as LLM providers. Added configuration templates and CLI scripts (`switch_provider.py`) for easy switching between them.
- **M12**: **Hybrid Retrieval & Reranking**: Upgraded the RAG pipeline with a sophisticated hybrid retrieval strategy, combining **FAISS** (semantic) and **BM25** (keyword) search. Added an **LLM-as-judge reranker** to score and reorder results for final context selection, significantly improving relevance.
- **M13**: **PDF & Text Ingestion**: Enhanced the data pipeline to ingest both **PDFs and plain text files** (`.txt`, `.md`). Implemented an **idempotent seeding process** with a manifest file to track content hashes, skipping unchanged files on subsequent runs for efficient, incremental index updates.

---

## Important Note

Each key block of the monorepo, like cloud-rag/ or edge-server/ has also its own focused README.me file. focused

## Quick Start (Local Development)

The recommended way to run the services directly on your machine for development and showcasing.

### Prerequisites
- **Python** and **Poetry** installed.
- **API Keys** for your chosen LLM provider (Nebius or OpenAI).

### Step 1: Configure LLM Provider

First, decide which provider you want to use (Nebius or OpenAI) and run the switcher scripts for both services. This step copies the correct configuration template into place.

```bash
# Configure the cloud service (e.g., for OpenAI)
python apps/cloud-rag/scripts/switch_provider.py openai

# Configure the edge service
python apps/edge-server/scripts/switch_provider.py openai
```
*Replace `openai` with `nebius` if you prefer to use Nebius.*

### Step 2: Set Up the Cloud RAG Service

```bash
# 1. Install dependencies
poetry -C apps/cloud-rag install

# 2. Set API Key and LangFuse credentials in your shell
export OPENAI_API_KEY="<your_key>" # or NEBIUS_API_KEY
export LANGFUSE_PUBLIC_KEY="<optional_public>"
export LANGFUSE_SECRET_KEY="<optional_secret>"

# 3. Seed the index with sample PDFs and text files
# This creates the vector database from documents in apps/cloud-rag/rag/data/seed/
poetry -C apps/cloud-rag run python -m scripts.seed_index

# 4. Run the server
poetry -C apps/cloud-rag run python main.py
```
The Cloud RAG service will now be running, typically on port 8000.

### Step 3: Set Up the Edge Server

```bash
# 1. Install dependencies
poetry -C apps/edge-server install

# 2. Set API Key (must match the cloud service)
export OPENAI_API_KEY="<your_key>" # or NEBIUS_API_KEY

# 3. Run the server
poetry -C apps/edge-server run python main.py
```
The Edge Server will now be running, typically on port 8080.

### Step 4: Access the Services

With both services running, you can now use the APIs and demo UIs:

- **Edge Server API**: http://localhost:8080/docs
- **Cloud RAG API**: http://localhost:8000/docs
- **Gradio Chat UI**: Run `poetry -C apps/gradio/chat install` then `poetry -C apps/gradio/chat run python app.py`
- **RAG Explorer UI**: Run `poetry -C apps/gradio/rag_explorer install` then `poetry -C apps/gradio/rag_explorer run python app.py`

---

## Alternative: Running with Docker

For users who prefer a containerized setup, Docker Compose can be used to run the full stack.

### 1. Configure Provider & Environment
Follow **Step 1** and **Step 2** from the local development guide to configure your provider and create your `.env` file with the necessary API keys.

### 2. Seed the Vector Index
This one-time setup ingests the sample documents and builds the FAISS index inside a container.

```bash
# Navigate to the compose directory
cd infra/compose

# Generate dynamic .env file with ports
python generate_env.py

# Run the seeding service
docker compose --profile seed run --rm seed-index
```

### 3. Start the Full Stack
```bash
# From the infra/compose directory
docker compose up --build
```

---

## Testing

Each service contains its own set of smoke tests in its `tests/` directory. To run them, use `pytest` from the project root.

```bash
# Run tests for the cloud-rag service
poetry run pytest apps/cloud-rag/tests/

# Run tests for the edge-server
poetry run pytest apps/edge-server/tests/
```

## References

- **Edge Server Details**: `apps/edge-server/README.md`
- **Cloud RAG Details**: `apps/cloud-rag/README.md`
- **Product Requirements**: `PRD.md`
- **Tasks & Milestones**: `.cursor/TASKS.md`