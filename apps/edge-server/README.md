# Edge Server

This is the edge FastAPI application (designed for Raspberry Pi) that powers the chat‑based assistant for smart device control and energy‑efficiency advice. For monorepo overview, see the root‑level README.

## Quick start (macOS)

1) Install dependencies
```
poetry install
```

2) Run the server
```
poetry run python main.py
```

The API will be available at `http://localhost:8080` and metrics at `/metrics`.

## Directory structure

```
api/                 # HTTP endpoints: /api/prompt, /api/context, /api/feedback/*
core/                # Orchestrator (routing) and classifier (intent)
pipelines/           # Device control and energy efficiency pipelines
llm_cloud/           # OpenAI-compatible client and tool system
provider_api/        # Integrator abstraction + mock provider (default)
services/            # History and feedback managers
monitoring/          # Prometheus metrics helpers
shared/              # Pydantic models and utilities
pyproject.toml       # Python project for edge
version.py           # App version
```

## Architecture overview

High-level flow of the edge application:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI API   │────│  Orchestrator    │────│   Classifier    │
│   Layer         │    │  (Coordination)  │    │  (Routing)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼────────┐  ┌───▼─────────┐    │
        │ Device Control │  │  Energy     │    │
        │   Pipeline     │  │ Efficiency  │    │
        │               │  │  Pipeline   │    │
        └────────────────┘  └─────────────┘    │
                │                              │
        ┌───────▼────────┐                     │
        │ Tool Manager   │                     │
        │ & Executors    │                     │
        └────────────────┘                     │
                │                              │
        ┌───────▼────────────────────────┐  ┌──▼──────────┐
        │ Smart Home Integrator Client   │  │ LLM Cloud   │
        │  (Adapter / Mock by default)   │  │  Provider   │
        └────────────────────────────────┘  └─────────────┘
```

## Key endpoints

- `POST /api/prompt`   → main chat endpoint
- `POST /api/context`  → refresh device context; inject daily digest
- `POST /api/reset`    → archive/reset active conversation
- `GET  /metrics`      → Prometheus metrics

## Pipelines

- Device Control (`pipelines/device_control/`): tool‑enabled operations (get devices, control on/off, utility stubs)
- Energy Efficiency (`pipelines/energy_efficiency/`): LLM‑only advisory with JSON schema validation

## LLM client

- LLM provider abstraction: `llm_cloud/provider.py` (OpenAI‑compatible)
- Do not modify this file unless explicitly required (see `.cursor/.cursorrules`).

## Getting started (detailed)

### Prerequisites
- Python 3.9+
- Nebius (OpenAI‑compatible) API key in env `LLM_API_KEY`

### Install dependencies (Poetry on macOS)
```
brew install poetry
poetry install --with dev --no-root
```

### Environment variables
```
export LLM_API_KEY="your_nebius_api_key"
export ENERGY_PROVIDER="mock"   # default; enables full local run without external creds
```

### Run the application
```
poetry run python main.py
```

API docs: http://localhost:8080/docs

### Quick API checks
```
# Initialize user context (loads devices, may inject daily digest)
curl -X POST "http://localhost:8080/api/context?token=demo&location_id=home1"

# Ask something
curl -X POST "http://localhost:8080/api/prompt?message=Show%20me%20my%20devices&token=demo&location_id=home1" | jq '.'

# Reset conversation history
curl -X POST "http://localhost:8080/api/reset"
```

## Configuration

- `config/config.json` controls server host/port, prompts, and model settings.
- Feature flag (planned, for cloud RAG integration):
  - `features.energy_efficiency_rag_enabled`
  - `cloud_rag.base_url`

## Tests

Run the test suite from this directory:
```
poetry run pytest -q
```

More options:
```
# With coverage
poetry run pytest --cov=. --cov-report=term-missing

# Run a specific module
poetry run pytest tests/core/test_orchestrator.py -q
```

## Notes

- Privacy: never send raw `user_email` to cloud services; use `user_hash`.
- Minimal changes: follow `.cursor/.cursorrules`; keep edits focused and avoid touching unrelated code.