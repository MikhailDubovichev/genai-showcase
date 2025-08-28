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
- Cloud RAG integration (feature-flagged):
  - `features.energy_efficiency_rag_enabled: true|false`
  - `cloud_rag.base_url: "http://localhost:8000"`
  - `cloud_rag.timeout_s: 5.0` (default; adjust for your environment)
  
### Classifier determinism

```
llm.models.classification.settings:
  max_tokens: 20
  temperature: 0.0
  top_p: 1.0
  top_k: 1
```

These settings ensure single-token label outputs like `DEVICE_CONTROL` or `ENERGY_EFFICIENCY` without reasoning.

### Provider switching (Nebius ↔ OpenAI)

You can switch LLM providers using templates and a small script. The templates contain full `config.json` variants to keep switching safe and consistent.

1) Switch provider (copies template → active config):
```
python scripts/switch_provider.py nebius
python scripts/switch_provider.py openai
```

2) Set environment variables (already supported):
- Nebius: `NEBIUS_API_KEY`
- OpenAI: `OPENAI_API_KEY`

3) Restart the server to apply changes:
```
poetry run python main.py
```

4) Verify selection in logs (INFO):
- "LLM provider selected: nebius | base_url=https://api.studio.nebius.com/v1/"
- or "LLM provider selected: openai | base_url=https://api.openai.com/v1"

Notes:
- Templates live under `config/templates/` and include per-model names for `classification`, `device_control`, and `energy_efficiency`.
- The code automatically handles small provider differences; no app code changes are required when switching.

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