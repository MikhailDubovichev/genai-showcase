# Gradio UIs

This directory contains Gradio web interfaces for the GenAI Showcase applications:
- **Edge Chat UI**: Interface for the edge FastAPI server
- **RAG Explorer UI**: Interface for the cloud RAG service

For monorepo overview, see the root-level README.

## Quick start (macOS)

1) Install dependencies
```bash
poetry install
```

2) Run Edge Chat UI
```bash
poetry run python chat/edge_chat.py
```
UI will be available at `http://localhost:7860`

3) Run RAG Explorer UI (in another terminal)
```bash
poetry run python rag_explorer/rag_explorer.py
```
UI will be available at `http://localhost:7861`

## Directory structure

```
chat/               # Edge Chat UI (edge_chat.py)
rag_explorer/       # RAG Explorer UI (rag_explorer.py)
config/             # Configuration (config.json)
shared/             # Shared utilities (config.py)
pyproject.toml     # Python project for Gradio UIs
poetry.lock        # Locked dependencies
```

## Project Structure Rationale

### Why One Poetry Project for Both UIs?

Both Gradio UIs are managed under a single Poetry project for the following reasons:

**Shared Technology Stack:**
- Both UIs use the same core dependency: `gradio`
- Consistent Python version requirements and environment
- Same deployment and dependency management approach

**Shared Configuration & Utilities:**
- Both UIs read from the same `config/config.json`
- Shared utility functions in `shared/config.py` for URL building, port extraction, and validation
- Consistent configuration patterns across both applications

**Related Functionality:**
- Both are frontend interfaces serving different backends
- Similar user interaction patterns and error handling
- Same architectural layer (presentation/UI) in the system

**Deployment Benefits:**
- Single `poetry install` for both UIs
- Consistent dependency versions across both applications
- Easier maintenance and updates
- Can be deployed together as one unit

**Comparison with Other Services:**
- **edge-server**: Separate FastAPI backend service
- **cloud-rag**: Separate RAG backend service
- **gradio**: Single frontend project containing multiple UIs

This structure reflects the architectural separation: backend services are separate (edge-server, cloud-rag) while frontend interfaces are consolidated (gradio).

## Applications

### Edge Chat UI (`chat/edge_chat.py`)
- Connects to edge FastAPI server (default: `http://localhost:8080`)
- Sends user messages and displays responses
- Shows raw JSON and extracted message
- Measures and displays latency

### RAG Explorer UI (`rag_explorer/rag_explorer.py`)
- Connects to cloud RAG service (default: `http://localhost:8000`)
- Sends questions and retrieves answers with context
- Shows retrieved chunks with similarity scores
- Displays raw JSON response and extracted message
- Measures and displays latency

## Configuration

Both UIs read from `config/config.json`:
- API base URLs and timeouts
- Gradio UI URLs (configurable ports)
- HTTP timeout settings

Example `config/config.json`:
```json
{
  "edge_api_base_url": "http://localhost:8080",
  "cloud_rag_base_url": "http://localhost:8000",
  "http_timeout_ms": 40000,
  "gradio_edge_chat_url": "http://localhost:7860",
  "gradio_rag_explorer_url": "http://localhost:7861"
}
```

## Getting started (detailed)

### Prerequisites
- Python 3.9+
- Edge server running (for Edge Chat UI)
- Cloud RAG service running (for RAG Explorer UI)
- API keys configured for the services

### Install dependencies (Poetry on macOS)
```bash
brew install poetry
poetry install --no-root
```

### Run the UIs

Edge Chat UI:
```bash
poetry run python chat/edge_chat.py
```

RAG Explorer UI:
```bash
poetry run python rag_explorer/rag_explorer.py
```

### Quick UI checks
```bash
# Check if Edge Chat UI is running
curl -s http://localhost:7860 | head -10

# Check if RAG Explorer UI is running
curl -s http://localhost:7861 | head -10
```

## Features

- **Configurable URLs**: All API endpoints and UI ports are configurable
- **Error handling**: Graceful handling of network errors and timeouts
- **Latency measurement**: Real-time latency display for API calls
- **Reset functionality**: Clear all fields and start fresh
- **JSON transparency**: View raw API responses for debugging

## Notes

- Privacy: UIs run locally and communicate with local services
- Minimal changes: Follow `.cursor/.cursorrules`; keep edits focused
- Consistency: UIs follow the same patterns as the backend services
