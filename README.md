# AI Assistant for Energy Efficiency - Technical Documentation

## About this repository (showcase context)

This repository is a vendor‑agnostic showcase derived from a real proof‑of‑concept for a Smart Home Integrator that wanted to embed an AI assistant into its product. The assistant was designed to run at the edge (for example, on a Raspberry Pi), so the system is intentionally lightweight and container‑friendly.

- **Edge‑first, container‑ready**: Built to run on resource‑constrained devices, optimized for small memory/CPU footprints, and easy to containerize for deployment on Raspberry Pi.
- **Lightweight stack**: Uses FastAPI and a small set of focused libraries; avoids heavyweight web frameworks (like LangChain) and large orchestration frameworks to keep runtime lean and predictable.
- **Device control + energy advice**: Provides smart device control via tools and delivers energy efficiency guidance.
- **Mock Smart Home Integrator by default**: Ships with a mock provider so the entire system runs without external credentials. A real integrator can be added later via the provider interface.

## Architecture Philosophy: Purpose-Built for the Edge

This project's architecture was not chosen in a vacuum; it is a direct and intentional result of its primary goal: **to run efficiently as a single-purpose AI assistant on a resource-constrained edge device (like a Raspberry Pi).**

This core mission dictated a "lightweight by design" philosophy, leading us to deliberately avoid heavyweight frameworks (like LangChain or LangGraph) and traditional databases. For a targeted application with a limited user scope (e.g., managing a single home), these choices offer significant advantages.

### Why This Architecture is Justified for the Edge

**✅ Performance & Resource Efficiency:**
- **Minimal Dependencies:** On an edge device with limited RAM and CPU, every dependency matters. We include only what is essential, resulting in a smaller memory footprint and faster startup time.
- **Direct LLM API Calls:** Bypassing framework abstractions reduces CPU overhead and request latency, which is critical for a responsive conversational experience.
- **File-Based Storage:** Using simple text/JSON files for history and context eliminates the resource cost (RAM, CPU, disk I/O) of running a database service, which is overkill for the data load of a single user or location.

**✅ Simplicity & Maintainability:**
- **Zero-Config Deployment:** The file-based approach requires no database setup or maintenance, simplifying deployment on an edge device.
- **Transparent Control Flow:** The code is straightforward and easy to debug. You can trace every step of the process without navigating complex framework internals, which is invaluable when troubleshooting on a remote device.
- **Reduced Attack Surface:** Fewer dependencies mean a smaller potential attack surface, an important consideration for an internet-connected device in a home.

**✅ Focused Functionality:**
- This assistant is designed to do a few things well: classify requests and route them to either a device control or energy efficiency pipeline.
- For this well-defined scope, the overhead of a general-purpose framework that supports countless integrations we don't need is an unnecessary tax on performance.

In summary, this architecture represents a series of deliberate engineering trade-offs. We sacrificed the development speed of generic frameworks for the runtime performance, efficiency, and simplicity required to build a reliable and robust AI assistant for an edge environment.

---

## 1. Project Overview

This is a sophisticated AI assistant backend service built with Python and FastAPI, designed specifically for energy management via a Smart Home Integrator. The system functions as an intelligent intermediary between users and their smart energy devices, providing conversational access to energy monitoring, device control, and efficiency optimization.

The core architecture follows a **pipeline-based design pattern** with clean separation of concerns, making it modular, testable, and maintainable.

## 2. Core Features

### 🤖 **Intelligent Request Routing**
- **Automatic classification** of user messages into appropriate processing pipelines
- **Context-aware responses** tailored to device control vs. energy efficiency queries
- **Graceful handling** of out-of-scope requests with polite rejections

### 🏠 **Smart Device Control**
- **Real-time device management** through a Smart Home Integrator adapter (Mock Integrator by default for demos/tests)
- **Intelligent tool calling** for device operations (on/off, scheduling, status queries)
- **Energy consumption monitoring** with live data access when a real integrator is configured

### 💡 **Energy Efficiency Advisory**
- **Personalized energy-saving tips** and recommendations
- **Educational content** about energy optimization strategies
- **Structured response format** for consistent user experience

### 📊 **Comprehensive Monitoring**
- **Prometheus metrics** for performance tracking and observability
- **Structured logging** with correlation IDs for request tracing
- **Pipeline-level metrics** for latency and error monitoring

### 🗣️ **Conversational Memory**
- **Per-user session management** with conversation history
- **Interaction tracking** with unique IDs for feedback correlation
- **Stateful dialogues** that maintain context across conversations

---

## 3. System Architecture

### High-Level Architecture Flow

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

### Directory Structure & Components

#### `📄 main.py`
**Application Entry Point**
- Initializes FastAPI application with CORS middleware
- Includes API routers for modular endpoint organization  
- Mounts Prometheus metrics endpoint at `/metrics`
- Configures Uvicorn server for production deployment

#### `📁 api/` - **HTTP Interface Layer**
**Clean REST API endpoints with request validation**

- **`prompt.py`**: Core chat endpoint (`POST /api/prompt`) that processes user messages through the orchestrator. Also handles conversation reset (`POST /api/reset`)
- **`context.py`**: Session initialization endpoint (`POST /api/context`) that loads user device context and injects daily energy digest
- **`feedback.py`**: User feedback collection (`POST /api/feedback/positive|negative`) for response quality tracking

#### `📁 core/` - **Business Logic Coordination**
**Central coordination and decision-making layer**

- **`orchestrator.py`**: **Main coordination hub** that:
  - Routes classified messages to appropriate pipelines
  - Manages session context and conversation history  
  - Handles errors and provides fallback responses
  - Generates unique interaction IDs for tracking

- **`classifier.py`**: **Message classification engine** that:
  - Uses LLM to categorize user requests (device control vs. energy efficiency vs. other)
  - Provides single source of truth for routing decisions
  - Handles direct rejection of unsupported query types

#### `📁 pipelines/` - **Specialized Processing Engines**
**Domain-specific message processing with clean abstractions**

- **`base.py`**: Abstract base class defining the pipeline interface with:
  - Standardized setup and processing methods
  - Built-in monitoring decorators for latency and error tracking
  - Common logging patterns for consistency

- **`device_control/pipeline.py`**: **Smart device management pipeline** featuring:
  - Full tool ecosystem for device operations (get devices, control, scheduling)
  - Smart Home Integrator adapter with authentication when enabled (Mock Integrator by default)
  - Tool call orchestration and result processing

- **`energy_efficiency/pipeline.py`**: **Energy advisory pipeline** providing:
  - Conversational energy-saving advice without tool dependencies
  - Structured response validation using Pydantic models
  - Educational content generation for energy optimization

#### `📁 llm_cloud/` - **AI Integration Layer**
**Clean abstraction for Large Language Model interactions**

- **`provider.py`**: **LLM client factory** that:
  - Provides configured OpenAI-compatible client for Nebius cloud
  - Avoids global state with function-based client creation
  - Enables easy testing with dependency injection

- **`tools/core.py`**: **Sophisticated tool management system** featuring:
  - `Tool` class for metadata and handler encapsulation
  - `ToolManager` for registration and definition retrieval  
  - `ToolExecutor` for execution with dependency injection and error handling

- **`tools/handlers.py`**: **Tool implementation library** with actual business logic for device operations

#### `📁 provider_api/` - **Integrator Abstraction and Mock**
**Provider-agnostic interface and default mock implementation**

- **`base.py`**: **Abstract interface** for Smart Home Integrator adapters (list devices, control devices)
- **`mock_client.py`**: **Deterministic mock client** enabling full local runs with no external credentials

#### `📁 monitoring/` - **Observability Infrastructure**
**Production-ready monitoring and metrics**

- **`metrics.py`**: **Prometheus metrics system** tracking:
  - HTTP request latency and counts by endpoint
  - Pipeline processing time by type
  - Tool execution time by tool name
  - External API latency (LLM and Integrator)
  - Error rates by type and location

#### `📁 services/` - **Supporting Business Services**
**Reusable business logic components**

- **`history_manager.py`**: **Conversation persistence** with JSON-based storage and archival
- **`feedback_manager.py`**: **User feedback collection** with contextual information capture
- **`daily_digest.py`**: **Proactive energy tips** with once-per-day delivery logic

#### `📁 shared/` - **Common Data Models**
**Type-safe data structures and utilities**

- **`models.py`**: **Pydantic models and enums** for:
  - Message classification categories (MessageCategory)
  - Session context management (SessionContext)  
  - Processing results (ProcessingResult)
  - Response validation (EnergyEfficiencyResponse)

- **`utils.py`**: **Common utilities** for error handling, tool execution, and session management

#### `📁 config/` - **Configuration Management**
**Externalized configuration with environment separation**

- **`config.json`**: **Central configuration** for API endpoints, model settings, and operational parameters
- **`*_system_prompt.txt`**: **LLM instruction templates** for different pipeline types
- **`logging_config.py`**: **Structured logging setup** with correlation ID support

---

## 4. Request Processing Flow

### Example: "Turn on my living room light"

1. **API Entry** (`api/prompt.py`)
   - FastAPI receives POST request with user message, token, and location_id
   - Request forwarded to PipelineOrchestrator

2. **Classification** (`core/classifier.py`) 
   - LLM analyzes message: "Turn on my living room light"
   - Classified as `MessageCategory.DEVICE_CONTROL`

3. **Pipeline Routing** (`core/orchestrator.py`)
   - Routes to DeviceControlPipeline based on classification
   - Generates unique interaction_id for tracking

4. **Pipeline Processing** (`pipelines/device_control/pipeline.py`)
   - Creates session context with user information
   - Calls LLM with device control system prompt and available tools
   - LLM decides to use `get_devices` tool to find living room light

5. **Tool Execution** (`llm_cloud/tools/`)
   - ToolExecutor runs `get_devices` handler
   - Integrator client (or Mock Integrator) fetches user's device list
   - Filters for lighting devices in living room

6. **Second LLM Call**
   - LLM receives device list as tool result
   - Decides to use `control_device` tool with device ID and "on" action
   - Tool executed to actually turn on the light

7. **Response Generation**
   - LLM generates human-friendly confirmation message
   - Response saved to conversation history with interaction_id
   - JSON response returned to frontend

---

## 5. Getting Started

### Prerequisites
- **Python 3.9+**
- **Nebius AI Studio account** with API key (or OpenAI-compatible provider)
  - Note: A Smart Home Integrator account is not required for local runs; the Mock Integrator is the default.

### Installation

1. **Install Poetry (recommended on macOS):**
   ```bash
   brew install poetry
   # or use the official installer
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository and install dependencies with Poetry:**
   ```bash
   git clone <repository_url>
   cd ai-assistant-energy-efficiency
   poetry install --with dev --no-root
   ```

3. **Configure environment variables:**
   ```bash
   export LLM_API_KEY="your_nebius_api_key"
   export ENERGY_PROVIDER="mock"  # default; keeps the app fully runnable without external credentials
   ```

4. **Update configuration:**
   - Review `config/config.json` for API endpoints and model settings (see `integrator` and `llm` sections)
   - Customize system prompts in `config/` directory as needed

5. **Run the application:**
   ```bash
   poetry run python main.py
   ```
   
   The API will be available at `http://localhost:8080` with auto-generated documentation at `http://localhost:8080/docs`
   
   Note: If you prefer `pip` and a manual virtual environment, you can still use `requirements.txt`; however, Poetry is the default workflow for this project.

### Testing the API

```bash
# Initialize user context (loads devices, shows daily digest)
curl -X POST "http://localhost:8080/api/context?token=your_token&location_id=1000299"

# Send a message to the assistant
curl -X POST "http://localhost:8080/api/prompt?message=Show%20me%20my%20devices&token=your_token&location_id=1000299" | jq '.'

# Reset conversation history
curl -X POST "http://localhost:8080/api/reset"
```

---

## 6. Production Considerations

### Monitoring
- **Prometheus metrics** available at `/metrics` endpoint
- **Structured logging** with JSON format for log aggregation
- **Request tracing** with correlation IDs across all components

### Security
- **Token-based authentication** for Smart Home Integrator access when enabled
- **Input validation** with Pydantic models
- **Error handling** that doesn't leak sensitive information

### Scalability
- **Stateless design** allows horizontal scaling
- **Pipeline architecture** enables easy feature addition
- **Dependency injection** supports testing and mocking

### Configuration
- **Environment-based secrets** (never commit API keys)
- **Externalized configuration** for different deployment environments
- **Model parameter tuning** per pipeline type for optimal performance

---

## 7. Tests

The project includes a `tests/` directory with automated tests that cover the most important units and pipelines:
- `tests/api/`: Validates API endpoints for `context`, `prompt`, and `feedback` flows
- `tests/core/`: Exercises the `classifier` and `orchestrator` coordination logic
- `tests/pipelines/`: Verifies the energy efficiency pipeline behavior
- `tests/services/`: Checks conversation history management

### Running the test suite (Poetry, macOS)

```bash
# Ensure dev dependencies (pytest, pytest-cov) are installed
poetry install --with dev --no-root

# Run the entire suite
poetry run pytest -q

# Run with coverage and show missing lines
poetry run pytest --cov=. --cov-report=term-missing

# Run a specific module
poetry run pytest tests/core/test_orchestrator.py -q
```

---

This architecture demonstrates that building sophisticated AI assistants doesn't require heavyweight frameworks. With clear abstractions, proper separation of concerns, and thoughtful design patterns, you can create maintainable, performant, and easily understood systems that solve real business problems.