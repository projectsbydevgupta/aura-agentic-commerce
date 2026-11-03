# AURA — AI-Native Luxury Fashion Concierge & Agentic Commerce Protocol

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq Cloud](https://img.shields.io/badge/Groq-Llama%203%20%2F%20GPT--OSS-f55036.svg)](https://groq.com/)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Vector%20Search-47A248.svg)](https://www.mongodb.com/atlas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CrewAI](https://img.shields.io/badge/CrewAI-Supervisor--Worker-red.svg)](https://crewai.com/)
[![Google ADK](https://img.shields.io/badge/Google--ADK-Autonomous--Buyer-blue.svg)](https://github.com/google)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-green.svg)](https://modelcontextprotocol.io/)

> An agentic multi-agent conversational commerce platform that replaces static keyword filters with collaborative cognitive agents: natural language query analysis, interactive Human-in-the-Loop context clarification, MongoDB Atlas hybrid vector search, critic validation gates, and dynamic AI styling.

---

## Key Features

- **Multi-Agent State Machine**: Built on LangGraph with deterministic state passing across 5 specialized cognitive agents.
- **Dynamic Context Clarification (HITL)**: Intelligently detects ambiguous queries (e.g., *"I want shoes"*, *"I need a dress"*) and prompts with dynamic, category-tailored suggestion pills before executing expensive vector searches.
- **Hybrid Vector Search**: Combines 512-dimensional dense embeddings (Voyage AI `voyage-3-lite`) with MongoDB Atlas scalar pre-filtering across 44,000+ fashion products.
- **Deterministic Guardrails & Price Normalization**: Enforces strict budget ceilings and automatic currency conversion between INR (₹) and catalog USD units ($).
- **Critic & Self-Correction Gatekeeper**: Audits retrieval relevance; discards out-of-domain hallucinations and reroutes failed queries.
- **WhatsApp Automation Engine**: Asynchronous queue-based customer engagement daemon powered by Baileys WebSocket protocol and MongoDB.
- **Autonomous Buyer & MCP Layer**: Reference implementations of Agent-to-Agent (A2A) procurement protocol and Model Context Protocol (MCP) tool integration.
- **Enterprise Observability**: Distributed OpenTelemetry tracing via Arize Phoenix, tracking agent spans, tokens, and per-node latencies.

---

## Multi-Agent Architecture

```
User / Client Request
        │
        ▼
┌───────────────────────────────┐
│  Security Guardrail Interceptor │  (Prompt injection, toxicity, spend ceiling filters)
└───────────────┬───────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              PRIMARY STATE ENGINE (LangGraph)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌───────────────┐        ┌────────────────────────────┐   │
│   │  Query Agent  │ ─────► │   Context Clarifier (HITL) │   │
│   │ (Intent/Parse)│        │ (Dynamic Suggestions / Opts)│   │
│   └───────────────┘        └─────────────┬──────────────┘   │
│                                          │ (Sufficient Context)
│                                          ▼                  │
│   ┌───────────────┐        ┌────────────────────────────┐   │
│   │Validation Gate│ ◄───── │   Hybrid Search Node       │   │
│   │ (Critic/Gate) │        │ (Voyage AI + Atlas Vector) │   │
│   └───────┬───────┘        └────────────────────────────┘   │
│           │                                                 │
│     (Pass)│   (Fail / Retry)                                │
│           ├─────────────────► [Self-Correction Loop]        │
│           ▼                                                 │
│   ┌───────────────┐                                         │
│   │ Stylist Agent │ ─────► Tailored Outfits & Smart Upsell │
│   └───────────────┘                                         │
└─────────────────────────────────────────────────────────────┘
```

---

---

## Agentic Paradigms & Supported Frameworks

AURA combines and benchmarks multiple foundational Multi-Agent System (MAS) paradigms and protocols:

| Framework / Pattern | Module | Role & Implementation Details |
| :--- | :--- | :--- |
| **Blackboard Architecture** | `src/blackboard/` | **Shared-Memory State Coordination**: Central shared state (`BlackboardState`) and event bus with an Arbiter control loop. Decoupled Knowledge Sources (`KS_Query`, `KS_Search`, `KS_Stylist`) asynchronously read and update shared state without peer-to-peer coupling. |
| **Hierarchical CrewAI** | `src/crews/` | **Supervisor-Worker Pattern**: Orchestrates a hierarchical team of specialized agents (Catalog Scout, Style Coordinator, QA Inspector) with structured task delegation, backed by deterministic fallbacks. |
| **Google ADK Autonomous Buyer** | `src/adk_agents/` | **Client-Side Buyer Representation**: Autonomous buyer agent acting on behalf of a human user to negotiate promo vouchers (`AURA20`), enforce spend ceilings, and execute programmatic orders. |
| **Agent-to-Agent (A2A) Protocol** | `src/protocol/` | **Cross-Framework Commerce**: Interoperability channel connecting heterogeneous agent frameworks (Google ADK Buyer <-> CrewAI Merchant) using standardized `A2AMessageEnvelope` payloads. |
| **Model Context Protocol (MCP)** | `src/mcp/` | **Standardized Tool Layer**: Standardized MCP-compliant tool servers exposing `catalog_hybrid_search` and `financial_fx_calculator` (currency conversion, discount logic, luxury tax). |

## Quick Start

### 1. Clone & Configure
```bash
git clone https://github.com/projectsbydevgupta/aura-agentic-commerce.git
cd aura-agentic-commerce

# Copy template and add your API keys
cp .env.example .env
```

### 2. Install Dependencies
```bash
# Python dependencies
pip install -r requirements.txt

# (Optional) Node.js dependencies for WhatsApp Baileys daemon
npm install
```

### 3. Launch the Server
```bash
# Option A: One-click launcher (Windows)
start.bat

# Option B: Manual uvicorn execution
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

---

## 🌐 Service Endpoints, Ports & Protocol Discovery

| Service / Interface | Port | Local Endpoint / URL | Description |
| :--- | :---: | :--- | :--- |
| 🛍️ **Storefront & Concierge** | `8000` | [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/) | Conversational Luxury Fashion Concierge & AI Styling Experience |
| 📊 **Executive Atelier Admin** | `8000` | [`http://127.0.0.1:8000/admin`](http://127.0.0.1:8000/admin) | Real-time revenue analytics, orders, patrons & A2A machine telemetry |
| 📜 **AP2 Agent Discovery Manifest** | `8000` | [`/.well-known/agent-protocol.json`](http://127.0.0.1:8000/.well-known/agent-protocol.json) | Machine-readable capability & endpoint manifest for external AI buyers |
| 🔌 **MCP Tool Schema Manifest** | `8000` | [`/.well-known/mcp.json`](http://127.0.0.1:8000/.well-known/mcp.json) | Model Context Protocol (MCP) declarations for Claude, AutoGPT & external agents |
| 📱 **WhatsApp QR Pairing** | `8000` / `5001` | [`http://127.0.0.1:8000/whatsapp`](http://127.0.0.1:8000/whatsapp) | Live Baileys WebSocket QR scanner & automated campaign dispatcher |
| 📈 **A2A Protocol Telemetry** | `8000` | [`/protocol/v1/telemetry`](http://127.0.0.1:8000/protocol/v1/telemetry) | Audit metrics for autonomous buyer spend ceilings & programmatic orders |
| 🔍 **Arize Phoenix Telemetry** | `6006` / Cloud | [`/api/monitoring/phoenix-status`](http://127.0.0.1:8000/api/monitoring/phoenix-status) | Distributed OpenTelemetry trace status, agent spans & latency monitoring |
| 💓 **Keep-Alive Health Probe** | `8000` | [`http://127.0.0.1:8000/health`](http://127.0.0.1:8000/health) | Uptime monitoring & MongoDB Atlas connectivity health check |
| 📖 **Interactive OpenAPI Docs** | `8000` | [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) | Interactive Swagger UI for testing all REST, MCP, and A2A API routes |

---

## Core Environment Configuration

See [`.env.example`](.env.example) for the complete list. Minimal required keys:

| Variable | Description |
| :--- | :--- |
| `GROQ_API_KEY` | Groq Cloud API Key (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`) |
| `VOYAGE_API_KEY` | Voyage AI Key for 512-dim embeddings (`voyage-3-lite`) |
| `MONGODB_URI` | MongoDB Atlas cluster connection string with Vector Search |
| `RAZORPAY_KEY_ID` | Razorpay test key ID for simulated checkout transactions |
| `PHOENIX_API_KEY` | *(Optional)* Arize Phoenix OpenTelemetry tracing key |

---

## Directory Structure

```
aura-agentic-commerce/
├── .env.example                 # Configuration template with all supported variables
├── server.py                    # FastAPI server exposing REST, WebSocket & UI endpoints
├── agent_cli.py                 # Interactive terminal shopping CLI
├── search_cli.py                # Direct CLI hybrid vector search utility
├── brain.md                     # Deep technical specification & architectural decisions
│
├── src/
│   ├── agents/                  # LangGraph MAS: Query, Context, Search, Validation, Stylist
│   │   ├── price_utils.py       # Deterministic INR/USD currency normalization
│   │   └── workflow.py          # StateGraph routing & edge transitions
│   ├── search/                  # Atlas Vector Search, Voyage AI embeddings & synonyms
│   ├── security/                # Guardrails: Prompt injection, toxicity, spend ceiling
│   ├── blackboard/              # Shared-memory blackboard architecture pattern
│   ├── crews/                   # Hierarchical supervisor-worker pattern
│   ├── protocol/                # A2A (Agent-to-Agent) autonomous commerce channel
│   ├── mcp/                     # Model Context Protocol (MCP) tools & FX calculator
│   ├── monitoring/              # Callbacks, JSONL telemetry & Arize Phoenix OpenTelemetry
│   └── whatsapp/                # Baileys daemon, MongoDB queue & rate-limited worker
│
├── docs/                        # Benchmark charts & evaluation documentation
│   ├── AURA - Agentic Commerce MAS Evaluation 2.pdf
│   ├── mas_heatmap.svg          # 2D agent latency & token utilization heatmap
│   └── evaluation_boxplot_latency.svg
│
├── static/                      # Vanilla JS/CSS glassmorphism frontend & admin views
└── tests/                       # Pytest test suite (14 test modules)
```

---

## Evaluation & Benchmarks

AURA was rigorously evaluated across **120 measured benchmark runs** over 10 Multi-Agent System (MAS) criteria:
- **Full Report**: [`docs/AURA - Agentic Commerce MAS Evaluation 2.pdf`](<docs/AURA - Agentic Commerce MAS Evaluation 2.pdf>)
- **Deep Technical Spec**: [`brain.md`](brain.md)
- **Telemetry Visualizations**:
  - `docs/mas_heatmap.svg`: 2D agent performance and token usage matrix.
  - `docs/evaluation_boxplot_latency.svg`: Latency comparison across Sequential, Hierarchical, and Blackboard architectures.
  - `docs/merge_agents_comparison.svg`: Empirical 60-prompt study demonstrating why decoupled agents mathematically outperform merged monolithic agents.

---

## Testing

Run the automated test suite across all agents, guardrails, and services:

```bash
# Run all tests
pytest tests/ -v

# Run context clarification & multi-turn dialog tests
pytest tests/test_context_clarification.py -v
```

---

## License & Attribution

Developed for **CSE3101: Agentic AI**.  
Built with LangGraph, Groq Cloud, MongoDB Atlas, Voyage AI, and FastAPI.
