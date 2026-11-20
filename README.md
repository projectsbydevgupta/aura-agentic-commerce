# AURA — AI-Native Luxury Fashion Concierge & Agentic Commerce Protocol

[![Live Deployment](https://img.shields.io/badge/Render-Live%20Production-00E599?style=for-the-badge&logo=render&logoColor=white)](https://aura-agentic-commerce.onrender.com)
[![Python 3.11](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Hierarchical%20Supervisor-FF4F00?style=for-the-badge)](https://crewai.com/)
[![Google ADK](https://img.shields.io/badge/Google--ADK-Autonomous%20Buyer-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-4B0082?style=for-the-badge)](https://modelcontextprotocol.io/)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas%20Vector%20Search-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/atlas)

> **Live Deployment**: [aura-agentic-commerce.onrender.com](https://aura-agentic-commerce.onrender.com)  
> An autonomous multi-agent conversational commerce system implementing **CrewAI**, **Google ADK**, **Blackboard Architecture**, **MCP Tool Calling**, and **Hybrid Vector Search** across 44,424 luxury fashion products.

---

## 🏛️ Multi-Agent System (MAS) Architectural Capabilities Matrix

AURA implements and benchmarks the following 10 multi-agent system engineering capabilities:

| # | Evaluation Criterion | Implementation Details | Core Source Files | Verification Status |
|:---:|:---|:---|:---|:---:|
| **C1** | **Context Sharing & State Updates / Blackboard Architecture** | Shared-memory `BlackboardState` with event bus and decoupled Knowledge Sources (`KS_Query`, `KS_Search`, `KS_Stylist`). Intermediate agent outputs feed into downstream consumers with zero point-to-point coupling. | `src/blackboard/blackboard.py`<br>`src/blackboard/event_bus.py` | ✅ **Verified** (`tests/test_blackboard_and_messaging.py`) |
| **C2** | **Tool Integration using MCP (2+ Tools)** | Standardized Model Context Protocol (MCP) tool servers exposing: 1) `catalog_hybrid_search` (dense vector + lexical prefilter) and 2) `financial_fx_calculator` (currency normalization, promo discounts, tax). | `src/mcp/mcp_server.py`<br>`src/mcp/mcp_client.py` | ✅ **Verified** (`tests/test_mcp_integration.py`) |
| **C3** | **Structured Output using Pydantic** | Every agent returns typed Pydantic models with `llm.with_structured_output()`: `QueryAnalysis`, `ContextEvaluation`, `ValidationDecision`, `UpsellRecommendation`, and `ADKOrderConfirmation`. | `src/agents/state.py`<br>`src/agents/base.py` | ✅ **Verified** (Zero unvalidated JSON strings) |
| **C4** | **Multi-Agent Design Patterns** | Benchmarked and implemented across three distinct agentic topologies: 1) **Sequential** (LangGraph StateGraph), 2) **Hierarchical** (CrewAI Supervisor-Worker), and 3) **Parallel** (Blackboard Knowledge Sources). | `src/agents/workflow.py`<br>`src/crews/fashion_crew.py`<br>`src/blackboard/` | ✅ **Verified** (`data/evaluation_120_runs.csv`) |
| **C5** | **Task Monitoring, Logging & 2D MAS Heatmap** | Comprehensive callback lifecycle tracking (`on_agent_start`, `on_agent_end`, `on_tool_call`). Produces a live 2D agent latency & token utilization heatmap and writes structured JSONL telemetry for provenance. | `src/monitoring/callbacks.py`<br>`static/evaluation_dashboard.html`<br>`docs/mas_heatmap.svg` | ✅ **Verified** (`tests/test_guardrails_and_monitoring.py`) |
| **C6** | **Agent-to-Agent (A2A) Protocol Interoperability** | Standardized `A2AMessageEnvelope` protocol enabling cross-framework autonomous negotiation between client-side **Google ADK Buyer** and merchant-side **CrewAI Concierge**. | `src/protocol/a2a_channel.py`<br>`src/protocol/router.py` | ✅ **Verified** (`tests/test_crewai_adk_interop.py`) |
| **C7** | **Dual Framework Compliance (CrewAI + Google ADK)** | **Mandatory requirement met**: Powered by **CrewAI** (hierarchical merchant concierge with Catalog Scout, Style Coordinator, and QA Inspector) and **Google ADK** (autonomous buyer with budget ceilings and auto-checkout). | `src/crews/fashion_crew.py`<br>`src/adk_agents/buyer_agent.py` | ✅ **Verified** (Both frameworks active and interoperable) |
| **C8** | **MAS Security: Guardrails, Personas & Spend Ceilings** | Multi-tier security interceptor detecting prompt injection, toxic/abusive persona inputs, adversarial jailbreaks, and hard budget overrun ceilings with audit logging. | `src/security/guardrails.py` | ✅ **Verified** (`tests/test_guardrails_and_monitoring.py`) |
| **C9** | **Enterprise Observability & Tracing** | Distributed OpenTelemetry spans exported to Arize Phoenix Cloud (Space: `aura-commerce`) and local port `6006`, capturing agent execution traces, token consumption, latency breakdowns, and prompt metadata. | `src/monitoring/phoenix_tracer.py` | ✅ **Verified** (`tests/test_phoenix_tracing.py`) |
| **C10** | **Empirical Evaluation: 120 Runs & Statistical Boxplots** | Evaluated across 120 empirical runs covering 6 user personas and 3 architectural topologies. Includes statistical latency boxplots, cost estimates, and LLM-as-a-Judge relevance scoring. | `data/evaluation_120_runs.csv`<br>`docs/evaluation_boxplot_latency.svg`<br>`scripts/run_benchmarks.py` | ✅ **Verified** (120 runs documented & visualized) |

---

## 🏗️ Multi-Agent Architecture

```
User Query / Client Request
          │
          ▼
┌─────────────────────────────────┐
│   Security Guardrail Layer      │  (Prompt Injection, Toxic Language, Spend Ceiling)
└─────────────────┬───────────────┘
                  │
                  ▼
┌───────────────────────────────────────────────────────────────┐
│            LANGGRAPH COGNITIVE STATE ENGINE                   │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌─────────────────┐        ┌────────────────────────────┐   │
│   │   Query Agent   │ ─────► │  Context Clarifier (HITL)  │   │
│   │  (Intent/Filter)│        │  (Ambiguity / Dynamic UI)  │   │
│   └─────────────────┘        └─────────────┬──────────────┘   │
│                                            │ (Sufficient Context)
│                                            ▼                  │
│   ┌─────────────────┐        ┌────────────────────────────┐   │
│   │ Validation Gate │ ◄───── │    Hybrid Search Node      │   │
│   │  (Critic Gate)  │        │ (Voyage AI + Atlas Vector) │   │
│   └────────┬────────┘        └────────────────────────────┘   │
│            │                                                  │
│      (Pass)│   (Fail / Retry)                                 │
│            ├─────────────────► [Self-Correction Loop]         │
│            ▼                                                  │
│   ┌─────────────────┐                                         │
│   │ Fashion Stylist │ ─────► Complete Outfit & Dynamic Upsell │
│   └─────────────────┘                                         │
└───────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│     Live Real-Time UI Stream    │  (Progressive multi-agent step animations)
└─────────────────────────────────┘
```

---

## ⚡ Agent-to-Agent (A2A) Dual Framework Interoperability

```
┌───────────────────────────────────┐               ┌───────────────────────────────────┐
│      Google ADK Autonomous        │               │         CrewAI Hierarchical       │
│           Buyer Agent             │               │          Merchant Concierge       │
├───────────────────────────────────┤               ├───────────────────────────────────┤
│ • Framework: Google ADK           │               │ • Framework: CrewAI               │
│ • Role: User's AI Procurement Rep │               │ • Manager: Fashion Supervisor     │
│ • Enforces Spend Ceiling ($200)   │               │ • Workers: Scout, Stylist, QA     │
│ • Injects Promo Voucher 'AURA20'  │               │ • MCP Tools: Catalog, FX Calc     │
└─────────────────┬─────────────────┘               └─────────────────▲─────────────────┘
                  │                                                   │
                  │        Standardized A2AMessageEnvelope             │
                  └──────────────────► [ A2A Bridge ] ────────────────┘
                                  (JSON-RPC 2.0 Payload)
```

---

## 🛡️ Ethical AI, Responsible Agentic Design & Safety Guardrails

In autonomous and multi-agent systems, engineering ethics and defensive boundaries are fundamental. AURA integrates four core ethical principles:

1. **Autonomous Financial Safety & Spend Ceilings**:
   Autonomous buyer agents operate strictly within human-specified spend ceilings (enforced deterministically in `src/security/guardrails.py`). The agent cannot execute checkout if the total exceeds the budget boundary, mitigating runaway purchasing behavior.
2. **Adversarial Input & Prompt Injection Immunity**:
   Adversarial prompts attempting system prompt extraction, rule overrides, or malicious redirection are intercepted at the perimeter before propagating to internal agent states.
3. **Anti-Toxicity & Respectful De-escalation**:
   User prompts with abusive, derogatory, or hostile sentiment (e.g., adversarial consumer personas) trigger respectful de-escalation responses without degrading service quality or compromising system integrity.
4. **Data Privacy & Ephemeral State**:
   User interactions and preferences are bound to session threads (`thread_id`) without permanent tracking or selling of personal identifiable information (PII). Payments utilize sandbox tokenization via Razorpay Test mode.

---

## 🖥️ Real-Time Progressive Frontend Pipeline

When a user submits an inquiry or clicks a suggestion, the frontend reveals the multi-agent pipeline in real-time with smooth state transitions:

1. **Query Agent**: Extracts intent, target gender, article type, color palettes, and budget limits.
2. **Context Agent**: Evaluates information sufficiency; prompts with category-tailored suggestion pills if ambiguous.
3. **Hybrid Search Node**: Executes 512-dimensional vector search fused with exact keyword filtering over 44,000+ products.
4. **Validation Agent**: Acts as an adversarial critic gate, auditing items for aesthetic coherence and discarding out-of-domain noise.
5. **Stylist Agent**: Applies Chain-of-Thought (CoT) reasoning to curate complete multi-piece fashion ensembles.

---

## 🌐 Live Deployments & Protocol Discovery Links

| Service / Interface | Method | Live Full URL | Description |
| :--- | :---: | :--- | :--- |
| 🛍️ **Storefront Concierge** | `GET` | [https://aura-agentic-commerce.onrender.com](https://aura-agentic-commerce.onrender.com) | Conversational luxury fashion concierge & AI styling experience |
| 📊 **MAS Evaluation Dashboard** | `GET` | [https://aura-agentic-commerce.onrender.com/evaluation](https://aura-agentic-commerce.onrender.com/evaluation) | 2D Agent latency/token heatmap & statistical benchmark boxplots |
| 📈 **Executive Atelier Admin** | `GET` | [https://aura-agentic-commerce.onrender.com/admin](https://aura-agentic-commerce.onrender.com/admin) | Real-time order telemetry, user management, and machine analytics |
| 💓 **Health Check Probe** | `GET` | [https://aura-agentic-commerce.onrender.com/health](https://aura-agentic-commerce.onrender.com/health) | Uptime monitoring and MongoDB Atlas database connection health |
| 🔌 **MCP Tool Declarations** | `GET` | [https://aura-agentic-commerce.onrender.com/.well-known/mcp.json](https://aura-agentic-commerce.onrender.com/.well-known/mcp.json) | Model Context Protocol declarations for external LLM tool consumers |
| 📜 **A2A Protocol Discovery** | `GET` | [https://aura-agentic-commerce.onrender.com/.well-known/agent-protocol.json](https://aura-agentic-commerce.onrender.com/.well-known/agent-protocol.json) | AP2 Agent Discovery Manifest for autonomous AI shoppers |
| 🔭 **Arize Phoenix Tracing** | `GET` | [https://aura-agentic-commerce.onrender.com/api/monitoring/phoenix-status](https://aura-agentic-commerce.onrender.com/api/monitoring/phoenix-status) | Live OpenTelemetry tracer status, emitted traces, and space (`aura-commerce`) configuration |
| 📑 **Presentation Slide Deck** | `PDF` | [`docs/AURA_Final_Presentation_Group_9.pdf`](docs/AURA_Final_Presentation_Group_9.pdf) | Official Group 9 Final Architecture & Evaluation Presentation Deck |
| 📖 **Interactive OpenAPI Docs** | `GET` | [https://aura-agentic-commerce.onrender.com/docs](https://aura-agentic-commerce.onrender.com/docs) | Interactive Swagger UI for testing all REST, MCP, and A2A routes |

### Evaluator Verification via cURL

```bash
# Health check & database probe
curl -s "https://aura-agentic-commerce.onrender.com/health"

# Arize Phoenix OpenTelemetry tracing status probe (confirms aura-commerce cloud space)
curl -s "https://aura-agentic-commerce.onrender.com/api/monitoring/phoenix-status"

# Model Context Protocol (MCP) tool schema
curl -s "https://aura-agentic-commerce.onrender.com/.well-known/mcp.json"

# Autonomous Agent-to-Agent (A2A) protocol manifest
curl -s "https://aura-agentic-commerce.onrender.com/.well-known/agent-protocol.json"
```

---

## 🔌 Service Port Allocation & Network Topology

AURA orchestrates microservices, agent protocols, and telemetry collectors across the following dedicated ports:

| Service / Subsystem | Local Dev Port | Live Production Port / URL | Protocol & Transport | Operational Responsibility |
| :--- | :---: | :---: | :---: | :--- |
| **FastAPI Core Application** | `8000` (`http://localhost:8000`) | Standard `443` HTTPS ([Live App](https://aura-agentic-commerce.onrender.com)) | HTTP/1.1 & SSE (Uvicorn ASGI) | Hosts Storefront UI, MAS Evaluation Dashboard (`/evaluation`), Swagger Docs (`/docs`), and REST APIs. |
| **FastMCP Tool Microservice** | `8001` (`http://localhost:8001`) | Routed via Core Port `443` (`/protocol/v1/mcp/call`) | JSON-RPC 2.0 over HTTP | Serves standardized Model Context Protocol tools (`catalog_hybrid_search`, `financial_fx_calculator`). |
| **Arize Phoenix Observability** | `6006` (`http://localhost:6006`) | Cloud Collector: `https://app.phoenix.arize.com/s/aura-commerce/v1/traces` (Status: `/api/monitoring/phoenix-status`) | OpenTelemetry / OTLP (Protobuf over HTTP) | Captures distributed traces, span hierarchies, token consumption, and agent latency for deep observability. |
| **WhatsApp Baileys Service** | `5001` (`http://localhost:5001`) | Internal Microservice Bridge | WebSockets & HTTP REST | Handles re-engagement campaign pushes, cart recovery, and coupon dispatching via WhatsApp protocol. |
| **MongoDB Atlas Cluster** | `27017` (SRV connection) | Cloud Cluster (`mongodb+srv://`) | TLS-Encrypted MongoDB Protocol | Stores 44,424 product records and serves 512-dim Voyage AI dense vector search queries. |

### Arize Phoenix Telemetry Configuration (Production & Local)

To route distributed multi-agent traces into the `aura-commerce` space:
```bash
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/aura-commerce/v1/traces
PHOENIX_PROJECT_NAME=aura-commerce
PHOENIX_LOCAL_PORT=6006
```

---

## 🔬 Benchmark Results & Statistical Validation (120 Runs)

AURA was evaluated across **120 measured benchmark runs** spanning 6 diverse user personas:

| Metric | Sequential (LangGraph) | Hierarchical (CrewAI) | Blackboard (Shared Memory) |
| :--- | :---: | :---: | :---: |
| **Mean Latency (ms)** | 6,854 ± 842 ms | 11,240 ± 1,420 ms | 4,210 ± 610 ms |
| **Relevance Score (0–100%)** | 94.6% | 96.2% | 89.4% |
| **Cost per 1k Inquiries** | $1.42 | $2.84 | $1.18 |
| **Guardrail Rejection Accuracy** | 99.2% | 98.8% | 99.1% |

Visualizations available in [`docs/`](docs/):
- `docs/AURA_Final_Presentation_Group_9.pdf`: Official Group 9 Final Architecture & Evaluation Presentation Slide Deck.
- `docs/mas_heatmap.svg`: 2D agent performance and token usage matrix.
- `docs/evaluation_boxplot_latency.svg`: Latency distribution across agentic architectures.
- `docs/merge_agents_comparison.svg`: Empirical study validating why decoupled multi-agent architectures outperform monolithic LLMs.

---

## 🏷️ Product Catalog & Benchmark Dataset Provenance

The production inventory powering AURA's retrieval, validation, and styling engines is built upon the **Myntra Fashion Product Dataset** (curated and distributed via [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) by Param Aggarwal):
- **Catalog Scale:** 44,424 authentic e-commerce inventory items across Apparel (21,397), Accessories (11,274), Footwear (9,219), and Personal Care (2,403).
- **Demographics & Usages:** Balanced representation across Men (22,147), Women (18,631), and Unisex (2,161), supporting Casual (34,723), Athletic/Sports (4,025), Ethnic/Traditional (3,208), and Formal (2,345) styles.
- **Production Asset Pipeline:** Preserves live high-resolution product imagery hosted directly on Myntra's CDN (`assets.myntassets.com`), along with authentic brand attributions (*Turtle, Peter England, Titan, Puma, Fabindia*), seasonal metadata, and retail pricing.
- **Hybrid Vector Indexing:** Precomputed 512-dimensional dense embeddings via **Voyage AI (`voyage-3-lite`)**, stored in **MongoDB Atlas Vector Search** with scalar compound filters on price, subcategory, and usage for sub-45ms hybrid retrieval.

---

## 🔄 Graceful Degradation & High-Availability Architecture

A senior engineering priority in production AI systems is resilience against third-party service degradation:
- **Distributed Groq Key Pool**: Agent queries cycle across distinct API keys to prevent rate limiting (`429 Too Many Requests`).
- **Zero-Downtime Catalog Fallback**: If MongoDB Atlas undergoes cloud maintenance or IP restrictions, the search subsystem automatically falls back to an in-memory lexical and dense index (`44,424 items`), guaranteeing continuous uptime during demonstrations.
- **Fail-Safe Framework Fallbacks**: If external agent framework packages experience remote gateway timeouts, the internal typed orchestration bridges seamlessly handle execution without crashing the user session.

---

## 🚀 Local Development Quickstart

```bash
# 1. Clone repository
git clone https://github.com/projectsbydevgupta/aura-agentic-commerce.git
cd aura-agentic-commerce

# 2. Configure environment
cp .env.example .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite (32 tests verifying all 10 criteria)
pytest tests/ -v

# 5. Launch local server (Port 8000)
uvicorn server:app --host 127.0.0.1 --port 8000 --reload

# 6. (Optional) Launch local Arize Phoenix trace viewer UI (Port 6006)
# Open http://127.0.0.1:6006 to inspect trace hierarchies and token spans locally
python -m phoenix.server.main serve
```

---

## 📦 Project Overview & Architecture Stack

- **Project**: AURA — AI-Native Luxury Fashion Concierge & Agentic Commerce Protocol
- **Repository**: [github.com/projectsbydevgupta/aura-agentic-commerce](https://github.com/projectsbydevgupta/aura-agentic-commerce)
- **Deployment**: Render Cloud (Python 3.11.9, Containerized Web Service)
- **Core Architecture**: CrewAI, Google ADK, LangGraph, MCP Protocol, MongoDB Atlas Vector Search, Groq Cloud, Voyage AI
- **License**: MIT
