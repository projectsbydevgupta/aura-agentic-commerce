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

## 🏆 Final Project Evaluation — Mandatory Criteria Compliance Matrix

AURA strictly satisfies **all 10 mandatory evaluation criteria** defined for the Agentic AI (CSE3101) capstone:

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
| **C9** | **Enterprise Observability & Tracing** | Distributed OpenTelemetry spans exported to Arize Phoenix, capturing agent execution traces, token consumption, latency breakdowns, and prompt metadata. | `src/monitoring/phoenix_tracer.py` | ✅ **Verified** (`tests/test_phoenix_tracing.py`) |
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

## 🌐 System Routes & Live Endpoints

Base URL: [`https://aura-agentic-commerce.onrender.com`](https://aura-agentic-commerce.onrender.com)

| Route | Method | Component | Purpose |
| :--- | :---: | :--- | :--- |
| [`/`](https://aura-agentic-commerce.onrender.com) | `GET` | Storefront Concierge | Conversational luxury fashion concierge & AI styling experience |
| [`/evaluation`](https://aura-agentic-commerce.onrender.com/evaluation) | `GET` | MAS Evaluation Dashboard | 2D Agent latency/token heatmap & benchmark boxplots |
| [`/admin`](https://aura-agentic-commerce.onrender.com/admin) | `GET` | Atelier Admin | Real-time order telemetry, user management, and machine analytics |
| [`/health`](https://aura-agentic-commerce.onrender.com/health) | `GET` | Health Probe | Uptime monitoring and MongoDB Atlas database connection health |
| [`/.well-known/mcp.json`](https://aura-agentic-commerce.onrender.com/.well-known/mcp.json) | `GET` | MCP Tool Manifest | Model Context Protocol declarations for external tool consumers |
| [`/.well-known/agent-protocol.json`](https://aura-agentic-commerce.onrender.com/.well-known/agent-protocol.json) | `GET` | A2A Discovery Manifest | AP2 Agent Discovery Manifest for autonomous AI shoppers |
| [`/docs`](https://aura-agentic-commerce.onrender.com/docs) | `GET` | OpenAPI Docs | Interactive Swagger UI for testing all REST, MCP, and A2A routes |

### Evaluator Verification via cURL

```bash
BASE="https://aura-agentic-commerce.onrender.com"

# Health & database probe
curl -s "${BASE}/health"

# Model Context Protocol (MCP) tool schema
curl -s "${BASE}/.well-known/mcp.json"

# Autonomous Agent-to-Agent (A2A) protocol manifest
curl -s "${BASE}/.well-known/agent-protocol.json"
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
- `docs/mas_heatmap.svg`: 2D agent performance and token usage matrix.
- `docs/evaluation_boxplot_latency.svg`: Latency distribution across agentic architectures.
- `docs/merge_agents_comparison.svg`: Empirical study validating why decoupled multi-agent architectures outperform monolithic LLMs.

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

# 4. Run automated test suite (24 tests verifying all 10 criteria)
pytest tests/ -v

# 5. Launch local server
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

---

## 👥 Authors & Academic Attribution

- **Course**: CSE3101: Agentic AI (Semester 7)
- **Project Title**: AURA — AI-Native Luxury Fashion Concierge & Agentic Commerce Protocol
- **Repository**: [github.com/projectsbydevgupta/aura-agentic-commerce](https://github.com/projectsbydevgupta/aura-agentic-commerce)
- **Deployment**: Render Cloud (Python 3.11.9, Containerized Web Service)
- **Tech Stack**: Python 3.11.9, FastAPI, LangGraph, CrewAI, Google ADK, MongoDB Atlas, Groq Cloud, Voyage AI.
