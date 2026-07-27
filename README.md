# Veriscan-Cortex — Advanced Fraud Intelligence & Private Multi-Agent Dashboard

> **Course:** CS 5588 — Data Science Capstone | **Date:** July 2026

---

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Visual Architecture](#visual-architecture)
- [Local AI Intelligence](#local-ai-intelligence)
- [Repository Structure](#repository-structure)
- [Snowflake Data Platform](#snowflake-data-platform)
- [Microservices Architecture](#microservices-architecture)
- [Quick Start](#quick-start)
- [Evaluation](#evaluation)

---

## Project Overview

Veriscan is an end-to-end **Fraud Detection & Security Platform** that processes transaction data through a multi-stage intelligence pipeline:

**Data Ingestion → Feature Engineering → Hybrid Fraud Modeling → Secure Identity Auth → Private Agentic AI**

### What is Veriscan?
The name **Veriscan** represents the fusion of two core security principles:
- **VERI** (*Verification & Veracity*): A commitment to absolute identity truth through dynamic authentication and data-backed evidence.
- **SCAN** (*Scanning & Surveillance*): The power of autonomous agentic "scans" that explore transaction history, risk profiles, and personalized financial advice.

### Specialized Agents
1. **Security AI Analyst** — Real-time fraud detection, shield monitoring, and anomaly protocols (`guard_agent_local`).
2. **Financial AI Advisor** — Credit health, savings plans, and spending optimization via a ReAct tool loop.
3. **Financial Orchestrator** — Routes questions to Historical Review, Math & Calc, and Current Analyst specialists.
4. **Multimodal Intelligence** — Session-isolated RAG over PDFs, images (OCR + Vision), CSVs, and transcripts.
5. **Spending DNA** — 8-axis behavioral fingerprinting for identity verification and trust scoring.

---

## System Architecture

Veriscan-Cortex is a **privacy-first multi-agent system** with a `src/` package layout. The Streamlit UI talks only to a FastAPI backend; models and agents never run inside the dashboard process.

### Unified Architectural Flow
```mermaid
graph TD
    subgraph Client_Layer [Frontend: User Interface]
        UI[Streamlit Dashboard]
        Session[Session State Manager]
    end

    subgraph API_Gateway [API Gateway & Security]
        FastAPI[FastAPI REST Router]
        AuthA[Auth Auditor]
        SessionStore[(In-Memory Session Store)]
    end

    subgraph Intelligence_Orchestrator [AI Intelligence Layer]
        Router[Agentic Router]
        SecAgent[Security Analyst]
        FinAgent[Financial Advisor + Orchestrator]
        MultiAgent[Multimodal Intelligence]
        DNAAgent[Spending DNA]
        React[ReAct Loop + Tool Registry]
    end

    subgraph Memory_Compute [Private Compute & Context]
        LLM[Meta-Llama-3-8B MLX]
        VisionLLM[LLaVA-1.5-7B MLX]
        RAG[Multimodal RAG + Relevance]
        VectorDB[(ChromaDB: Session Isolated)]
        Traj[(logs/trajectories.jsonl)]
    end

    subgraph Persistence [Data Tier]
        Snowflake[(Snowflake Data Cloud)]
        LocalCSV[(data/csv_data)]
    end

    UI -->|REST: session_id| FastAPI
    FastAPI --> AuthA
    AuthA --> SessionStore
    FastAPI --> Router
    Router --> SecAgent & FinAgent & MultiAgent & DNAAgent
    FinAgent --> React
    SecAgent & FinAgent & MultiAgent --> LLM
    MultiAgent --> VisionLLM
    MultiAgent --> RAG
    RAG --> VectorDB
    React --> Traj
    SecAgent & FinAgent & DNAAgent --> LocalCSV
    LocalCSV -.->|ETL| Snowflake
```

### Security & Session Lifecycle
- **Auth Auditor**: Login requests are scored for risk before a unique `session_id` is issued.
- **State Propagation**: `session_id` isolates agent memory and multimodal uploads for the session lifetime.
- **Zero-Trust RAG**: PDFs, images, and CSVs stay on-device. Embeddings live in session-filtered ChromaDB collections under `.chroma_db_*`.
- **HITL gating**: Sensitive tools (e.g. `challenge_auth`) require explicit `approved=True` via the tool registry.

### Layer Specifications

| Layer | Responsibility | Technology Stack |
|-------|----------------|------------------|
| **Client** | Visualization & chat UI | Streamlit, Plotly Express |
| **Gateway** | Session management & routing | FastAPI, Pydantic v2 |
| **Agents** | ReAct tool use & orchestration | `react_loop`, `tool_registry`, specialist agents |
| **Multimodal** | Image/doc analysis | RapidOCR / Tesseract, PyPDF, LLaVA (optional) |
| **Inference** | Local accelerated LLM | MLX-LM (Apple Silicon) |
| **RAG** | Session-isolated retrieval + rerank | ChromaDB, MiniLM, optional cross-encoder |
| **Observability** | Agent step traces | `logs/trajectories.jsonl` |

---

## Visual Architecture

### How the AI Brain Works
Specialized agents collaborate instead of one monolithic model doing everything.

```mermaid
graph LR
    User([User Query]) --> ModelSelector{Model Selector}
    
    ModelSelector -->|Security| SecAnalyst[Security Analyst]
    ModelSelector -->|Financial| FinOrchestrator[Financial Orchestrator]
    ModelSelector -->|Multimodal| MultiAnalyst[Multimodal Expert]

    subgraph Security_Domain [Security Intelligence]
        direction LR
        SecAnalyst --> Scanner[Scanner]
        SecAnalyst --> Profile[Investigator]
    end

    subgraph Financial_Domain [Multi-Agent Advisory]
        direction LR
        FinOrchestrator --> HistAgent[Historical Review]
        FinOrchestrator --> CalcAgent[Math and Calc]
        FinOrchestrator --> CurrAgent[Current Analyst]
        FinOrchestrator --> ReactLoop[ReAct Advisor]
    end

    subgraph Multimodal_Domain [Document and Visual Intel]
        direction LR
        MultiAnalyst --> RAG[RAG Search + Rerank]
        MultiAnalyst --> Vision[Vision Analyzer]
    end

    Security_Domain --> Report[Security Audit]
    Financial_Domain --> Report2[Synthesized Advisory Report]
    Multimodal_Domain --> Report3[Evidence Analysis Report]
```

| Agent | Role | Specialized Tools |
| :--- | :--- | :--- |
| **Orchestrator** | Routes to specialists; keyword or LLM supervisor mode | Multi-agent synthesis |
| **Historical Review** | Long-term spending patterns | Monthly / YoY comparisons |
| **Math & Calculation** | Totals, averages, forecasts | Cash-flow forecast, surplus optimizer |
| **Current Analyst** | Recent activity windows (30/60/90d) | Price hikes, tax-deductible finder |
| **Financial Advisor** | Bounded ReAct JSON tool loop | Profile, risk, category, DNA tools |
| **Multimodal Expert** | Uploaded evidence | Semantic search, Vision OCR, CSV summarizer |
| **Security Guard** | Fraud / shield queries | High-risk scan, user risk profile |
| **Spending DNA** | Behavioral fingerprint | Profile, compare, challenge (HITL) |

### Multimodal RAG Pipeline
Session-isolated indexing with page-aware PDF extract, OCR fallback, lexical boost, distance thresholds, and optional cross-encoder rerank (`rag_relevance.py`).

```mermaid
graph LR
    subgraph Ingestion [Privacy-First Ingestion]
        Docs[(PDF/CSV/TXT)]
        IMG[(Images)]
        TXN[(Transactions)]
    end

    subgraph RAG_Engine [Local Intelligence]
        direction TB
        OCR[Tesseract/Vision OCR]
        Chunk[Semantic Chunking]
        Embed[all-MiniLM-L6-v2]
        Rank[Relevance filter + rerank]
        Chroma[(ChromaDB: Session Filtered)]
    end

    subgraph Inference [Agentic Response]
        LLM[Meta-Llama-3-8B]
        Vision[LLaVA-1.5-7B]
    end

    Docs & TXN --> Chunk
    IMG --> OCR
    OCR --> Chunk
    Chunk --> Embed
    Embed --> Chroma
    Chroma --> Rank
    Rank --> LLM & Vision
    LLM & Vision --> Reply[Grounded Evidence Analysis]
```

---

## Local AI Intelligence

- **Text LLM**: `Meta-Llama-3-8B-Instruct` (MLX / 4-bit)
- **Vision LLM**: `LLaVA-1.5-7B` (MLX / 4-bit)
- **Inference**: MLX-LM on Apple Silicon
- **Embeddings**: `all-MiniLM-L6-v2`
- **Optional reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Vector DB**: ChromaDB (session metadata filtering)
- **Agent loop**: Bounded ReAct (`max_steps` / `max_tool_calls`) with JSON actions and keyword fallback

---

## Repository Structure

```
Veriscan-Cortex/
├── app/
│   └── streamlit_app.py              # Streamlit dashboard (API client)
├── src/                              # Import root (PYTHONPATH=src)
│   ├── paths.py                      # Shared data / log / Chroma paths
│   ├── agents/
│   │   ├── base.py                   # AgentAction / AgentResult interfaces
│   │   ├── react_loop.py             # Bounded ReAct JSON tool loop
│   │   ├── tool_registry.py          # Tool schemas + HITL gating
│   │   ├── trajectory_log.py         # JSONL run traces
│   │   ├── financial_advisor_agent.py
│   │   ├── financial_orchestrator.py
│   │   ├── historical_review_agent.py
│   │   ├── current_transaction_analyst.py
│   │   ├── transaction_calculation_agent.py
│   │   ├── spending_dna_agent.py
│   │   └── memory.py
│   ├── api/
│   │   ├── main.py                   # FastAPI routers & lifespan
│   │   └── schemas.py                # Pydantic models
│   └── models/
│       ├── local_llm.py              # MLX Llama-3 wrapper
│       ├── vision_llm.py             # LLaVA wrapper
│       ├── guard_agent_local.py      # Security analyst facade
│       ├── rag_engine_local.py       # Knowledge-base RAG
│       ├── multimodal_rag.py         # Session-isolated multimodal RAG
│       ├── rag_relevance.py          # Score, filter, rerank, grounded prompts
│       ├── agent_tools_data.py       # Risk / profile data tools
│       └── auth_store.py             # Demo user store
├── data/
│   ├── csv_data/                     # Transactions, scores, DNA, QA
│   ├── ic3_2024_csvs/                # IC3 crime statistics
│   ├── pdf_data/                     # RAG source PDFs
│   └── user_uploads/                 # Per-session upload sandbox
├── scripts/                          # ETL & synthetic generators
├── sql/                              # Snowflake DDL + analytical queries
├── configs/ingest_config.yaml
├── tests/                            # Offline agent & RAG evaluations
├── docs/                             # Capstone report
├── logs/                             # pipeline_logs.csv, trajectories.jsonl
├── artifacts/                        # Trained model binaries (.joblib)
├── notebooks/
├── poster/
├── run_api.sh                        # FastAPI (reload watches src/ only)
├── run_dashboard.sh                  # Streamlit dashboard
├── streamlit_app.py                  # Thin backward-compatible launcher
├── pyproject.toml                    # Package metadata (src layout)
└── requirements.txt
```

---

## Snowflake Data Platform

| Table | Purpose |
|-------|--------|
| `RAW_TRANSACTIONS` | Source transaction data |
| `TRANSACTION_FEATURES` | 19 engineered signals |
| `FRAUD_SCORES` | ML + heuristic risk scores |
| `AUTH_PROFILES` | User security profiles |
| `PIPELINE_RUNS` | Pipeline audit trail |

**Views:** `ENRICHED_TRANSACTIONS`, `USER_RISK_DASHBOARD`

See `sql/create_tables.sql` and `sql/analytical_queries.sql`.

---

## Microservices Architecture

```mermaid
graph LR
    subgraph Frontend [Streamlit Dashboard Port 8502]
        UI[Dashboard UI]
    end

    subgraph Backend [FastAPI Backend Port 8000]
        API[REST API Router]
        Agent[Specialized AI Agents]
        RAG[RAG Engine + Relevance]
    end

    UI -->|POST /api/advisor/chat| API
    UI -->|POST /api/security/chat| API
    UI -->|POST /api/rag/chat| API
    UI -->|GET /api/dna/profile/ID| API
    UI -->|GET /api/user/ID/risk| API
    API --> Agent
    API --> RAG
```

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Health check & loaded services |
| `POST` | `/api/auth/login` | Demo login → `session_id` + risk scoring |
| `GET` | `/api/fraud/high-risk` | Top riskiest transactions |
| `GET` | `/api/user/{user_id}/risk` | User risk profile |
| `POST` | `/api/rag/query` | Knowledge-base semantic search |
| `POST` | `/api/rag/upload` | Multi-file upload & session indexing |
| `POST` | `/api/rag/chat` | Grounded chat over uploaded docs |
| `POST` | `/api/advisor/chat` | Financial advisor (ReAct / orchestrator) |
| `GET` | `/api/advisor/users` | Available advisor user IDs |
| `POST` | `/api/advisor/reset` | Reset advisor conversation memory |
| `POST` | `/api/security/chat` | Security analyst chat |
| `GET` | `/api/dna/profile/{user_id}` | Spending DNA fingerprint |
| `POST` | `/api/dna/compare` | Compare DNA profiles |
| `POST` | `/api/dna/challenge` | DNA challenge (HITL-gated) |

Interactive docs: `http://127.0.0.1:8000/docs`

---

## Quick Start

### 1. Requirements
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python **3.11+** (see `pyproject.toml`)
- OCR: Tesseract (`brew install tesseract`) **or** the bundled RapidOCR Python fallback (`rapidocr-onnxruntime`)
- Optional Vision: `pip install mlx-vlm` (LLaVA via MLX; do not set `VERISCAN_SKIP_VISION=1`)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare / refresh agent datasets (optional)
CSV products under `data/csv_data/` are already present for demo. To regenerate:
```bash
python scripts/generate_financial_advisor_dataset.py
python scripts/generate_spending_dna_dataset.py
python scripts/generate_cfpb_dataset.py
python scripts/feature_engineering.py
python scripts/fix_agent_data.py
```

### 4. Launch the API Backend
```bash
./run_api.sh
# equivalent:
# PYTHONPATH=src python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir src
```

### 5. Launch the Dashboard (separate terminal)
```bash
./run_dashboard.sh --server.port 8502
# equivalent:
# PYTHONPATH=src streamlit run app/streamlit_app.py --server.port 8502
```

On first run, Llama-3 (~4.9GB) downloads automatically via Hugging Face.

### Trajectory logging
Agent runs append to `logs/trajectories.jsonl` by default. Disable with:
```bash
export VERISCAN_TRAJECTORY_LOG=0
```

---

## Evaluation

Offline suites live under `tests/` (they prepend `src/` to `sys.path`):

```bash
PYTHONPATH=src python tests/evaluate_rag_local.py
PYTHONPATH=src python tests/evaluate_agent_local.py
PYTHONPATH=src python tests/evaluate_advisor_agent_local.py
PYTHONPATH=src python src/models/evaluate_multimodal_rag_local.py
```

---

## Reproducibility & Deployment

| Aspect | Details |
|--------|--------|
| **Environment** | Python 3.11+, `requirements.txt` |
| **Layout** | `src/` packages; launch via `run_*.sh` or `PYTHONPATH=src` |
| **Model Versioning** | `fraud_model_rf.joblib` + `encoders.joblib` (`random_state=42`) |
| **Dataset** | Synthetic + CFPB/IC3-aligned CSVs under `data/` |
| **Vector Store** | ChromaDB under `.chroma_db_local` / `.chroma_db_multimodal` |
| **Config** | `configs/ingest_config.yaml` (env overrides supported) |
| **Secrets** | Credentials via environment variables; `.env` gitignored |

## Project Data Realism

- **78% Online Skew**: Simulated fraud losses concentrated in Online Shopping / Electronics.
- **Scale**: ~**90,000 transactions** across **1,000** user archetypes.
- **DNA Fingerprinting**: Each user maps to an 8-axis behavioral vector for trust scoring.

### Source
https://consumerfed.org/press_release/americans-estimated-to-lose-119-billion-annually-to-online-scams/

Full write-up: [`docs/VERISCAN-CORTEX-FINAL-REPORT.md`](docs/VERISCAN-CORTEX-FINAL-REPORT.md)
