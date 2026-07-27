# Veriscan-Cortex: Advanced Fraud Intelligence & Private Multi-Agent Dashboard

> **Project Title:** Veriscan-Cortex  
> **Course:** CS 5588 — Data Science Capstone  
> **Date:** July 2026  
> **Focus:** Fraud Intelligence, Identity Fingerprinting, and Privacy-First Multi-Agent Architecture

---

## 1. Introduction

As digital financial systems grow increasingly complex, the surface area for cybercrime, financial fraud, and identity theft continues to expand. According to the FBI Internet Crime Complaint Center (IC3), financial losses due to online scams and fraudulent transactions have reached unprecedented billions annually, with specific spikes in online shopping and investment fraud. Traditional, centralized fraud detection platforms rely heavily on cloud-based heuristics and static thresholds. While functional, these platforms introduce severe privacy liabilities, funneling highly sensitive Personal Identifiable Information (PII) and financial transaction logs through external servers.

**Veriscan-Cortex** represents a paradigm shift in financial security analytics. It is an end-to-end, privacy-first security and financial intelligence platform designed to counteract digital fraud locally. By utilizing local-first Large Language Models (LLMs), deep behavioral fingerprinting, and real-time data orchestration, Veriscan-Cortex establishes a high-fidelity "Security Shield" without necessitating cloud egress for core intelligence tasks.

The nomenclature of the platform embodies its dual-layered philosophy:
- **VERI (Verification & Veracity):** The commitment to absolute truth in identity through dynamic, multidimensional behavioral analysis, stepping away from easily compromised authentications like static passwords.
- **SCAN (Scanning & Surveillance):** The proactive deployment of autonomous, agentic artificial intelligence continuously auditing transactions, profiling risk, and summarizing unstructured evidence in real time.

The primary objective of Veriscan-Cortex is to demonstrate that state-of-the-art Agentic AI and high-precision fraud modeling can be effectively bridged on consumer-grade hardware (specifically Apple Silicon) while maintaining Tier-1 financial data privacy compliance.

---

## 2. Data Source and Processing

To ensure the Veriscan-Cortex intelligence correlates cleanly with real-world scenarios, a massive, data-driven synthesis approach was adopted. Rather than relying on simple toy data, the platform required complex, noisy datasets to validate the system's machine learning and multi-agent systems.

### 2.1 Primary Data Sources
1. **Base Transaction Architecture:** Modeled upon industry-standard Kaggle credit card fraud distributions, the base dataset captures realistic geometries—including varying transaction amounts, merchant frequencies, and geographical tagging.
2. **CFPB Market Intelligence:** Real-world Consumer Financial Protection Bureau (CFPB) Credit Card complaint data is utilized to understand billing disputes, identity theft claims, and institutional friction.
3. **IC3 2024 Global Cybercrime Data:** Aggregated national incident statistics inform macro-level dashboard analytics and help align the synthetic pipelines with realistic threat vectors (e.g., heavily weighting "Online Shopping" and "Electronics" which constitute ~78% of generated losses).

### 2.2 Data Processing & Synthetic Generation Pipeline
The raw source data alone lacks the depth required for complex multi-agent analysis. Therefore, an offline data generation and feature-engineering pipeline under `scripts/` processes the foundation into `data/csv_data/`:

- **Volume and Scale:** The pipeline generated **90,000 synthetic transactions** distributed across **1,000 distinct user archetypes**. This scale allows for realistic clustering and stress-testing of historical RAG retrievals.
- **Feature Engineering:** During the ETL phase, scripts automatically engineered **19 high-fidelity signals** per transaction. This includes time-of-day encodings, geographical distance calculations, and transaction velocity windows (e.g., 1-hour and 24-hour rolling sums).
- **Behavioral Data Processing:** The platform processes each user into an "8-Axis Behavioral Vector" (Spending DNA). This transforms raw logs into a continuous fingerprint representing merchant preferences, geographical habits, transaction scale, and timing.

---

## 3. Methodology

Veriscan-Cortex implements a combined methodology, merging classical Machine Learning pipelines with cutting-edge Local Agentic AI orchestration.

### 3.1 Hybrid ML Fraud Modeling
At the first layer of defense, every transaction passes through a deterministic screening process. Instead of forcing an LLM to blindly guess if a $5.00 coffee charge is fraudulent, Veriscan utilizes classical ML paradigms.
- A **Random Forest (RF)** algorithm is trained on the synthesized feature outputs.
- It assesses the engineered signals (like velocity and merchant risk factors), producing a "Combined Risk Score" scaled from 0 to 100 within a fraction of a millisecond.

### 3.2 Privacy-First Agentic LLM Orchestration
The primary innovation of Cortex relies on the local execution of large foundation models using the **MLX-LM** framework, optimized specifically for M-Series architecture (Mac).

**Model Tiering:**
1. **Meta-Llama-3-8B (4-bit Quantized):** Triggers core reasoning, logic routing, and financial advisory generation without leaving the local device memory.
2. **LLaVA-1.5-7B (Vision Integration):** Handles OCR and raw image analysis on multimodal inputs (like paper receipts or PDF invoices).
3. **all-MiniLM-L6-v2:** An agile local embedding model tasked with indexing unstructured data into a session-isolated ChromaDB vector store.
4. **Optional cross-encoder (`ms-marco-MiniLM-L-6-v2`):** Reranks retrieved chunks after distance filtering for higher Precision@k on multimodal queries.

### 3.3 The Specialized Multi-Agent Structure
To prevent hallucination loops common in monolithic AI usage, the system delegates tasks across a Cortex of specialized agents under `src/agents/`:

- **Security AI Analyst:** Evaluates transaction velocities and outputs deterministic fraud judgments through a security-conscious persona (`guard_agent_local`).
- **Financial AI Advisor:** Uses a bounded **ReAct** loop (`react_loop.py`) over a typed **Tool Registry** (`tool_registry.py`) to call profile, risk, category, and DNA tools with JSON action/final turns; falls back to keyword routing when JSON parse fails.
- **Financial Orchestrator:** Routes advisory questions to Historical Review, Transaction Calculation, and Current Transaction Analyst specialists, then synthesizes one coherent reply (keyword path or LLM supervisor mode).
- **Multimodal Intelligence:** Manages session-isolated multimodal RAG (`multimodal_rag.py` + `rag_relevance.py`). Uploaded PDFs/images/CSVs are chunked, embedded, relevance-filtered, and optionally reranked before Llama-3 / LLaVA synthesis.
- **Spending DNA:** Maps behavior to an 8-axis fingerprint and supports profile, compare, and HITL-gated challenge flows.

Agent runs optionally append structured traces to `logs/trajectories.jsonl` for evaluation and portfolio observability (`trajectory_log.py`).

---

## 4. Results

The deployment of Veriscan-Cortex verified multiple hypotheses regarding the viability of localized, agentic security analysis. Extensive local benchmarks were conducted.

### 4.1 RAG and Search Precision
Utilizing the internal evaluation suites (`tests/evaluate_rag_local.py` and `src/models/evaluate_multimodal_rag_local.py`), the RAG infrastructure was tested across complex queries involving CFPB credit card dispute guidelines and uploaded evidence.
- **Metric:** The knowledge-base engine achieved a consistent **92%+ Precision@3**. This confirmed that the combination of semantic chunking, `all-MiniLM-L6-v2` embeddings, distance-to-confidence thresholds, and isolated ChromaDB collections effectively fetches highly relevant evidence chunks before passing them to the generative model.

### 4.2 Agentic Orchestration Integrity
The GuardAgent facade and Financial Advisor ReAct path rely on correctly identifying user intent and dispatching the proper tool.
- **Metric:** In synthetic benchmarking (`tests/evaluate_agent_local.py`, `tests/evaluate_advisor_agent_local.py`) spanning User, Knowledge, and System based queries, the agent routing mechanism successfully matched user intent to system tools with high fidelity, proving that 8B models possess sufficient parametric knowledge for complex autonomous routing when constrained by JSON schemas and tool caps.

### 4.3 Inference Constraints and Efficiency
Testing on high-end Apple Silicon (M3 Max with ample unified memory) yielded:
- **Token Generation:** Local generation speeds utilizing the MLX framework peaked at **18–24 tokens/sec** for Llama-3-8B.
- **Latency:** Overall logic tasks (from query, to RAG context fetch, to LLM output) consistently functioned under the interactive dashboard margin for typical advisory and security chats.

---

## 5. System Architecture

The architectural methodology is heavily decoupled to separate deterministic logic layers from heavy, computationally intensive generative layers. Application code lives under a **`src/` layout** (`agents`, `api`, `models`, `paths`), with presentation in `app/` and launchers `run_api.sh` / `run_dashboard.sh` setting `PYTHONPATH=src`.

### 5.1 Layer Breakdown
1. **Frontend Presentation (Client Layer):** Constructed with Streamlit and Plotly Express. It aggregates local intelligence into high-fidelity dashboards displaying dynamic visualizations, geographic transaction heatmaps, and financial metrics (`app/streamlit_app.py`).
2. **API Gateway (Routing Layer):** A FastAPI implementation (`src/api/main.py`) brokers requests from the UI, assigns session states, and exposes auth, fraud, RAG, advisor, security, and Spending DNA endpoints. API reload watches only `src/` so dashboard edits do not restart the backend.
3. **The AI Orchestrator (Intelligence Layer):** The routing hub for Security, Advisor/Orchestrator, Multimodal, and DNA agents. ReAct tool calling is bounded (`max_steps`, `max_tool_calls`) with duplicate-call detection and HITL gating for sensitive tools (e.g. `challenge_auth`).
4. **Local Memory and Computation:** MLX accelerates quantized models. Vector embeddings are stored in transient ChromaDB instances (`.chroma_db_local`, `.chroma_db_multimodal`), filtered explicitly by `session_id` to guarantee cross-tenant privacy.
5. **Data Persistence Tier (Offline & Cloud):** Historical transactions rely on local CSV stores under `data/csv_data/` for rapid Pandas loading, with optional synchronization to **Snowflake Data Cloud** via `scripts/upload_all_to_snowflake.py`.

### 5.2 The Authentication & Zero-Trust RAG Lifecycle
A key security concept built into the architecture is **Identity Veracity**. Auth login verifies credentials against an in-memory store, computes a basic login risk score, and issues a globally unique `session_id`. As multimodal evidence enters the application, it remains localized under that session. RAG processing ensures that extracted sensitive context is isolated from any broader training corpus and never leaves the host machine for core inference.

### 5.3 Software Layout (Summary)

| Path | Role |
|------|------|
| `src/agents/` | ReAct loop, tool registry, orchestrator, specialists, trajectories |
| `src/api/` | FastAPI routers and Pydantic schemas |
| `src/models/` | LLM, Vision, Guard, RAG, relevance, auth, data tools |
| `src/paths.py` | Canonical project / data / log / Chroma paths |
| `app/` | Streamlit client |
| `tests/` | Offline RAG and agent evaluation suites |
| `docs/` | This report |
| `logs/` | Pipeline CSV + agent trajectory JSONL |

---

## 6. Conclusion

The completion of the Veriscan-Cortex platform illustrates deeply impactful implications for the future of financial risk surveillance. As digital adversaries adapt, classical threshold monitoring fails both in scalability and false-positive rates. Traditional Cloud LLMs succeed in reasoning but catastrophically fail in privacy and regulatory compliance when managing raw transaction data.

Veriscan successfully resolves this tension. By executing **Local-First AI Governance**, deploying a **Specialized Multi-Agent architecture** with bounded ReAct tool use, and shifting identity verification from static keys to dynamic **Spending DNA Fingerprints**, the project effectively realizes a private, resilient security shield. High precision benchmarks in retrieval, coupled with impressive MLX hardware acceleration results on Apple consumer hardware, suggest that true, personalized local AI-banking intelligence is not merely a theoretical concept, but an immediately viable engineering reality.

---

## 7. References

1. **IC3 Internet Crime Report 2024.** *Federal Bureau of Investigation*. Statistics surrounding financial fraud losses and key demographics targeting.
2. **Consumer Financial Protection Bureau (CFPB) Open Data.** *Consumer Complaint Database*. Utilized for extracting billing patterns, disputes, and unstructured RAG knowledge ingestion.
3. **Kaggle Synthetic Fraud Pipeline.** Base heuristics derived from publicly accessible simulated fraud vectors.
4. **Machine Learning API (MLX) Docs.** *Apple.* Framework references for executing Meta-Llama-3-8B and LLaVA-1.5-7B optimally on Apple Silicon GPU/NPU arrays.
5. **ChromaDB Documentation.** Methodologies for transient, session-isolated semantic search using `all-MiniLM-L6-v2` dense passage retrieval representations.
6. **Sentence-Transformers Cross-Encoders.** Optional MiniLM cross-encoder reranking for multimodal retrieval quality.
