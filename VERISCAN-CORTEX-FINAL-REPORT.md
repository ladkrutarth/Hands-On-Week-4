# Veriscan-Cortex: Advanced Fraud Intelligence & Private Multi-Agent Dashboard

> **Project Title:** Veriscan-Cortex  
> **Course:** CS 5588 — Data Science Capstone  
> **Date:** April 2026  
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
2. **CFPB Market Intelligence:** Real-world Consumer Financial Protection Bureau (CFPB) Credit Card complaint data (over 50,000 complaints) is utilized to understand billing disputes, identity theft claims, and institutional friction. 
3. **IC3 2024 Global Cybercrime Data:** Aggregated national incident statistics inform macro-level dashboard analytics and help align the synthetic pipelines with realistic threat vectors (e.g., heavily weighting "Online Shopping" and "Electronics" which constitute ~78% of generated losses).

### 2.2 Data Processing & Synthetic Generation Pipeline
The raw source data alone lacks the depth required for complex multi-agent analysis. Therefore, a massive offline data generation and feature-engineering pipeline processes the foundation:

- **Volume and Scale:** The pipeline generated **90,000 synthetic transactions** distributed across **1,000 distinct user archetypes**. This scale allows for realistic clustering and stress-testing of historical RAG retrievals.
- **Feature Engineering:** During the ETL (Extract, Transform, Load) phase, scripts automatically engineered **19 high-fidelity signals** per transaction. This includes time-of-day encodings, geographical distance calculations, and transaction velocity windows (e.g., 1-hour and 24-hour rolling sums).
- **Behavioral Data Processing:** The data represents specific behavioral tendencies. The platform processes each transaction into an "8-Axis Behavioral Vector" (Spending DNA). This transforms raw logs into a continuous "fingerprint" representing merchant preferences, geographical habits, transaction scale, and timing.

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

### 3.3 The Specialized Multi-Agent Structure
To prevent "hallucination loops" common in monolithic AI usage, the system delegates tasks across a "Cortex" of specialized agents:

- **🛡️ Security AI Analyst:** Evaluates the transaction velocities and outputs deterministic fraud judgments through a security-conscious persona.
- **💰 Financial AI Advisor:** Reviews 60–90 days of personal user context to highlight subscription creep, overspending trends, and fiscal optimization.
- **🧬 Multimodal Intelligence:** Manages the system's "Local RAG Engine". If a user questions a policy, this engine retrieves relevant CFPB documents or internal policies, passing only the most pertinent chunks up to Llama-3 for final synthesis.
- **🧬 Spending DNA:** Maps behavior. It continually processes a user's transaction history to update their multidimensional fingerprint, raising flags if sudden spatial or categorical deviation occurs (e.g., an account transitioning abruptly from domestic grocery purchases to international software licenses).

---

## 4. Results

The deployment of Veriscan-Cortex verified multiple hypotheses regarding the viability of localized, agentic security analysis. Extensive local benchmarks were conducted.

### 4.1 RAG and Search Precision
Utilizing the internal evaluation suites (`evaluate_rag_local.py`), the RAG (Retrieval-Augmented Generation) infrastructure was tested across complex queries involving CFPB credit card dispute guidelines.
- **Metric:** The engine achieved a consistent **92%+ Precision@3**. This confirmed that the combination of semantic chunking, `all-MiniLM-L6-v2` embeddings, and isolated ChromaDB collections effectively fetches highly relevant evidence chunks before passing them to the generative model.

### 4.2 Agentic Orchestration Integrity
The GuardAgent Facade relies on correctly identifying user intent and dispatching the proper tool (e.g., `query_rag` vs. `get_user_risk_profile`). 
- **Metric:** In synthetic benchmarking (`evaluate_agent_local.py`) spanning "User", "Knowledge", and "System" based queries, the agent routing mechanism successfully matched user intent to system tools with high fidelity, proving that 8B models possess sufficient parametric knowledge for complex autonomous routing.

### 4.3 Inference Constraints and Efficiency
Testing on High-end Apple Silicon (M3 Max with ample unified memory) yielded:
- **Token Generation:** Local generation speeds utilizing the MLX framework peaked at **18-24 tokens/sec** for Llama-3-8B.
- **Latency:** Overall logic tasks (from query, to RAG context fetch, to LLM output) consistently functioned under the 2-second margin, classifying the platform as fit for interactive dashboard usage.

---

## 5. System Architecture

The architectural methodology is heavily decoupled to separate deterministic logic layers from heavy, computationally intensive generative layers. 

### 5.1 Layer Breakdown
1. **Frontend Presentation (Client Layer):** Constructed with Streamlit and Plotly Express. It aggregates local intelligence into high-fidelity "Mobbin-inspired" dashboards displaying dynamic visualizations, geographic transaction heatmaps, and financial metrics.
2. **API Gateway (Routing Layer):** A high-speed FastAPI implementation functioning as the backend. It brokers requests from the UI and assigns strict session states. It intercepts auth commands to instantiate global `session_id` tags.
3. **The AI Orchestrator (Intelligence Layer):** The routing hub for the core agents (Security, Advisor, Multimodal, DNA). Based on the REST API endpoint struck, this layer spins up the necessary sub-agents and tool chains.
4. **Local Memory and Computation:** At the center of the hardware execution is the MLX acceleration loading quantized models. Vector embeddings are stored in a transient ChromaDB instance, effectively filtering and isolating context explicitly by user-session to guarantee absolute cross-tenant privacy.
5. **Data Persistence Tier (Offline & Cloud):** The 90k processed historical transactions heavily rely on local CSV stores prioritizing rapid Python Pandas loading, but synchronizes analytics up to **Snowflake Data Cloud** via backend ETL integration for data warehouse warehousing functionality.

### 5.2 The Authentication & Zero-Trust RAG Lifecycle
A key security concept built into the architecture is **Identity Veracity**. A dedicated Auth Auditor verifies logins. As multimodal evidence (such as a picture of a passport or a bank PDF) enters the application, it remains completely localized over the `session_id`. RAG processing ensures that extracted sensitive context is isolated from broader LLM training corpus.

---

## 6. Conclusion

The completion of the Veriscan-Cortex platform illustrates deeply impactful implications for the future of financial risk surveillance. As digital adversaries adapt, classical threshold monitoring fails both in scalability and false-positive rates. Traditional Cloud LLMs succeed in reasoning but catastrophically fail in privacy and regulatory compliance when managing raw transaction data.

Veriscan successfully resolves this tension. By executing **Local-First AI Governance**, deploying a **Specialized Multi-Agent architecture**, and shifting identity verification from static keys to dynamic **Spending DNA Fingerprints**, the project effectively realizes a private, resilient security shield. High precision benchmarks in retrieval, coupled with impressive MLX hardware acceleration results on Apple consumer hardware, suggest that true, personalized local AI-banking intelligence is not merely a theoretical concept, but an immediately viable engineering reality.

---

## 7. References

1. **IC3 Internet Crime Report 2024.** *Federal Bureau of Investigation*. Statistics surrounding financial fraud losses and key demographics targeting.
2. **Consumer Financial Protection Bureau (CFPB) Open Data.** *Consumer Complaint Database*. Utilized for extracting billing patterns, disputes, and unstructured RAG knowledge ingestion.
3. **Kaggle Synthetic Fraud Pipeline.** Base heuristics derived from publicly accessible simulated fraud vectors. 
4. **Machine Learning API (MLX) Docs.** *Apple.* Framework references for executing Meta-Llama-3-8B and LLaVA-1.5-7B optimally on Apple Silicon GPU/NPU arrays. 
5. **ChromaDB Documentation.** Methodologies for transient, session-isolated semantic search using `all-MiniLM-L6-v2` dense passage retrieval representations.
