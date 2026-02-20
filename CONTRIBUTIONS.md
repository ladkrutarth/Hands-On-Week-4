# CONTRIBUTIONS.md — GraphGuard Team Responsibilities

> **Note:** Update team member names before final submission.

## Team Members & Responsibilities

| Team Member | Role | Components Owned |
|-------------|------|------------------|
| **Member 1** | Data Engineer | Data ingestion scripts, Snowflake schema design, pipeline logging |
| **Member 2** | ML Engineer | Feature engineering, fraud scoring model (IsolationForest + rules) |
| **Member 3** | Application Developer | Streamlit dashboard, authentication UI, visualization |
| **Member 4** | DevOps / Analytics | SQL analytical queries, deployment, monitoring, documentation |

---

### Member 1 — Data Engineer
**Files owned:**
- `scripts/ingest_csv_to_snowflake.py` — CSV to Snowflake ingestion pipeline
- `scripts/ingest_config.yaml` — ingestion configuration
- `sql/create_tables.sql` — Snowflake DDL for all 5 tables
- `pipeline_logs.csv` — pipeline execution log

**Responsibilities:**
- Designed the raw data ingestion pipeline with dry-run mode
- Created Snowflake table schemas (RAW_TRANSACTIONS, PIPELINE_RUNS)
- Implemented pipeline logging to track every run
- Ensured reproducibility with config-driven connections

---

### Member 2 — ML Engineer
**Files owned:**
- `scripts/feature_engineering.py` — feature computation pipeline
- `models/fraud_model.py` — fraud scoring (rules + IsolationForest)
- `models/auth_decision.py` — authentication decision module + Gemini integration
- `models/rag_engine.py` — RAG engine (ChromaDB + Gemini embeddings)
- `models/gemini_question_gen.py` — dynamic question generation (Gemini LLM)

**Responsibilities:**
- Designed 19 transaction features (amount, velocity, geographic, time, category)
- Built hybrid fraud model combining rule-based heuristics and ML
- Implemented z-score anomaly detection, velocity checks, and geographic risk
- Created authentication decision logic with adaptive security levels
- Built RAG engine with ChromaDB vector store and Gemini embeddings
- Implemented dynamic question generation using Gemini 2.0 Flash LLM

---

### Member 3 — Application Developer
**Files owned:**
- `streamlit_app.py` — main Streamlit application (Gemini + RAG integrated)
- `requirements.txt` — project dependencies

**Responsibilities:**
- Built the Streamlit dashboard with authentication, fraud detection, dashboard, and RAG explorer tabs
- Integrated Google Gemini LLM for AI-powered question generation and answer verification
- Integrated RAG for context-aware fraud analysis and free-form Q&A
- Designed the user interface with premium dark-themed CSS and visualization components

---

### Member 4 — DevOps / Analytics
**Files owned:**
- `sql/analytical_queries.sql` — 8 analytical SQL queries
- `docs/architecture_diagram.png` — system architecture diagram
- `README.md` — project documentation
- `CONTRIBUTIONS.md` — this file

**Responsibilities:**
- Wrote analytical queries for spending patterns, anomaly detection, merchant risk
- Created the architecture diagram showing end-to-end pipeline
- Documented system workflow, architecture, and extensions
- Set up monitoring and evaluation logging
