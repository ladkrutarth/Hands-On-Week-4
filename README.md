# GraphGuard — AI-Powered Fraud Detection & Dynamic Authentication System

> **Course:** CS 5588 — Data Science Capstone | **Date:** February 2026

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Pipeline Workflow](#pipeline-workflow)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Component Details](#component-details)
- [Monitoring & Logging](#monitoring--logging)
- [Implemented Extensions](#implemented-extensions)
- [Demo Video](#demo-video)

---

## Project Overview

GraphGuard is an end-to-end **fraud detection and dynamic authentication system** that processes financial transaction data through a complete data pipeline:

**Data Sources → Cloud Warehouse (Snowflake) → Feature Engineering → Modeling / Decision Layer → Streamlit Dashboard**

The system uses a hybrid approach combining **rule-based heuristics**, **statistical anomaly detection** (z-score, velocity checks), and **machine learning** (IsolationForest) to score transaction risk in real-time. An AI-powered authentication module uses **Google Gemini LLM** with **RAG (Retrieval-Augmented Generation)** over project data to dynamically generate security challenges adapted to each user's risk profile.

### Key Capabilities
- **Real Kaggle dataset** — [kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) with 10 users sampled from ~1.8M Sparkov credit card transactions
- **Ground-truth fraud labels** (`IS_FRAUD_ACTUAL`) enable real model evaluation: Precision, Recall, F1
- **19 engineered features** per transaction (amount, velocity, geographic, temporal, categorical)
- **Hybrid fraud scoring** with weighted combination of 5 risk signals
- **Adaptive authentication** with meaningful questions about stores, locations, and categories (no risk scores)
- **Transaction habit prediction model** that learns each user's spending patterns
- **Production-ready RAG pipeline** with query rewriting, re-ranking, confidence scoring, and automated evaluation
- **Full pipeline monitoring** with execution logs and performance metrics

---

## System Architecture

![GraphGuard Architecture](docs/architecture_diagram.png)

```
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐
│ CSV Data     │────▶│ Ingestion    │────▶│ Snowflake         │
│ (301 txns)   │     │ Script       │     │ RAW_TRANSACTIONS  │
└──────────────┘     └──────────────┘     └─────────┬─────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                   Feature Engineering                        │
│  19 features: amount z-score, velocity (1h/24h/7d),         │
│  category risk, geographic entropy, time-of-day, …          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌─────────────────┐   ┌─────────────────┐
          │ Fraud Model     │   │ Auth Decision   │
          │ IsolationForest │   │ Risk Profiles   │
          │ + Rule Scoring  │   │ + Gemini LLM    │
          └────────┬────────┘   └────────┬────────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
          ┌───────────────────────────────────┐
          │        RAG Engine (ChromaDB)      │
          │  Embeds CSV + SQL → Vector Store  │
          └──────────────┬────────────────────┘
                         ▼
              ┌───────────────────────────┐
              │  Streamlit Dashboard      │
              │  🔑 Auth  🚨 Fraud       │
              │  📊 Dashboard  🔍 RAG Q&A│
              └───────────────────────────┘
                              │
              ┌───────────────────────────┐
              │  Gemini LLM (2.0 Flash)   │
              │  Dynamic Questions + RAG  │
              │  Answer Verification      │
              └───────────────────────────┘
                              │
              ┌───────────────────────────┐
              │  Monitoring & Logging     │
              │  pipeline_logs.csv        │
              └───────────────────────────┘
```

---

## Pipeline Workflow

| Stage | Script / File | Input | Output |
|-------|--------------|-------|--------|
| **0. Kaggle Adapter** | `scripts/load_kaggle_data.py` | `dataset/raw/fraudTrain.csv` | `transactions_kaggle.csv` |
| **1. Ingestion** | `scripts/ingest_csv_to_snowflake.py` | CSV file | Snowflake `RAW_TRANSACTIONS` |
| **2. Feature Engineering** | `scripts/feature_engineering.py` | Raw transactions | `features_output.csv` (19 features) |
| **3. Fraud Scoring** | `models/fraud_model.py` | Feature CSV | `fraud_scores_output.csv` + Precision/Recall/F1 |
| **4. Auth Profiling** | `models/auth_decision.py` | Fraud scores | `auth_profiles_output.csv` |
| **5. Dashboard** | `streamlit_app.py` | All outputs | Interactive web app (Gemini + RAG) |

### Reproducing the Pipeline
```bash
# Step 1: Ingest data (dry-run mode — no Snowflake credentials needed)
python scripts/ingest_csv_to_snowflake.py --dry-run

# Step 2: Compute features
python scripts/feature_engineering.py

# Step 3: Score transactions
python models/fraud_model.py

# Step 4: Generate auth profiles
python models/auth_decision.py

# Step 5: Launch dashboard
streamlit run streamlit_app.py
```

---

## Repository Structure

```
Hands-On-Week-4/
├── README.md                          # This file
├── CONTRIBUTIONS.md                   # Team member responsibilities
├── requirements.txt                   # Python dependencies
├── streamlit_app.py                   # Main Streamlit dashboard (Gemini + RAG)
├── pipeline_logs.csv                  # Pipeline execution log
│
├── scripts/                          # Data pipeline scripts
│   ├── ingest_csv_to_snowflake.py     # CSV → Snowflake ingestion
│   ├── ingest_config.yaml             # Connection configuration
│   └── feature_engineering.py         # Feature computation (19 features)
│
├── sql/                              # SQL schemas & queries
│   ├── create_tables.sql              # DDL for 5 Snowflake tables
│   └── analytical_queries.sql         # 8 analytical queries
│
├── models/                           # Modeling / AI / Decision layer
│   ├── fraud_model.py                 # Hybrid fraud scoring model
│   ├── auth_decision.py               # Auth decision + Gemini integration
│   ├── rag_engine.py                  # Production RAG engine (ChromaDB + hybrid retrieval)
│   ├── rag_evaluator.py               # Automated RAG accuracy evaluation (15 tests)
│   ├── gemini_question_gen.py         # Dynamic question generation (Gemini LLM)
│   └── habit_model.py                 # Transaction habit prediction model
│
└── docs/                             # Documentation assets
    └── architecture_diagram.png       # System architecture diagram
```

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Enable AI Features (Optional)
```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
# Or enter it in the Streamlit sidebar at runtime
```

### Run the Dashboard
```bash
streamlit run streamlit_app.py
# Open http://localhost:8501
```

### Run the Full Pipeline
```bash
# Feature engineering + scoring (no Snowflake needed)
python scripts/feature_engineering.py
python models/fraud_model.py
python models/auth_decision.py
```

---

## Component Details

### Data Ingestion (`scripts/`)
- Reads transaction CSV (301 rows, 10 users, 8 categories)
- Validates schema against expected columns
- Supports `--dry-run` mode for testing without Snowflake
- Logs every pipeline run to `pipeline_logs.csv`

### SQL Layer (`sql/`)
**5 tables** defined in `create_tables.sql`:
| Table | Purpose |
|-------|---------|
| `RAW_TRANSACTIONS` | Ingested transaction data |
| `TRANSACTION_FEATURES` | Computed features (19 columns) |
| `FRAUD_SCORES` | Model output risk scores |
| `AUTH_EVENTS` | Authentication event log |
| `PIPELINE_RUNS` | Pipeline execution metadata |

**8 analytical queries** in `analytical_queries.sql`:
User spending summaries, anomaly detection (>2σ), velocity checks, merchant risk profiles, geographic anomalies, daily trends, and category risk weights.

### Feature Engineering (`scripts/feature_engineering.py`)
Computes **19 features** per transaction:
- **Amount**: z-score, is_high_value flag
- **User-level**: avg/std spend, total transactions
- **Velocity**: transaction counts in 1h / 24h / 7d windows
- **Category**: risk weight (Jewelry=0.9, Coffee=0.05)
- **Time**: hour of day, day of week, is_weekend
- **Geographic**: is_new_location, location entropy

### Fraud Model (`models/fraud_model.py`)
Hybrid scoring with **5 weighted signals**:
| Signal | Weight | Method |
|--------|--------|--------|
| Z-Score Flag | 25% | Continuous scoring from amount deviation |
| Velocity Flag | 20% | Burst detection (1h, 24h thresholds) |
| Category Risk | 15% | Category-based risk weights |
| Geographic Risk | 15% | New location + location entropy |
| IsolationForest | 25% | Unsupervised anomaly detection |

Output: combined score 0.0–1.0 → risk levels: LOW / MEDIUM / HIGH / CRITICAL

### Auth Decision (`models/auth_decision.py`)
- Computes user risk profiles from fraud scores
- Recommends security level and number of auth questions
- **New:** Integrates Gemini LLM for dynamic question generation via `generate_dynamic_questions()`
- **New:** LLM-powered answer verification via `verify_answers_with_llm()`

### RAG Engine (`models/rag_engine.py`)
- **Production-ready 5-step pipeline**: query rewriting → metadata filtering → expanded retrieval → re-ranking → confidence scoring
- Indexes project CSV outputs and SQL queries into **ChromaDB** vector store
- Uses **Google Generative AI embeddings** (`models/embedding-001`) when API key available
- **Aggregate document indexing**: category analysis, location heatmap, portfolio overview
- Provides `query()`, `get_context_for_user()`, `get_context_for_query()`, and `get_detailed_results()` methods

### RAG Evaluator (`models/rag_evaluator.py`)
- **15 ground truth test cases** covering user profiles, transactions, categories, locations, and portfolio queries
- **Retrieval metrics**: Hit Rate, MRR (Mean Reciprocal Rank), Type Match Rate, Average Latency
- **Answer quality scoring**: Gemini-judged accuracy, completeness, and readability (1-5 scale)
- Integrated into the Streamlit dashboard for one-click evaluation

### Gemini Question Generator (`models/gemini_question_gen.py`)
- **Dynamic security questions** about stores, locations, categories, and spending from the last 5 days
- **No risk scores** — questions feel like a real bank verifying identity
- **30-second live countdown timer** per question (JavaScript-powered, runs independently)
- **Auto-replacement on miss** — wrong answer or timeout generates a brand-new LLM question
- **Always unique** — used-question tracking ensures no question is ever repeated
- Question difficulty scales with security level (LOW → easy, CRITICAL → very hard)
- **Structured output** with Key Findings, Analysis, Recommendations sections
- Falls back to static questions when no API key is available

### Transaction Habit Model (`models/habit_model.py`)
- Learns each user's transaction habits from historical data:
  - **Top 5 most visited stores** with visit count and average spend per store
  - **Preferred categories** ranked by frequency
  - **Typical locations** (top cities)
  - **Spending patterns** (avg/median amount, spending range)
  - **Time preferences** (peak hour, peak day, weekend vs weekday)
- **Habit consistency score** (0-100) measuring behavioral predictability
- **Next purchase prediction** based on historical frequency distribution
- **Similar user finder** using KNN on normalized spending features
- **Anomaly detection** — checks if a new transaction matches learned habits

### Dashboard (`streamlit_app.py`)
| Tab | Function |
|-----|----------|
| 🔑 Authentication | Timed questions (30s countdown), auto-replacement on miss, always-unique LLM questions |
| 🚨 Fraud Detection | Per-transaction AI fraud analysis with RAG-enhanced structured explanations |
| 📊 Dashboard | Risk distribution charts, user profiles, pipeline monitoring |
| 🔍 RAG Explorer | Free-form Q&A with confidence meter, source attribution, and RAG evaluation dashboard |
| 🧠 User Habits | Top 5 stores with avg spend, next purchase predictions, similar users, anomaly checker |

---

## Monitoring & Logging

### Pipeline Logs (`pipeline_logs.csv`)
Every pipeline execution is logged with:
```
run_id, timestamp, stage, status, records_processed, duration_ms, error_message
```

### Product Metrics (`logs/product_metrics.csv`)
Application-level metrics including:
- Authentication generation/verification events
- Fraud detection analysis events
- Latency measurements, confidence scores, success/failure status

---

## Implemented Extensions

1. **Hybrid ML + Rule-Based Model** — Combines IsolationForest with interpretable rules
2. **Adaptive Authentication** — Security level dynamically adjusts based on user risk profile
3. **Gemini LLM Question Generation** — Meaningful questions about stores, locations, and categories (no risk scores)
4. **30-Second Live Timer** — JavaScript-powered countdown per auth question with auto-expiry
5. **Auto-Replacement Questions** — Wrong answer or timeout triggers a new unique LLM question (never repeats)
6. **Production RAG Pipeline** — 5-step hybrid retrieval: query rewriting, metadata filtering, re-ranking, confidence scoring
7. **RAG Evaluation Suite** — 15 ground truth tests with Hit Rate, MRR, Type Match, and LLM-judged answer quality
8. **Transaction Habit Prediction** — Per-user habit learning with top 5 stores, KNN similarity, anomaly detection
9. **Top Stores with Avg Spend** — Visual breakdown of most visited store categories with per-visit spending
10. **Structured AI Output** — All AI answers formatted with Key Findings, Analysis, Recommendations, and Confidence
11. **RAG-Powered Q&A Explorer** — Free-form natural-language queries with confidence meter and source attribution
12. **Full Pipeline Monitoring** — Every stage logged with status, duration, and record counts
13. **Reproducible Pipeline** — Config-driven, dry-run mode, documented step-by-step execution
14. **Geographic Anomaly Detection** — Location entropy and new-location flagging
15. **Velocity-Based Detection** — Multi-window (1h/24h/7d) transaction burst detection

---

## Demo Video

📹 **[Demo Video Link]** — *(To be added before submission)*

---

## Team

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for detailed team member responsibilities.

**Course:** CS 5588 — Data Science Capstone  
**Date:** February 2026
