"""
GraphGuard — Production-Ready RAG Engine
Retrieval-Augmented Generation over customer transaction data.
Uses Google GenAI embeddings + ChromaDB for vector storage.

Production features:
  • Hybrid retrieval with metadata filtering
  • Query rewriting for better semantic matching
  • Cross-encoder-style re-ranking via Gemini
  • Confidence scoring per result
  • Aggregated statistical summaries for richer retrieval
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = PROJECT_ROOT / ".chroma_db"

# Only index CSV files with actual customer / transaction data.
DATA_SOURCES = {
    "features": PROJECT_ROOT / "dataset" / "csv_data" / "features_output.csv",
    "fraud_scores": PROJECT_ROOT / "dataset" / "csv_data" / "fraud_scores_output.csv",
    "auth_profiles": PROJECT_ROOT / "dataset" / "csv_data" / "auth_profiles_output.csv",
}


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Concise, customer-focused document builders (~100 words each)
# ---------------------------------------------------------------------------

def _build_user_summaries(
    features_path: Path,
    fraud_path: Path,
    auth_path: Path,
) -> List[dict]:
    """
    Build ONE concise ~100-word summary per user.
    Combines key stats from features, fraud scores, and auth profile.
    """
    docs: List[dict] = []
    try:
        feat_df = pd.read_csv(features_path) if features_path.exists() else pd.DataFrame()
        fraud_df = pd.read_csv(fraud_path) if fraud_path.exists() else pd.DataFrame()
        auth_df = pd.read_csv(auth_path) if auth_path.exists() else pd.DataFrame()
    except Exception:
        return docs

    for df in (feat_df, fraud_df, auth_df):
        df.columns = [c.upper() for c in df.columns]

    all_users = set()
    for df in (feat_df, fraud_df, auth_df):
        if "USER_ID" in df.columns:
            all_users.update(df["USER_ID"].unique())

    for user in sorted(all_users):
        uf = feat_df[feat_df["USER_ID"] == user] if not feat_df.empty else pd.DataFrame()
        us = fraud_df[fraud_df["USER_ID"] == user] if not fraud_df.empty else pd.DataFrame()
        ua = auth_df[auth_df["USER_ID"] == user] if not auth_df.empty else pd.DataFrame()

        # --- Build concise summary ---
        parts = [f"Customer {user}:"]

        # Auth level
        if not ua.empty:
            row = ua.iloc[0]
            parts.append(
                f"Security level {row.get('RECOMMENDED_SECURITY_LEVEL', 'N/A')}. "
                f"Avg risk {row.get('AVG_RISK', 0):.2f}, max risk {row.get('MAX_RISK', 0):.2f}, "
                f"{int(row.get('HIGH_RISK_COUNT', 0))} high-risk transactions."
            )

        # Transaction summary (key numbers only)
        if not uf.empty:
            n = len(uf)
            avg_amt = uf["AMOUNT"].mean()
            total = uf["AMOUNT"].sum()
            top_cat = uf["CATEGORY"].value_counts().index[0] if "CATEGORY" in uf.columns else "N/A"
            top_loc = uf["LOCATION"].value_counts().index[0] if "LOCATION" in uf.columns else "N/A"
            hv = int(uf["IS_HIGH_VALUE"].sum()) if "IS_HIGH_VALUE" in uf.columns else 0
            parts.append(
                f"{n} transactions totaling ${total:,.0f} (avg ${avg_amt:.0f}). "
                f"Top category: {top_cat}. Primary location: {top_loc}. "
                f"{hv} high-value transactions."
            )

        # Fraud level distribution
        if not us.empty and "RISK_LEVEL" in us.columns:
            levels = us["RISK_LEVEL"].value_counts()
            level_str = ", ".join(f"{l} {c}" for l, c in levels.items())
            parts.append(f"Risk breakdown: {level_str}.")

        docs.append({
            "text": " ".join(parts),
            "metadata": {"source": "customer_summary", "type": "profile", "user_id": user},
        })

    return docs


def _build_transaction_summaries(
    features_path: Path,
    fraud_path: Path,
) -> List[dict]:
    """
    Build compact per-transaction docs (one line each) for detailed queries.
    """
    docs: List[dict] = []
    try:
        feat_df = pd.read_csv(features_path) if features_path.exists() else pd.DataFrame()
        fraud_df = pd.read_csv(fraud_path) if fraud_path.exists() else pd.DataFrame()
    except Exception:
        return docs

    if feat_df.empty or fraud_df.empty:
        return docs

    for df in (feat_df, fraud_df):
        df.columns = [c.upper() for c in df.columns]

    merged = feat_df.merge(
        fraud_df[["TRANSACTION_ID", "COMBINED_RISK_SCORE", "RISK_LEVEL", "RECOMMENDATION"]],
        on="TRANSACTION_ID", how="left",
    )

    for _, t in merged.iterrows():
        text = (
            f"{t.get('USER_ID','?')} txn {t.get('TRANSACTION_ID','?')}: "
            f"${t.get('AMOUNT', 0):.2f} at {t.get('CATEGORY','?')} in {t.get('LOCATION','?')}. "
            f"Risk {t.get('RISK_LEVEL','?')} ({t.get('COMBINED_RISK_SCORE',0):.3f})."
        )
        docs.append({
            "text": text,
            "metadata": {
                "source": "transaction",
                "type": "transaction",
                "user_id": str(t.get("USER_ID", "")),
                "risk_level": str(t.get("RISK_LEVEL", "")),
                "category": str(t.get("CATEGORY", "")),
                "location": str(t.get("LOCATION", "")),
            },
        })

    return docs


# ---------------------------------------------------------------------------
# NEW: Aggregated statistical summaries for richer retrieval
# ---------------------------------------------------------------------------

def _build_aggregate_summaries(
    features_path: Path,
    fraud_path: Path,
) -> List[dict]:
    """
    Build aggregated statistical summaries:
      - Category-level risk analysis
      - Location-level risk heatmap
      - Overall portfolio statistics
    These give the RAG engine richer context for analytical questions.
    """
    docs: List[dict] = []
    try:
        feat_df = pd.read_csv(features_path) if features_path.exists() else pd.DataFrame()
        fraud_df = pd.read_csv(fraud_path) if fraud_path.exists() else pd.DataFrame()
    except Exception:
        return docs

    if feat_df.empty or fraud_df.empty:
        return docs

    for df in (feat_df, fraud_df):
        df.columns = [c.upper() for c in df.columns]

    merged = feat_df.merge(
        fraud_df[["TRANSACTION_ID", "COMBINED_RISK_SCORE", "RISK_LEVEL"]],
        on="TRANSACTION_ID", how="left",
    )

    # --- Category-level risk analysis ---
    if "CATEGORY" in merged.columns:
        cat_stats = merged.groupby("CATEGORY").agg(
            txn_count=("TRANSACTION_ID", "count"),
            avg_amount=("AMOUNT", "mean"),
            total_amount=("AMOUNT", "sum"),
            avg_risk=("COMBINED_RISK_SCORE", "mean"),
            max_risk=("COMBINED_RISK_SCORE", "max"),
            high_risk_count=("RISK_LEVEL", lambda x: (x.isin(["HIGH", "CRITICAL"])).sum()),
        ).reset_index()

        parts = ["Category Risk Analysis:"]
        for _, row in cat_stats.sort_values("avg_risk", ascending=False).iterrows():
            parts.append(
                f"{row['CATEGORY']}: {int(row['txn_count'])} txns, "
                f"avg ${row['avg_amount']:.0f}, avg risk {row['avg_risk']:.3f}, "
                f"max risk {row['max_risk']:.3f}, "
                f"{int(row['high_risk_count'])} high-risk."
            )
        docs.append({
            "text": " ".join(parts),
            "metadata": {"source": "aggregate", "type": "category_analysis"},
        })

    # --- Location-level risk analysis ---
    if "LOCATION" in merged.columns:
        loc_stats = merged.groupby("LOCATION").agg(
            txn_count=("TRANSACTION_ID", "count"),
            avg_amount=("AMOUNT", "mean"),
            avg_risk=("COMBINED_RISK_SCORE", "mean"),
            max_risk=("COMBINED_RISK_SCORE", "max"),
            high_risk_count=("RISK_LEVEL", lambda x: (x.isin(["HIGH", "CRITICAL"])).sum()),
        ).reset_index()

        parts = ["Location Risk Heatmap:"]
        for _, row in loc_stats.sort_values("avg_risk", ascending=False).iterrows():
            parts.append(
                f"{row['LOCATION']}: {int(row['txn_count'])} txns, "
                f"avg risk {row['avg_risk']:.3f}, max risk {row['max_risk']:.3f}, "
                f"{int(row['high_risk_count'])} high-risk."
            )
        docs.append({
            "text": " ".join(parts),
            "metadata": {"source": "aggregate", "type": "location_analysis"},
        })

    # --- Overall portfolio statistics ---
    total_txns = len(merged)
    risk_dist = merged["RISK_LEVEL"].value_counts()
    risk_str = ", ".join(f"{level}: {count}" for level, count in risk_dist.items())
    avg_risk = merged["COMBINED_RISK_SCORE"].mean()
    total_users = merged["USER_ID"].nunique()

    # Find riskiest users
    user_risk = merged.groupby("USER_ID")["COMBINED_RISK_SCORE"].mean().sort_values(ascending=False)
    top_risky = ", ".join(f"{uid} ({score:.3f})" for uid, score in user_risk.head(3).items())

    portfolio_text = (
        f"Portfolio Overview: {total_txns} total transactions across {total_users} users. "
        f"Average risk score: {avg_risk:.3f}. Risk distribution: {risk_str}. "
        f"Highest risk users: {top_risky}. "
        f"Total transaction volume: ${merged['AMOUNT'].sum():,.0f}, "
        f"average transaction: ${merged['AMOUNT'].mean():.0f}."
    )
    docs.append({
        "text": portfolio_text,
        "metadata": {"source": "aggregate", "type": "portfolio_overview"},
    })

    return docs


# ---------------------------------------------------------------------------
# RAG Engine — Production-Ready
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Production-ready RAG engine with:
      • Hybrid retrieval (metadata filtering + vector search)
      • Query rewriting for better semantic matching
      • Cross-encoder re-ranking via Gemini
      • Per-result confidence scoring
      • Aggregated statistical document indexing
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._collection = None
        self._use_gemini_embeddings = bool(self.api_key)
        self._genai_client = None
        self._setup_chromadb()

    def _setup_chromadb(self):
        try:
            import chromadb
        except ImportError:
            print("⚠️  chromadb not installed. RAG features disabled.")
            return

        self._client = chromadb.Client()

        if self._use_gemini_embeddings:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
                client = self._genai_client

                class GeminiEmbedder:
                    name = "gemini-embedding"
                    def __call__(self, input: List[str]) -> List[List[float]]:
                        results = []
                        for text in input:
                            resp = client.models.embed_content(
                                model="gemini-embedding-exp-03-07",
                                contents=text,
                            )
                            results.append(resp.embeddings[0].values)
                        return results

                self._collection = self._client.get_or_create_collection(
                    name="graphguard_data",
                    embedding_function=GeminiEmbedder(),
                )
                print("✅ RAG engine initialized with Gemini embeddings")
            except Exception as e:
                print(f"⚠️  Gemini embeddings failed ({e}), using default embeddings")
                self._use_gemini_embeddings = False
                self._collection = self._client.get_or_create_collection(
                    name="graphguard_data"
                )
        else:
            self._collection = self._client.get_or_create_collection(
                name="graphguard_data"
            )
            print("ℹ️  RAG engine initialized with default embeddings (no API key)")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_data(self, force: bool = False) -> int:
        if self._collection is None:
            return 0

        current_count = self._collection.count()
        if current_count > 0 and not force:
            return current_count

        if current_count > 0:
            all_ids = self._collection.get()["ids"]
            if all_ids:
                self._collection.delete(ids=all_ids)

        # Build all document types
        documents = []
        documents.extend(_build_user_summaries(
            DATA_SOURCES["features"],
            DATA_SOURCES["fraud_scores"],
            DATA_SOURCES["auth_profiles"],
        ))
        documents.extend(_build_transaction_summaries(
            DATA_SOURCES["features"],
            DATA_SOURCES["fraud_scores"],
        ))
        # NEW: Aggregated statistical summaries
        documents.extend(_build_aggregate_summaries(
            DATA_SOURCES["features"],
            DATA_SOURCES["fraud_scores"],
        ))

        if not documents:
            return 0

        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [d["text"] for d in documents]
        metadatas = [d["metadata"] for d in documents]

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self._collection.add(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        n_profiles = sum(1 for d in documents if d["metadata"]["type"] == "profile")
        n_txns = sum(1 for d in documents if d["metadata"]["type"] == "transaction")
        n_agg = sum(1 for d in documents if d["metadata"]["source"] == "aggregate")
        print(f"✅ Indexed {len(documents)} documents ({n_profiles} profiles, {n_txns} transactions, {n_agg} aggregates)")
        return len(documents)

    # ------------------------------------------------------------------
    # Query Rewriting — optimize query for indexed data format
    # ------------------------------------------------------------------

    def _rewrite_query(self, question: str) -> str:
        """
        Use Gemini to rewrite a conversational question into a
        search-optimized query that matches the indexed document format.
        Falls back to the original question if LLM is unavailable.
        """
        if not self._genai_client:
            return question

        prompt = f"""Rewrite this user question into a concise search query optimized for retrieving fraud detection data.
The database contains: customer profiles (user IDs, risk scores, transaction counts, categories, locations),
individual transactions (amounts, categories, locations, risk levels), and aggregate statistics (category risk, location risk, portfolio overview).

User question: {question}

Return ONLY the rewritten search query, nothing else. Keep it under 30 words. Focus on key entities and data terms."""

        try:
            resp = self._genai_client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
            rewritten = resp.text.strip().strip('"').strip("'")
            if rewritten and len(rewritten) < 200:
                return rewritten
        except Exception:
            pass
        return question

    # ------------------------------------------------------------------
    # Metadata-filtered retrieval
    # ------------------------------------------------------------------

    def _build_where_filter(self, question: str, user_id: Optional[str] = None) -> Optional[dict]:
        """
        Build ChromaDB `where` filter from the question content.
        Extracts user IDs, risk levels, and doc types for precision filtering.
        """
        filters = []

        # Filter by explicit user ID
        if user_id:
            filters.append({"user_id": user_id})
        else:
            # Try to extract user ID from question
            user_match = re.findall(r'USER_\d+', question.upper())
            if user_match:
                filters.append({"user_id": user_match[0]})

        # Filter by risk level if mentioned
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if level.lower() in question.lower():
                filters.append({"risk_level": level})
                break

        # If asking about categories or locations, prefer aggregate docs
        aggregate_keywords = ["category", "categories", "location", "locations",
                              "distribution", "overview", "portfolio", "heatmap", "overall"]
        if any(kw in question.lower() for kw in aggregate_keywords):
            filters.append({"source": "aggregate"})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$or": filters}

    # ------------------------------------------------------------------
    # Re-ranking via Gemini
    # ------------------------------------------------------------------

    def _rerank_results(
        self, question: str, candidates: List[dict], top_k: int = 5
    ) -> List[dict]:
        """
        Re-rank retrieved candidates using Gemini for relevance scoring.
        Falls back to distance-based ranking if LLM is unavailable.
        """
        if not self._genai_client or len(candidates) <= top_k:
            return candidates[:top_k]

        # Build candidate list for re-ranking
        candidate_texts = []
        for i, c in enumerate(candidates):
            candidate_texts.append(f"[{i}] {c['text'][:200]}")

        prompt = f"""You are a search relevance judge for a fraud detection system.
Given the user's question and a list of retrieved documents, rank the documents by relevance.

Question: {question}

Documents:
{chr(10).join(candidate_texts)}

Return ONLY a JSON array of document indices ordered by relevance (most relevant first).
Example: [3, 0, 7, 1, 5]
Return the top {top_k} most relevant indices only."""

        try:
            resp = self._genai_client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
            text = resp.text.strip()
            # Extract JSON array
            match = re.search(r'\[[\d,\s]+\]', text)
            if match:
                indices = json.loads(match.group())
                reranked = []
                seen = set()
                for idx in indices[:top_k]:
                    if 0 <= idx < len(candidates) and idx not in seen:
                        reranked.append(candidates[idx])
                        seen.add(idx)
                # Fill remaining slots if needed
                for c in candidates:
                    if len(reranked) >= top_k:
                        break
                    if c not in reranked:
                        reranked.append(c)
                return reranked
        except Exception:
            pass

        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(results: List[dict]) -> List[dict]:
        """
        Assign a normalized confidence score (0.0–1.0) to each result.
        Based on ChromaDB distance (lower = better match).
        """
        if not results:
            return results

        distances = [r.get("distance", 1.0) for r in results]
        max_dist = max(distances) if distances else 1.0
        min_dist = min(distances) if distances else 0.0
        dist_range = max_dist - min_dist if max_dist != min_dist else 1.0

        for i, r in enumerate(results):
            dist = r.get("distance", 1.0)
            # Normalize: closer distance = higher confidence
            raw_confidence = 1.0 - ((dist - min_dist) / dist_range)
            # Apply position decay — earlier results get slight boost
            position_boost = max(0, 0.1 * (1 - i / len(results)))
            r["confidence"] = round(min(1.0, raw_confidence + position_boost), 3)

        return results

    # ------------------------------------------------------------------
    # Core query — production pipeline
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        n_results: int = 5,
        user_id: Optional[str] = None,
        rewrite: bool = True,
        rerank: bool = True,
    ) -> List[dict]:
        """
        Production query pipeline:
          1. Rewrite query for better semantic matching
          2. Build metadata filters for precision
          3. Retrieve top-K candidates (expanded set)
          4. Re-rank via Gemini cross-encoder
          5. Score confidence per result
        """
        if self._collection is None or self._collection.count() == 0:
            self.index_data()

        if self._collection is None or self._collection.count() == 0:
            return []

        # Step 1: Query rewriting
        search_query = self._rewrite_query(question) if rewrite else question

        # Step 2: Build metadata filters
        where_filter = self._build_where_filter(question, user_id)

        # Step 3: Retrieve expanded candidate set
        retrieve_n = min(n_results * 3, self._collection.count())  # 3x over-retrieve

        query_args = {
            "query_texts": [search_query],
            "n_results": retrieve_n,
        }
        if where_filter:
            try:
                results = self._collection.query(**query_args, where=where_filter)
                # If filter returned too few results, fall back to unfiltered
                if not results["documents"][0] or len(results["documents"][0]) < 2:
                    results = self._collection.query(**query_args)
            except Exception:
                results = self._collection.query(**query_args)
        else:
            results = self._collection.query(**query_args)

        # Parse results into structured format
        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })

        # Step 4: Re-rank
        if rerank:
            docs = self._rerank_results(question, docs, top_k=n_results)
        else:
            docs = docs[:n_results]

        # Step 5: Confidence scoring
        docs = self._compute_confidence(docs)

        return docs

    # ------------------------------------------------------------------
    # High-level context retrieval
    # ------------------------------------------------------------------

    def get_context_for_user(self, user_id: str) -> str:
        """Get rich RAG context for a specific user with metadata filtering."""
        results = self.query(
            f"customer {user_id} profile risk transactions",
            n_results=5,
            user_id=user_id,
        )
        if not results:
            return "No data available for this user."

        # Prioritize the user's profile summary
        context_parts = []
        for doc in results:
            if user_id.lower() in doc["text"].lower():
                context_parts.append(doc["text"])

        if not context_parts:
            context_parts = [doc["text"] for doc in results[:2]]

        return "\n".join(context_parts[:3])

    def get_context_for_query(self, question: str) -> str:
        """Get rich RAG context for a free-form question with full pipeline."""
        results = self.query(question, n_results=5)
        if not results:
            return "No relevant data found."
        return "\n".join(doc["text"] for doc in results)

    def get_detailed_results(self, question: str, n_results: int = 5) -> dict:
        """
        Get full detailed results including confidence scores and metadata.
        Used by the enhanced Streamlit UI for structured display.
        """
        results = self.query(question, n_results=n_results)

        avg_confidence = (
            sum(r.get("confidence", 0) for r in results) / len(results)
            if results else 0
        )

        return {
            "results": results,
            "context": "\n".join(r["text"] for r in results),
            "avg_confidence": round(avg_confidence, 3),
            "num_sources": len(results),
            "source_types": list(set(r["metadata"].get("type", "unknown") for r in results)),
        }


if __name__ == "__main__":
    engine = RAGEngine()
    count = engine.index_data(force=True)
    print(f"\nIndexed {count} documents.\n")

    test_queries = [
        "Which users have the highest risk?",
        "Tell me about USER_004",
        "What categories have the most fraud?",
        "What is the overall risk distribution?",
        "Show high risk transactions in Electronics",
    ]
    for q in test_queries:
        print(f"Q: {q}")
        results = engine.query(q, n_results=3)
        for r in results:
            conf = r.get("confidence", 0)
            print(f"  [{conf:.0%}] {r['text'][:120]}...")
        print()
