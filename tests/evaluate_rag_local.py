from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import pandas as pd
from models.rag_engine_local import RAGEngineLocal


def evaluate_rag():
    print("Initializing Local RAG Engine...")
    rag = RAGEngineLocal()

    test_cases = [
        {
            "query": "What are common credit card fraud trends?",
            "keywords": ["fraud", "credit", "card", "complaint", "scam"],
            "min_confidence": 0.25,
        },
        {
            "query": "How to dispute a charge?",
            "keywords": ["dispute", "charge", "billing", "error", "unauthorized"],
            "min_confidence": 0.25,
        },
        {
            "query": "Identity theft protection",
            "keywords": ["identity", "theft", "protection", "security", "fraud"],
            "min_confidence": 0.25,
        },
    ]

    total_score = 0
    print("\n--- RAG Retrieval Evaluation ---")

    for case in test_cases:
        query = case["query"]
        print(f"\nQuery: '{query}'")

        results = rag.query(query, n_results=3)

        hit_count = 0
        for i, res in enumerate(results):
            doc = res["text"]
            conf = float(res.get("confidence", 0.0))
            is_hit = any(kw.lower() in doc.lower() for kw in case["keywords"])
            strong = conf >= case["min_confidence"]
            if is_hit and strong:
                hit_count += 1

            snippet = doc[:100].replace("\n", " ")
            status = "✅ RELEVANT" if (is_hit and strong) else "❌ WEAK/IRRELEVANT"
            print(
                f"  [{i+1}] {status} | conf={conf:.3f} | type={res.get('type')} | {snippet}..."
            )

        precision = hit_count / 3 if results else 0.0
        total_score += precision
        print(f"Precision@3 (keyword+confidence): {precision:.2f}")

    avg_precision = total_score / len(test_cases)
    print(f"\nAverage Precision@3: {avg_precision:.2f}")

    return avg_precision


if __name__ == "__main__":
    evaluate_rag()
