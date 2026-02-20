"""
GraphGuard — RAG Evaluator
Automated evaluation of RAG pipeline accuracy.
Measures retrieval quality (Hit Rate, MRR) and answer quality (LLM-judged).

Run:
    python -m models.rag_evaluator
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Ground Truth Test Suite
# Each test has: question, expected_keywords (must appear in retrieved context),
# expected_type (document type that should be retrieved), and description.
# ---------------------------------------------------------------------------

GROUND_TRUTH_TESTS = [
    {
        "id": "T01",
        "question": "Which users have the highest fraud risk?",
        "expected_keywords": ["risk", "user"],
        "expected_type": ["profile", "portfolio_overview"],
        "description": "Should retrieve user profiles sorted by risk",
    },
    {
        "id": "T02",
        "question": "Tell me about USER_004",
        "expected_keywords": ["USER_004"],
        "expected_type": ["profile", "transaction"],
        "description": "Should retrieve USER_004 profile and transactions",
    },
    {
        "id": "T03",
        "question": "What categories have the most fraud?",
        "expected_keywords": ["category", "risk"],
        "expected_type": ["category_analysis", "transaction"],
        "description": "Should retrieve category analysis aggregate",
    },
    {
        "id": "T04",
        "question": "What is the overall risk distribution?",
        "expected_keywords": ["risk", "distribution"],
        "expected_type": ["portfolio_overview"],
        "description": "Should retrieve portfolio overview",
    },
    {
        "id": "T05",
        "question": "Show me high risk transactions",
        "expected_keywords": ["HIGH", "risk"],
        "expected_type": ["transaction"],
        "description": "Should retrieve HIGH/CRITICAL risk transactions",
    },
    {
        "id": "T06",
        "question": "Which locations are most risky?",
        "expected_keywords": ["location", "risk"],
        "expected_type": ["location_analysis"],
        "description": "Should retrieve location risk heatmap",
    },
    {
        "id": "T07",
        "question": "What is USER_001's security level?",
        "expected_keywords": ["USER_001", "security"],
        "expected_type": ["profile"],
        "description": "Should retrieve USER_001 profile with security level",
    },
    {
        "id": "T08",
        "question": "How many high-value transactions are there?",
        "expected_keywords": ["high-value", "transaction"],
        "expected_type": ["profile", "portfolio_overview"],
        "description": "Should retrieve summary with high-value counts",
    },
    {
        "id": "T09",
        "question": "What is the average transaction amount?",
        "expected_keywords": ["avg", "amount"],
        "expected_type": ["portfolio_overview", "profile"],
        "description": "Should retrieve portfolio or profile with avg amounts",
    },
    {
        "id": "T10",
        "question": "Show critical risk transactions for USER_003",
        "expected_keywords": ["USER_003"],
        "expected_type": ["transaction", "profile"],
        "description": "Should filter to USER_003 critical transactions",
    },
    {
        "id": "T11",
        "question": "Compare risk across all users",
        "expected_keywords": ["risk", "user"],
        "expected_type": ["portfolio_overview", "profile"],
        "description": "Should retrieve comparative user risk data",
    },
    {
        "id": "T12",
        "question": "What is the total transaction volume?",
        "expected_keywords": ["total", "transaction"],
        "expected_type": ["portfolio_overview"],
        "description": "Should retrieve portfolio with total volume",
    },
    {
        "id": "T13",
        "question": "Which user has the most transactions?",
        "expected_keywords": ["transaction", "user"],
        "expected_type": ["profile", "portfolio_overview"],
        "description": "Should retrieve user profiles with transaction counts",
    },
    {
        "id": "T14",
        "question": "Explain the risk factors for the riskiest transaction",
        "expected_keywords": ["risk"],
        "expected_type": ["transaction"],
        "description": "Should retrieve highest-risk transaction details",
    },
    {
        "id": "T15",
        "question": "What are the most common transaction categories?",
        "expected_keywords": ["category"],
        "expected_type": ["category_analysis", "profile"],
        "description": "Should retrieve category analysis",
    },
]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Evaluates RAG pipeline accuracy using:
      1. Hit Rate — proportion of tests where expected keywords appear in results
      2. MRR — Mean Reciprocal Rank of the first relevant result
      3. Type Match Rate — how often the expected doc type is retrieved
      4. Answer Quality — LLM-judged quality score (1-5) via Gemini
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._genai_client = None
        if self.api_key:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
            except Exception:
                pass

    def evaluate_retrieval(self, rag_engine, tests: List[dict] = None) -> dict:
        """
        Run retrieval evaluation against ground truth tests.
        Returns detailed per-test results and aggregate metrics.
        """
        if tests is None:
            tests = GROUND_TRUTH_TESTS

        results = []
        total_hit = 0
        total_type_match = 0
        reciprocal_ranks = []

        for test in tests:
            t0 = time.time()
            retrieved = rag_engine.query(test["question"], n_results=5)
            retrieval_time = time.time() - t0

            # Combine all retrieved text for keyword checking
            all_text = " ".join(r["text"] for r in retrieved).lower()
            all_types = [r["metadata"].get("type", "") for r in retrieved]

            # Hit rate — did any expected keyword appear?
            keyword_hits = [kw.lower() in all_text for kw in test["expected_keywords"]]
            is_hit = all(keyword_hits)
            if is_hit:
                total_hit += 1

            # Type match — did expected doc type appear?
            type_match = any(t in all_types for t in test["expected_type"])
            if type_match:
                total_type_match += 1

            # MRR — position of first fully relevant result
            rr = 0
            for i, r in enumerate(retrieved):
                r_text = r["text"].lower()
                if all(kw.lower() in r_text for kw in test["expected_keywords"]):
                    rr = 1.0 / (i + 1)
                    break
            reciprocal_ranks.append(rr)

            # Average confidence of results
            avg_conf = (
                sum(r.get("confidence", 0) for r in retrieved) / len(retrieved)
                if retrieved else 0
            )

            results.append({
                "test_id": test["id"],
                "question": test["question"],
                "description": test["description"],
                "hit": is_hit,
                "type_match": type_match,
                "reciprocal_rank": round(rr, 3),
                "avg_confidence": round(avg_conf, 3),
                "retrieval_time_ms": round(retrieval_time * 1000, 1),
                "num_retrieved": len(retrieved),
                "keyword_hits": keyword_hits,
            })

        n = len(tests)
        metrics = {
            "hit_rate": round(total_hit / n, 3) if n else 0,
            "mrr": round(sum(reciprocal_ranks) / n, 3) if n else 0,
            "type_match_rate": round(total_type_match / n, 3) if n else 0,
            "total_tests": n,
            "passed": total_hit,
            "avg_retrieval_ms": round(
                sum(r["retrieval_time_ms"] for r in results) / n, 1
            ) if n else 0,
        }

        return {"metrics": metrics, "details": results}

    def evaluate_answer_quality(
        self, rag_engine, question_generator, tests: List[dict] = None, max_tests: int = 5
    ) -> dict:
        """
        Use Gemini to judge answer quality on a 1-5 scale.
        Only runs if a Gemini API key is available.
        """
        if not self._genai_client:
            return {"error": "No API key — answer quality evaluation requires Gemini."}

        if tests is None:
            tests = GROUND_TRUTH_TESTS[:max_tests]

        scores = []
        details = []

        for test in tests[:max_tests]:
            # Get RAG answer
            context = rag_engine.get_context_for_query(test["question"])
            answer = question_generator.answer_question_with_rag(test["question"], context)

            # Judge quality with Gemini
            judge_prompt = f"""Rate this RAG answer on a 1-5 scale across three dimensions.
Question: {test["question"]}
Expected to cover: {', '.join(test["expected_keywords"])}

Answer:
{answer}

Rate each dimension (1=poor, 5=excellent):
1. Accuracy — Does the answer contain correct, relevant information?
2. Completeness — Does it cover the key aspects of the question?
3. Readability — Is it well-structured, concise, and easy to understand?

Return ONLY a JSON object like: {{"accuracy": 4, "completeness": 3, "readability": 5, "overall": 4}}"""

            try:
                resp = self._genai_client.models.generate_content(
                    model="gemini-2.0-flash", contents=judge_prompt
                )
                text = resp.text.strip()
                # Extract JSON
                import re
                match = re.search(r'\{[^}]+\}', text)
                if match:
                    score_dict = json.loads(match.group())
                    overall = score_dict.get("overall", 3)
                    scores.append(overall)
                    details.append({
                        "test_id": test["id"],
                        "question": test["question"],
                        "scores": score_dict,
                        "answer_preview": answer[:200],
                    })
                else:
                    scores.append(3)
                    details.append({
                        "test_id": test["id"],
                        "question": test["question"],
                        "scores": {"overall": 3, "note": "parse_failed"},
                        "answer_preview": answer[:200],
                    })
            except Exception as e:
                scores.append(3)
                details.append({
                    "test_id": test["id"],
                    "question": test["question"],
                    "scores": {"overall": 3, "error": str(e)},
                })

        avg_score = round(sum(scores) / len(scores), 2) if scores else 0

        return {
            "avg_quality_score": avg_score,
            "num_evaluated": len(scores),
            "details": details,
        }

    def full_evaluation(self, rag_engine, question_generator=None) -> dict:
        """Run complete evaluation suite and return structured report."""
        print("🔍 Running RAG retrieval evaluation...")
        retrieval = self.evaluate_retrieval(rag_engine)

        print(f"\n📊 Retrieval Results:")
        m = retrieval["metrics"]
        print(f"   Hit Rate:       {m['hit_rate']:.1%} ({m['passed']}/{m['total_tests']})")
        print(f"   MRR:            {m['mrr']:.3f}")
        print(f"   Type Match:     {m['type_match_rate']:.1%}")
        print(f"   Avg Latency:    {m['avg_retrieval_ms']:.0f} ms")

        # Show per-test details
        print(f"\n{'ID':<5} {'Hit':>4} {'Type':>5} {'RR':>6} {'Conf':>6}  Question")
        print("─" * 80)
        for d in retrieval["details"]:
            hit_icon = "✅" if d["hit"] else "❌"
            type_icon = "✅" if d["type_match"] else "❌"
            print(
                f"{d['test_id']:<5} {hit_icon:>4} {type_icon:>5} "
                f"{d['reciprocal_rank']:>6.3f} {d['avg_confidence']:>6.3f}  "
                f"{d['question'][:50]}"
            )

        report = {"retrieval": retrieval}

        # Answer quality (requires Gemini + question generator)
        if question_generator and self._genai_client:
            print("\n🧠 Running answer quality evaluation (Gemini-judged)...")
            quality = self.evaluate_answer_quality(rag_engine, question_generator)
            report["answer_quality"] = quality

            if "avg_quality_score" in quality:
                print(f"\n📊 Answer Quality Score: {quality['avg_quality_score']:.1f}/5.0")
                for d in quality.get("details", []):
                    scores = d.get("scores", {})
                    print(
                        f"   {d['test_id']}: acc={scores.get('accuracy','?')} "
                        f"comp={scores.get('completeness','?')} "
                        f"read={scores.get('readability','?')} "
                        f"→ {scores.get('overall','?')}/5"
                    )

        return report


if __name__ == "__main__":
    from models.rag_engine import RAGEngine

    api_key = os.environ.get("GOOGLE_API_KEY", "")

    engine = RAGEngine(api_key=api_key)
    engine.index_data(force=True)

    question_gen = None
    if api_key:
        try:
            from models.gemini_question_gen import GeminiQuestionGenerator
            question_gen = GeminiQuestionGenerator(api_key=api_key)
        except Exception:
            pass

    evaluator = RAGEvaluator(api_key=api_key)
    report = evaluator.full_evaluation(engine, question_gen)

    print("\n" + "=" * 60)
    print("📋 EVALUATION COMPLETE")
    print("=" * 60)
