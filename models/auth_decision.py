"""
GraphGuard — Authentication Decision Module
Determines security level and manages auth events based on user risk profile.
"""

import csv
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_LOG_PATH = PROJECT_ROOT / "pipeline_logs.csv"

PIPELINE_LOG_COLUMNS = [
    "run_id", "timestamp", "stage", "status",
    "records_processed", "duration_ms", "error_message",
]


def log_pipeline_event(stage, status, records=0, duration_ms=0.0, error_message=""):
    exists = PIPELINE_LOG_PATH.exists()
    with open(PIPELINE_LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PIPELINE_LOG_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "stage": stage,
            "status": status,
            "records_processed": records,
            "duration_ms": round(duration_ms, 2),
            "error_message": error_message,
        })


# ---------------------------------------------------------------------------
# Risk Profile Calculation
# ---------------------------------------------------------------------------

def compute_user_risk_profile(
    user_id: str,
    fraud_scores_df: pd.DataFrame,
) -> dict:
    """
    Compute an aggregate risk profile for a user based on recent fraud scores.

    Returns a dict with:
        - avg_risk: average combined risk score
        - max_risk: max combined risk score
        - high_risk_count: count of HIGH/CRITICAL scored transactions
        - recommended_security_level: LOW / MEDIUM / HIGH / CRITICAL
        - num_questions: how many auth questions to ask
    """
    user_scores = fraud_scores_df[
        fraud_scores_df["USER_ID"] == user_id
    ]

    if user_scores.empty:
        return {
            "user_id": user_id,
            "avg_risk": 0.0,
            "max_risk": 0.0,
            "high_risk_count": 0,
            "recommended_security_level": "LOW",
            "num_questions": 2,
        }

    avg_risk = float(user_scores["COMBINED_RISK_SCORE"].mean())
    max_risk = float(user_scores["COMBINED_RISK_SCORE"].max())
    high_risk_count = int(
        user_scores["RISK_LEVEL"].isin(["HIGH", "CRITICAL"]).sum()
    )

    # Determine security level from user's overall profile
    if avg_risk >= 0.6 or max_risk >= 0.75 or high_risk_count >= 3:
        level = "CRITICAL"
        num_q = 5
    elif avg_risk >= 0.4 or max_risk >= 0.5 or high_risk_count >= 2:
        level = "HIGH"
        num_q = 4
    elif avg_risk >= 0.2 or high_risk_count >= 1:
        level = "MEDIUM"
        num_q = 3
    else:
        level = "LOW"
        num_q = 2

    return {
        "user_id": user_id,
        "avg_risk": round(avg_risk, 4),
        "max_risk": round(max_risk, 4),
        "high_risk_count": high_risk_count,
        "recommended_security_level": level,
        "num_questions": num_q,
    }


# ---------------------------------------------------------------------------
# Authentication Decision
# ---------------------------------------------------------------------------

def evaluate_auth_outcome(
    user_id: str,
    security_level: str,
    num_questions: int,
    correct_answers: int,
    latency_ms: float,
) -> dict:
    """
    Evaluate the authentication outcome and return a decision dict.
    Also logs the event to pipeline_logs.csv.
    """
    score = correct_answers / max(num_questions, 1)

    # Pass thresholds by level
    thresholds = {
        "LOW": 0.5,
        "MEDIUM": 0.6,
        "HIGH": 0.75,
        "CRITICAL": 0.9,
    }
    threshold = thresholds.get(security_level, 0.6)
    passed = score >= threshold

    event = {
        "event_id": str(uuid.uuid4())[:12],
        "user_id": user_id,
        "security_level": security_level,
        "num_questions": num_questions,
        "correct_answers": correct_answers,
        "score": round(score, 4),
        "threshold": threshold,
        "passed": passed,
        "latency_ms": round(latency_ms, 2),
        "timestamp": datetime.utcnow().isoformat(),
    }

    status = "success" if passed else "failed"
    log_pipeline_event("authentication", status, records=1, duration_ms=latency_ms)

    return event


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def generate_auth_events_report(
    fraud_scores_path: str,
    output_path: str = None,
):
    """
    Generate a report of recommended security levels for all users.
    Good for pipeline dashboards.
    """
    scores_df = pd.read_csv(fraud_scores_path)
    users = scores_df["USER_ID"].unique()

    profiles = []
    for uid in sorted(users):
        profile = compute_user_risk_profile(uid, scores_df)
        profiles.append(profile)

    report_df = pd.DataFrame(profiles)

    if output_path is None:
        output_path = str(PROJECT_ROOT / "auth_profiles_output.csv")

    report_df.to_csv(output_path, index=False)
    print(f"✅ Auth profiles saved to {output_path}")
    print(f"\n📊 Security Level Distribution:")
    print(report_df["recommended_security_level"].value_counts().to_string())

    log_pipeline_event("auth_profiling", "success", len(profiles))
    return report_df


# ---------------------------------------------------------------------------
# Gemini-Powered Dynamic Question Generation
# ---------------------------------------------------------------------------

def generate_dynamic_questions(
    user_id: str,
    fraud_scores_df: pd.DataFrame,
    api_key: str = "",
) -> dict:
    """
    Generate dynamic security questions using Gemini LLM + RAG.

    Returns a dict with:
        - profile: user risk profile dict
        - questions: list of question dicts
        - source: 'gemini' or 'static'
    """
    from models.rag_engine import RAGEngine
    from models.gemini_question_gen import GeminiQuestionGenerator

    # Compute user risk profile
    profile = compute_user_risk_profile(user_id, fraud_scores_df)

    # Initialize RAG and question generator
    rag = RAGEngine(api_key=api_key)
    rag.index_data()
    gen = GeminiQuestionGenerator(api_key=api_key)

    # Get user-specific context from RAG
    rag_context = rag.get_context_for_user(user_id)

    # Generate questions
    questions = gen.generate_questions(
        user_id=user_id,
        security_level=profile["recommended_security_level"],
        num_questions=profile["num_questions"],
        rag_context=rag_context,
        user_profile=profile,
    )

    source = "gemini" if gen.model else "static"
    return {
        "profile": profile,
        "questions": questions,
        "source": source,
    }


def verify_answers_with_llm(
    questions: list,
    answers: list,
    user_id: str,
    api_key: str = "",
) -> list:
    """
    Verify user answers using Gemini LLM.

    Returns a list of verification result dicts.
    """
    from models.rag_engine import RAGEngine
    from models.gemini_question_gen import GeminiQuestionGenerator

    rag = RAGEngine(api_key=api_key)
    rag.index_data()
    gen = GeminiQuestionGenerator(api_key=api_key)

    rag_context = rag.get_context_for_user(user_id)

    results = []
    for q, a in zip(questions, answers):
        question_text = q["question"] if isinstance(q, dict) else q
        result = gen.verify_answer(question_text, a, rag_context)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auth Decision Module")
    parser.add_argument(
        "--scores", type=str,
        default=str(PROJECT_ROOT / "fraud_scores_output.csv"),
        help="Path to fraud scores CSV",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for auth profiles",
    )
    args = parser.parse_args()

    generate_auth_events_report(args.scores, args.output)
