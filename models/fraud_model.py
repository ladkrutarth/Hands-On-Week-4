"""
GraphGuard — Fraud Scoring Model
Combines rule-based heuristics with IsolationForest for fraud risk scoring.
Produces a combined risk score (0.0–1.0) per transaction.
"""

import argparse
import csv
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
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
# Rule-based scoring components
# ---------------------------------------------------------------------------

def zscore_flag(zscore: float) -> float:
    """Convert z-score to a 0–1 risk signal."""
    if zscore <= 1.0:
        return 0.0
    elif zscore <= 2.0:
        return round((zscore - 1.0) * 0.3, 4)
    elif zscore <= 3.0:
        return round(0.3 + (zscore - 2.0) * 0.4, 4)
    else:
        return min(1.0, round(0.7 + (zscore - 3.0) * 0.15, 4))


def velocity_flag(txn_1h: int, txn_24h: int) -> float:
    """Risk signal from velocity features."""
    score = 0.0
    if txn_1h >= 3:
        score += 0.4
    elif txn_1h >= 2:
        score += 0.2
    if txn_24h >= 8:
        score += 0.4
    elif txn_24h >= 5:
        score += 0.2
    return min(1.0, score)


def geographic_risk(is_new_location: bool, location_entropy: float) -> float:
    """Risk signal from geographic features."""
    score = 0.0
    if is_new_location:
        score += 0.3
    if location_entropy > 2.0:
        score += 0.2
    elif location_entropy > 1.5:
        score += 0.1
    return min(1.0, score)


def risk_level_label(score: float) -> str:
    """Map combined score to risk level label."""
    if score >= 0.75:
        return "CRITICAL"
    elif score >= 0.5:
        return "HIGH"
    elif score >= 0.25:
        return "MEDIUM"
    return "LOW"


def recommendation_text(level: str) -> str:
    """Generate recommendation based on risk level."""
    mapping = {
        "CRITICAL": "Block transaction. Require multi-factor authentication and manual review.",
        "HIGH": "Flag for review. Require step-up authentication before approval.",
        "MEDIUM": "Allow with enhanced monitoring. Send user notification.",
        "LOW": "Allow transaction. Standard monitoring.",
    }
    return mapping.get(level, "Allow transaction.")


# ---------------------------------------------------------------------------
# IsolationForest wrapper
# ---------------------------------------------------------------------------

def run_isolation_forest(features_df: pd.DataFrame) -> np.ndarray:
    """Run IsolationForest and return anomaly scores (0–1 scale, higher = riskier)."""
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        print("⚠️  scikit-learn not installed. Skipping IsolationForest.")
        return np.zeros(len(features_df))

    numeric_cols = [
        "AMOUNT", "AMOUNT_ZSCORE", "USER_AVG_AMOUNT",
        "TXN_COUNT_1H", "TXN_COUNT_24H", "TXN_COUNT_7D",
        "CATEGORY_RISK_WEIGHT", "HOUR_OF_DAY", "LOCATION_ENTROPY",
    ]
    X = features_df[numeric_cols].fillna(0).values

    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
    )
    model.fit(X)

    # decision_function: lower = more anomalous; convert to 0–1 risk scale
    raw_scores = model.decision_function(X)
    # Normalize: map min→1.0, max→0.0
    mn, mx = raw_scores.min(), raw_scores.max()
    if mx - mn == 0:
        return np.zeros(len(features_df))
    normalized = 1.0 - (raw_scores - mn) / (mx - mn)
    return np.clip(normalized, 0, 1)


# ---------------------------------------------------------------------------
# Combined scoring
# ---------------------------------------------------------------------------

WEIGHTS = {
    "zscore": 0.25,
    "velocity": 0.20,
    "category": 0.15,
    "geographic": 0.15,
    "isolation_forest": 0.25,
}


def score_transactions(features_df: pd.DataFrame) -> pd.DataFrame:
    """Score all transactions and return a DataFrame of fraud scores."""
    df = features_df.copy()

    # Rule-based components
    df["ZSCORE_FLAG"] = df["AMOUNT_ZSCORE"].apply(zscore_flag)
    df["VELOCITY_FLAG"] = df.apply(
        lambda r: velocity_flag(r.get("TXN_COUNT_1H", 0), r.get("TXN_COUNT_24H", 0)),
        axis=1,
    )
    df["CATEGORY_RISK_SCORE"] = df["CATEGORY_RISK_WEIGHT"]
    df["GEOGRAPHIC_RISK_SCORE"] = df.apply(
        lambda r: geographic_risk(
            bool(r.get("IS_NEW_LOCATION", False)),
            float(r.get("LOCATION_ENTROPY", 0)),
        ),
        axis=1,
    )

    # ML component
    df["ISOLATION_FOREST_SCORE"] = run_isolation_forest(df)

    # Weighted combination
    df["COMBINED_RISK_SCORE"] = (
        WEIGHTS["zscore"] * df["ZSCORE_FLAG"] +
        WEIGHTS["velocity"] * df["VELOCITY_FLAG"] +
        WEIGHTS["category"] * df["CATEGORY_RISK_SCORE"] +
        WEIGHTS["geographic"] * df["GEOGRAPHIC_RISK_SCORE"] +
        WEIGHTS["isolation_forest"] * df["ISOLATION_FOREST_SCORE"]
    ).clip(0, 1).round(4)

    df["RISK_LEVEL"] = df["COMBINED_RISK_SCORE"].apply(risk_level_label)
    df["RECOMMENDATION"] = df["RISK_LEVEL"].apply(recommendation_text)
    df["SCORE_ID"] = [str(uuid.uuid4())[:12] for _ in range(len(df))]
    df["SCORED_AT"] = datetime.utcnow().isoformat()

    output_cols = [
        "SCORE_ID", "TRANSACTION_ID", "USER_ID",
        "ZSCORE_FLAG", "VELOCITY_FLAG", "CATEGORY_RISK_SCORE",
        "GEOGRAPHIC_RISK_SCORE", "ISOLATION_FOREST_SCORE",
        "COMBINED_RISK_SCORE", "RISK_LEVEL", "RECOMMENDATION",
        "SCORED_AT",
    ]
    return df[output_cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fraud Scoring Model")
    parser.add_argument(
        "--input", type=str,
        default=str(PROJECT_ROOT / "features_output.csv"),
        help="Path to features CSV (output of feature_engineering.py)",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "fraud_scores_output.csv"),
        help="Output path for scored results",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("  Run feature_engineering.py first.")
        sys.exit(1)

    t0 = time.time()
    try:
        print("🔄 Loading features …")
        features_df = pd.read_csv(input_path)
        print(f"   {len(features_df)} rows loaded.")

        print("🔄 Scoring transactions …")
        scores_df = score_transactions(features_df)
        elapsed = (time.time() - t0) * 1000

        scores_df.to_csv(args.output, index=False)
        print(f"✅ Scores saved to {args.output}")
        print(f"\n📊 Risk Distribution:")
        print(scores_df["RISK_LEVEL"].value_counts().to_string())
        print(f"\n   Avg risk score: {scores_df['COMBINED_RISK_SCORE'].mean():.4f}")
        print(f"   Max risk score: {scores_df['COMBINED_RISK_SCORE'].max():.4f}")

        log_pipeline_event("scoring", "success", len(scores_df), elapsed)

    except Exception as exc:
        elapsed = (time.time() - t0) * 1000
        log_pipeline_event("scoring", "failed", error_message=str(exc))
        print(f"❌ Scoring failed: {exc}")
        raise


if __name__ == "__main__":
    main()
