"""
Data-only tools for risk and user profile. No MLX/LLM imports.
Used by the API so /api/fraud/high-risk and /api/user/{id}/risk work without loading GuardAgent.
"""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATHS = {
    "features": PROJECT_ROOT / "data" / "csv_data" / "features_output.csv",
    "fraud_scores": PROJECT_ROOT / "data" / "csv_data" / "fraud_scores_output.csv",
    "auth_profiles": PROJECT_ROOT / "data" / "csv_data" / "auth_profiles_output.csv",
    "transactions": PROJECT_ROOT / "data" / "csv_data" / "transactions_3000.csv",
}

_fraud_df = None
_auth_df = None

# Global Scam Statistics (Losses in $B) - FBI/IC3 2024 Estimates
GLOBAL_SCAM_STATS = {
    "Investment Scams": 46.9,
    "Business Email Compromise": 19.8,
    "Tech Support Scams": 10.5,
    "Non-Payment/Delivery": 5.6,
    "Confidence/Romance Scams": 4.8,
    "Govt. Impersonation": 2.9,
    "Employment Scams": 2.0,
    "Credit Card/Check Fraud": 1.9,
    "Real Estate/Rental": 1.4,
    "Misc. Cyber-enabled": 1.2
}


def _load_cache():
    global _fraud_df, _auth_df
    fraud_path = DATA_PATHS["fraud_scores"]
    if fraud_path.exists() and _fraud_df is None:
        _fraud_df = pd.read_csv(fraud_path)
        _fraud_df.columns = [c.upper() for c in _fraud_df.columns]
    auth_path = DATA_PATHS["auth_profiles"]
    if auth_path.exists() and _auth_df is None:
        _auth_df = pd.read_csv(auth_path)
        _auth_df.columns = [c.upper() for c in _auth_df.columns]


def tool_get_user_risk_profile(user_id: str) -> Dict[str, Any]:
    _load_cache()
    result = {"user_id": user_id, "found": False}
    if _auth_df is not None:
        user_data = _auth_df[_auth_df["USER_ID"] == user_id]
        if not user_data.empty:
            row = user_data.iloc[0]
            result.update({
                "found": True,
                "security_level": str(row.get("RECOMMENDED_SECURITY_LEVEL", "N/A")),
                "avg_risk": float(row.get("AVG_RISK", 0)),
                "high_risk_count": int(row.get("HIGH_RISK_COUNT", 0))
            })
    if _fraud_df is not None:
        user_data = _fraud_df[_fraud_df["USER_ID"] == user_id]
        if not user_data.empty:
            result.update({
                "found": True,
                "txn_count": len(user_data),
                "avg_combined_score": float(user_data["COMBINED_RISK_SCORE"].mean()),
                "risk_distribution": user_data["RISK_LEVEL"].value_counts().to_dict()
            })
    return result


def tool_get_high_risk_transactions(limit: int = 10) -> List[Dict]:
    _load_cache()
    if _fraud_df is None:
        return []
    df = _fraud_df.sort_values("COMBINED_RISK_SCORE", ascending=False).head(limit)
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# Transaction validation (risk score for single txn) — fast path for ADDF
# ---------------------------------------------------------------------------
_model_cache = None
_encoders_cache = None


def _load_fraud_model():
    global _model_cache, _encoders_cache
    if _model_cache is not None:
        return _model_cache, _encoders_cache
    model_path = PROJECT_ROOT / "models" / "fraud_model_rf.joblib"
    enc_path = PROJECT_ROOT / "models" / "encoders.joblib"
    if model_path.exists() and enc_path.exists():
        import joblib
        _model_cache = joblib.load(model_path)
        _encoders_cache = joblib.load(enc_path)
    return _model_cache, _encoders_cache


# Heuristic when no trained model: category + amount (fast, no disk)
_CATEGORY_RISK = {"shopping_net": 0.5, "travel": 0.4, "entertainment": 0.35, "health": 0.3, "transfer": 0.6, "gas_transport": 0.2}


def score_transaction(category: str, amt: float, merchant: str, hour: int, day_of_week: int) -> float:
    """
    Returns COMBINED_RISK_SCORE (0–100). Uses RF model if available, else fast heuristic.
    """
    model, encoders = _load_fraud_model()
    if model is not None and encoders is not None:
        try:
            import numpy as np
            X = pd.DataFrame([{
                "category": category, "amt": amt, "gender": "M", "state": "CA",
                "merchant": merchant or "unknown", "hour": hour, "day_of_week": day_of_week,
            }])
            for col in ["category", "gender", "state", "merchant"]:
                if col in encoders:
                    X[col] = encoders[col].transform(X[col].astype(str))
            features = ["category", "amt", "gender", "state", "merchant", "hour", "day_of_week"]
            X = X[[c for c in features if c in X.columns]]
            prob = model.predict_proba(X)[0, 1]
            return float(prob * 100)
        except Exception:
            pass
    # Fast heuristic
    base = _CATEGORY_RISK.get(category, 0.25)
    amount_factor = min(1.0, amt / 3000.0) * 0.4
    fraud_merchant = 0.3 if merchant and "fraud_" in str(merchant).lower() else 0
    return (base + amount_factor + fraud_merchant) * 100
