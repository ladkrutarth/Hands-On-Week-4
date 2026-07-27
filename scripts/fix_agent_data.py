import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "csv_data"
MODEL_DIR = PROJECT_ROOT / "artifacts"

# Output Paths requested by GuardAgent
FRAUD_SCORES_PATH = DATA_DIR / "fraud_scores_output.csv"
AUTH_PROFILES_PATH = DATA_DIR / "auth_profiles_output.csv"
TXN_3000_PATH = DATA_DIR / "transactions_3000.csv"

# Input Paths
FEATURES_PATH = DATA_DIR / "processed_fraud_train.csv"
MODEL_PATH = MODEL_DIR / "fraud_model_rf.joblib"
ENCODERS_PATH = MODEL_DIR / "encoders.joblib"

def fix_data():
    print("🚀 Veriscan GuardAgent Data Synchronization...")
    
    if not FEATURES_PATH.exists():
        print(f"❌ Cannot find {FEATURES_PATH}. Please run prepare_fraud_data.py first.")
        return

    if not MODEL_PATH.exists():
        print(f"❌ Cannot find {MODEL_PATH}. Please run train_fraud_model.py first.")
        return

    # 1. Load data and model
    print("📂 Loading data and model...")
    df = pd.read_csv(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    # 2. Add synthetic User IDs if missing
    # (The processed data from Kaggle has first/last but the agent likes USER_ID)
    if "USER_ID" not in df.columns:
        print("🆔 Generating synthetic User IDs...")
        user_map = {name: f"USER_{i:03d}" for i, name in enumerate(df.groupby(['first', 'last']).groups.keys())}
        df["USER_ID"] = df.apply(lambda r: user_map[(r['first'], r['last'])], axis=1)

    # 3. Predict Scores
    print("🔮 Generating fraud scores...")
    features = ["category", "amt", "gender", "state", "merchant", "hour", "day_of_week"]
    X = df[features].copy()
    for col in ["category", "gender", "state", "merchant"]:
        X[col] = encoders[col].transform(X[col].astype(str))
    
    probs = model.predict_proba(X)[:, 1]
    
    # 4. Create Fraud Scores Output
    print(f"💾 Saving {FRAUD_SCORES_PATH}...")
    scores_df = pd.DataFrame({
        "TRANSACTION_ID": [f"TXN_{i:06d}" for i in range(len(df))],
        "USER_ID": df["USER_ID"],
        "CATEGORY": df["category"],
        "MERCHANT": df["merchant"],
        "COMBINED_RISK_SCORE": probs * 100,
        "RISK_LEVEL": ["CRITICAL" if p > 0.8 else "HIGH" if p > 0.5 else "MEDIUM" if p > 0.2 else "LOW" for p in probs],
        "ZSCORE_FLAG": np.random.uniform(0, 5, len(df)), # Synthetic for analyst flavor
        "VELOCITY_FLAG": np.random.uniform(0, 5, len(df)),
        "GEOGRAPHIC_RISK_SCORE": np.random.uniform(0, 10, len(df))
    })
    scores_df.to_csv(FRAUD_SCORES_PATH, index=False)

    # 5. Create Auth Profiles Output
    print(f"💾 Saving {AUTH_PROFILES_PATH}...")
    profiles = scores_df.groupby("USER_ID").agg(
        AVG_RISK=("COMBINED_RISK_SCORE", "mean"),
        HIGH_RISK_COUNT=("RISK_LEVEL", lambda x: (x == "HIGH").sum() + (x == "CRITICAL").sum()),
        TXN_COUNT=("USER_ID", "count")
    ).reset_index()
    profiles["RECOMMENDED_SECURITY_LEVEL"] = profiles["AVG_RISK"].apply(
        lambda x: "MFA_REQUIRED" if x > 40 else "Standard"
    )
    profiles.to_csv(AUTH_PROFILES_PATH, index=False)

    # 6. Ensure transactions_3000.csv exists (satisfies HabitModel)
    print(f"💾 Saving {TXN_3000_PATH}...")
    # Add dummy location if missing
    if "location" not in df.columns:
        df["location"] = df["state"].apply(lambda s: f"Sample City, {s}")
    
    # Standardize columns for HabitModel
    df_mini = df.copy()
    df_mini.columns = [c.upper() for c in df_mini.columns]
    df_mini.to_csv(TXN_3000_PATH, index=False)

    print("\n✅ Sync complete. GuardAgent tools are now ready to use.")

if __name__ == "__main__":
    fix_data()
