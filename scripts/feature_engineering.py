"""
GraphGuard — Feature Engineering Pipeline
Computes transaction features from raw CSV or Snowflake table.
Outputs features as CSV or loads into Snowflake TRANSACTION_FEATURES table.
"""

import argparse
import csv
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
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

# Category risk weights — mirrors sql/analytical_queries.sql query #8
CATEGORY_RISK = {
    "Jewelry": 0.9,
    "Electronics": 0.7,
    "Clothing": 0.4,
    "Retail": 0.3,
    "Restaurants": 0.2,
    "Grocery": 0.15,
    "Gas Stations": 0.1,
    "Coffee Shops": 0.05,
}


def log_pipeline_event(stage, status, records=0, duration_ms=0.0, error_message=""):
    """Append to pipeline_logs.csv."""
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
# Feature computation
# ---------------------------------------------------------------------------

def compute_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user aggregate statistics."""
    stats = df.groupby("USER_ID")["AMOUNT"].agg(
        USER_AVG_AMOUNT="mean",
        USER_STD_AMOUNT="std",
        USER_TOTAL_TRANSACTIONS="count",
    ).reset_index()
    stats["USER_STD_AMOUNT"] = stats["USER_STD_AMOUNT"].fillna(0)
    return stats


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute transaction counts in rolling windows (1h, 24h, 7d)."""
    df = df.sort_values(["USER_ID", "TRANSACTION_DATE"])
    df["TRANSACTION_DATE"] = pd.to_datetime(df["TRANSACTION_DATE"])

    results = []
    for user_id, group in df.groupby("USER_ID"):
        group = group.sort_values("TRANSACTION_DATE")
        dates = group["TRANSACTION_DATE"].values

        count_1h, count_24h, count_7d = [], [], []
        for i, dt in enumerate(dates):
            dt_ts = pd.Timestamp(dt)
            c1 = ((dates[:i] >= (dt_ts - pd.Timedelta(hours=1)).to_numpy()) &
                   (dates[:i] <= dt)).sum()
            c24 = ((dates[:i] >= (dt_ts - pd.Timedelta(hours=24)).to_numpy()) &
                    (dates[:i] <= dt)).sum()
            c7 = ((dates[:i] >= (dt_ts - pd.Timedelta(days=7)).to_numpy()) &
                   (dates[:i] <= dt)).sum()
            count_1h.append(int(c1))
            count_24h.append(int(c24))
            count_7d.append(int(c7))

        group = group.copy()
        group["TXN_COUNT_1H"] = count_1h
        group["TXN_COUNT_24H"] = count_24h
        group["TXN_COUNT_7D"] = count_7d
        results.append(group)

    return pd.concat(results, ignore_index=True)


def compute_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute geographic features: is_new_location and location_entropy."""
    # Primary location per user
    user_loc_counts = df.groupby(["USER_ID", "LOCATION"]).size().reset_index(name="loc_count")
    primary = user_loc_counts.sort_values("loc_count", ascending=False).drop_duplicates("USER_ID")
    primary = primary.rename(columns={"LOCATION": "PRIMARY_LOCATION"})[["USER_ID", "PRIMARY_LOCATION"]]

    df = df.merge(primary, on="USER_ID", how="left")
    df["IS_NEW_LOCATION"] = df["LOCATION"] != df["PRIMARY_LOCATION"]

    # Location entropy per user
    def entropy(group):
        probs = group["LOCATION"].value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs + 1e-9))

    ent = df.groupby("USER_ID").apply(entropy).reset_index(name="LOCATION_ENTROPY")
    df = df.merge(ent, on="USER_ID", how="left")
    df.drop(columns=["PRIMARY_LOCATION"], inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = df.copy()
    df["TRANSACTION_DATE"] = pd.to_datetime(df["TRANSACTION_DATE"])

    # User statistics
    user_stats = compute_user_stats(df)
    df = df.merge(user_stats, on="USER_ID", how="left")

    # Amount features
    df["AMOUNT_ZSCORE"] = (df["AMOUNT"] - df["USER_AVG_AMOUNT"]) / df["USER_STD_AMOUNT"].replace(0, 1)
    df["AMOUNT_ZSCORE"] = df["AMOUNT_ZSCORE"].round(4)
    df["IS_HIGH_VALUE"] = df["AMOUNT_ZSCORE"] > 2

    # Category risk
    df["CATEGORY_RISK_WEIGHT"] = df["CATEGORY"].map(CATEGORY_RISK).fillna(0.3)

    # Time features
    df["HOUR_OF_DAY"] = df["TRANSACTION_DATE"].dt.hour
    df["DAY_OF_WEEK"] = df["TRANSACTION_DATE"].dt.dayofweek
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([5, 6])

    # Velocity
    df = compute_velocity_features(df)

    # Geographic
    df = compute_location_features(df)

    # Computed timestamp
    df["COMPUTED_AT"] = datetime.utcnow().isoformat()

    # Select output columns
    output_cols = [
        "TRANSACTION_ID", "USER_ID", "AMOUNT", "AMOUNT_ZSCORE", "IS_HIGH_VALUE",
        "USER_AVG_AMOUNT", "USER_STD_AMOUNT", "USER_TOTAL_TRANSACTIONS",
        "TXN_COUNT_1H", "TXN_COUNT_24H", "TXN_COUNT_7D",
        "CATEGORY", "CATEGORY_RISK_WEIGHT",
        "HOUR_OF_DAY", "DAY_OF_WEEK", "IS_WEEKEND",
        "LOCATION", "IS_NEW_LOCATION", "LOCATION_ENTROPY",
        "COMPUTED_AT",
    ]
    # Add optional Kaggle-specific columns if present
    for opt_col in ["IS_FRAUD_ACTUAL", "MERCHANT_NAME", "CARDHOLDER_NAME"]:
        if opt_col in df.columns:
            output_cols.append(opt_col)
    return df[output_cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument(
        "--input", type=str,
        default=str(PROJECT_ROOT / "dataset" / "csv_data" / "transactions_kaggle.csv"),
        help="Path to raw transactions CSV (default: transactions_kaggle.csv)",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Output features to CSV file (default: features_output.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    t0 = time.time()
    try:
        print("🔄 Loading raw transactions …")
        df = pd.read_csv(input_path)
        print(f"   {len(df)} transactions loaded.")

        print("🔄 Computing features …")
        features_df = engineer_features(df)
        elapsed = (time.time() - t0) * 1000

        output_path = args.output_csv or str(PROJECT_ROOT / "dataset" / "csv_data" / "features_output.csv")
        features_df.to_csv(output_path, index=False)
        print(f"✅ Features saved to {output_path}")
        print(f"   {len(features_df)} rows, {len(features_df.columns)} columns")
        print(f"   High-value transactions: {features_df['IS_HIGH_VALUE'].sum()}")
        print(f"   New-location transactions: {features_df['IS_NEW_LOCATION'].sum()}")

        log_pipeline_event("feature_engineering", "success", len(features_df), elapsed)

    except Exception as exc:
        elapsed = (time.time() - t0) * 1000
        log_pipeline_event("feature_engineering", "failed", error_message=str(exc))
        print(f"❌ Feature engineering failed: {exc}")
        raise


if __name__ == "__main__":
    main()
