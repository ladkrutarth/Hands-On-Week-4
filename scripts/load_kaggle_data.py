"""
GraphGuard — Kaggle Dataset Adapter
Converts the kartik2112/fraud-detection Kaggle CSV (fraudTrain.csv / fraudTest.csv)
into the project's internal transaction schema.

Dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection

Kaggle columns:
  Unnamed:0, trans_date_trans_time, cc_num, merchant, category, amt,
  first, last, gender, street, city, state, zip, lat, long, city_pop,
  job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud

Internal columns produced:
  TRANSACTION_ID, USER_ID, AMOUNT, CATEGORY, LOCATION,
  TRANSACTION_DATE, IS_FRAUD_ACTUAL, MERCHANT_NAME, CARDHOLDER_NAME

Usage:
  python scripts/load_kaggle_data.py --input dataset/raw/fraudTrain.csv --sample 10000
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT  = PROJECT_ROOT / "dataset" / "raw" / "fraudTrain.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "csv_data" / "transactions_kaggle.csv"
OLD_DATASET    = PROJECT_ROOT / "dataset" / "csv_data" / "transactions_3000.csv"

# ---------------------------------------------------------------------------
# Category mapping — Kaggle categories → cleaner display names
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "grocery_pos":           "Grocery",
    "grocery_net":           "Grocery",
    "gas_transport":         "Gas Stations",
    "entertainment":         "Entertainment",
    "food_dining":           "Restaurants",
    "shopping_pos":          "Retail",
    "shopping_net":          "Online Shopping",
    "health_fitness":        "Health & Fitness",
    "personal_care":         "Personal Care",
    "home":                  "Home & Garden",
    "kids_pets":             "Kids & Pets",
    "travel":                "Travel",
    "misc_pos":              "Miscellaneous",
    "misc_net":              "Miscellaneous",
}


def map_category(raw: str) -> str:
    return CATEGORY_MAP.get(str(raw).strip().lower(), str(raw).title())


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

def load_and_normalize(
    input_path: Path,
    output_path: Path,
    n_users: int = 10,
    sample_size: int = 10_000,
    fraud_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Read Kaggle CSV and produce the normalized internal schema."""

    print(f"📂 Loading Kaggle dataset from: {input_path}")
    if not input_path.exists():
        print(f"\n❌ File not found: {input_path}")
        print("   Please download fraudTrain.csv from:")
        print("   https://www.kaggle.com/datasets/kartik2112/fraud-detection")
        print(f"   and place it at: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path, low_memory=False)
    print(f"   {len(df):,} rows loaded, {df.shape[1]} columns")

    # --- Step 1: Select top-N most active credit cards as our "users" ---
    top_cards = (
        df["cc_num"]
        .value_counts()
        .head(n_users)
        .index
        .tolist()
    )
    df = df[df["cc_num"].isin(top_cards)].copy()
    print(f"   {len(df):,} rows after filtering top {n_users} cardholders")

    # --- Step 2: Sample balanced fraud/legit ---
    n_fraud = int(sample_size * fraud_ratio)
    n_legit = sample_size - n_fraud

    fraud_df = df[df["is_fraud"] == 1].sample(
        min(n_fraud, df["is_fraud"].sum()), random_state=seed
    )
    legit_df = df[df["is_fraud"] == 0].sample(
        min(n_legit, (df["is_fraud"] == 0).sum()), random_state=seed
    )
    df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"   Sampled {len(df):,} rows ({len(fraud_df)} fraud, {len(legit_df)} legit)")

    # --- Step 3: Build USER_ID mapping ---
    card_to_user = {
        card: f"USER_{str(i+1).zfill(3)}"
        for i, card in enumerate(top_cards)
    }
    df["USER_ID"] = df["cc_num"].map(card_to_user)

    # --- Step 4: Map all columns to internal schema ---
    out = pd.DataFrame()
    out["TRANSACTION_ID"]  = df["trans_num"].astype(str).str[:16].str.upper()
    out["USER_ID"]         = df["USER_ID"]
    out["AMOUNT"]          = df["amt"].round(2)
    out["CATEGORY"]        = df["category"].apply(map_category)
    out["LOCATION"]        = df["city"].str.strip() + ", " + df["state"].str.strip()
    out["TRANSACTION_DATE"]= pd.to_datetime(df["trans_date_trans_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["IS_FRAUD_ACTUAL"] = df["is_fraud"].astype(int)
    out["MERCHANT_NAME"]   = df["merchant"].str.replace("fraud_", "", regex=False).str.strip()
    out["CARDHOLDER_NAME"] = df["first"].str.strip() + " " + df["last"].str.strip()

    # --- Step 5: Sort by date ---
    out = out.sort_values("TRANSACTION_DATE").reset_index(drop=True)

    print(f"\n📊 Category distribution:")
    print(out["CATEGORY"].value_counts().to_string())
    print(f"\n📊 Users: {out['USER_ID'].unique().tolist()}")
    print(f"📊 Fraud rate: {out['IS_FRAUD_ACTUAL'].mean():.1%}")
    print(f"📊 Avg amount: ${out['AMOUNT'].mean():.2f}")
    print(f"📊 Date range: {out['TRANSACTION_DATE'].min()} → {out['TRANSACTION_DATE'].max()}")

    return out


def main():
    parser = argparse.ArgumentParser(description="Kaggle Fraud Dataset Adapter")
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT),
        help=f"Path to Kaggle fraudTrain.csv (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--sample", type=int, default=10_000,
        help="Number of rows to sample (default: 10000)",
    )
    parser.add_argument(
        "--users", type=int, default=10,
        help="Number of top cardholders to use as users (default: 10)",
    )
    parser.add_argument(
        "--fraud-ratio", type=float, default=0.15,
        help="Fraction of fraudulent transactions in sample (default: 0.15)",
    )
    parser.add_argument(
        "--remove-old", action="store_true", default=True,
        help="Remove old transactions_3000.csv if present",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_normalize(
        input_path, output_path,
        n_users=args.users,
        sample_size=args.sample,
        fraud_ratio=args.fraud_ratio,
    )

    df.to_csv(output_path, index=False)
    print(f"\n✅ Normalized dataset saved to: {output_path}")
    print(f"   {len(df):,} transactions, {df.shape[1]} columns")

    # Remove old synthetic dataset
    if args.remove_old and OLD_DATASET.exists():
        OLD_DATASET.unlink()
        print(f"🗑️  Removed old dataset: {OLD_DATASET}")

    print("\n🚀 Next steps:")
    print("   python scripts/feature_engineering.py")
    print("   python models/fraud_model.py")
    print("   python models/auth_decision.py")


if __name__ == "__main__":
    main()
