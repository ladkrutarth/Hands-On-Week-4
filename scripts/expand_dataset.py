"""
GraphGuard — Dataset Expansion Utility
Clones existing user data to expand the dataset from 10 to 50+ users.
Adds jitter to amounts and dates to create unique behavior patterns.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "transactions_3000.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "transactions_3000.csv"  # Overwrite original

def expand_dataset(target_users=50, jitter_amount=0.1):
    print(f"📂 Reading base dataset from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        print("❌ Base dataset not found. Please ensure transactions_3000.csv exists.")
        return

    df = pd.read_csv(INPUT_PATH)
    current_users = sorted(df["USER_ID"].unique())
    n_current = len(current_users)
    
    if n_current >= target_users:
        print(f"✅ Dataset already has {n_current} users. No expansion needed.")
        return

    print(f"📈 Expanding dataset from {n_current} to {target_users} users...")
    
    new_dfs = [df]
    orig_df = df.copy()
    
    for i in range(n_current + 1, target_users + 1):
        new_user_id = f"USER_{str(i).zfill(3)}"
        # Clone a random original user
        source_user = current_users[(i - 1) % n_current]
        cloned_df = orig_df[orig_df["USER_ID"] == source_user].copy()
        
        # Update User ID
        cloned_df["USER_ID"] = new_user_id
        
        # Jitter amount (±10% by default)
        cloned_df["AMOUNT"] = cloned_df["AMOUNT"] * (1 + np.random.uniform(-jitter_amount, jitter_amount, len(cloned_df)))
        cloned_df["AMOUNT"] = cloned_df["AMOUNT"].round(2)
        
        # Jitter dates (± few hours)
        cloned_df["TRANSACTION_DATE"] = pd.to_datetime(cloned_df["TRANSACTION_DATE"])
        cloned_df["TRANSACTION_DATE"] = cloned_df["TRANSACTION_DATE"] + pd.to_timedelta(np.random.randint(-12, 12, len(cloned_df)), unit='h')
        cloned_df["TRANSACTION_DATE"] = cloned_df["TRANSACTION_DATE"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        
        # New Transaction IDs (deterministic-ish suffix)
        cloned_df["TRANSACTION_ID"] = cloned_df["TRANSACTION_ID"].apply(lambda x: f"{x[:12]}_{str(i).zfill(3)}")
        
        new_dfs.append(cloned_df)
        
    expanded_df = pd.concat(new_dfs).sort_values("TRANSACTION_DATE").reset_index(drop=True)
    
    print(f"💾 Saving expanded dataset to: {OUTPUT_PATH}")
    expanded_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Expanded dataset ready! {len(expanded_df)} transactions for {target_users} users.")

if __name__ == "__main__":
    expand_dataset(target_users=50)
