"""
Generate Spending DNA Dataset — 90,000 realistic rows.
Aligned with the Financial Advisor Dataset (Aug 2024 - Dec 2025).

Simulates the 8-axis financial fingerprint per user:
  1. avg_txn_amount       — Average transaction size
  2. location_entropy     — Geographic diversity score
  3. weekend_ratio        — Weekend spending fraction
  4. category_diversity   — Number of unique categories used (normalized)
  5. time_of_day_pref     — Preferred hour bucket (0=morning, 1=afternoon, 2=evening, 3=night)
  6. risk_appetite_score  — Weighted % of high-risk category spend
  7. spending_velocity    — Avg transactions per week
  8. merchant_loyalty_score — Repeat merchant fraction

Run: python scripts/generate_spending_dna_dataset.py
Output: dataset/csv_data/spending_dna_dataset.csv
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "spending_dna_dataset.csv"

random.seed(7)
np.random.seed(7)

# ── Config (Aligned with Advisor Dataset) ──────────────────────────────────
N_USERS = 1000
TARGET_ROWS = 90_000

# Same Categories as Advisor Dataset
CATEGORY_WEIGHTS = {
    "Housing (Rent/Mortgage)": 0.25,
    "Groceries":               0.15,
    "Utilities & Bills":       0.10,
    "Dining & Restaurants":    0.08,
    "Transportation & Gas":    0.08,
    "Healthcare":              0.05,
    "Online Shopping":         0.07,
    "Subscriptions":           0.04,
    "Travel & Leisure":        0.05,
    "Clothing & Fashion":      0.03,
    "Electronics":             0.02,
    "Entertainment":           0.03,
    "Education":               0.02,
    "Insurance":               0.02,
    "Misc / Cash":             0.01,
}

CATEGORIES = list(CATEGORY_WEIGHTS.keys())

RISK_WEIGHTS = {
    "Electronics":             0.85,
    "Online Shopping":         0.70,
    "Travel & Leisure":        0.60,
    "Entertainment":           0.45,
    "Clothing & Fashion":      0.40,
    "Dining & Restaurants":    0.30,
    "Misc / Cash":             0.25,
    "Groceries":               0.10,
    "Healthcare":              0.05,
    "Utilities & Bills":       0.05,
    "Housing (Rent/Mortgage)": 0.02,
    "Education":               0.10,
    "Insurance":               0.02,
    "Transportation & Gas":    0.08,
    "Subscriptions":           0.15,
}

CITIES = [
    ("Los Angeles", "CA"), ("New York", "NY"), ("Houston", "TX"), ("Miami", "FL"), ("Chicago", "IL"),
    ("Philadelphia", "PA"), ("Phoenix", "AZ"), ("Atlanta", "GA"), ("Charlotte", "NC"), ("Seattle", "WA"),
    ("Denver", "CO"), ("Boston", "MA"), ("Dallas", "TX"), ("Detroit", "MI"), ("Newark", "NJ")
]

REGULAR_MERCHANTS = {
    "Groceries":        ["Whole Foods", "Kroger", "Walmart", "Trader Joe's", "ALDI", "Costco", "Publix"],
    "Dining & Restaurants": ["McDonald's", "Chipotle", "Olive Garden", "Chick-fil-A", "Local Diner", "Sushi Zen", "Steakhouse"],
    "Entertainment":    ["AMC Theaters", "Regal Cinema", "Dave & Buster's", "Sky Zone", "Ticketmaster"],
    "Travel & Leisure": ["Delta Airlines", "United Airlines", "Marriott", "Airbnb", "Uber", "Expedia"],
    "Clothing & Fashion": ["H&M", "Zara", "Nike", "Forever 21", "Nordstrom", "Lululemon"],
    "Electronics":      ["Best Buy", "Apple Store", "B&H Photo", "Micro Center", "GameStop"],
    "Transportation & Gas": ["Shell", "Chevron", "ExxonMobil", "BP", "Tesla Supercharger", "Lyft"],
    "Healthcare":       ["CVS Pharmacy", "Walgreens", "LabCorp", "Urgent Care", "Kaiser Permanente"],
    "Utilities & Bills": ["AT&T", "Comcast", "Duke Energy", "Verizon", "ConEd"],
    "Housing (Rent/Mortgage)": ["Property Mgmt", "Chase Mortgage", "Wells Fargo Home", "Local Landlord"],
    "Online Shopping":  ["Amazon", "eBay", "Etsy", "Wayfair", "Target", "Temu", "Shein"],
    "Education":        ["Coursera", "Udemy", "Local College", "Student Loan Svc"],
    "Insurance":        ["GEICO", "State Farm", "Progressive", "BlueCross BlueShield"],
}

SUBSCRIPTION_MERCHANTS = [
    "Netflix", "Spotify", "Amazon Prime", "Hulu", "Disney+", "Apple One", "HBO Max"
]

# ── DNA Generation Logic ───────────────────────────────────────────────────

def build_user_dna_profile(user_id: str) -> dict:
    """Generate a stable per-user DNA fingerprint baseline."""
    return {
        "user_id":              user_id,
        "primary_city":        random.choice(CITIES),
        "avg_txn_amount":      round(np.random.lognormal(4.0, 0.8), 2),
        "location_entropy":    round(random.uniform(0.1, 3.0), 4),
        "weekend_ratio":       round(random.uniform(0.15, 0.75), 4),
        "category_diversity":  round(random.uniform(0.3, 1.0), 4),
        "time_of_day_pref":    random.choice([0, 1, 2, 3]), # 0: Morn, 1: Aft, 2: Eve, 3: Night
        "risk_appetite_score": round(random.uniform(0.05, 0.50), 4),
        "spending_velocity":   round(random.uniform(3.0, 25.0), 2),
        "merchant_loyalty":    round(random.uniform(0.2, 0.85), 4),
        "fav_categories":      random.sample(CATEGORIES, k=random.randint(4, 8)),
    }

def get_random_date_in_range():
    # August 2024 to December 2025
    start_date = datetime(2024, 8, 1)
    end_date = datetime(2025, 12, 31)
    delta = end_date - start_date
    return start_date + timedelta(days=random.randint(0, delta.days))

def hour_from_pref(pref: int) -> int:
    buckets = {0: (6, 11), 1: (11, 16), 2: (16, 21), 3: (21, 24)}
    lo, hi = buckets[pref]
    return random.randint(lo, hi - 1)

def generate_session_row(dna: dict, is_anomalous: bool = False) -> dict:
    txn_date = get_random_date_in_range()
    is_weekend = txn_date.weekday() >= 5

    # Location drift
    if is_anomalous and random.random() < 0.65:
        city_obj = random.choice([c for c in CITIES if c != dna["primary_city"]])
    else:
        city_obj = dna["primary_city"]
    
    city, state = city_obj

    # Category choices
    if is_anomalous and random.random() < 0.5:
        category = random.choice(["Electronics", "Online Shopping", "Travel & Leisure"])
    else:
        category = random.choice(dna["fav_categories"])

    merchant_list = REGULAR_MERCHANTS.get(category, ["Miscellaneous"])
    if category == "Subscriptions":
        merchant = random.choice(SUBSCRIPTION_MERCHANTS)
    else:
        merchant = random.choice(merchant_list)

    # Amount drift
    base_amt = dna["avg_txn_amount"]
    if is_anomalous:
        amount = round(base_amt * random.uniform(2.5, 15.0), 2)
    else:
        amount = round(max(5.0, np.random.normal(base_amt, base_amt * 0.35)), 2)

    # Time of day drift
    hour = hour_from_pref(dna["time_of_day_pref"])
    if is_anomalous and random.random() < 0.4:
        hour = random.choice([0, 1, 2, 3, 4, 5]) # Early morning / Late night

    # ── DNA Scoring Logic ───────────────────────────────────────────────
    amount_dev = min(abs(amount - dna["avg_txn_amount"]) / max(dna["avg_txn_amount"], 1), 1.0)
    loc_dev    = 0.95 if city_obj != dna["primary_city"] else 0.05
    time_pref  = dna["time_of_day_pref"]
    current_bucket = 0 if hour < 11 else (1 if hour < 16 else (2 if hour < 21 else 3))
    time_dev   = min(abs(current_bucket - time_pref) / 3, 1.0)
    cat_dev    = 0.1 if category in dna["fav_categories"] else 0.8
    
    dna_deviation = round((amount_dev * 0.3 + loc_dev * 0.35 + time_dev * 0.15 + cat_dev * 0.20), 4)
    trust_score   = round(max(0.0, 1.0 - dna_deviation + random.uniform(-0.04, 0.04)), 4)

    return {
        "user_id":              dna["user_id"],
        "session_id":           f"SES_{random.randint(100000, 999999)}",
        "transaction_date":     txn_date.strftime("%Y-%m-%d"),
        "transaction_time":     f"{hour:02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
        "hour_of_day":          hour,
        "is_weekend":           is_weekend,
        "city":                 city,
        "state":                state,
        "category":             category,
        "merchant":             merchant,
        "amount":               amount,
        # ── 8 DNA Axes (Baselines) ───────────────────────────────────────
        "avg_txn_amount":       dna["avg_txn_amount"],
        "location_entropy":     dna["location_entropy"],
        "weekend_ratio":        dna["weekend_ratio"],
        "category_diversity":   dna["category_diversity"],
        "time_of_day_pref":     dna["time_of_day_pref"],
        "risk_appetite_score":  dna["risk_appetite_score"],
        "spending_velocity":    dna["spending_velocity"],
        "merchant_loyalty_score": dna["merchant_loyalty"],
        # ── Metrics ──────────────────────────────────────────────────────
        "dna_deviation_score":  dna_deviation,
        "trust_score":          trust_score,
        "is_anomalous_session": is_anomalous,
    }

def build_dataset() -> pd.DataFrame:
    dna_profiles = [build_user_dna_profile(f"USER_{i:04d}") for i in range(N_USERS)]
    rows_per_user = TARGET_ROWS // N_USERS
    rows = []

    for dna in dna_profiles:
        for _ in range(rows_per_user):
            # 8% Anomalous sessions (aligned with higher fraud interest)
            is_anomalous = random.random() < 0.08
            rows.append(generate_session_row(dna, is_anomalous=is_anomalous))

    df = pd.DataFrame(rows)
    # Sort for consistency
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df = df.sort_values(['user_id', 'transaction_date']).reset_index(drop=True)
    df['transaction_date'] = df['transaction_date'].dt.strftime('%Y-%m-%d')
    
    return df.head(TARGET_ROWS)

if __name__ == "__main__":
    print(f"🔄 Generating Aligned Spending DNA dataset ({TARGET_ROWS} rows)…")
    print(f"📅 Range: Aug 2024 to Dec 2025")
    
    df = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Success: {len(df):,} rows → {OUTPUT_PATH}")
    print(f"📊 Anomalous Sessions: {df['is_anomalous_session'].sum()} ({(df['is_anomalous_session'].mean()*100):.1f}%)")
    print(f"👥 Users: {df['user_id'].nunique()}")
    print(f"📅 Dates: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
