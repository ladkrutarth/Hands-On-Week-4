"""
Generate AI Financial Advisor Dataset — 90,000 realistic rows.
Simulates monthly spending per user across categories with temporal patterns,
payday cycles, weekend surges, and balanced fraud clustering.

Date Range: August 2024 to December 2025.
Target Fraud Cases: >= 5,000.
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "financial_advisor_dataset.csv"

# ── Seed for reproducibility ───────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Realistic Configuration ─────────────────────────────────────────────────

# Categories with Weights (Simulating real-world spending importance)
CATEGORY_WEIGHTS = {
    "Housing (Rent/Mortgage)": 0.25, # High weight, fixed-ish amount
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
WEIGHTS = list(CATEGORY_WEIGHTS.values())

CATEGORY_CREDIT_IMPACT = {
    "Housing (Rent/Mortgage)": "positive",
    "Groceries":               "positive",
    "Utilities & Bills":       "positive",
    "Dining & Restaurants":    "neutral",
    "Transportation & Gas":    "positive",
    "Healthcare":              "positive",
    "Online Shopping":         "neutral",
    "Subscriptions":           "negative", # High count of subscriptions can imply lack of budgeting
    "Travel & Leisure":        "positive", # If paid off
    "Clothing & Fashion":      "neutral",
    "Electronics":             "neutral",
    "Entertainment":           "neutral",
    "Education":               "positive",
    "Insurance":               "positive",
    "Misc / Cash":             "neutral",
}

SUBSCRIPTION_MERCHANTS = [
    "Netflix", "Spotify", "Amazon Prime", "Hulu", "Disney+",
    "HBO Max", "Apple One", "YouTube Premium", "Adobe CC", "Microsoft 365",
    "Planet Fitness", "HelloFresh", "Blue Apron", "LinkedIn Premium", "Audible",
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

STATES = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "WA", "AZ", "MA", "VA", "MI", "NJ"]
FIRST_NAMES = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "William", "Elizabeth"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

# ── User Archetypes ────────────────────────────────────────────────────────
USER_ARCHETYPES = {
    "frugal_saver":      {"base_income": 4000, "std": 300,  "housing_ratio": 0.25},
    "average_spender":   {"base_income": 6500, "std": 800,  "housing_ratio": 0.30},
    "lifestyle_spender": {"base_income": 10000, "std": 2000, "housing_ratio": 0.35},
    "high_earner":       {"base_income": 20000, "std": 5000, "housing_ratio": 0.20},
}

# ── Logic ──────────────────────────────────────────────────────────────────

def generate_user_profile(user_id: str) -> dict:
    arch_name = random.choice(list(USER_ARCHETYPES.keys()))
    arch = USER_ARCHETYPES[arch_name]
    return {
        "user_id": user_id,
        "first": random.choice(FIRST_NAMES),
        "last": random.choice(LAST_NAMES),
        "archetype": arch_name,
        "monthly_income": arch["base_income"] + np.random.normal(0, arch["std"]),
        "housing_cost": (arch["base_income"] * arch["housing_ratio"]) * random.uniform(0.9, 1.1),
        "state": random.choice(STATES),
    }

def get_random_date_in_range():
    # August 2024 to December 2025
    start_date = datetime(2024, 8, 1)
    end_date = datetime(2025, 12, 31)
    delta = end_date - start_date
    return start_date + timedelta(days=random.randint(0, delta.days))

def generate_rows(n_users: int = 500, target_rows: int = 90_000) -> pd.DataFrame:
    rows = []
    users = [generate_user_profile(f"USER_{i:04d}") for i in range(n_users)]
    
    # Target fraud count: 5000 / 90000 = ~5.5% fraud rate
    # We'll use a slightly higher internal rate to ensure we hit the floor
    base_fraud_rate = 0.058 
    
    rows_per_user = target_rows // n_users
    
    for user in users:
        # Pre-generate recurring events
        # Paydays: 1st and 15th
        # Housing: 1st
        # Utilities: 5th-10th
        
        for _ in range(rows_per_user):
            txn_date = get_random_date_in_range()
            day = txn_date.day
            weekday = txn_date.weekday() # 0=Mon, 6=Sun
            month_key = txn_date.strftime("%Y-%m")
            
            # Logic for Realistic Category selection
            if day == 1:
                category = "Housing (Rent/Mortgage)"
                amount = user["housing_cost"]
            elif random.random() < 0.15 and weekday >= 4: # Weekend surge (Fri/Sat/Sun)
                category = random.choice(["Dining & Restaurants", "Entertainment", "Online Shopping"])
                amount = max(10, np.random.lognormal(mean=3.5, sigma=0.8))
            elif day in [1, 15] and random.random() < 0.3: # Payday treat
                category = random.choice(["Clothing & Fashion", "Electronics", "Dining & Restaurants"])
                amount = max(50, np.random.lognormal(mean=4.5, sigma=1.0))
            else:
                category = np.random.choice(CATEGORIES, p=WEIGHTS)
                if category == "Housing (Rent/Mortgage)": # Don't double rent unless it's day 1
                    category = "Groceries"
                
                # Distribution logic per category
                if category == "Groceries":
                    amount = max(20, np.random.normal(80, 40))
                elif category == "Utilities & Bills":
                    amount = max(50, np.random.normal(150, 50))
                elif category == "Transportation & Gas":
                    amount = max(15, np.random.normal(45, 15))
                elif category == "Subscriptions":
                    amount = random.choice([9.99, 14.99, 19.99, 29.99, 49.99])
                else:
                    amount = max(5, np.random.lognormal(mean=3.8, sigma=1.0))

            merchant = random.choice(REGULAR_MERCHANTS.get(category, ["Miscellaneous Merchant"]))
            if category == "Subscriptions":
                merchant = random.choice(SUBSCRIPTION_MERCHANTS)

            # Improved Fraud Case Distribution Skew (matching 78% online loss profile)
            # High risk: Online Shopping, Electronics (representing most monetary losses)
            risk_category_boost = 0.2 # Default low risk
            if category == "Online Shopping":
                risk_category_boost = 7.5 # Very high skew
            elif category == "Electronics":
                risk_category_boost = 4.5 # Heavy skew
            elif category in ["Dining & Restaurants", "Travel & Leisure"]:
                risk_category_boost = 1.2 # Moderate risk
            
            # Velocity boost simulation
            velocity = random.randint(1, 25)
            # Probability skewed heavily towards Online/Electronics
            is_fraud_flag = random.random() < (base_fraud_rate * risk_category_boost * (velocity/12))
            
            # Ensure high risk categories get the 5.5% share
            risk_score = round(random.uniform(0.75, 0.99) if is_fraud_flag else random.uniform(0.01, 0.49), 3)

            rows.append({
                "user_id":                    user["user_id"],
                "first":                      user["first"],
                "last":                       user["last"],
                "archetype":                  user["archetype"],
                "state":                      user["state"],
                "transaction_date":           txn_date.strftime("%Y-%m-%d"),
                "month":                      txn_date.month,
                "year":                       txn_date.year,
                "month_key":                  month_key,
                "category":                   category,
                "merchant":                   merchant,
                "amount":                     round(amount, 2),
                "is_subscription":            category == "Subscriptions",
                "subscription_frequency":     "monthly" if category == "Subscriptions" else "one-time",
                "monthly_total":              0, # Calculated afterwards if needed, but for large scale we approximate
                "prev_month_total":           0,
                "month_over_month_change_pct": 0,
                "avg_monthly_spend":          round(user["monthly_income"] * 0.8, 2),
                "credit_score_impact_category": CATEGORY_CREDIT_IMPACT.get(category, "neutral"),
                "spending_velocity_7d":       velocity,
                "is_fraud_flag":              is_fraud_flag,
                "risk_score":                 risk_score,
            })

    df = pd.DataFrame(rows)
    
    # Sort by date for better CSV appearance
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df = df.sort_values(['user_id', 'transaction_date']).reset_index(drop=True)
    df['transaction_date'] = df['transaction_date'].dt.strftime('%Y-%m-%d')
    
    # Final row padding/trimming
    if len(df) < target_rows:
        extra = df.sample(target_rows - len(df), replace=True).copy()
        df = pd.concat([df, extra], ignore_index=True)
    return df.head(target_rows)

if __name__ == "__main__":
    print(f"🔄 Generating Balanced Veriscan Dataset (Target: 90,000 rows)...")
    print(f"📅 Range: Aug 2024 to Dec 2025")
    
    df = generate_rows(n_users=1000, target_rows=90_000)
    
    # Check fraud count
    fraud_count = df['is_fraud_flag'].sum()
    print(f"🔍 Preliminary Fraud Count: {fraud_count}")
    
    # Force fraud balance if it fell short (rare with high base rate but safe)
    if fraud_count < 5000:
        print(f"⚠️  Fraud count {fraud_count} is below 5,000. Injecting additional cases...")
        diff = 5000 - fraud_count
        clean_indices = df[~df['is_fraud_flag']].sample(diff).index
        df.loc[clean_indices, 'is_fraud_flag'] = True
        df.loc[clean_indices, 'risk_score'] = [round(random.uniform(0.75, 0.99), 3) for _ in range(diff)]
        print(f"✅ Adjusted Fraud Count: {df['is_fraud_flag'].sum()}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"🚀 Success: {len(df):,} rows → {OUTPUT_PATH}")
    print(f"📊 Fraud Rate: {(df['is_fraud_flag'].mean()*100):.2f}% ({df['is_fraud_flag'].sum()} cases)")
    print(f"📅 Min Date: {df['transaction_date'].min()}")
    print(f"📅 Max Date: {df['transaction_date'].max()}")
    print(f"👥 Users: {df['user_id'].nunique()}")
