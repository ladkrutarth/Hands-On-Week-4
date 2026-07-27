"""
Generate Synthetic CFPB Credit Card Complaint Dataset (50,000 rows).
Focuses on major credit bureaus (Equifax, TransUnion) and top issuers.

Output: data/csv_data/cfpb_credit_card.csv
"""

import random
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# --- Config ---
TARGET_ROWS = 50_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "csv_data" / "cfpb_credit_card.csv"

# --- Constants ---
COMPANIES = [
    "EQUIFAX, INC.",
    "TRANSUNION INTERMEDIATE HOLDINGS, INC.",
    "EXPERIAN INFORMATION SOLUTIONS, INC.",
    "LEXISNEXIS RISK DATA MANAGEMENT INC.",
    "CITIBANK, N.A.",
    "JPMORGAN CHASE & CO.",
    "CAPITAL ONE FINANCIAL CORPORATION",
    "WELLS FARGO & COMPANY",
    "BANK OF AMERICA, NATIONAL ASSOCIATION",
    "DISCOVER BANK",
    "SYNCHRONY FINANCIAL",
    "AMERICAN EXPRESS COMPANY"
]

# Bias weights for Reporting vs Issuers
# Equifax, TU, Experian, LexisNexis (0.15 each = 0.60 total for reporting)
# Others (0.05 each = 0.40 total for issuers)
COMPANY_WEIGHTS = [0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

ISSUES = {
    "Incorrect information on your report": ["Account status incorrect", "Information belongs to someone else", "Old information still on report"],
    "Problem with a company's investigation": ["Was not notified of investigation status", "Investigation took too long", "Did not receive result"],
    "Application denied": ["Application denied for unknown reason", "Inaccurate info on report led to denial", "Credit limit too low"],
    "Advertising and marketing": ["Confusing advertising", "Misleading promotional offer", "Promotion not applied"],
    "Problem with a purchase shown on your statement": ["Incorrect amount", "Duplicate charge", "Merchant dispute"],
    "Fees or interest": ["Unexpected fee", "Interest rate higher than expected", "Grace period problem"]
}

STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA", "MD"]

RESPONSES = [
    "Closed with explanation",
    "Closed with non-monetary relief",
    "Closed with monetary relief",
    "In progress",
    "Untimely response"
]

def generate_row(idx):
    company = random.choices(COMPANIES, weights=COMPANY_WEIGHTS)[0]
    issue = random.choice(list(ISSUES.keys()))
    sub_issue = random.choice(ISSUES[issue])
    
    # Date range: 2024 to 2026
    days_ago = random.randint(0, 730)
    date_received = datetime.now() - timedelta(days=days_ago)
    date_sent = date_received + timedelta(days=random.randint(0, 3))
    
    return {
        "Date received": date_received.strftime("%m/%d/%y"),
        "Product": "Credit card",
        "Sub-product": "General-purpose credit card or charge card",
        "Issue": issue,
        "Sub-issue": sub_issue,
        "Consumer complaint narrative": None,
        "Company public response": "Company has responded to the consumer",
        "Company": company,
        "State": random.choice(STATES),
        "ZIP code": str(random.randint(10000, 99999)),
        "Tags": random.choice(["None", "Servicemember", "Older American"]),
        "Consumer consent provided?": "Consent provided",
        "Submitted via": "Web",
        "Date sent to company": date_sent.strftime("%m/%d/%y"),
        "Company response to consumer": random.choice(RESPONSES),
        "Timely response?": "Yes",
        "Consumer disputed?": "N/A",
        "Complaint ID": 8000000 + idx
    }

def main():
    print(f"🔄 Generating {TARGET_ROWS:,} CFPB complaint rows...")
    rows = [generate_row(i) for i in range(TARGET_ROWS)]
    df = pd.DataFrame(rows)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved to {OUTPUT_PATH}")
    
    # Quick Summary
    print("\n--- Summary ---")
    print(df['Company'].value_counts().head(5))

if __name__ == "__main__":
    main()
