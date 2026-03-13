"""
Spending DNA Agent — Computes the 8-axis financial fingerprint per user.
Used for identity verification and anomaly detection.
"""

from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DNA_DATA_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "spending_dna_dataset.csv"

# The 8 canonical DNA axes and their display labels
DNA_AXES = [
    ("avg_txn_amount",        "Avg Transaction Amount"),
    ("location_entropy",      "Location Entropy"),
    ("weekend_ratio",         "Weekend Ratio"),
    ("category_diversity",    "Category Diversity"),
    ("time_of_day_pref",      "Time-of-Day Preference"),
    ("risk_appetite_score",   "Risk Appetite"),
    ("spending_velocity",     "Spending Velocity"),
    ("merchant_loyalty_score","Merchant Loyalty"),
]

TIME_LABELS = {0: "Morning", 1: "Afternoon", 2: "Evening", 3: "Night"}


class SpendingDNAAgent:
    """Computes and compares user Spending DNA profiles."""

    def __init__(self):
        self._df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(DNA_DATA_PATH)
            self._df["transaction_date"] = pd.to_datetime(self._df["transaction_date"])
            self._df["year"] = self._df["transaction_date"].dt.year
            self._df["month"] = self._df["transaction_date"].dt.month
        return self._df

    # ── Normalise a single axis to [0, 1] for radar chart ─────────────────
    def _normalize(self, col: str, value: float) -> float:
        col_data = self.df[col]
        mn, mx = col_data.min(), col_data.max()
        if mx == mn:
            return 0.5
        return round((value - mn) / (mx - mn), 4)

    def compute_dna(self, user_id: str) -> dict[str, Any]:
        """Return the 8-axis DNA fingerprint for a user (for radar chart)."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {"error": f"No DNA data found for {user_id}"}

        raw: dict[str, float] = {}
        normalized: dict[str, float] = {}

        for col, _ in DNA_AXES:
            if col == "time_of_day_pref":
                val = float(user_df[col].mode().iloc[0])
            else:
                val = float(user_df[col].mean())
            raw[col] = round(val, 4)
            normalized[col] = self._normalize(col, val)

        avg_trust  = round(float(user_df["trust_score"].mean()), 4)
        avg_dev    = round(float(user_df["dna_deviation_score"].mean()), 4)
        anomalous  = int(user_df["is_anomalous_session"].sum())
        total_sess = len(user_df)

        trust_pct = round(avg_trust * 100, 1)
        trust_grade = (
            "A — Trusted Profile ✅" if trust_pct >= 80 else
            "B — Normal Variance 🟡" if trust_pct >= 60 else
            "C — Elevated Risk 🟠" if trust_pct >= 40 else
            "D — High Risk 🔴"
        )

        radar_labels = [label for _, label in DNA_AXES]
        radar_values = [normalized[col] for col, _ in DNA_AXES]

        return {
            "user_id":          user_id,
            "radar_labels":     radar_labels,
            "radar_values":     radar_values,
            "raw_axes":         raw,
            "normalized_axes":  normalized,
            "avg_trust_score":  avg_trust,
            "avg_deviation":    avg_dev,
            "anomalous_count":  anomalous,
            "total_sessions":   total_sess,
            "trust_grade":      trust_grade,
            "time_preference":  TIME_LABELS.get(int(raw.get("time_of_day_pref", 1)), "Afternoon"),
        }

    def compute_yearly_dna(self, user_id: str, year: int) -> dict[str, Any]:
        """Compute DNA for a specific year."""
        user_df = self.df[(self.df["user_id"] == user_id) & (self.df["year"] == year)]
        if user_df.empty:
            return {"error": f"No data for {year}"}

        normalized: dict[str, float] = {}
        for col, _ in DNA_AXES:
            if col == "time_of_day_pref":
                val = float(user_df[col].mode().iloc[0]) if not user_df[col].empty else 1
            else:
                val = float(user_df[col].mean())
            normalized[col] = self._normalize(col, val)
        
        return {
            "year": year,
            "radar_values": [normalized[col] for col, _ in DNA_AXES],
            "avg_trust": round(float(user_df["trust_score"].mean()), 3) if not user_df.empty else 0
        }

    def compute_monthly_evolution(self, user_id: str) -> dict[str, Any]:
        """Compute monthly trust score and deviation trend."""
        user_df = self.df[self.df["user_id"] == user_id].copy()
        if user_df.empty:
            return {"error": "No data"}

        # Group by year-month
        trend = user_df.groupby(["year", "month"]).agg({
            "trust_score": "mean",
            "dna_deviation_score": "mean"
        }).reset_index()
        
        trend["label"] = trend.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)
        
        return {
            "labels": trend["label"].tolist(),
            "trust_scores": [round(x, 3) for x in trend["trust_score"].tolist()],
            "deviations": [round(x, 3) for x in trend["dna_deviation_score"].tolist()]
        }

    def compare_session(self, user_id: str, session_overrides: dict | None = None) -> dict[str, Any]:
        """
        Compare a new session's axes against the user's DNA baseline.
        session_overrides: optional dict of axis -> value to simulate a new session.
        """
        dna = self.compute_dna(user_id)
        if "error" in dna:
            return dna

        baseline_norm = dna["normalized_axes"]

        # Use overrides if provided, else draw a random sample session
        if session_overrides:
            session_norm = {}
            for col, _ in DNA_AXES:
                if col in session_overrides:
                    session_norm[col] = self._normalize(col, float(session_overrides[col]))
                else:
                    session_norm[col] = baseline_norm[col]
        else:
            # Simulate a new session from the last 20 rows of this user
            sample = self.df[self.df["user_id"] == user_id].tail(20).sample(1).iloc[0]
            session_norm = {col: self._normalize(col, float(sample[col])) for col, _ in DNA_AXES if col in sample}

        # Deviation per axis
        deviations = {
            col: round(abs(baseline_norm.get(col, 0) - session_norm.get(col, 0)), 4)
            for col, _ in DNA_AXES
        }
        composite_deviation = round(float(np.mean(list(deviations.values()))), 4)
        trust_score = round(max(0.0, 1.0 - composite_deviation * 1.8), 4)

        return {
            "user_id":              user_id,
            "baseline_radar":       [baseline_norm.get(col, 0) for col, _ in DNA_AXES],
            "session_radar":        [session_norm.get(col, 0) for col, _ in DNA_AXES],
            "radar_labels":         [label for _, label in DNA_AXES],
            "axis_deviations":      deviations,
            "composite_deviation":  composite_deviation,
            "session_trust_score":  trust_score,
            "verdict":              (
                "✅ Session matches DNA — Trusted"    if trust_score >= 0.75 else
                "⚠️ Moderate deviation — Verify"     if trust_score >= 0.55 else
                "🔴 High deviation — Challenge Auth"
            ),
        }

    def get_all_users(self) -> list[str]:
        return sorted(self.df["user_id"].unique().tolist())
