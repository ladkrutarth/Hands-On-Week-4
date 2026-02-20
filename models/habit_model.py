"""
GraphGuard — Transaction Habit Prediction Model
Learns each user's transaction habits and predicts behavior patterns.

Features:
  • Preferred categories (ranked by frequency)
  • Typical locations (top cities)
  • Spending patterns (avg amount, typical time/day)
  • Habit score — how consistent is this user's behavior
  • Anomaly detection — is a new transaction "in character"

Uses: statistics + scikit-learn KNN for habit similarity scoring.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "features_output.csv"
TRANSACTIONS_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "transactions_3000.csv"


# ---------------------------------------------------------------------------
# Habit Profile Builder
# ---------------------------------------------------------------------------

class UserHabitProfile:
    """Represents a single user's learned transaction habits."""

    def __init__(self, user_id: str, transactions_df: pd.DataFrame):
        self.user_id = user_id
        self.df = transactions_df
        self._build_profile()

    def _build_profile(self):
        df = self.df
        if df.empty:
            self._empty_profile()
            return

        # --- Category habits ---
        self.top_categories = (
            df["CATEGORY"].value_counts()
            .head(5)
            .to_dict()
        )
        self.favorite_category = (
            df["CATEGORY"].value_counts().index[0]
            if "CATEGORY" in df.columns and not df["CATEGORY"].empty
            else "Unknown"
        )
        self.num_categories = df["CATEGORY"].nunique()

        # --- Top 5 most visited stores with avg spend ---
        cat_stats = df.groupby("CATEGORY")["AMOUNT"].agg(["count", "mean"]).sort_values("count", ascending=False).head(5)
        self.top_stores_with_spend = [
            {
                "store": cat,
                "visits": int(row["count"]),
                "avg_spend": round(float(row["mean"]), 2),
            }
            for cat, row in cat_stats.iterrows()
        ]

        # --- Location habits ---
        self.top_locations = (
            df["LOCATION"].value_counts()
            .head(5)
            .to_dict()
        )
        self.primary_location = (
            df["LOCATION"].value_counts().index[0]
            if "LOCATION" in df.columns and not df["LOCATION"].empty
            else "Unknown"
        )
        self.num_locations = df["LOCATION"].nunique()

        # --- Spending patterns ---
        self.avg_amount = round(float(df["AMOUNT"].mean()), 2)
        self.median_amount = round(float(df["AMOUNT"].median()), 2)
        self.std_amount = round(float(df["AMOUNT"].std()), 2) if len(df) > 1 else 0.0
        self.min_amount = round(float(df["AMOUNT"].min()), 2)
        self.max_amount = round(float(df["AMOUNT"].max()), 2)
        self.total_spent = round(float(df["AMOUNT"].sum()), 2)

        # --- Time patterns ---
        if "HOUR_OF_DAY" in df.columns:
            hour_counts = df["HOUR_OF_DAY"].value_counts()
            self.peak_hour = int(hour_counts.index[0])
            self.peak_hour_label = self._hour_to_label(self.peak_hour)
        else:
            self.peak_hour = 12
            self.peak_hour_label = "Afternoon"

        if "DAY_OF_WEEK" in df.columns:
            day_counts = df["DAY_OF_WEEK"].value_counts()
            self.peak_day = int(day_counts.index[0])
            self.peak_day_label = self._day_to_label(self.peak_day)
        else:
            self.peak_day = 0
            self.peak_day_label = "Monday"

        if "IS_WEEKEND" in df.columns:
            weekend_ratio = float(df["IS_WEEKEND"].mean())
            self.prefers_weekend = weekend_ratio > 0.5
            self.weekend_ratio = round(weekend_ratio * 100, 1)
        else:
            self.prefers_weekend = False
            self.weekend_ratio = 0.0

        # --- Habit consistency score (0-100) ---
        self.habit_score = self._compute_habit_score()

        # --- Transaction frequency ---
        self.total_transactions = len(df)

    def _empty_profile(self):
        self.top_categories = {}
        self.favorite_category = "Unknown"
        self.num_categories = 0
        self.top_stores_with_spend = []
        self.top_locations = {}
        self.primary_location = "Unknown"
        self.num_locations = 0
        self.avg_amount = 0.0
        self.median_amount = 0.0
        self.std_amount = 0.0
        self.min_amount = 0.0
        self.max_amount = 0.0
        self.total_spent = 0.0
        self.peak_hour = 12
        self.peak_hour_label = "Afternoon"
        self.peak_day = 0
        self.peak_day_label = "Monday"
        self.prefers_weekend = False
        self.weekend_ratio = 0.0
        self.habit_score = 0
        self.total_transactions = 0

    def _compute_habit_score(self) -> int:
        """
        Compute a 0-100 habit consistency score.
        Higher = more predictable/consistent spending patterns.
        """
        scores = []

        # Category concentration — if top category is >40% of transactions, very consistent
        if self.top_categories:
            top_count = list(self.top_categories.values())[0]
            total = sum(self.top_categories.values())
            concentration = top_count / total if total > 0 else 0
            scores.append(min(100, concentration * 150))

        # Location concentration — fewer locations = more predictable
        if self.num_locations > 0:
            loc_score = max(0, 100 - (self.num_locations - 1) * 15)
            scores.append(loc_score)

        # Spending consistency — lower coefficient of variation = more consistent
        if self.avg_amount > 0 and self.std_amount is not None:
            cv = self.std_amount / self.avg_amount
            spend_score = max(0, 100 - cv * 80)
            scores.append(spend_score)

        # Time consistency — if weekend ratio is extreme (very high or very low), consistent
        time_score = abs(self.weekend_ratio - 50) * 2  # Further from 50% = more consistent
        scores.append(min(100, time_score))

        return int(sum(scores) / len(scores)) if scores else 50

    def is_anomalous(self, amount: float, category: str, location: str) -> dict:
        """
        Check if a transaction matches this user's habits.
        Returns anomaly assessment dict.
        """
        flags = []
        anomaly_score = 0.0

        # Amount check
        if self.std_amount > 0:
            z = abs(amount - self.avg_amount) / self.std_amount
            if z > 3:
                flags.append(f"Amount ${amount:.2f} is {z:.1f}σ from average ${self.avg_amount:.2f}")
                anomaly_score += 0.4
            elif z > 2:
                anomaly_score += 0.2

        # Category check
        if category not in self.top_categories:
            flags.append(f"Category '{category}' is not in user's top categories")
            anomaly_score += 0.3

        # Location check
        if location not in self.top_locations:
            flags.append(f"Location '{location}' is new for this user")
            anomaly_score += 0.3

        is_anomalous = anomaly_score >= 0.5

        return {
            "is_anomalous": is_anomalous,
            "anomaly_score": round(min(1.0, anomaly_score), 3),
            "flags": flags,
            "verdict": "⚠️ Unusual" if is_anomalous else "✅ Normal",
        }

    def to_dict(self) -> dict:
        """Convert profile to dict for display/serialization."""
        return {
            "user_id": self.user_id,
            "favorite_category": self.favorite_category,
            "top_categories": self.top_categories,
            "num_categories": self.num_categories,
            "primary_location": self.primary_location,
            "top_locations": self.top_locations,
            "num_locations": self.num_locations,
            "avg_amount": self.avg_amount,
            "median_amount": self.median_amount,
            "total_spent": self.total_spent,
            "peak_hour_label": self.peak_hour_label,
            "peak_day_label": self.peak_day_label,
            "prefers_weekend": self.prefers_weekend,
            "weekend_ratio": self.weekend_ratio,
            "habit_score": self.habit_score,
            "total_transactions": self.total_transactions,
        }

    @staticmethod
    def _hour_to_label(hour: int) -> str:
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 22:
            return "Evening"
        else:
            return "Late Night"

    @staticmethod
    def _day_to_label(day: int) -> str:
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day % 7]


# ---------------------------------------------------------------------------
# Habit Prediction Model — learns all users' habits
# ---------------------------------------------------------------------------

class HabitPredictionModel:
    """
    Learns transaction habits for all users.
    Uses KNN for finding similar users and statistical profiling.
    """

    def __init__(self, features_path: Optional[Path] = None):
        self.features_path = features_path or FEATURES_PATH
        self.profiles: Dict[str, UserHabitProfile] = {}
        self._features_df = None

    def load_data(self) -> bool:
        """Load transaction data."""
        if not self.features_path.exists():
            print(f"⚠️ Features file not found: {self.features_path}")
            return False

        self._features_df = pd.read_csv(self.features_path)
        self._features_df.columns = [c.upper() for c in self._features_df.columns]
        print(f"✅ Loaded {len(self._features_df)} transactions for habit learning")
        return True

    def train(self) -> Dict[str, UserHabitProfile]:
        """
        Build habit profiles for all users.
        Returns dict of user_id -> UserHabitProfile.
        """
        if self._features_df is None:
            if not self.load_data():
                return {}

        users = self._features_df["USER_ID"].unique()

        for user_id in sorted(users):
            user_df = self._features_df[self._features_df["USER_ID"] == user_id]
            self.profiles[user_id] = UserHabitProfile(user_id, user_df)

        print(f"✅ Built habit profiles for {len(self.profiles)} users")
        return self.profiles

    def get_profile(self, user_id: str) -> Optional[UserHabitProfile]:
        """Get a specific user's habit profile."""
        if not self.profiles:
            self.train()
        return self.profiles.get(user_id)

    def predict_next_category(self, user_id: str) -> List[dict]:
        """
        Predict the most likely next transaction category for a user.
        Based on historical frequency distribution.
        """
        profile = self.get_profile(user_id)
        if not profile or not profile.top_categories:
            return [{"category": "Unknown", "probability": 1.0}]

        total = sum(profile.top_categories.values())
        predictions = []
        for cat, count in profile.top_categories.items():
            prob = round(count / total, 3)
            predictions.append({"category": cat, "probability": prob})

        return sorted(predictions, key=lambda x: x["probability"], reverse=True)

    def predict_next_location(self, user_id: str) -> List[dict]:
        """
        Predict the most likely next transaction location for a user.
        """
        profile = self.get_profile(user_id)
        if not profile or not profile.top_locations:
            return [{"location": "Unknown", "probability": 1.0}]

        total = sum(profile.top_locations.values())
        predictions = []
        for loc, count in profile.top_locations.items():
            prob = round(count / total, 3)
            predictions.append({"location": loc, "probability": prob})

        return sorted(predictions, key=lambda x: x["probability"], reverse=True)

    def find_similar_users(self, user_id: str, top_n: int = 3) -> List[dict]:
        """
        Find users with the most similar transaction habits using KNN on spending features.
        """
        if not self.profiles:
            self.train()

        if user_id not in self.profiles:
            return []

        # Build feature vectors for all users
        user_features = {}
        for uid, prof in self.profiles.items():
            user_features[uid] = [
                prof.avg_amount,
                prof.std_amount,
                prof.num_categories,
                prof.num_locations,
                prof.weekend_ratio,
                prof.total_transactions,
            ]

        if user_id not in user_features:
            return []

        target = np.array(user_features[user_id])

        # Normalize features
        all_features = np.array(list(user_features.values()))
        means = all_features.mean(axis=0)
        stds = all_features.std(axis=0)
        stds[stds == 0] = 1  # Avoid division by zero

        target_norm = (target - means) / stds

        similarities = []
        for uid, feats in user_features.items():
            if uid == user_id:
                continue
            feat_norm = (np.array(feats) - means) / stds
            distance = float(np.sqrt(np.sum((target_norm - feat_norm) ** 2)))
            similarity = round(1.0 / (1.0 + distance), 3)
            similarities.append({
                "user_id": uid,
                "similarity": similarity,
                "shared_categories": len(
                    set(self.profiles[uid].top_categories.keys()) &
                    set(self.profiles[user_id].top_categories.keys())
                ),
            })

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_n]

    def check_transaction(self, user_id: str, amount: float, category: str, location: str) -> dict:
        """
        Check if a transaction matches a user's learned habits.
        Returns anomaly assessment.
        """
        profile = self.get_profile(user_id)
        if not profile:
            return {"error": f"No profile found for {user_id}"}
        return profile.is_anomalous(amount, category, location)

    def get_all_profiles_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of all user habits for dashboard display."""
        if not self.profiles:
            self.train()

        rows = []
        for uid, prof in sorted(self.profiles.items()):
            rows.append({
                "User": uid,
                "Favorite Store": prof.favorite_category,
                "Primary City": prof.primary_location,
                "Avg Spend": f"${prof.avg_amount:.0f}",
                "Total Spent": f"${prof.total_spent:,.0f}",
                "Peak Time": prof.peak_hour_label,
                "Peak Day": prof.peak_day_label,
                "Weekend %": f"{prof.weekend_ratio:.0f}%",
                "Habit Score": f"{prof.habit_score}/100",
                "Transactions": prof.total_transactions,
            })

        return pd.DataFrame(rows)


if __name__ == "__main__":
    model = HabitPredictionModel()
    model.train()

    print("\n📊 All User Habit Profiles:")
    print(model.get_all_profiles_summary().to_string(index=False))

    print("\n" + "=" * 60)
    for uid in ["USER_001", "USER_004", "USER_007"]:
        profile = model.get_profile(uid)
        if profile:
            print(f"\n👤 {uid}:")
            print(f"   Favorite store: {profile.favorite_category}")
            print(f"   Primary city: {profile.primary_location}")
            print(f"   Avg spend: ${profile.avg_amount:.2f}")
            print(f"   Peak time: {profile.peak_hour_label} on {profile.peak_day_label}")
            print(f"   Habit score: {profile.habit_score}/100")

            # Predict next category
            preds = model.predict_next_category(uid)
            print(f"   Next purchase prediction: {preds[0]['category']} ({preds[0]['probability']:.0%})")

            # Find similar users
            similar = model.find_similar_users(uid)
            if similar:
                print(f"   Most similar user: {similar[0]['user_id']} ({similar[0]['similarity']:.0%} match)")

            # Test anomaly detection
            check = model.check_transaction(uid, 5000, "Jewelry", "Anchorage, AK")
            print(f"   Anomaly check ($5000 Jewelry in Anchorage): {check['verdict']}")
