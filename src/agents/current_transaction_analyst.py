"""
Current Transaction Analyst Agent — analyzes current/recent transactions only.
Focus: this month, last 30/60/90 days, latest activity, recent fraud/risk flags.
Returns structured dicts for the orchestrator to synthesize.
"""

from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADVISOR_DATA_PATH = PROJECT_ROOT / "data" / "csv_data" / "financial_advisor_dataset.csv"


class CurrentTransactionAnalyst:
    """Analyzes current and recent transaction activity for a user."""

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(ADVISOR_DATA_PATH)
            self._df["transaction_date"] = pd.to_datetime(self._df["transaction_date"])
        return self._df

    def _user_df(self, user_id: str) -> pd.DataFrame:
        return self.df[self.df["user_id"] == user_id].copy()

    def recent_transactions(
        self, user_id: str, limit: int = 20, window_days: Optional[int] = None
    ) -> dict[str, Any]:
        """List most recent transactions, optionally within last N days."""
        user_df = self._user_df(user_id)
        if user_df.empty:
            return {"agent": "current_analyst", "error": f"No data for {user_id}"}
        user_df = user_df.sort_values("transaction_date", ascending=False)
        if window_days:
            cutoff = user_df["transaction_date"].max() - timedelta(days=window_days)
            user_df = user_df[user_df["transaction_date"] >= cutoff]
        head = user_df.head(limit)
        rows = head[["transaction_date", "category", "merchant", "amount"]].to_dict("records")
        for r in rows:
            r["transaction_date"] = str(r["transaction_date"])[:10]
        return {
            "agent": "current_analyst",
            "tool": "recent_transactions",
            "count": len(rows),
            "window_days": window_days,
            "recent_txns": rows,
            "total_in_window": round(float(head["amount"].sum()), 2),
        }

    def current_month_summary(self, user_id: str) -> dict[str, Any]:
        """Summary of spending for the current month (latest month in data)."""
        user_df = self._user_df(user_id)
        if user_df.empty:
            return {"agent": "current_analyst", "error": f"No data for {user_id}"}
        months = sorted(user_df["month_key"].unique())
        if not months:
            return {"agent": "current_analyst", "error": "No month data"}
        current_key = months[-1]
        curr = user_df[user_df["month_key"] == current_key]
        by_cat = (
            curr.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .round(2)
            .to_dict()
        )
        return {
            "agent": "current_analyst",
            "tool": "current_month_summary",
            "current_month": current_key,
            "current_month_total": round(float(curr["amount"].sum()), 2),
            "transaction_count": len(curr),
            "by_category": by_cat,
        }

    def last_n_days(self, user_id: str, days: int = 30) -> dict[str, Any]:
        """Totals and category breakdown for the last N days."""
        user_df = self._user_df(user_id)
        if user_df.empty:
            return {"agent": "current_analyst", "error": f"No data for {user_id}"}
        max_date = user_df["transaction_date"].max()
        cutoff = max_date - timedelta(days=days)
        recent = user_df[user_df["transaction_date"] >= cutoff]
        by_cat = (
            recent.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .round(2)
            .to_dict()
        )
        return {
            "agent": "current_analyst",
            "tool": "last_n_days",
            "days": days,
            "total": round(float(recent["amount"].sum()), 2),
            "transaction_count": len(recent),
            "by_category": by_cat,
        }

    def recent_fraud_risk_flags(self, user_id: str, last_n: int = 50) -> dict[str, Any]:
        """Recent transactions with fraud/risk flags or high-risk categories."""
        user_df = self._user_df(user_id)
        if user_df.empty:
            return {"agent": "current_analyst", "error": f"No data for {user_id}"}
        recent = user_df.sort_values("transaction_date", ascending=False).head(last_n)
        alerts = []
        if "is_fraud_flag" in recent.columns:
            fraud = recent[recent["is_fraud_flag"] == True]
            for _, row in fraud.iterrows():
                alerts.append(
                    {
                        "date": str(row["transaction_date"])[:10],
                        "reason": "fraud_flag",
                        "category": row.get("category"),
                        "amount": round(float(row["amount"]), 2),
                    }
                )
        if "risk_score" in recent.columns:
            high_risk = recent[recent["risk_score"] > 0.7]
            for _, row in high_risk.iterrows():
                if row.get("is_fraud_flag") != True:
                    alerts.append(
                        {
                            "date": str(row["transaction_date"])[:10],
                            "reason": "high_risk_score",
                            "risk_score": round(float(row["risk_score"]), 3),
                            "category": row.get("category"),
                            "amount": round(float(row["amount"]), 2),
                        }
                    )
        return {
            "agent": "current_analyst",
            "tool": "recent_fraud_risk_flags",
            "alerts_count": len(alerts),
            "alerts": alerts[:20],
        }

    def run(self, user_id: str, period: str = "last_30") -> dict[str, Any]:
        """
        Run current analysis for the user. period: this_month | last_30 | last_60 | last_90.
        Returns a combined structured summary for the orchestrator.
        """
        out = {"agent": "current_analyst", "user_id": user_id}
        out["current_month"] = self.current_month_summary(user_id)
        days = 30 if period == "last_30" else (60 if period == "last_60" else 90)
        out["last_n_days"] = self.last_n_days(user_id, days=days)
        out["recent_fraud_risk"] = self.recent_fraud_risk_flags(user_id)
        return out
