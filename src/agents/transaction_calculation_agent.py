"""
Transaction Calculation Agent — pure calculations on transaction data.
Totals, averages, growth rates, category breakdowns, forecasts, comparisons.
Returns structured result dicts (numbers, breakdowns, time series).
"""

from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADVISOR_DATA_PATH = PROJECT_ROOT / "data" / "csv_data" / "financial_advisor_dataset.csv"


class TransactionCalculationAgent:
    """Performs calculations on transaction data; no current vs historical semantics."""

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

    def total_in_range(
        self, user_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> dict[str, Any]:
        """Total spend in a date range. If no dates, use all data."""
        user_df = self._user_df(user_id)
        if user_df.empty:
            return {"agent": "calculation", "error": f"No data for {user_id}"}
        if start_date:
            user_df = user_df[user_df["transaction_date"] >= pd.to_datetime(start_date)]
        if end_date:
            user_df = user_df[user_df["transaction_date"] <= pd.to_datetime(end_date)]
        total = float(user_df["amount"].sum())
        by_cat = (
            user_df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .round(2)
            .to_dict()
        )
        return {
            "agent": "calculation",
            "tool": "total_in_range",
            "total": round(total, 2),
            "transaction_count": len(user_df),
            "by_category": by_cat,
        }

    def average_by_category(self, user_id: str, months: Optional[int] = None) -> dict[str, Any]:
        """Average monthly spend by category. Optionally last N months."""
        user_df = self._user_df(user_id)
        if user_df.empty:
            return {"agent": "calculation", "error": f"No data for {user_id}"}
        if months:
            months_list = sorted(user_df["month_key"].unique())[-months:]
            user_df = user_df[user_df["month_key"].isin(months_list)]
        by_cat = user_df.groupby("category")["amount"].sum()
        n_months = user_df["month_key"].nunique() or 1
        avg_by_cat = (by_cat / n_months).round(2).sort_values(ascending=False).to_dict()
        return {
            "agent": "calculation",
            "tool": "average_by_category",
            "avg_monthly_by_category": avg_by_cat,
            "months_used": n_months,
        }

    def month_over_month_change(self, user_id: str) -> dict[str, Any]:
        """Month-over-month total spend change (latest two months)."""
        user_df = self._user_df(user_id)
        if user_df.empty:
            return {"agent": "calculation", "error": f"No data for {user_id}"}
        months = sorted(user_df["month_key"].unique())
        if len(months) < 2:
            return {"agent": "calculation", "error": "Insufficient months"}
        curr_key, prev_key = months[-1], months[-2]
        curr = user_df[user_df["month_key"] == curr_key]["amount"].sum()
        prev = user_df[user_df["month_key"] == prev_key]["amount"].sum()
        change_pct = ((curr - prev) / max(prev, 1)) * 100
        return {
            "agent": "calculation",
            "tool": "month_over_month_change",
            "current_month": curr_key,
            "previous_month": prev_key,
            "current_total": round(float(curr), 2),
            "previous_total": round(float(prev), 2),
            "growth_pct": round(change_pct, 2),
        }

    def forecast_next_month(self, user_id: str) -> dict[str, Any]:
        """Simple forecast: avg monthly total and subscription vs variable."""
        user_df = self._user_df(user_id)
        if user_df.empty:
            return {"agent": "calculation", "error": f"No data for {user_id}"}
        subs = user_df[user_df["is_subscription"] == True] if "is_subscription" in user_df.columns else pd.DataFrame()
        sub_total = float(subs.groupby("merchant")["amount"].mean().sum()) if not subs.empty else 0.0
        by_month = user_df.groupby("month_key")["amount"].sum()
        avg_total = float(by_month.mean())
        avg_variable = avg_total - sub_total
        return {
            "agent": "calculation",
            "tool": "forecast_next_month",
            "forecast_total": round(avg_total, 2),
            "subscription_component": round(sub_total, 2),
            "variable_component": round(avg_variable, 2),
        }

    def subscription_totals(self, user_id: str) -> dict[str, Any]:
        """Total recurring/subscription spend by merchant."""
        user_df = self._user_df(user_id)
        if user_df.empty:
            return {"agent": "calculation", "error": f"No data for {user_id}"}
        if "is_subscription" not in user_df.columns:
            return {"agent": "calculation", "tool": "subscription_totals", "by_merchant": {}, "monthly_total": 0}
        subs = user_df[user_df["is_subscription"] == True]
        if subs.empty:
            return {"agent": "calculation", "tool": "subscription_totals", "by_merchant": {}, "monthly_total": 0}
        by_merchant = subs.groupby("merchant")["amount"].mean().round(2).to_dict()
        monthly_total = round(sum(by_merchant.values()), 2)
        return {
            "agent": "calculation",
            "tool": "subscription_totals",
            "by_merchant": by_merchant,
            "monthly_total": monthly_total,
        }

    def run(
        self,
        user_id: str,
        calculation: str = "summary",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run requested calculation. calculation: total | average_by_category | mom_change | forecast | subscriptions.
        """
        if calculation == "total" or calculation == "total_in_range":
            return self.total_in_range(user_id, start_date, end_date)
        if calculation == "average_by_category":
            months = int(end_date) if end_date and end_date.isdigit() else None
            return self.average_by_category(user_id, months=months)
        if calculation == "mom_change" or calculation == "month_over_month":
            return self.month_over_month_change(user_id)
        if calculation == "forecast":
            return self.forecast_next_month(user_id)
        if calculation == "subscriptions":
            return self.subscription_totals(user_id)
        return self.total_in_range(user_id, start_date, end_date)
