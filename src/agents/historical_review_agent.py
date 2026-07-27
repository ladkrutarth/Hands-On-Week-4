"""
Historical Review Agent — reviews past behavior over the last 2 years (24 months).
Bank-statement/credit-card style: yearly totals, YoY comparison, monthly trends,
category evolution, statement-style summaries. Returns structured dicts.
"""

from pathlib import Path
from typing import Any, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADVISOR_DATA_PATH = PROJECT_ROOT / "data" / "csv_data" / "financial_advisor_dataset.csv"
HISTORY_MONTHS = 24


class HistoricalReviewAgent:
    """Reviews transaction history over the last 24 months."""

    def __init__(self, window_months: int = HISTORY_MONTHS):
        self._df: Optional[pd.DataFrame] = None
        self.window_months = window_months

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(ADVISOR_DATA_PATH)
            self._df["transaction_date"] = pd.to_datetime(self._df["transaction_date"])
        return self._df

    def _user_history(self, user_id: str) -> pd.DataFrame:
        user_df = self.df[self.df["user_id"] == user_id].copy()
        if user_df.empty:
            return user_df
        months = sorted(user_df["month_key"].unique())
        if len(months) <= self.window_months:
            return user_df
        keep = set(months[-self.window_months :])
        return user_df[user_df["month_key"].isin(keep)]

    def yearly_totals(self, user_id: str) -> dict[str, Any]:
        """Total spend by year within the 24-month window."""
        hist = self._user_history(user_id)
        if hist.empty:
            return {"agent": "historical", "error": f"No data for {user_id}"}
        by_year = hist.groupby("year")["amount"].sum().round(2).to_dict()
        return {
            "agent": "historical",
            "tool": "yearly_totals",
            "yearly_totals": {str(k): v for k, v in by_year.items()},
        }

    def year_over_year_change(self, user_id: str) -> dict[str, Any]:
        """Year-over-year comparison (last two full years in window)."""
        hist = self._user_history(user_id)
        if hist.empty:
            return {"agent": "historical", "error": f"No data for {user_id}"}
        years = sorted(hist["year"].unique())
        if len(years) < 2:
            return {"agent": "historical", "tool": "yoy", "message": "Insufficient years"}
        y2, y1 = years[-1], years[-2]
        t2 = hist[hist["year"] == y2]["amount"].sum()
        t1 = hist[hist["year"] == y1]["amount"].sum()
        change_pct = ((t2 - t1) / max(t1, 1)) * 100
        return {
            "agent": "historical",
            "tool": "year_over_year",
            "year_latest": int(y2),
            "year_previous": int(y1),
            "total_latest": round(float(t2), 2),
            "total_previous": round(float(t1), 2),
            "yoy_change_pct": round(change_pct, 2),
        }

    def monthly_trends(self, user_id: str) -> dict[str, Any]:
        """Monthly totals over the 24-month window (time series)."""
        hist = self._user_history(user_id)
        if hist.empty:
            return {"agent": "historical", "error": f"No data for {user_id}"}
        by_month = hist.groupby("month_key")["amount"].sum().round(2)
        series = by_month.sort_index().to_dict()
        return {
            "agent": "historical",
            "tool": "monthly_trends",
            "monthly_totals": {str(k): v for k, v in series.items()},
        }

    def category_evolution(self, user_id: str) -> dict[str, Any]:
        """Average monthly spend by category over the history window."""
        hist = self._user_history(user_id)
        if hist.empty:
            return {"agent": "historical", "error": f"No data for {user_id}"}
        n_months = hist["month_key"].nunique() or 1
        by_cat = hist.groupby("category")["amount"].sum() / n_months
        avg_by_cat = by_cat.round(2).sort_values(ascending=False).to_dict()
        return {
            "agent": "historical",
            "tool": "category_evolution",
            "monthly_avg_by_category": avg_by_cat,
            "months_in_window": n_months,
        }

    def summary_by_year(self, user_id: str) -> dict[str, Any]:
        """Statement-style summary: totals and transaction count by year."""
        hist = self._user_history(user_id)
        if hist.empty:
            return {"agent": "historical", "error": f"No data for {user_id}"}
        by_year = hist.groupby("year").agg(
            total=("amount", "sum"),
            transaction_count=("amount", "count"),
        ).round(2)
        summary = {
            str(int(y)): {"total": float(row["total"]), "transaction_count": int(row["transaction_count"])}
            for y, row in by_year.iterrows()
        }
        return {
            "agent": "historical",
            "tool": "summary_by_year",
            "summary_by_year": summary,
        }

    def run(
        self,
        user_id: str,
        year: Optional[int] = None,
        month_range: Optional[tuple[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Run full historical review. Optional year or month_range for filtering.
        Returns combined structured output for the orchestrator.
        """
        out = {"agent": "historical", "user_id": user_id, "window_months": self.window_months}
        out["yearly_totals"] = self.yearly_totals(user_id)
        out["yoy_change"] = self.year_over_year_change(user_id)
        out["monthly_trends"] = self.monthly_trends(user_id)
        out["category_evolution"] = self.category_evolution(user_id)
        out["summary_by_year"] = self.summary_by_year(user_id)
        return out
