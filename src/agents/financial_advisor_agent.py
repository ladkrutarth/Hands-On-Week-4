"""
Financial Advisor Agent — Tool-based conversational financial analysis.
Supports: fraud detection, category advice, savings plan, spending chart, suspicious activity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re
import pandas as pd
import numpy as np
import random
from agents.memory import get_memory
from models.agent_tools_data import GLOBAL_SCAM_STATS

PROJECT_ROOT       = Path(__file__).resolve().parents[2]
ADVISOR_DATA_PATH  = PROJECT_ROOT / "data" / "csv_data" / "financial_advisor_dataset.csv"

# Category advice knowledge-base
CATEGORY_ADVICE: dict[str, dict] = {
    "coffee": {
        "label": "☕ Coffee Shops",
        "avg_weekly": 35,
        "tips": [
            "Brew at home: a $5 bag of beans brews ~30 cups vs 5–6 café visits.",
            "Switch to a 3x/week café limit — saves ~$300/year.",
            "Use a loyalty card (e.g. Starbucks Stars) to offset costs.",
            "Try office coffee or a French press for work-day caffeine.",
        ],
        "annual_waste_estimate": "$900–$1,800 for daily buyers.",
        "risk": "low",
    },
    "dining": {
        "label": "🍽️ Dining & Restaurants",
        "avg_weekly": 120,
        "tips": [
            "Meal-prep Sunday: cover 4–5 weekday lunches and save $200+/month.",
            "Limit restaurant visits to 2×/week — budget $40 per outing.",
            "Use apps like Too Good To Go for discounted restaurant leftovers.",
            "Avoid weekday dinner delivery fees — cook or batch-cook instead.",
        ],
        "annual_waste_estimate": "$2,400–$4,800 for frequent diners.",
        "risk": "medium",
    },
    "club": {
        "label": "🎉 Clubs & Entertainment",
        "avg_weekly": 80,
        "tips": [
            "Set a hard monthly entertainment budget ($150 cap recommended).",
            "Drink at home before going out — pre-gaming saves 40–60% on bar spend.",
            "Look for free local events, open-mic nights, or free museum days.",
            "Audit recurring club memberships — cancel the ones you visit < 2×/month.",
        ],
        "annual_waste_estimate": "$3,000–$6,000+ for frequent club-goers.",
        "risk": "high",
    },
    "gambling": {
        "label": "🎲 Gambling & Casinos",
        "avg_weekly": 150,
        "tips": [
            "Set a hard session loss limit — walk away when it's hit.",
            "Treat gambling budget as entertainment expense, never 'investment'.",
            "Avoid chasing losses — it statistically increases total loss.",
            "If habitual, consider a self-exclusion program or cooling-off period.",
            "Redirect even 50% of gambling spend to an emergency fund.",
        ],
        "annual_waste_estimate": "$5,000–$20,000+ for regular gamblers.",
        "risk": "critical",
        "warning": "⚠️ Gambling-related spending has a **negative credit score impact** and is flagged by fraud systems.",
    },
    "shopping": {
        "label": "🛍️ Shopping & Retail",
        "avg_weekly": 90,
        "tips": [
            "Practice the 24-hour rule: wait a day before non-essential purchases.",
            "Unsubscribe from retailer emails to reduce impulse buys.",
            "Buy off-season clothing (up to 70% off in clearance).",
            "Use cashback cards (1–5% back on purchases you'd make anyway).",
        ],
        "annual_waste_estimate": "$2,000–$5,000 for impulse buyers.",
        "risk": "medium",
    },
    "gas": {
        "label": "⛽ Gas & Transportation",
        "avg_weekly": 60,
        "tips": [
            "Use GasBuddy or Waze to find cheapest nearby fuel.",
            "Inflate tires properly — saves up to 3% on fuel economy.",
            "Carpool or combine errands into single trips.",
            "Consider an EV or hybrid for long commutes.",
        ],
        "annual_waste_estimate": "$500–$1,500 vs optimized alternatives.",
        "risk": "low",
    },
    "subscriptions": {
        "label": "📦 Subscriptions",
        "avg_weekly": 30,
        "tips": [
            "Audit ALL subscriptions — the average American pays for 12+ they forgot about.",
            "Share streaming plans (Netflix, Spotify, etc.) with family.",
            "Rotate services: subscribe for 1 month, binge, cancel, repeat.",
            "Use card-linked offers to get cash back on subscriptions you keep.",
        ],
        "annual_waste_estimate": "$800–$2,400 on forgotten subscriptions.",
        "risk": "medium",
    },
}

# Fraud signal categories that are inherently suspicious
HIGH_RISK_CATEGORIES = {"gambling", "alcohol", "adult_entertainment", "wire_transfer", "crypto"}
SUSPICIOUS_MERCHANTS  = {"cash advance", "western union", "wire", "pawn", "payday loan"}


class FinancialAdvisorAgent:
    """Conversational agent that answers spending questions via tool calls."""

    def __init__(self, llm=None):
        self._df: pd.DataFrame | None = None
        self.llm = llm  # Optional; shared with GuardAgent for smart-path synthesis

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(ADVISOR_DATA_PATH)
            self._df["transaction_date"] = pd.to_datetime(self._df["transaction_date"])
        return self._df

    # ── TOOLS ──────────────────────────────────────────────────────────────

    def tool_monthly_comparison(self, user_id: str) -> dict[str, Any]:
        """Compare current month vs. previous month spending."""
        user_df = self.df[self.df["user_id"] == user_id].copy()
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        months = sorted(user_df["month_key"].unique())
        if len(months) < 2:
            return {"error": "Insufficient history for comparison"}

        curr_key, prev_key = months[-1], months[-2]
        curr = user_df[user_df["month_key"] == curr_key]["amount"].sum()
        prev = user_df[user_df["month_key"] == prev_key]["amount"].sum()
        change_pct = ((curr - prev) / max(prev, 1)) * 100

        curr_breakdown = (
            user_df[user_df["month_key"] == curr_key]
            .groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )

        return {
            "tool": "monthly_comparison",
            "current_month": curr_key,
            "previous_month": prev_key,
            "current_spend": round(curr, 2),
            "previous_spend": round(prev, 2),
            "change_pct": round(change_pct, 1),
            "trend": "📈 UP" if change_pct > 5 else ("📉 DOWN" if change_pct < -5 else "➡️ STABLE"),
            "top_categories_this_month": {k: round(v, 2) for k, v in curr_breakdown.items()},
            "show_chart": True,
        }

    def tool_find_cancellable_subscriptions(self, user_id: str, target_savings: float = 100.0) -> dict[str, Any]:
        """Find subscriptions to cancel to reach a savings target."""
        subs = self.df[
            (self.df["user_id"] == user_id) & (self.df["is_subscription"] == True)
        ].copy()
        if subs.empty:
            return {"tool": "find_cancellable_subscriptions", "message": "No subscriptions found.", "items": []}

        by_merchant = (
            subs.groupby("merchant")["amount"]
            .agg(["sum", "count", "mean"])
            .rename(columns={"sum": "total", "count": "transactions", "mean": "avg"})
            .reset_index()
            .sort_values("total", ascending=False)
        )

        selected, running_total = [], 0.0
        for _, row in by_merchant.iterrows():
            if running_total >= target_savings:
                break
            selected.append({
                "merchant": row["merchant"],
                "monthly_cost": round(row["avg"], 2),
                "annual_cost": round(row["avg"] * 12, 2),
            })
            running_total += row["avg"]

        return {
            "tool": "find_cancellable_subscriptions",
            "target_savings": target_savings,
            "projected_monthly_savings": round(running_total, 2),
            "recommended_cancellations": selected,
        }

    def tool_credit_score_impact(self, user_id: str, target_category: str | None = None) -> dict[str, Any]:
        """Estimate credit score impact from spending patterns, utilization, and payment history."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        impact_counts = user_df["credit_score_impact_category"].value_counts().to_dict()
        total = len(user_df)
        pos_ratio = impact_counts.get("positive", 0) / max(total, 1)
        neg_ratio = impact_counts.get("negative", 0) / max(total, 1)

        # Credit Limit and Utilization Simulation
        avg_monthly_spend = user_df.groupby("month_key")["amount"].sum().mean()
        base_credit_limit = max(5000.0, avg_monthly_spend * 3)
        utilization = avg_monthly_spend / base_credit_limit
        
        # Credit Score calculation
        base = 680
        score_adj = round((pos_ratio * 40) - (neg_ratio * 60))
        if utilization > 0.3:
            score_adj -= round((utilization - 0.3) * 100)
        
        estimated_score = base + score_adj

        result: dict[str, Any] = {
            "tool": "credit_score_impact",
            "user_id": user_id,
            "estimated_credit_score": min(850, max(580, estimated_score)),
            "credit_limit": round(base_credit_limit, 2),
            "utilization_pct": round(utilization * 100, 1),
            "monthly_spend": round(avg_monthly_spend, 2)
        }

        # Analyze where to purchase and how much
        high_risk_cats = user_df[user_df["credit_score_impact_category"] == "negative"]["category"].unique()
        good_cats = user_df[user_df["credit_score_impact_category"] == "positive"]["category"].unique()
        
        # Calculate additional metrics for detailed analysis
        num_months = max(len(user_df["month_key"].unique()), 1)
        total_txns = len(user_df)
        avg_txn = round(user_df["amount"].mean(), 2)
        pos_count = impact_counts.get("positive", 0)
        neg_count = impact_counts.get("negative", 0)
        neutral_count = impact_counts.get("neutral", 0)
        
        good_cats_str = ", ".join(good_cats[:3]) if len(good_cats) > 0 else "Groceries, Utilities"
        risk_cats_str = ", ".join(high_risk_cats[:3]) if len(high_risk_cats) > 0 else "Gambling, Cash advances"
        
        # Utilization advice
        if utilization <= 0.1:
            util_advice = "This is excellent and well below the recommended 30% threshold. Maintaining low utilization is one of the strongest contributors to a high credit score."
        elif utilization <= 0.3:
            util_advice = "This is within the healthy range. Aim to keep it below 30% consistently. Lenders view low utilization as a sign of responsible credit management."
        else:
            util_advice = f"This exceeds the recommended 30% threshold. Reducing your monthly spend by ${(avg_monthly_spend - base_credit_limit * 0.3):,.0f} would bring you into the optimal range and could boost your score by 20-40 points."
        
        result["recommendation"] = (
            "**Credit Building Strategy:**\n\n"
            f"**📊 Utilization Analysis**\n"
            f"- You are using **{result['utilization_pct']}%** of your estimated **${result['credit_limit']:,.0f}** credit limit (spending ~${avg_monthly_spend:,.0f}/mo). {util_advice}\n\n"
            f"**💳 Payment Discipline**\n"
            f"- Always pay your **full statement balance** on or before the due date every month. Payment history accounts for approximately 35% of your credit score. Even a single missed payment can drop your score by 60-100 points and stays on your report for up to 7 years.\n\n"
            f"**📈 Spending Mix Impact**\n"
            f"- Out of **{total_txns}** transactions analyzed over {num_months} months (avg **${avg_txn}** per transaction), **{pos_count}** had a positive credit impact, **{neg_count}** had negative impact, and **{neutral_count}** were neutral.\n\n"
            f"**✅ Where to Purchase**\n"
            f"- Favor reliable, recurring merchants in stable categories like **{good_cats_str}**. Consistent, predictable spending patterns on essentials signal financial stability to credit bureaus.\n\n"
            f"**⚠️ Categories to Minimize**\n"
            f"- Reduce spending in **{risk_cats_str}**. These categories are flagged by credit models as high-risk and can negatively impact your score over time. Limit discretionary purchases in these areas.\n\n"
            f"**🎯 Action Plan**\n"
            f"1. Keep utilization under 30% by tracking your balance weekly.\n"
            f"2. Set up autopay to never miss a due date.\n"
            f"3. Redirect spending from high-risk categories to stable ones.\n"
            f"4. Request a credit limit increase to naturally lower your utilization ratio."
        )
        return result

    def tool_spending_summary(self, user_id: str) -> dict[str, Any]:
        """Get a full spending summary for the user."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        by_cat = (
            user_df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
            .to_dict()
        )
        top_merchant = user_df.groupby("merchant")["amount"].sum().idxmax()
        total = round(user_df["amount"].sum(), 2)
        avg_monthly = round(user_df.groupby("month_key")["amount"].sum().mean(), 2)

        return {
            "tool": "spending_summary",
            "user_id": user_id,
            "total_spend": total,
            "avg_monthly_spend": avg_monthly,
            "top_categories": {k: round(v, 2) for k, v in by_cat.items()},
            "top_merchant": top_merchant,
            "archetype": user_df["archetype"].iloc[0],
            "velocity_7d": int(user_df["spending_velocity_7d"].mean()),
            "show_chart": True,
        }

    # ── NEW TOOL: Category Advice ───────────────────────────────────────────

    def tool_category_advice(self, user_id: str, category_keyword: str) -> dict[str, Any]:
        """Give personalized spending advice for a specific category (coffee, dining, club, gambling…)."""
        user_df = self.df[self.df["user_id"] == user_id]
        key = category_keyword.lower().strip()

        # Map keyword synonyms to our advice keys
        synonym_map = {
            "coffee shop": "coffee", "café": "coffee", "cafe": "coffee", "starbucks": "coffee",
            "food": "dining", "restaurant": "dining", "eat": "dining", "takeout": "dining",
            "bar": "club", "nightclub": "club", "party": "club", "entertainment": "club",
            "casino": "gambling", "bet": "gambling", "lottery": "gambling", "poker": "gambling",
            "shop": "shopping", "retail": "shopping", "amazon": "shopping",
            "petrol": "gas", "fuel": "gas", "uber": "gas", "transport": "gas",
            "subscription": "subscriptions", "netflix": "subscriptions", "spotify": "subscriptions",
        }
        resolved = key
        for syn, mapped in synonym_map.items():
            if syn in key:
                resolved = mapped
                break

        advice = CATEGORY_ADVICE.get(resolved, None)

        # Actual user spend in this category (fuzzy match)
        user_spend_this_cat = 0.0
        if not user_df.empty:
            mask = user_df["category"].str.lower().str.contains(resolved, na=False)
            user_spend_this_cat = round(user_df[mask]["amount"].sum(), 2)

        return {
            "tool": "category_advice",
            "category": resolved,
            "label": advice["label"] if advice else f"📂 {resolved.title()}",
            "user_spend": user_spend_this_cat,
            "tips": advice["tips"] if advice else ["Track this category for 30 days to identify saving opportunities."],
            "annual_waste_estimate": advice.get("annual_waste_estimate", "N/A") if advice else "N/A",
            "risk_level": advice.get("risk", "medium") if advice else "medium",
            "warning": advice.get("warning", "") if advice else "",
        }

    # ── NEW TOOL: Savings Plan ──────────────────────────────────────────────

    def tool_savings_plan(self, user_id: str) -> dict[str, Any]:
        """Generate a personalized month-by-month savings plan for every single spending category."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        monthly_avg = round(user_df.groupby("month_key")["amount"].sum().mean(), 2)

        by_cat = (
            user_df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .to_dict()
        )

        savings_opportunities = []
        total_potential_savings = 0.0
        
        # Analyze ALL categories
        for cat, total in by_cat.items():
            monthly = round(total / max(len(user_df["month_key"].unique()), 1), 2)
            reduction = 0.0
            tip = ""
            cat_lower = cat.lower()

            if any(k in cat_lower for k in ["coffee", "café", "cafe"]):
                reduction = round(monthly * 0.5, 2)
                tip = "Brew at home 5 days/week"
            elif any(k in cat_lower for k in ["dining", "food", "restaurant", "takeout"]):
                reduction = round(monthly * 0.35, 2)
                tip = "Meal prep + limit dine-out to twice weekly"
            elif any(k in cat_lower for k in ["shopping", "retail", "clothing"]):
                reduction = round(monthly * 0.30, 2)
                tip = "Apply 24-hour rule before purchases"
            elif any(k in cat_lower for k in ["gambling", "casino", "bet"]):
                reduction = round(monthly * 0.70, 2)
                tip = "Set hard session limits"
            elif any(k in cat_lower for k in ["club", "entertainment", "bar"]):
                reduction = round(monthly * 0.40, 2)
                tip = "Set $150/month entertainment cap"
            elif any(k in cat_lower for k in ["subscription"]):
                reduction = round(monthly * 0.45, 2)
                tip = "Audit and cancel unused subscriptions"
            elif any(k in cat_lower for k in ["gas", "fuel", "transport"]):
                reduction = round(monthly * 0.15, 2)
                tip = "Carpool + use GasBuddy"
            elif any(k in cat_lower for k in ["grocery", "groceries"]):
                reduction = round(monthly * 0.10, 2)
                tip = "Use generic brands and buy in bulk"
            elif any(k in cat_lower for k in ["utilities", "electric", "water"]):
                reduction = round(monthly * 0.05, 2)
                tip = "Use smart thermostat and energy-efficient bulbs"
            elif any(k in cat_lower for k in ["healthcare", "medical", "pharmacy"]):
                reduction = round(monthly * 0.05, 2)
                tip = "Ask for generic prescriptions"
            else:
                reduction = round(monthly * 0.10, 2) # General 10% cut
                tip = f"Review {cat} spend for strict 10% reduction"

            if reduction > 0:
                savings_opportunities.append({
                    "category": cat,
                    "monthly_spend": monthly,
                    "potential_saving": reduction,
                    "tip": tip,
                })
                total_potential_savings += reduction

        return {
            "tool": "savings_plan",
            "user_id": user_id,
            "monthly_avg_spend": monthly_avg,
            "potential_monthly_savings": round(total_potential_savings, 2),
            "potential_annual_savings": round(total_potential_savings * 12, 2),
            "opportunities": savings_opportunities,  # Return ALL categories
            "archetype": user_df["archetype"].iloc[0] if "archetype" in user_df.columns else "unknown",
        }

    # ── NEW TOOL: Real-time Fraud Detection ────────────────────────────────

    def tool_realtime_fraud_check(self, user_id: str, latest_n: int = 10) -> dict[str, Any]:
        """Scan the most recent transactions for suspicious/fraudulent patterns."""
        user_df = self.df[self.df["user_id"] == user_id].copy()
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        # Sort by date, take last N
        recent = user_df.sort_values("transaction_date", ascending=False).head(latest_n)

        alerts = []
        for _, row in recent.iterrows():
            flags = []
            cat = str(row.get("category", "")).lower()
            merchant = str(row.get("merchant", "")).lower()
            amt = float(row.get("amount", 0))
            risk = float(row.get("risk_score", 0))
            fraud_flag = bool(row.get("is_fraud_flag", False))
            hour = int(pd.to_datetime(row["transaction_date"]).hour) if pd.notna(row.get("transaction_date")) else 12

            if fraud_flag or risk > 0.75:
                flags.append(f"🚨 High fraud score ({risk:.2f}) detected by ML model")
            if any(k in cat for k in HIGH_RISK_CATEGORIES):
                flags.append(f"⚠️ High-risk category: {cat}")
            if any(k in merchant for k in SUSPICIOUS_MERCHANTS):
                flags.append(f"🔴 Suspicious merchant keyword: {merchant}")
            if hour < 2 or hour > 23:
                flags.append(f"🌙 Late-night transaction at {hour:02d}:00")
            if amt > 500 and risk > 0.5:
                flags.append(f"💰 Large transaction (${amt:.2f}) with elevated risk")

            if flags:
                alerts.append({
                    "transaction_date": str(row["transaction_date"])[:10],
                    "merchant": row.get("merchant", "Unknown"),
                    "category": row.get("category", "Unknown"),
                    "amount": round(amt, 2),
                    "risk_score": round(risk, 3),
                    "flags": flags,
                    "severity": "CRITICAL" if (fraud_flag or risk > 0.85) else ("HIGH" if risk > 0.65 else "MEDIUM"),
                })

        avg_risk = round(float(recent["risk_score"].mean()), 3) if "risk_score" in recent.columns else 0.0

        return {
            "tool": "realtime_fraud_check",
            "user_id": user_id,
            "transactions_scanned": len(recent),
            "alerts_found": len(alerts),
            "avg_risk_score": avg_risk,
            "overall_status": (
                "🚨 CRITICAL — Immediate action required!" if any(a["severity"] == "CRITICAL" for a in alerts)
                else ("⚠️ WARNING — Suspicious activity detected" if alerts
                else "✅ CLEAR — No suspicious activity detected")
            ),
            "alerts": alerts,
        }

    # ── NEW TOOL: Suspicious Activity Monitor (real-time watch) ───────────

    def tool_suspicious_activity_monitor(self, user_id: str) -> dict[str, Any]:
        """Watch for account-level suspicious patterns: velocity, location, category shifts."""
        user_df = self.df[self.df["user_id"] == user_id].copy()
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        alerts = []
        recent = user_df.sort_values("transaction_date", ascending=False).head(30)

        # 1. Velocity spike
        velocity_7d = float(recent["spending_velocity_7d"].mean()) if "spending_velocity_7d" in recent.columns else 0
        velocity_baseline = float(user_df["spending_velocity_7d"].mean()) if "spending_velocity_7d" in user_df.columns else 0
        if velocity_baseline > 0 and velocity_7d > velocity_baseline * 1.8:
            alerts.append({
                "type": "velocity_spike",
                "emoji": "⚡",
                "title": "Spending Velocity Spike",
                "detail": f"7-day spend velocity is {velocity_7d:.0f} — {(velocity_7d/velocity_baseline - 1)*100:.0f}% above your norm.",
                "severity": "HIGH",
            })

        # 2. High-risk category surge
        if "category" in recent.columns:
            risky_txns = recent[recent["category"].str.lower().isin(HIGH_RISK_CATEGORIES)]
            if len(risky_txns) >= 3:
                alerts.append({
                    "type": "risky_category",
                    "emoji": "🎲",
                    "title": f"Risky Category Activity",
                    "detail": f"{len(risky_txns)} transactions in high-risk categories (gambling/cash/wire) in last 30 txns.",
                    "severity": "HIGH",
                })

        # 3. High fraud-flag rate
        if "is_fraud_flag" in recent.columns:
            flag_rate = recent["is_fraud_flag"].mean()
            if flag_rate > 0.15:
                alerts.append({
                    "type": "fraud_flags",
                    "emoji": "🚩",
                    "title": "Multiple Fraud Flags",
                    "detail": f"{flag_rate:.0%} of your recent transactions triggered the fraud model.",
                    "severity": "CRITICAL",
                })

        # 4. Unusual hour activity
        if "transaction_date" in recent.columns:
            recent = recent.copy()
            recent["hour"] = pd.to_datetime(recent["transaction_date"]).dt.hour
            late_night = recent[(recent["hour"] >= 0) & (recent["hour"] <= 4)]
            if len(late_night) >= 3:
                alerts.append({
                    "type": "late_night",
                    "emoji": "🌙",
                    "title": "Late-Night Transaction Cluster",
                    "detail": f"{len(late_night)} transactions between midnight–4am in recent activity.",
                    "severity": "MEDIUM",
                })

        # 5. Large one-off transactions
        amt_mean = float(recent["amount"].mean())
        large_txns = recent[recent["amount"] > amt_mean * 3]
        if len(large_txns) >= 2:
            alerts.append({
                "type": "large_txn",
                "emoji": "💸",
                "title": "Unusually Large Transactions",
                "detail": f"{len(large_txns)} transactions 3× above your average (avg: ${amt_mean:.2f}).",
                "severity": "MEDIUM",
            })

        overall = (
            "🚨 CRITICAL" if any(a["severity"] == "CRITICAL" for a in alerts)
            else ("⚠️ WARNING" if alerts else "✅ ALL CLEAR")
        )

        return {
            "tool": "suspicious_activity_monitor",
            "user_id": user_id,
            "alert_count": len(alerts),
            "overall_status": overall,
            "alerts": alerts,
        }

    # ── NEW ADVANCED TOOLS ──────────────────────────────────────────────────

    def tool_cash_flow_forecast(self, user_id: str) -> dict[str, Any]:
        """Projects next month's total spend based on historical averages and recurring subscriptions."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {"error": f"No data for {user_id}"}
        
        subs = user_df[user_df["is_subscription"] == True]
        avg_monthly_subs = float(subs.groupby("merchant")["amount"].mean().sum()) if not subs.empty else 0.0
        
        non_subs = user_df[user_df["is_subscription"] == False]
        avg_monthly_non_subs = float(non_subs.groupby("month_key")["amount"].sum().mean()) if not non_subs.empty else 0.0
        
        forecast = avg_monthly_subs + avg_monthly_non_subs
        
        return {
            "tool": "cash_flow_forecast",
            "forecast_amount": round(forecast, 2),
            "subs_amount": round(avg_monthly_subs, 2),
            "variable_amount": round(avg_monthly_non_subs, 2)
        }

    def tool_detect_price_hikes(self, user_id: str) -> dict[str, Any]:
        """Analyzes merchant + amount for recurring subscriptions to flag recent price increases."""
        user_df = self.df[(self.df["user_id"] == user_id) & (self.df["is_subscription"] == True)]
        alerts = []
        if not user_df.empty:
            for merchant, group in user_df.groupby("merchant"):
                group = group.sort_values("transaction_date")
                if len(group) >= 2:
                    last_two = group["amount"].tail(2).values
                    if last_two[1] > last_two[0] * 1.05: # 5% hike
                        alerts.append({
                            "merchant": merchant,
                            "old_price": round(last_two[0], 2),
                            "new_price": round(last_two[1], 2),
                            "increase_pct": round((last_two[1] / last_two[0] - 1) * 100, 1)
                        })
        return {
            "tool": "detect_price_hikes",
            "alerts": alerts
        }


    def tool_tax_deductible_finder(self, user_id: str) -> dict[str, Any]:
        """Flags potential tax-deductible expenses like Healthcare, Charity, or Business travel."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {"error": f"No data for {user_id}"}
            
        tax_cats = ["healthcare", "medical", "pharmacy", "charity", "donation", "travel", "office", "education"]
        mask = user_df["category"].str.lower().apply(lambda x: any(tc in x for tc in tax_cats))
        
        tax_df = user_df[mask]
        total_potential = float(tax_df["amount"].sum())
        
        breakdown = tax_df.groupby("category")["amount"].sum().to_dict()
        
        return {
            "tool": "tax_deductible_finder",
            "total_potential_deduction": round(total_potential, 2),
            "breakdown": {k: round(v, 2) for k, v in breakdown.items() if v > 0}
        }

    def tool_surplus_optimizer(self, user_id: str) -> dict[str, Any]:
        """Calculates estimated leftover monthly surplus (Income - Spend) and suggests optimization strategies."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {"error": f"No data for {user_id}"}
            
        avg_monthly_spend = float(user_df.groupby("month_key")["amount"].sum().mean())
        assumed_income = max(5000.0, avg_monthly_spend * 1.5) # Fake income simulation
        surplus = assumed_income - avg_monthly_spend
        
        return {
            "tool": "surplus_optimizer",
            "assumed_monthly_income": round(assumed_income, 2),
            "avg_monthly_spend": round(avg_monthly_spend, 2),
            "monthly_surplus": round(surplus, 2),
            "suggestions": [
                f"Invest ${round(surplus * 0.5, 2)} into an Index Fund.",
                f"Allocate ${round(surplus * 0.3, 2)} to a High-Yield Savings.",
                f"Keep ${round(surplus * 0.2, 2)} as liquid fun money."
            ] if surplus > 0 else ["Reduce spending to create a surplus."]
        }

    def tool_liquidity_guard(self, user_id: str) -> dict[str, Any]:
        """Predictive alerts for upcoming bills vs assumed available balance."""
        user_df = self.df[(self.df["user_id"] == user_id) & (self.df["is_subscription"] == True)]
        if user_df.empty:
            return {"error": f"No data for {user_id}"}
            
        avg_monthly_spend = float(self.df[self.df["user_id"] == user_id].groupby("month_key")["amount"].sum().mean())
        assumed_balance = avg_monthly_spend * 0.5 # Fake balance
        
        # upcoming bills in next 7 days (mocked as total sub spend / 4)
        total_sub_spend = float(user_df.groupby("merchant")["amount"].mean().sum())
        upcoming_bills = total_sub_spend / 4
        
        risk = "HIGH" if upcoming_bills > assumed_balance * 0.8 else "LOW"
        
        return {
            "tool": "liquidity_guard",
            "assumed_balance": round(assumed_balance, 2),
            "upcoming_7d_bills": round(upcoming_bills, 2),
            "liquidity_risk": risk
        }

    def tool_market_fraud_insights(self) -> dict[str, Any]:
        """
        Provides high-level market fraud context and category-level fraud frequencies.
        Enables the advisor to reason about global scam trends vs local dataset risk.
        Uses fraud_scores_output.csv for local risk calculation.
        """
        try:
            # 1. Load the authoritative risk dataset
            risk_path = PROJECT_ROOT / "data" / "csv_data" / "fraud_scores_output.csv"
            if not risk_path.exists():
                return {"error": "Risk dataset not found"}
            
            risk_df = pd.read_csv(risk_path)
            risk_df.columns = [c.upper() for c in risk_df.columns]
            
            # 2. Dataset-specific Category Fraud Rates (Local Heatmap)
            # We look for HIGH risk transactions per category
            cat_stats = risk_df.groupby('CATEGORY').agg(
                total_txns=('COMBINED_RISK_SCORE', 'count'),
                high_risk_cases=('RISK_LEVEL', lambda x: (x == 'HIGH').sum())
            ).reset_index()
            cat_stats['risk_density'] = (cat_stats['high_risk_cases'] / cat_stats['total_txns']) * 100
            
            heatmap_data = cat_stats.sort_values(by='risk_density', ascending=False).to_dict('records')
            
            # 3. Global Scam Stats (from central model)
            return {
                "tool": "market_fraud_insights",
                "global_scam_trends": GLOBAL_SCAM_STATS,
                "local_cat_fraud_heatmap": [
                    {
                        "category": r['CATEGORY'],
                        "risk_density_pct": round(r['risk_density'], 2),
                        "high_risk_cases": int(r['high_risk_cases'])
                    } for r in heatmap_data[:10]
                ],
                "summary": "Investment and BEC scams lead global losses ($65B+), while local data shows highest risk density in " + 
                           f"{heatmap_data[0]['CATEGORY'] if not cat_stats.empty else 'shopping_net'}."
            }
        except Exception as e:
            return {"error": str(e)}


    # ── CHAT ROUTER ────────────────────────────────────────────────────────

    _CHART_KEYWORDS = [
        "chart", "graph", "show me", "visuali", "bar", "plot",
        "category", "categories", "breakdown", "spending", "how much",
        "summary", "overview", "what am i", "total",
    ]

    def _multi_agent_enabled(self) -> bool:
        """True when ENABLE_MULTI_AGENT_ADVISOR is set (e.g. 1, true, yes)."""
        v = os.environ.get("ENABLE_MULTI_AGENT_ADVISOR", "").strip().lower()
        return v in ("1", "true", "yes")

    def _agentic_enabled(self) -> bool:
        """ReAct tool loop when LLM is available unless ENABLE_AGENTIC_ADVISOR=0."""
        if not self.llm or not hasattr(self.llm, "generate"):
            return False
        v = os.environ.get("ENABLE_AGENTIC_ADVISOR", "1").strip().lower()
        return v not in ("0", "false", "no", "off")

    def build_tool_registry(self, user_id: str):
        """Register deterministic tool_* methods for the ReAct loop (user_id bound)."""
        from agents.tool_registry import ToolRegistry

        reg = ToolRegistry()
        reg.register(
            "monthly_comparison",
            "Compare current vs previous month spending.",
            lambda: self.tool_monthly_comparison(user_id),
            {"type": "object", "properties": {}},
        )
        reg.register(
            "find_cancellable_subscriptions",
            "Find subscriptions to cancel to reach a savings target.",
            lambda target_savings=100.0: self.tool_find_cancellable_subscriptions(
                user_id, target_savings=float(target_savings)
            ),
            {
                "type": "object",
                "properties": {
                    "target_savings": {"type": "number", "description": "Monthly savings goal"},
                },
            },
        )
        reg.register(
            "credit_score_impact",
            "Estimate credit score impact from spending patterns.",
            lambda target_category=None: self.tool_credit_score_impact(
                user_id, target_category=target_category
            ),
            {
                "type": "object",
                "properties": {
                    "target_category": {"type": "string", "description": "Optional category focus"},
                },
            },
        )
        reg.register(
            "spending_summary",
            "Full spending overview: totals, top categories, archetype.",
            lambda: self.tool_spending_summary(user_id),
            {"type": "object", "properties": {}},
        )
        reg.register(
            "category_advice",
            "Personalized tips for a spending category (coffee, dining, gambling, etc.).",
            lambda category_keyword="dining": self.tool_category_advice(
                user_id, str(category_keyword)
            ),
            {
                "type": "object",
                "properties": {
                    "category_keyword": {
                        "type": "string",
                        "description": "coffee|dining|club|gambling|shopping|gas|subscriptions",
                    },
                },
            },
        )
        reg.register(
            "savings_plan",
            "Comprehensive category-level savings strategy.",
            lambda: self.tool_savings_plan(user_id),
            {"type": "object", "properties": {}},
        )
        reg.register(
            "realtime_fraud_check",
            "Scan recent transactions for fraud signals.",
            lambda latest_n=10: self.tool_realtime_fraud_check(user_id, latest_n=int(latest_n)),
            {
                "type": "object",
                "properties": {"latest_n": {"type": "integer"}},
            },
        )
        reg.register(
            "suspicious_activity_monitor",
            "Monitor account for suspicious activity patterns.",
            lambda: self.tool_suspicious_activity_monitor(user_id),
            {"type": "object", "properties": {}},
        )
        reg.register(
            "cash_flow_forecast",
            "Forecast cash flow for the next 30 days.",
            lambda: self.tool_cash_flow_forecast(user_id),
            {"type": "object", "properties": {}},
        )
        reg.register(
            "detect_price_hikes",
            "Detect recent subscription price increases.",
            lambda: self.tool_detect_price_hikes(user_id),
            {"type": "object", "properties": {}},
        )
        reg.register(
            "tax_deductible_finder",
            "Find potentially tax-deductible expenses.",
            lambda: self.tool_tax_deductible_finder(user_id),
            {"type": "object", "properties": {}},
        )
        reg.register(
            "surplus_optimizer",
            "Estimate surplus and suggest optimization.",
            lambda: self.tool_surplus_optimizer(user_id),
            {"type": "object", "properties": {}},
        )
        reg.register(
            "liquidity_guard",
            "Check short-term liquidity vs upcoming bills.",
            lambda: self.tool_liquidity_guard(user_id),
            {"type": "object", "properties": {}},
        )
        reg.register(
            "market_fraud_insights",
            "Global scam trends and local fraud heatmap.",
            lambda: self.tool_market_fraud_insights(),
            {"type": "object", "properties": {}},
        )
        return reg

    # Intent → tool allowlists (keep prompts small; curriculum: intent→group routing)
    _INTENT_TOOL_GROUPS: dict[str, list[str]] = {
        "fraud": [
            "realtime_fraud_check",
            "suspicious_activity_monitor",
            "market_fraud_insights",
        ],
        "savings": [
            "savings_plan",
            "find_cancellable_subscriptions",
            "surplus_optimizer",
            "detect_price_hikes",
        ],
        "credit": ["credit_score_impact", "spending_summary"],
        "category": ["category_advice", "spending_summary"],
        "forecast": ["cash_flow_forecast", "monthly_comparison", "spending_summary"],
        "tax": ["tax_deductible_finder", "spending_summary"],
        "liquidity": ["liquidity_guard", "cash_flow_forecast", "spending_summary"],
        "overview": ["spending_summary", "monthly_comparison"],
    }

    def _infer_intent(self, message: str) -> str:
        msg = message.lower()
        if any(k in msg for k in [
            "fraud", "suspicious", "hacked", "stolen", "scam", "alert",
            "phish", "unauthorized", "risk score", "monitor activity",
        ]):
            return "fraud"
        if any(k in msg for k in [
            "credit score", "fico", "credit impact", "credit risk",
        ]):
            return "credit"
        if any(k in msg for k in [
            "save money", "savings", "how to save", "cut back", "reduce spending",
            "budget", "cancel subscription", "cancellable", "frugal",
        ]):
            return "savings"
        if any(k in msg for k in ["forecast", "cash flow", "next month", "project spend"]):
            return "forecast"
        if any(k in msg for k in ["tax", "deduction", "deductible", "write-off", "write off"]):
            return "tax"
        if any(k in msg for k in ["liquidity", "liquid", "upcoming bills", "runway", "cash cushion"]):
            return "liquidity"
        category_tokens = [
            "coffee", "café", "cafe", "dining", "restaurant", "club", "bar",
            "gambl", "casino", "shopping", "amazon", "gas", "fuel", "uber",
            "netflix", "spotify", "subscription tip", "eat out",
        ]
        if any(t in msg for t in category_tokens) and any(
            k in msg for k in ["tip", "advice", "cut", "reduce", "spend less", "too much", "habit"]
        ):
            return "category"
        if any(t in msg for t in category_tokens) and not any(
            k in msg for k in ["this month", "last month", "total", "how much", "summary"]
        ):
            return "category"
        return "overview"

    def build_tool_registry_for_intent(self, user_id: str, message: str):
        """Full registry filtered to the inferred intent group (+ always allow spending_summary)."""
        full = self.build_tool_registry(user_id)
        intent = self._infer_intent(message)
        names = list(self._INTENT_TOOL_GROUPS.get(intent, self._INTENT_TOOL_GROUPS["overview"]))
        if "spending_summary" not in names:
            names.append("spending_summary")
        # Category queries need the advice tool even if overview was chosen loosely
        if intent == "category" and "category_advice" not in names:
            names.insert(0, "category_advice")
        subset = full.subset(names)
        return subset if len(subset) else full, intent

    def _keyword_route(self, message: str, user_id: str) -> tuple[list[dict], bool]:
        """Legacy keyword router — used as ReAct fallback and when no LLM."""
        msg = message.lower()
        results: list[dict] = []
        show_chart = any(k in msg for k in self._CHART_KEYWORDS)

        # Tight month/compare triggers — avoid bare "month" matching every dining Q
        month_compare_triggers = [
            "more this month", "spending this month", "this month vs", "vs last month",
            "last month", "month over month", "mom", "compare months", "compared to last",
            "increase", "decreased", "spend trend", "spending trend",
        ]
        if any(k in msg for k in month_compare_triggers):
            results.append(self.tool_monthly_comparison(user_id))

        is_price_hike_query = any(k in msg for k in ["price hike", "price increase", "raised price", "cost went up"])
        if any(k in msg for k in [
            "cancel subscription", "cancellable", "save $", "cut subscriptions",
            "which subscription", "subscriptions to cancel",
        ]) and not is_price_hike_query:
            match = re.search(r"\$(\d+)", message)
            target = float(match.group(1)) if match else 100.0
            results.append(self.tool_find_cancellable_subscriptions(user_id, target_savings=target))

        if any(k in msg for k in ["credit score", "fico", "credit impact", "impact on credit"]):
            known_cats = ["jewelry", "electronics", "groceries", "dining", "travel", "subscriptions", "gas", "clothing"]
            target_cat = next((c for c in known_cats if c in msg), None)
            results.append(self.tool_credit_score_impact(user_id, target_category=target_cat))

        category_synonyms = {
            "coffee": ["coffee", "café", "cafe", "starbucks", "latte"],
            "dining": ["dining", "restaurant", "eat out", "takeout"],
            "club": ["club", "nightclub", "going out"],
            "gambling": ["gambl", "casino", "lottery", "poker"],
            "shopping": ["shopping", "retail", "amazon", "mall"],
            "gas": ["gas station", "petrol", "fuel cost"],
            "subscriptions": ["netflix", "spotify", "hulu", "streaming"],
        }
        # Only fire category advice for tip/habit style questions, not pure totals
        wants_category_advice = any(
            k in msg for k in ["tip", "advice", "cut back", "too much", "habit", "how to reduce", "save on"]
        ) or (
            any(any(s in msg for s in syns) for syns in category_synonyms.values())
            and not any(k in msg for k in ["how much", "total", "summary", "overview", "this month"])
        )
        if wants_category_advice:
            for cat_key, synonyms in category_synonyms.items():
                if any(s in msg for s in synonyms):
                    results.append(self.tool_category_advice(user_id, cat_key))
                    break

        if any(k in msg for k in [
            "save money", "savings plan", "savings strategy", "how to save",
            "budget plan", "save more", "cut back spending", "reduce spending", "financial plan",
        ]):
            results.append(self.tool_savings_plan(user_id))

        if any(k in msg for k in [
            "fraud", "suspicious", "hacked", "stolen", "scam", "unauthorized",
            "fraud check", "fraud alert", "risk scan",
        ]):
            results.append(self.tool_realtime_fraud_check(user_id))
            results.append(self.tool_suspicious_activity_monitor(user_id))

        if any(k in msg for k in ["spending hierarchy", "vector path", "archetype chart"]):
            results.append(self.tool_spending_summary(user_id))
            show_chart = True

        if any(k in msg for k in ["forecast", "cash flow", "project next month", "next month spend"]):
            results.append(self.tool_cash_flow_forecast(user_id))
        if is_price_hike_query or "subscription price" in msg:
            results.append(self.tool_detect_price_hikes(user_id))
        if any(k in msg for k in ["tax deduction", "deductible", "write-off", "write off"]):
            results.append(self.tool_tax_deductible_finder(user_id))
        if any(k in msg for k in ["surplus", "optimize surplus"]):
            results.append(self.tool_surplus_optimizer(user_id))
        if any(k in msg for k in ["liquidity", "liquid cash", "upcoming bills", "cash cushion"]):
            results.append(self.tool_liquidity_guard(user_id))
        if any(k in msg for k in ["market fraud", "scam heatmap", "global scam", "fraud overview"]):
            results.append(self.tool_market_fraud_insights())
            show_chart = True

        # Pure spending totals / overview
        if not results and any(k in msg for k in [
            "how much", "total spend", "spending summary", "overview", "summary", "what did i spend",
        ]):
            results.append(self.tool_spending_summary(user_id))
            show_chart = True

        if not results:
            results.append(self.tool_spending_summary(user_id))
            show_chart = True

        # Cap noisy multi-tool keyword piles
        if len(results) > 3:
            results = results[:3]

        return results, show_chart

    def chat(
        self,
        message: str,
        user_id: str,
        session_id: str | None = None,
        approved: bool = False,
    ) -> dict[str, Any]:
        """Route a natural-language question via ReAct (LLM), supervisor, or keyword fallback."""
        if session_id:
            memory = get_memory()
            memory.add_message(session_id, "user", message)

        if self._multi_agent_enabled():
            from agents.financial_orchestrator import FinancialOrchestrator
            if not hasattr(self, "_orchestrator") or getattr(self._orchestrator, "llm", None) is not self.llm:
                self._orchestrator = FinancialOrchestrator(llm=self.llm)
            return self._orchestrator.chat(message, user_id, session_id=session_id, approved=approved)

        if self._agentic_enabled():
            return self._chat_react(message, user_id, session_id=session_id, approved=approved)

        results, show_chart = self._keyword_route(message, user_id)
        reply = self._compose_reply(message, results, session_id=session_id)
        if session_id:
            get_memory().add_message(session_id, "assistant", reply)

        return {
            "reply": reply,
            "tool_results": results,
            "user_id": user_id,
            "show_chart": show_chart,
            "trace": ["keyword_router"],
            "step_count": len(results),
            "tools_used": [r.get("tool", "") for r in results if isinstance(r, dict)],
            "actions": [],
            "pending_approval": None,
            "status": "success",
        }

    def _llm_synthesize(
        self,
        message: str,
        tool_results: list[dict],
        session_id: str | None = None,
    ) -> str:
        """LLM-only final answer from tool observations (no template sections)."""
        import json

        if not self.llm or not hasattr(self.llm, "generate"):
            return "LLM unavailable — cannot synthesize agent reply."
        history = ""
        if session_id:
            history = get_memory().get_history(session_id)
        context = json.dumps(tool_results, default=str)
        if len(context) > 4000:
            context = context[:3997] + "..."
        prompt = (
            "You are Veriscan's Financial Advisor AI agent. "
            "Answer using ONLY the tool results below. Be specific with numbers. "
            "Do not invent data. Write a clear professional reply (not JSON).\n\n"
        )
        if history:
            prompt += f"Conversation history:\n{history}\n\n"
        prompt += f"Tool results:\n{context}\n\nUser question: {message}\n\nAdvisor reply:"
        try:
            out = self.llm.generate(prompt, max_tokens=450, temp=0.3)
            if out and out.strip():
                return out.strip()
        except Exception as e:
            return f"LLM synthesis failed: {e}"
        return "I ran tools but could not generate a natural-language summary."

    def _oneshot_tool_plan(self, message: str, user_id: str) -> list[dict]:
        """If ReAct JSON parse fails with no tools yet, ask LLM once for a tool list and execute it."""
        from agents.react_loop import extract_json

        registry, intent = self.build_tool_registry_for_intent(user_id, message)
        prompt = (
            "Pick 1–2 tools to answer the user. Reply with JSON only:\n"
            '{"tools":[{"name":"<tool>","args":{...}}]}\n\n'
            f"Intent hint: {intent}\n"
            f"Available tools:\n{registry.schemas_for_prompt()}\n\n"
            f"User: {message}\nJSON:"
        )
        try:
            raw = self.llm.generate(prompt, max_tokens=180, temp=0.0)
        except Exception:
            return []
        decision = extract_json(raw) or {}
        planned = decision.get("tools") or []
        if isinstance(decision.get("tool"), str):
            planned = [{"name": decision["tool"], "args": decision.get("args") or {}}]
        results = []
        for item in planned[:2]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("tool") or "").strip()
            args = item.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            if name in registry:
                results.append(registry.call(name, args, approved=True))
        if not results and "spending_summary" in registry:
            results.append(registry.call("spending_summary", {}, approved=True))
        return results

    def _chat_react(
        self,
        message: str,
        user_id: str,
        session_id: str | None = None,
        approved: bool = False,
    ) -> dict[str, Any]:
        from agents.react_loop import ReactLoop
        from agents.trajectory_log import log_trajectory

        registry, intent = self.build_tool_registry_for_intent(user_id, message)
        history = get_memory().get_history(session_id) if session_id else ""

        def _fallback() -> dict[str, Any]:
            # Only used when ReAct collected zero tools
            results = self._oneshot_tool_plan(message, user_id)
            if not results:
                results, _ = self._keyword_route(message, user_id)
            reply = self._llm_synthesize(message, results, session_id=session_id)
            if not reply or reply.startswith("LLM"):
                reply = self._compose_reply(message, results, session_id=session_id)
            show_chart = any(
                isinstance(r, dict)
                and r.get("tool") in ("spending_summary", "market_fraud_insights", "monthly_comparison")
                for r in results
            )
            return {
                "reply": reply,
                "tool_results": results,
                "show_chart": show_chart,
                "actions": [
                    {
                        "step": i + 1,
                        "tool": r.get("tool", "tool"),
                        "args": {},
                        "result": "oneshot",
                    }
                    for i, r in enumerate(results)
                    if isinstance(r, dict)
                ],
                "trace": [f"intent:{intent}", "oneshot_llm_plan"],
                "step_count": len(results),
                "tools_used": [r.get("tool", "") for r in results if isinstance(r, dict)],
                "pending_approval": None,
                "status": "oneshot_fallback",
            }

        loop = ReactLoop(
            self.llm,
            registry,
            max_steps=4,
            max_tool_calls=2,
            default_args={},
        )
        out = loop.run(
            message,
            session_history=history,
            approved=approved,
            on_parse_fail=_fallback,
        )
        out.setdefault("trace", [])
        if out["trace"] and not str(out["trace"][0]).startswith("intent:"):
            out["trace"] = [f"intent:{intent}"] + list(out["trace"])

        results = out.get("tool_results") or []
        show_chart = any(
            (isinstance(r, dict) and r.get("show_chart"))
            or (
                isinstance(r, dict)
                and r.get("tool")
                in ("spending_summary", "market_fraud_insights", "monthly_comparison")
            )
            for r in results
        )
        if any(k in message.lower() for k in self._CHART_KEYWORDS):
            show_chart = True

        reply = (out.get("reply") or "").strip()
        needs_synth = bool(results) and (
            not reply
            or out.get("status") in (
                "max_steps",
                "needs_synth",
                "fallback_tool_error",
                "oneshot_fallback",
            )
            or reply.startswith("Reached step limit")
            or reply.startswith("Analysis complete")
            or reply.startswith("Tool results ready")
            or reply.startswith("Tool budget reached")
            or out.get("used_fallback")
        )
        if needs_synth:
            if out.get("status") == "oneshot_fallback" and reply and not reply.startswith("Tool"):
                pass  # already synthesized in fallback
            else:
                synth = self._llm_synthesize(message, results, session_id=session_id)
                if synth and not synth.startswith("LLM"):
                    reply = synth
                else:
                    reply = self._compose_reply(message, results, session_id=session_id)
                if out.get("status") in ("max_steps", "needs_synth"):
                    out["status"] = "success_synth"
        elif not reply and results:
            reply = self._compose_reply(message, results, session_id=session_id)
        elif not reply and not results:
            results = [registry.call("spending_summary", {}, approved=True)]
            reply = self._llm_synthesize(message, results, session_id=session_id)
            if not reply or reply.startswith("LLM"):
                reply = self._compose_reply(message, results, session_id=session_id)
            out["tools_used"] = ["spending_summary"]
            out["trace"] = list(out.get("trace") or []) + ["forced_spending_summary"]

        if session_id:
            get_memory().add_message(session_id, "assistant", reply)

        log_trajectory(
            agent="financial_advisor",
            message=message,
            user_id=user_id,
            session_id=session_id,
            tools_used=out.get("tools_used"),
            step_count=out.get("step_count", 0),
            status=out.get("status", "success"),
            trace=out.get("trace"),
            reply_preview=reply,
        )

        return {
            "reply": reply,
            "tool_results": results,
            "user_id": user_id,
            "show_chart": show_chart,
            "trace": out.get("trace", []),
            "step_count": out.get("step_count", 0),
            "tools_used": out.get("tools_used", []),
            "actions": out.get("actions", []),
            "pending_approval": out.get("pending_approval"),
            "status": out.get("status", "success"),
            "used_fallback": out.get("used_fallback", False),
            "intent": intent,
        }

    def _compose_reply(self, question: str, tool_results: list[dict], session_id: str | None = None) -> str:
        """Compose a detailed, professional reply. Uses LLM when available, else templates."""
        if self.llm and hasattr(self.llm, "generate"):
            try:
                history = ""
                if session_id:
                    history = get_memory().get_history(session_id)
                
                context = "\n".join(str(r) for r in tool_results[:6])
                prompt = f"System: You are a professional Financial Advisor. Use tool data to answer specifically. Stay concise.\n\n"
                if history:
                    prompt += f"Conversation History:\n{history}\n\n"
                
                prompt += f"Recent Data:\n{context}\n\nNew Question: {question}\n\nAssistant:"
                
                out = self.llm.generate(prompt, max_tokens=400, temp=0.4)
                if out and out.strip():
                    return out.strip()
            except Exception:
                pass
        parts = []
        MAX_SECTIONS = 4  # Allow full detailed reports

        for r in tool_results:
            if len(parts) >= MAX_SECTIONS:
                break
            tool = r.get("tool", "")
            if "error" in r:
                parts.append(f"⚠️ {r['error']}")
                continue

            if tool == "spending_summary":
                cats = ", ".join(list(r['top_categories'].keys())[:4])
                parts.append(
                    f"**📊 Your Spending Overview**\n\n"
                    f"Total spend: **${r['total_spend']:,.2f}** | Monthly avg: **${r['avg_monthly_spend']:,.2f}**\n\n"
                    f"Archetype: *{r['archetype'].replace('_', ' ').title()}* | Top merchant: {r['top_merchant']}\n\n"
                    f"Top categories: {cats}"
                )

            elif tool == "monthly_comparison":
                chg = r.get("change_pct", 0)
                arrow = "↑" if chg > 0 else "↓"
                top_cats = ", ".join(f"{k} (${v:,.0f})" for k, v in list(r['top_categories_this_month'].items())[:3])
                parts.append(
                    f"**📈 Monthly Comparison** {'📉 DOWN' if chg < 0 else '📈 UP'}\n\n"
                    f"You spent **${r['current_spend']:,.2f}** in {r['current_month']} vs **${r['previous_spend']:,.2f}** in {r['previous_month']} ({arrow}{abs(chg):.1f}%).\n\n"
                    f"Top categories: {top_cats}"
                )

            elif tool == "find_cancellable_subscriptions":
                subs = r.get("recommended_cancellations", [])[:3]
                if subs:
                    sub_lines = "\n".join(f"  - **{s['merchant']}** — ${s['monthly_cost']:.2f}/mo (${s['annual_cost']:,.2f}/yr)" for s in subs)
                    parts.append(
                        f"**💡 Subscription Optimizer**\n\n"
                        f"Cancel these to save **${r['projected_monthly_savings']:.2f}/month**:\n\n{sub_lines}"
                    )

            elif tool == "credit_score_impact":
                score = r.get("estimated_credit_score", 0)
                badge = "🟢" if score >= 740 else ("🟡" if score >= 670 else "🔴")
                parts.append(
                    f"**💳 Credit Score Analysis** {badge} **{score}**\n\n"
                    f"{r.get('recommendation', '')}"
                )

            elif tool == "savings_plan":
                ops = r.get("opportunities", [])
                ops.sort(key=lambda x: x["potential_saving"], reverse=True)
                archetype = r.get("archetype", "unknown").replace("_", " ").title()
                monthly_avg = r.get("monthly_avg_spend", 0)
                monthly_save = r.get("potential_monthly_savings", 0)
                annual_save = r.get("potential_annual_savings", 0)

                # --- Header ---
                header = (
                    f"**💰 Comprehensive Savings Strategy**\n\n"
                    f"Based on your **{archetype}** spending profile, your average monthly expenditure "
                    f"is **${monthly_avg:,.2f}**. After analyzing every category in your transaction history, "
                    f"we have identified a realistic savings target of **${monthly_save:,.2f}/month** "
                    f"(**${annual_save:,.0f}/year**). Below is a detailed, category-by-category breakdown "
                    f"with professional recommendations.\n"
                )

                # --- High-Impact Categories (top 3) ---
                high_impact = ops[:3]
                high_lines = "\n".join(
                    f"  - **{o['category']}** — You spend **${o['monthly_spend']:,.2f}/mo**. "
                    f"Reduce by **${o['potential_saving']:.2f}/mo** by: *{o['tip']}*."
                    for o in high_impact
                )
                high_section = f"\n**🔴 High-Impact Categories (Biggest Savings)**\n\n{high_lines}\n" if high_impact else ""

                # --- Everyday Essentials ---
                everyday_keys = ["grocery", "groceries", "gas", "fuel", "utilities", "electric", "water", "healthcare", "medical"]
                everyday = [o for o in ops[3:] if any(k in o["category"].lower() for k in everyday_keys)]
                everyday_lines = "\n".join(
                    f"  - **{o['category']}** — ${o['monthly_spend']:,.2f}/mo → Save ${o['potential_saving']:.2f}/mo. *{o['tip']}*."
                    for o in everyday
                )
                everyday_sum = sum(o["potential_saving"] for o in everyday)
                everyday_section = (
                    f"\n**🟢 Everyday Essentials (Total: ${everyday_sum:.2f}/mo)**\n\n"
                    f"These are necessary expenses but still have room for optimization:\n\n{everyday_lines}\n"
                ) if everyday else ""

                # --- Subscriptions ---
                subs = [o for o in ops[3:] if "subscription" in o["category"].lower()]
                sub_sum = sum(o["potential_saving"] for o in subs)
                sub_lines = "\n".join(
                    f"  - **{o['category']}** — ${o['monthly_spend']:,.2f}/mo → Save ${o['potential_saving']:.2f}/mo. *{o['tip']}*."
                    for o in subs
                )
                sub_section = (
                    f"\n**📦 Subscriptions & Recurring (Total: ${sub_sum:.2f}/mo)**\n\n"
                    f"Recurring charges are often overlooked. Audit each one:\n\n{sub_lines}\n"
                ) if subs else ""

                # --- Discretionary / Other ---
                others = [o for o in ops[3:] if o not in everyday and o not in subs]
                other_sum = sum(o["potential_saving"] for o in others)
                other_lines = "\n".join(
                    f"  - **{o['category']}** — ${o['monthly_spend']:,.2f}/mo → Save ${o['potential_saving']:.2f}/mo. *{o['tip']}*."
                    for o in others
                )
                other_section = (
                    f"\n**🟡 Discretionary & Lifestyle (Total: ${other_sum:.2f}/mo)**\n\n"
                    f"Non-essential spending where budgeting discipline pays off:\n\n{other_lines}\n"
                ) if others else ""

                # --- Action Plan ---
                action = (
                    f"\n**✅ Action Plan**\n\n"
                    f"1. **Immediate:** Focus on **{high_impact[0]['category']}** and **{high_impact[1]['category'] if len(high_impact) > 1 else 'subscriptions'}** — these two alone account for the largest portion of your potential savings.\n"
                    f"2. **This Month:** Audit all subscriptions and cancel any you have not used in the last 30 days.\n"
                    f"3. **Ongoing:** Track spending weekly using a budgeting app. Set category-level spending caps based on the targets above.\n"
                    f"4. **Goal:** Redirect the saved **${monthly_save:,.2f}/mo** into an emergency fund or high-yield savings account to build **${annual_save:,.0f}** in annual wealth.\n"
                )

                parts.append(header + high_section + everyday_section + sub_section + other_section + action)

            elif tool == "category_advice":
                tips = "\n".join(f"  - {t}" for t in r["tips"][:3])
                risk_badge = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}.get(r["risk_level"], "⚪")
                parts.append(
                    f"**{r['label']}** {risk_badge}\n\n"
                    f"Your spend: **${r['user_spend']:,.2f}** | Annual waste: *{r['annual_waste_estimate']}*\n\n"
                    f"Tips:\n{tips}"
                )

            elif tool == "realtime_fraud_check":
                parts.append(
                    f"**🔍 Fraud Scan**\n\n"
                    f"{r['overall_status']}\n\n"
                    f"Scanned: {r['transactions_scanned']} transactions | Alerts: {r['alerts_found']}"
                )

            elif tool == "suspicious_activity_monitor":
                parts.append(f"**👁️ Activity Monitor**\n\n{r['overall_status']}")

            elif tool == "cash_flow_forecast":
                parts.append(
                    f"**📉 Cash Flow Forecast (Next 30 Days)**\n\n"
                    f"Expected Total: **${r['forecast_amount']:,.2f}**\n"
                    f"- Fixed Bills/Subs: ${r['subs_amount']:,.2f}\n"
                    f"- Variable Spend: ${r['variable_amount']:,.2f}"
                )

            elif tool == "detect_price_hikes":
                if r['alerts']:
                    hikes = "\n".join(f"  - **{a['merchant']}**: ${a['old_price']:.2f} ➡️ ${a['new_price']:.2f} (+{a['increase_pct']}%)" for a in r['alerts'])
                    parts.append(f"**🚨 Price Hike Alerts**\n\nThe following subscriptions recently increased in price:\n\n{hikes}")
                else:
                    parts.append("**✅ No Price Hikes Detected**\n\nAll of your recurring subscriptions have remained stable.")


            elif tool == "tax_deductible_finder":
                if r['breakdown']:
                    cats = "\n".join(f"  - **{k}**: ${v:.2f}" for k, v in r['breakdown'].items())
                    parts.append(f"**🧾 Tax-Deductible Finder**\n\nYou have **${r['total_potential_deduction']:,.2f}** in potential write-offs:\n\n{cats}")
                else:
                    parts.append("**🧾 No Deductibles**\n\nWe did not detect any obvious tax-deductible expenses.")

            elif tool == "surplus_optimizer":
                sugs = "\n".join(f"  - {s}" for s in r['suggestions'])
                parts.append(
                    f"**📈 Surplus Optimizer**\n\n"
                    f"Estimated Income: ${r['assumed_monthly_income']:,.2f}\n"
                    f"Avg monthly spend: ${r['avg_monthly_spend']:,.2f}\n"
                    f"Monthly Surplus: **${r['monthly_surplus']:,.2f}**\n\n"
                    f"**Optimization Strategy:**\n{sugs}"
                )

            elif tool == "liquidity_guard":
                badge = "🔴" if r["liquidity_risk"] == "HIGH" else "🟢"
                parts.append(
                    f"**💧 Liquidity Guard** {badge}\n\n"
                    f"Assumed Balance: ${r['assumed_balance']:,.2f}\n"
                    f"Upcoming Bills (7 Days): **${r['upcoming_7d_bills']:,.2f}**\n"
                    f"Risk Status: **{r['liquidity_risk']}**"
                )

            elif tool == "market_fraud_insights":
                trends = "\n".join(f"  - **{k}**: ${v}B" for k, v in list(r['global_scam_trends'].items())[:4])
                local = "\n".join(f"  - **{it['category']}**: {it['risk_density_pct']}% risk density" for it in r['local_cat_fraud_heatmap'][:4])
                parts.append(
                    f"**📊 Market Fraud Intelligence**\n\n"
                    f"**Global Scam Losses (FBI/IC3):**\n{trends}\n\n"
                    f"**Local Data Risk Density (Heatmap):**\n{local}\n\n"
                    f"💡 *Summary:* {r['summary']}"
                )

        return "\n\n---\n\n".join(parts) if parts else "No insights found for your query. Try asking about spending, savings, or fraud detection."


    def get_all_users(self) -> list[str]:
        return sorted(self.df["user_id"].unique().tolist())

    def get_user_names_map(self) -> dict[str, str]:
        """Return a mapping of user_id -> Full Name."""
        mapping = self.df.drop_duplicates("user_id")[["user_id", "first", "last"]]
        return {row["user_id"]: f"{row['first']} {row['last']}" for _, row in mapping.iterrows()}

    def get_chart_data(self, user_id: str) -> dict[str, float]:
        """Return category → amount dict for bar chart rendering."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {}
        return (
            user_df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .round(2)
            .to_dict()
        )
