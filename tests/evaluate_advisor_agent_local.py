"""
Veriscan — Financial Advisor trajectory / tool-routing evaluation.

Runs without MLX by default (keyword router + tool registry).
Golden set checks that expected tools appear in tool_results / tools_used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path as _Path
from typing import Any

_ROOT = _Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

# Force keyword path so eval is offline-friendly
os.environ.setdefault("ENABLE_AGENTIC_ADVISOR", "0")
os.environ.setdefault("ENABLE_MULTI_AGENT_ADVISOR", "0")

from agents.financial_advisor_agent import FinancialAdvisorAgent
from agents.react_loop import extract_json
from agents.tool_registry import ToolRegistry, REQUIRE_HITL


GOLDEN = [
    {
        "query": "Am I spending more this month than last?",
        "expected_tools": ["monthly_comparison"],
        "user_id": "USER_0001",
    },
    {
        "query": "Show my spending summary overview",
        "expected_tools": ["spending_summary"],
        "user_id": "USER_0001",
    },
    {
        "query": "Help me save money and cut spending",
        "expected_tools": ["savings_plan"],
        "user_id": "USER_0001",
    },
    {
        "query": "Check for fraud and suspicious activity on my account",
        "expected_tools": ["realtime_fraud_check", "suspicious_activity_monitor"],
        "user_id": "USER_0001",
    },
    {
        "query": "How can I cut dining and restaurant spending?",
        "expected_tools": ["category_advice"],
        "user_id": "USER_0001",
    },
    {
        "query": "What subscriptions can I cancel to save $50?",
        "expected_tools": ["find_cancellable_subscriptions"],
        "user_id": "USER_0001",
    },
    {
        "query": "Forecast my cash flow for next month",
        "expected_tools": ["cash_flow_forecast"],
        "user_id": "USER_0001",
    },
    {
        "query": "Any subscription price hikes?",
        "expected_tools": ["detect_price_hikes"],
        "user_id": "USER_0001",
    },
    {
        "query": "Find tax deductible expenses",
        "expected_tools": ["tax_deductible_finder"],
        "user_id": "USER_0001",
    },
    {
        "query": "Show market fraud heatmap and global scam trends",
        "expected_tools": ["market_fraud_insights"],
        "user_id": "USER_0001",
    },
    {
        "query": "Am I at fraud risk this month and how can I cut dining?",
        "expected_tools": ["realtime_fraud_check", "category_advice"],
        "user_id": "USER_0001",
        "any_of": False,
    },
    {
        "query": "Check my credit score impact",
        "expected_tools": ["credit_score_impact"],
        "user_id": "USER_0001",
    },
    {
        "query": "Liquidity guard — upcoming bills and balance",
        "expected_tools": ["liquidity_guard"],
        "user_id": "USER_0001",
    },
    {
        "query": "Optimize my surplus income",
        "expected_tools": ["surplus_optimizer"],
        "user_id": "USER_0001",
    },
    {
        "query": "Give me coffee spending tips",
        "expected_tools": ["category_advice"],
        "user_id": "USER_0001",
    },
]


def _tools_from_result(result: dict[str, Any]) -> set[str]:
    names = set(result.get("tools_used") or [])
    for r in result.get("tool_results") or []:
        if isinstance(r, dict) and r.get("tool"):
            names.add(r["tool"])
    return {n for n in names if n}


def evaluate_advisor(user_id: str | None = None) -> float:
    print("Initializing FinancialAdvisorAgent (keyword / registry eval)...")
    agent = FinancialAdvisorAgent(llm=None)

    # Registry smoke test
    uid = user_id or "USER_0001"
    reg = agent.build_tool_registry(uid)
    assert len(reg) >= 10, "Expected advisor tools registered"
    assert "challenge_auth" in REQUIRE_HITL
    assert extract_json('{"action":"final","answer":"ok"}')["action"] == "final"
    assert extract_json("Here is JSON:\n```json\n{\"action\":\"tool\",\"tool\":\"x\",\"args\":{}}\n```")["tool"] == "x"

    # HITL block smoke
    hitl_reg = ToolRegistry()
    hitl_reg.register(
        "challenge_auth",
        "test",
        lambda: {"ok": True},
        requires_hitl=True,
    )
    blocked = hitl_reg.call("challenge_auth", {}, approved=False)
    assert blocked.get("pending_approval"), "HITL should block without approval"
    allowed = hitl_reg.call("challenge_auth", {}, approved=True)
    assert allowed.get("ok") is True

    success = 0
    print(f"\n{'='*60}")
    print(f" Advisor Trajectory Eval — {len(GOLDEN)} Golden Queries")
    print(f"{'='*60}")

    for i, case in enumerate(GOLDEN, 1):
        q = case["query"]
        expected = case["expected_tools"]
        uid = case.get("user_id", uid)
        result = agent.chat(q, uid)
        got = _tools_from_result(result)
        require_all = not case.get("any_of", False)
        if require_all:
            ok = all(t in got for t in expected)
        else:
            ok = any(t in got for t in expected)

        status = "✅" if ok else "❌"
        print(f"[{i}/{len(GOLDEN)}] {status} {q[:60]}")
        if not ok:
            print(f"     expected ⊆ {expected}, got={sorted(got)}")
        else:
            success += 1

    accuracy = success / len(GOLDEN)
    print(f"\n{'='*60}")
    print(f" Accuracy: {success}/{len(GOLDEN)} = {accuracy*100:.0f}%")
    print(f"{'='*60}")
    return accuracy


if __name__ == "__main__":
    evaluate_advisor()
