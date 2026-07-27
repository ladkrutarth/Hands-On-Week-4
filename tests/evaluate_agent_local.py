"""
Veriscan — GuardAgent Evaluation Suite
Measures tool-selection accuracy across User, Knowledge, and System query types.

Set VERISCAN_EVAL_KEYWORD_ONLY=1 to skip the LLM ReAct loop and score the
keyword/tool fallback path (fast, no GPU).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from models.guard_agent_local import LocalGuardAgent


def evaluate_agent():
    keyword_only = os.environ.get("VERISCAN_EVAL_KEYWORD_ONLY", "").strip() == "1"
    print("Initializing GuardAgent...")
    if keyword_only:
        agent = LocalGuardAgent(load_llm=False)
        print("Mode: keyword/tool fallback only (VERISCAN_EVAL_KEYWORD_ONLY=1)")
    else:
        agent = LocalGuardAgent()
        print("Mode: agentic ReAct (falls back to keyword on parse failure)")

    scenarios = [
        # ── User Queries (expect: get_user_risk_profile) ──
        {"query": "Investigate USER_123 for potential fraud.",
         "expected_tool": "get_user_risk_profile", "category": "User"},
        {"query": "Does USER_456 have a high risk score?",
         "expected_tool": "get_user_risk_profile", "category": "User"},
        {"query": "Show risk profile for USER_0",
         "expected_tool": "get_user_risk_profile", "category": "User"},
        {"query": "Analyze risk for USER_789",
         "expected_tool": "get_user_risk_profile", "category": "User"},

        # ── Knowledge Queries (expect: query_rag) ──
        {"query": "What are the latest CFPB trends for credit card disputes?",
         "expected_tool": "query_rag", "category": "Knowledge"},
        {"query": "Explain what 1h velocity means in fraud detection.",
         "expected_tool": "query_rag", "category": "Knowledge"},
        {"query": "How to dispute a charge?",
         "expected_tool": "query_rag", "category": "Knowledge"},
        {"query": "What is identity theft protection?",
         "expected_tool": "query_rag", "category": "Knowledge"},

        # ── System Queries (expect: get_high_risk_transactions) ──
        {"query": "Show the top high risk transactions in the system.",
         "expected_tool": "get_high_risk_transactions", "category": "System"},
        {"query": "What are the most dangerous transactions?",
         "expected_tool": "get_high_risk_transactions", "category": "System"},

        # ── HITL (expect pending without approval) ──
        {"query": "Issue challenge auth for USER_1 due to takeover risk",
         "expected_tool": "challenge_auth", "category": "HITL",
         "expect_pending": True},
    ]

    success_count = 0
    category_results = {"User": [0, 0], "Knowledge": [0, 0], "System": [0, 0], "HITL": [0, 0]}

    print(f"\n{'='*60}")
    print(f" GuardAgent Evaluation Suite — {len(scenarios)} Scenarios")
    print(f"{'='*60}")

    for i, scenario in enumerate(scenarios):
        query = scenario["query"]
        expected = scenario["expected_tool"]
        cat = scenario["category"]
        print(f"\n[{i+1}/{len(scenarios)}] ({cat}) '{query}'")

        if keyword_only:
            if cat == "HITL":
                # Direct registry HITL check
                reg = agent.build_tool_registry()
                blocked = reg.call("challenge_auth", {"user_id": "USER_1"}, approved=False)
                found_tool = bool(blocked.get("pending_approval"))
                actions = [{"tool": "challenge_auth"}] if found_tool else []
                result = {"actions": actions, "pending_approval": blocked}
            else:
                tool_results = agent._keyword_tools(query)
                actions = [
                    {"tool": r.get("tool", "unknown")}
                    for r in tool_results
                ]
                result = {"actions": actions}
        else:
            result = agent.analyze(
                query,
                approved=False,
                use_agentic=True,
            )

        actions = result.get("actions", [])
        found_tool = any(
            (a.get("tool") if isinstance(a, dict) else getattr(a, "tool", None)) == expected
            for a in actions
        )

        # HITL: also accept pending_approval for challenge_auth
        if scenario.get("expect_pending") and result.get("pending_approval"):
            pending_tool = (result.get("pending_approval") or {}).get("tool")
            if pending_tool == expected:
                found_tool = True

        if cat not in category_results:
            category_results[cat] = [0, 0]
        category_results[cat][1] += 1

        if found_tool:
            print(f"  ✅ Correct Tool Called: {expected}")
            success_count += 1
            category_results[cat][0] += 1
        else:
            print(f"  ❌ Expected '{expected}' — not called.")
            if actions:
                tools = [
                    a.get("tool") if isinstance(a, dict) else getattr(a, "tool", None)
                    for a in actions
                ]
                print(f"     Called instead: {tools}")
            else:
                print("     No tools were called.")

    accuracy = success_count / len(scenarios)
    print(f"\n{'='*60}")
    print(" FINAL RESULTS")
    print(f"{'='*60}")
    print(f" Overall Accuracy: {success_count}/{len(scenarios)} = {accuracy*100:.0f}%")
    print("")
    for cat, (correct, total) in category_results.items():
        pct = (correct / total * 100) if total > 0 else 0
        print(f"  {cat:12s}: {correct}/{total} ({pct:.0f}%)")
    print(f"{'='*60}")

    return accuracy


if __name__ == "__main__":
    evaluate_agent()
