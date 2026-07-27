"""
Veriscan — Multi-Agent GuardAgent with tool-calling ReAct loop.

Tools:
  - get_user_risk_profile
  - get_high_risk_transactions
  - query_rag
  - challenge_auth (HITL — requires approved=True)
"""

from __future__ import annotations

import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from models.rag_engine_local import RAGEngineLocal
from models.agent_tools_data import (
    tool_get_user_risk_profile,
    tool_get_high_risk_transactions,
)
from agents.tool_registry import ToolRegistry
from agents.react_loop import ReactLoop
from agents.trajectory_log import log_trajectory
from agents.memory import get_memory

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_rag_engine = None


def _get_rag():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngineLocal()
        _rag_engine.index_data()
    return _rag_engine


def _get_local_llm():
    from models.local_llm import LocalLLM
    return LocalLLM()


def tool_query_rag(query: str, n_results: int = 4) -> Dict[str, Any]:
    """Retrieve knowledge-base context for a security / compliance question."""
    rag = _get_rag()
    try:
        if hasattr(rag, "query"):
            hits = rag.query(query, n_results=n_results)
            return {"tool": "query_rag", "query": query, "results": hits}
        context = rag.get_context_for_query(query)
        return {"tool": "query_rag", "query": query, "context": context}
    except Exception as e:
        return {"tool": "query_rag", "error": str(e)}


def tool_challenge_auth(user_id: str, reason: str = "high_risk") -> Dict[str, Any]:
    """
    High-impact step-up auth challenge. HITL-gated — must be approved by a human.
    """
    return {
        "tool": "challenge_auth",
        "user_id": user_id,
        "reason": reason,
        "status": "challenge_issued",
        "actions": [
            "Require MFA / step-up authentication",
            "Temporarily limit high-risk transaction types",
            "Notify account owner of suspicious activity",
            "Log event for fraud operations review",
        ],
        "message": f"Challenge Auth issued for {user_id} ({reason}).",
    }


class LocalGuardAgent:
    """Security analyst agent with bounded tool-calling loop."""

    def __init__(self, llm=None, load_llm: bool = True):
        if llm is not None:
            self.llm = llm
        elif load_llm:
            self.llm = _get_local_llm()
        else:
            self.llm = None
        self._rag = None

    def build_tool_registry(self, default_user_id: Optional[str] = None) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(
            "get_user_risk_profile",
            "Load risk / auth profile for a user_id (e.g. USER_123).",
            lambda user_id="USER_0": tool_get_user_risk_profile(str(user_id)),
            {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User id like USER_123"},
                },
            },
        )
        reg.register(
            "get_high_risk_transactions",
            "List top high-risk transactions system-wide.",
            lambda limit=10: tool_get_high_risk_transactions(limit=int(limit)),
            {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            },
        )
        reg.register(
            "query_rag",
            "Search CFPB / fraud knowledge base for policy and definitions.",
            lambda query="", n_results=4: tool_query_rag(str(query) or "fraud", int(n_results)),
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "n_results": {"type": "integer"},
                },
            },
        )
        uid = default_user_id or "USER_0"
        reg.register(
            "challenge_auth",
            "Issue step-up Challenge Auth for a risky user. REQUIRES human approval.",
            lambda user_id=uid, reason="high_risk": tool_challenge_auth(str(user_id), str(reason)),
            {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
            requires_hitl=True,
        )
        return reg

    def _keyword_tools(self, question: str) -> list[dict]:
        """Offline / fallback tool selection for eval and LLM failures."""
        q = question.lower()
        results = []
        m = re.search(r"(user[_\s-]?\d+)", q, re.I)
        user_id = None
        if m:
            user_id = m.group(1).upper().replace(" ", "_").replace("-", "_")
            if not user_id.startswith("USER"):
                user_id = "USER_" + re.sub(r"\D", "", user_id)

        if any(k in q for k in ["risk profile", "investigate", "analyze risk", "high risk score", "fraud"]):
            if user_id or "user" in q:
                profile = tool_get_user_risk_profile(user_id or "USER_0")
                if isinstance(profile, dict):
                    profile = {**profile, "tool": "get_user_risk_profile"}
                results.append(profile)

        if any(k in q for k in ["high risk transaction", "dangerous", "top high risk", "most dangerous"]):
            items = tool_get_high_risk_transactions(limit=10)
            results.append({"tool": "get_high_risk_transactions", "items": items})

        if any(
            k in q
            for k in [
                "cfpb",
                "dispute",
                "identity theft",
                "velocity",
                "explain",
                "what is",
                "what are",
                "how to",
                "trends",
                "protection",
            ]
        ):
            # Avoid loading embedding models during offline keyword eval
            if os.environ.get("VERISCAN_EVAL_KEYWORD_ONLY", "").strip() == "1":
                results.append(
                    {
                        "tool": "query_rag",
                        "query": question,
                        "context": "[eval stub] knowledge retrieval skipped",
                    }
                )
            else:
                results.append(tool_query_rag(question))

        if not results:
            # Default: try user profile if USER mentioned else RAG
            if user_id:
                r = tool_get_user_risk_profile(user_id)
                if isinstance(r, dict):
                    r = {**r, "tool": "get_user_risk_profile"}
                results.append(r)
            elif os.environ.get("VERISCAN_EVAL_KEYWORD_ONLY", "").strip() == "1":
                results.append(
                    {"tool": "query_rag", "query": question, "context": "[eval stub]"}
                )
            else:
                results.append(tool_query_rag(question))
        return results

    def analyze(
        self,
        question: str,
        session_id: Optional[str] = None,
        approved: bool = False,
        use_agentic: bool = True,
    ) -> Dict[str, Any]:
        """
        Tool-calling analysis loop. Falls back to keyword tools + LLM summary.
        """
        print(f"⚡ GuardAgent ReAct (Session: {session_id}, approved={approved})")
        if session_id:
            get_memory().add_message(session_id, "user", question)

        try:
            registry = self.build_tool_registry()

            def _fallback() -> dict[str, Any]:
                tool_results = self._keyword_tools(question)
                actions = [
                    {
                        "step": i + 1,
                        "tool": r.get("tool", "unknown"),
                        "args": {},
                        "result": str(r)[:400],
                    }
                    for i, r in enumerate(tool_results)
                ]
                context = "\n".join(str(r) for r in tool_results[:4])
                prompt = (
                    "You are an AI Security Analyst. Use the tool data to answer concisely.\n\n"
                    f"Tool data:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                )
                try:
                    answer = self.llm.generate(prompt, max_tokens=180, temp=0.2)
                except Exception:
                    answer = context[:800]
                return {
                    "reply": answer,
                    "tool_results": tool_results,
                    "actions": actions,
                    "trace": ["keyword_fallback"],
                    "step_count": len(actions),
                    "tools_used": [a["tool"] for a in actions],
                    "pending_approval": None,
                    "status": "fallback",
                }

            if use_agentic and self.llm and hasattr(self.llm, "generate"):
                loop = ReactLoop(self.llm, registry, max_steps=5)
                out = loop.run(
                    question,
                    session_history=get_memory().get_history(session_id) if session_id else "",
                    approved=approved,
                    on_parse_fail=_fallback,
                )
                # If final answer empty but tools ran, compose
                reply = out.get("reply") or ""
                if not reply.strip() and out.get("tool_results"):
                    fb = _fallback()
                    reply = fb["reply"]
                    if not out.get("actions"):
                        out["actions"] = fb["actions"]
                        out["tools_used"] = fb["tools_used"]
            else:
                out = _fallback()
                reply = out.get("reply", "")

            if session_id:
                get_memory().add_message(session_id, "assistant", reply)

            log_trajectory(
                agent="guard_agent",
                message=question,
                session_id=session_id,
                tools_used=out.get("tools_used"),
                step_count=out.get("step_count", 0),
                status=out.get("status", "success"),
                trace=out.get("trace"),
                reply_preview=reply,
            )

            return {
                "answer": reply,
                "reply": reply,
                "actions": out.get("actions", []),
                "status": out.get("status", "completed"),
                "session_id": session_id,
                "trace": out.get("trace", []),
                "tool_results": out.get("tool_results", []),
                "tools_used": out.get("tools_used", []),
                "step_count": out.get("step_count", 0),
                "pending_approval": out.get("pending_approval"),
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "answer": f"Error in GuardAgent: {e}",
                "reply": f"Error in GuardAgent: {e}",
                "actions": [],
                "status": "error",
                "session_id": session_id,
                "trace": [f"error:{e}"],
            }


if __name__ == "__main__":
    agent = LocalGuardAgent()
    print(agent.analyze("Investigate USER_1")["answer"])
