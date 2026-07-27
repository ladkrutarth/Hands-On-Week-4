"""
Financial Orchestrator — routes user questions to specialist agents and synthesizes
one coherent reply. Supports keyword routing (offline) and LLM supervisor mode.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.current_transaction_analyst import CurrentTransactionAnalyst
from agents.historical_review_agent import HistoricalReviewAgent
from agents.memory import get_memory
from agents.tool_registry import ToolRegistry, truncate_observation
from agents.trajectory_log import log_trajectory
from agents.transaction_calculation_agent import TransactionCalculationAgent


CURRENT_KEYWORDS = [
    "current", "this month", "recent", "last 30 days", "last 60", "last 90",
    "latest transactions", "right now", "latest", "this month's", "recent activity",
]
CALCULATION_KEYWORDS = [
    "total", "average", "how much", "calculate", "forecast", "compare months",
    "breakdown", "percentage", "growth", "subscription", "subscriptions",
    "month over month", "mom", "spending by category",
]
HISTORICAL_KEYWORDS = [
    "last 2 years", "past year", "year over year", "yoy", "history", "historical",
    "bank statement", "credit card history", "long term", "trend over time",
    "last year vs", "two years", "24 months", "statement",
]

MAX_SUPERVISOR_DELEGATIONS = 3

SUPERVISOR_PROMPT = """You are a Veriscan financial supervisor. Reply with ONE JSON object only.

Workers (tools):
{tools}

Output either:
{{"action":"tool","tool":"<worker_name>","args":{{...}}}}
or
{{"action":"final","answer":"<coherent reply using worker results>"}}

Rules: max {max_delegations} worker calls. Prefer condensed facts. Do not invent numbers.
user_id is already bound — omit it from args.
"""


class FinancialOrchestrator:
    """Routes to current analyst, calculation agent, and/or historical reviewer; synthesizes reply."""

    def __init__(self, llm=None):
        self.llm = llm
        self.current_analyst = CurrentTransactionAnalyst()
        self.calc_agent = TransactionCalculationAgent()
        self.historical_agent = HistoricalReviewAgent()

    def _route(self, message: str) -> list[str]:
        """Return list of agent keys to call: current, calculation, historical."""
        msg = message.lower()
        agents = []
        if any(k in msg for k in CURRENT_KEYWORDS):
            agents.append("current")
        if any(k in msg for k in CALCULATION_KEYWORDS):
            agents.append("calculation")
        if any(k in msg for k in HISTORICAL_KEYWORDS):
            agents.append("historical")
        if not agents:
            agents.append("current")
        return agents

    def _parse_period(self, message: str) -> str:
        msg = message.lower()
        if "last 60" in msg or "60 days" in msg:
            return "last_60"
        if "last 90" in msg or "90 days" in msg:
            return "last_90"
        return "last_30"

    def _parse_calculation(self, message: str, instruction: str = "") -> str:
        blob = f"{message} {instruction}".lower()
        if "forecast" in blob or "next month" in blob:
            return "forecast"
        if "subscription" in blob:
            return "subscriptions"
        if "month over month" in blob or "mom" in blob or "compare months" in blob:
            return "mom_change"
        if "average" in blob:
            return "average_by_category"
        return "summary"

    def _run_agents(self, user_id: str, message: str, agent_keys: list[str]) -> list[dict]:
        """Execute chosen agents and collect structured outputs."""
        results = []
        if "current" in agent_keys:
            out = self.current_analyst.run(user_id, period=self._parse_period(message))
            results.append({"agent": "current_analyst", "data": out})
        if "calculation" in agent_keys:
            out = self.calc_agent.run(user_id, calculation=self._parse_calculation(message))
            results.append({"agent": "calculation", "data": out})
        if "historical" in agent_keys:
            out = self.historical_agent.run(user_id)
            results.append({"agent": "historical", "data": out})
        return results

    def build_worker_registry(self, user_id: str, message: str) -> ToolRegistry:
        """Register specialist workers as tools for the LLM supervisor."""
        reg = ToolRegistry()

        def run_current(period: str = "last_30", instruction: str = "") -> dict:
            p = period if period in ("last_30", "last_60", "last_90") else self._parse_period(
                f"{message} {instruction} {period}"
            )
            return self.current_analyst.run(user_id, period=p)

        def run_calculation(calculation: str = "summary", instruction: str = "") -> dict:
            calc = calculation if calculation in (
                "summary", "total", "average_by_category", "mom_change", "forecast", "subscriptions"
            ) else self._parse_calculation(message, instruction)
            return self.calc_agent.run(user_id, calculation=calc)

        def run_historical(instruction: str = "") -> dict:
            return self.historical_agent.run(user_id)

        reg.register(
            "run_current_analyst",
            "Recent / current-month activity, last N days, fraud flags.",
            run_current,
            {
                "type": "object",
                "properties": {
                    "period": {"type": "string", "description": "last_30|last_60|last_90"},
                    "instruction": {"type": "string"},
                },
            },
        )
        reg.register(
            "run_calculation",
            "Totals, MoM change, forecasts, subscription totals.",
            run_calculation,
            {
                "type": "object",
                "properties": {
                    "calculation": {
                        "type": "string",
                        "description": "summary|total|average_by_category|mom_change|forecast|subscriptions",
                    },
                    "instruction": {"type": "string"},
                },
            },
        )
        reg.register(
            "run_historical",
            "24-month history: YoY, yearly totals, category evolution.",
            run_historical,
            {
                "type": "object",
                "properties": {"instruction": {"type": "string"}},
            },
        )
        return reg

    def _synthesize(self, message: str, agent_results: list[dict]) -> str:
        """Turn structured agent results into one coherent reply (template-based)."""
        parts = []
        for item in agent_results:
            agent_name = item.get("agent", "")
            data = item.get("data", item)
            if isinstance(data, dict) and data.get("error"):
                parts.append(f"⚠️ {data['error']}")
                continue

            if agent_name == "current_analyst":
                cm = data.get("current_month") or {}
                if isinstance(cm, dict) and "error" not in cm and "current_month_total" in cm:
                    parts.append(
                        f"**Current month** ({cm.get('current_month', 'N/A')}): "
                        f"**${cm['current_month_total']:,.2f}** across {cm.get('transaction_count', 0)} transactions. "
                        f"Top categories: {', '.join(list(cm.get('by_category', {}).keys())[:5])}."
                    )
                ln = data.get("last_n_days") or {}
                if isinstance(ln, dict) and "error" not in ln and "total" in ln:
                    parts.append(
                        f"**Last {ln.get('days', 30)} days:** **${ln['total']:,.2f}** total. "
                        f"By category: {ln.get('by_category', {})}."
                    )
                fr = data.get("recent_fraud_risk") or {}
                if isinstance(fr, dict) and fr.get("alerts_count", 0) > 0:
                    parts.append(
                        f"**Fraud/risk:** {fr['alerts_count']} alert(s) in recent transactions."
                    )

            elif agent_name == "calculation":
                if "total" in data:
                    parts.append(f"**Total spend (range):** **${data['total']:,.2f}**.")
                if "growth_pct" in data:
                    parts.append(
                        f"**Month-over-month:** {data.get('growth_pct', 0):.1f}% change "
                        f"({data.get('previous_month')} → {data.get('current_month')})."
                    )
                if "forecast_total" in data:
                    parts.append(
                        f"**Forecast (next month):** ~**${data['forecast_total']:,.2f}** "
                        f"(subscriptions ~${data.get('subscription_component', 0):,.2f}, variable ~${data.get('variable_component', 0):,.2f})."
                    )
                if "monthly_total" in data and data.get("tool") == "subscription_totals":
                    parts.append(
                        f"**Subscriptions:** **${data['monthly_total']:,.2f}/month** by merchant: {data.get('by_merchant', {})}."
                    )

            elif agent_name == "historical":
                yt_block = data.get("yearly_totals") or {}
                if isinstance(yt_block, dict) and "yearly_totals" in yt_block and "error" not in yt_block:
                    yt = yt_block.get("yearly_totals", {})
                    parts.append(f"**Yearly totals (last 2 years):** {yt}.")
                yoy_block = data.get("yoy_change") or {}
                if isinstance(yoy_block, dict) and "yoy_change_pct" in yoy_block:
                    yoy = yoy_block
                    parts.append(
                        f"**Year-over-year:** {yoy.get('year_previous')} → {yoy.get('year_latest')}: "
                        f"**{yoy['yoy_change_pct']:.1f}%** change "
                        f"(${yoy.get('total_previous', 0):,.2f} → ${yoy.get('total_latest', 0):,.2f})."
                    )
                evo_block = data.get("category_evolution") or {}
                if isinstance(evo_block, dict) and "monthly_avg_by_category" in evo_block:
                    evo = evo_block.get("monthly_avg_by_category", {})
                    top = dict(list(evo.items())[:5])
                    parts.append(f"**Historical monthly average by category (top 5):** {top}.")

        if not parts:
            return (
                "I've run the requested analysis but couldn't summarize results. "
                "Please try a more specific question (e.g. 'current month spending', "
                "'total last 30 days', 'year over year comparison')."
            )
        return "\n\n".join(parts)

    def _chat_keyword(self, message: str, user_id: str) -> dict[str, Any]:
        agent_keys = self._route(message)
        agent_results = self._run_agents(user_id, message, agent_keys)
        tool_results = []
        for ar in agent_results:
            data = ar.get("data", ar)
            if isinstance(data, dict):
                tool_results.append(data)
            else:
                tool_results.append({"agent": ar.get("agent"), "result": data})
        reply = self._synthesize(message, agent_results)
        show_chart = any(
            k in message.lower()
            for k in ["chart", "graph", "breakdown", "category", "trend", "overview", "summary"]
        )
        return {
            "reply": reply,
            "tool_results": tool_results,
            "user_id": user_id,
            "show_chart": show_chart,
            "trace": [f"keyword_route:{','.join(agent_keys)}"],
            "step_count": len(agent_keys),
            "tools_used": agent_keys,
            "actions": [
                {"step": i + 1, "tool": k, "args": {"user_id": user_id}, "result": "ok"}
                for i, k in enumerate(agent_keys)
            ],
            "pending_approval": None,
            "status": "success",
        }

    def _chat_supervisor(
        self,
        message: str,
        user_id: str,
        session_id: Optional[str] = None,
        approved: bool = False,
    ) -> dict[str, Any]:
        """LLM supervisor: delegate to workers, then synthesize."""
        from agents.react_loop import ReactLoop, extract_json

        registry = self.build_worker_registry(user_id, message)
        history = get_memory().get_history(session_id) if session_id else ""
        observations: list[str] = []
        tool_results: list[dict] = []
        actions: list[dict] = []
        tools_used: list[str] = []
        trace: list[str] = ["supervisor"]

        system = SUPERVISOR_PROMPT.format(
            tools=registry.schemas_for_prompt(),
            max_delegations=MAX_SUPERVISOR_DELEGATIONS,
        )

        for step in range(1, MAX_SUPERVISOR_DELEGATIONS + 2):  # +1 for final
            prompt_parts = [system, ""]
            if history:
                prompt_parts.append(f"Conversation history:\n{history}\n")
            prompt_parts.append(f"User question: {message}\n")
            if observations:
                prompt_parts.append("Worker results so far:")
                for i, obs in enumerate(observations, 1):
                    prompt_parts.append(f"{i}. {obs}")
                prompt_parts.append("")
            if len(tools_used) >= MAX_SUPERVISOR_DELEGATIONS or (
                observations and step > MAX_SUPERVISOR_DELEGATIONS
            ):
                prompt_parts.append(
                    'CRITICAL: Enough worker results. Reply ONLY '
                    '{"action":"final","answer":"<coherent reply grounded in worker results>"}.'
                )
            prompt_parts.append("Respond with JSON only:")
            prompt = "\n".join(prompt_parts)

            try:
                raw = self.llm.generate(prompt, max_tokens=320, temp=0.0)
            except Exception as e:
                trace.append(f"llm_error:{e}")
                fb = self._chat_keyword(message, user_id)
                fb["used_fallback"] = True
                fb["trace"] = trace + fb.get("trace", [])
                fb["status"] = "fallback_llm_error"
                return fb

            decision = extract_json(raw)
            if decision is None:
                trace.append(f"step{step}:parse_fail")
                if tool_results:
                    # Synthesize from what we have
                    agent_results = [
                        {"agent": r.get("agent", "worker"), "data": r} for r in tool_results
                    ]
                    reply = self._synthesize(message, agent_results)
                    if self.llm and hasattr(self.llm, "generate"):
                        try:
                            synth_prompt = (
                                "Summarize these financial worker results for the user briefly.\n"
                                f"Question: {message}\nData: {truncate_observation(tool_results, 2000)}\nSummary:"
                            )
                            out = self.llm.generate(synth_prompt, max_tokens=350, temp=0.3)
                            if out and out.strip():
                                reply = out.strip()
                        except Exception:
                            pass
                    return self._pack(reply, tool_results, actions, tools_used, trace, user_id, message, "partial")
                fb = self._chat_keyword(message, user_id)
                fb["used_fallback"] = True
                fb["trace"] = trace + fb.get("trace", [])
                fb["status"] = "fallback_parse"
                return fb

            action = str(decision.get("action", "")).lower().strip()
            trace.append(f"step{step}:{action}")

            if action == "final":
                answer = str(decision.get("answer") or "").strip()
                if not answer and tool_results:
                    agent_results = [
                        {"agent": r.get("agent", "worker"), "data": r} for r in tool_results
                    ]
                    answer = self._synthesize(message, agent_results)
                return self._pack(answer, tool_results, actions, tools_used, trace, user_id, message, "success")

            if action != "tool":
                continue

            if len(tools_used) >= MAX_SUPERVISOR_DELEGATIONS:
                observations.append("Max worker delegations reached. Finish with action=final.")
                continue

            tool_name = str(decision.get("tool") or "").strip()
            args = decision.get("args") or {}
            if not isinstance(args, dict):
                args = {}

            result = registry.call(tool_name, args, approved=approved)
            obs = truncate_observation(result)
            observations.append(f"{tool_name} → {obs}")
            if isinstance(result, dict):
                tool_results.append(result)
            else:
                tool_results.append({"agent": tool_name, "result": result})
            tools_used.append(tool_name)
            actions.append({"step": step, "tool": tool_name, "args": args, "result": obs[:500]})

            if isinstance(result, dict) and result.get("error"):
                observations.append("Worker failed. Try another worker or finish.")

        # Cap hit — template synthesize
        agent_results = [{"agent": r.get("agent", "worker"), "data": r} for r in tool_results]
        reply = self._synthesize(message, agent_results) if tool_results else self._chat_keyword(message, user_id)["reply"]
        return self._pack(reply, tool_results, actions, tools_used, trace, user_id, message, "max_steps")

    def _pack(
        self,
        reply: str,
        tool_results: list,
        actions: list,
        tools_used: list,
        trace: list,
        user_id: str,
        message: str,
        status: str,
    ) -> dict[str, Any]:
        show_chart = any(
            k in message.lower()
            for k in ["chart", "graph", "breakdown", "category", "trend", "overview", "summary"]
        )
        return {
            "reply": reply,
            "tool_results": tool_results,
            "user_id": user_id,
            "show_chart": show_chart,
            "trace": trace,
            "step_count": len(actions),
            "tools_used": tools_used,
            "actions": actions,
            "pending_approval": None,
            "status": status,
            "used_fallback": False,
        }

    def chat(
        self,
        message: str,
        user_id: str,
        session_id: Optional[str] = None,
        approved: bool = False,
    ) -> dict[str, Any]:
        """Supervisor (LLM) or keyword route → execute → synthesize."""
        if self.llm and hasattr(self.llm, "generate"):
            out = self._chat_supervisor(message, user_id, session_id=session_id, approved=approved)
        else:
            out = self._chat_keyword(message, user_id)

        if session_id:
            get_memory().add_message(session_id, "assistant", out.get("reply", ""))

        log_trajectory(
            agent="financial_orchestrator",
            message=message,
            user_id=user_id,
            session_id=session_id,
            tools_used=out.get("tools_used"),
            step_count=out.get("step_count", 0),
            status=out.get("status", "success"),
            trace=out.get("trace"),
            reply_preview=out.get("reply", ""),
        )
        return out
