"""
Bounded ReAct-style agent loop with JSON tool calls.

LLM returns:
  {"action":"tool","tool":"<name>","args":{...}}
  or
  {"action":"final","answer":"..."}

Falls back to a caller-provided keyword path when JSON parse fails
*and* no useful tool results exist yet.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from agents.base import AgentAction
from agents.tool_registry import ToolRegistry, truncate_observation

DEFAULT_MAX_STEPS = 4
DEFAULT_MAX_TOOL_CALLS = 2
MAX_REPLAN_ON_ERROR = 1

SYSTEM_INSTRUCTIONS = """You are a Veriscan tool-using agent. You MUST reply with a single JSON object only.
No markdown fences, no prose outside JSON.

Available tools:
{tools}

Rules:
1. To call a tool: {{"action":"tool","tool":"<name>","args":{{...}}}}
2. To finish: {{"action":"final","answer":"<concise helpful reply using tool observations>"}}
3. Prefer 1 tool, max {max_tools} tools, then finish with action=final. Do not invent tool names.
4. user_id is injected automatically; you may omit it from args.
5. Never call the same tool with the same args twice.
6. After you have observations, you MUST finish with action=final (do not keep calling tools).
7. Ground answers in tool numbers; do not invent amounts.
"""


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Best-effort extract of a JSON object from LLM output."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _tool_fingerprint(name: str, args: dict[str, Any]) -> str:
    try:
        return name + ":" + json.dumps(args, sort_keys=True, default=str)
    except TypeError:
        return f"{name}:{args}"


class ReactLoop:
    """Observe → think (LLM JSON) → act, with step/tool caps and duplicate detection."""

    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        default_args: Optional[dict[str, Any]] = None,
    ):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.max_tool_calls = max(1, max_tool_calls)
        self.default_args = dict(default_args or {})

    def run(
        self,
        message: str,
        *,
        session_history: str = "",
        approved: bool = False,
        on_parse_fail: Optional[Callable[[], dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Returns:
          {
            reply, tool_results, actions, trace, step_count, tools_used,
            pending_approval, status, used_fallback
          }
        """
        observations: list[str] = []
        tool_results: list[dict[str, Any]] = []
        actions: list[AgentAction] = []
        trace: list[str] = []
        tools_used: list[str] = []
        pending_approval: Optional[dict[str, Any]] = None
        error_replans = 0
        seen_calls: set[str] = set()

        system = SYSTEM_INSTRUCTIONS.format(
            tools=self.registry.schemas_for_prompt(),
            max_tools=self.max_tool_calls,
        )

        for step in range(1, self.max_steps + 1):
            force_final = len(tools_used) >= self.max_tool_calls
            prompt = self._build_prompt(
                system,
                message,
                session_history,
                observations,
                force_final=force_final,
            )
            try:
                raw = self.llm.generate(prompt, max_tokens=320, temp=0.0)
            except Exception as e:
                trace.append(f"llm_error:{e}")
                if tool_results:
                    return self._finish(
                        "Tool results ready; summarizing.",
                        tool_results,
                        actions,
                        trace,
                        tools_used,
                        pending_approval,
                        status="needs_synth",
                    )
                if on_parse_fail:
                    fb = on_parse_fail()
                    fb["used_fallback"] = True
                    fb.setdefault("trace", []).extend(trace)
                    fb.setdefault("actions", [])
                    fb["status"] = "fallback_llm_error"
                    return fb
                return self._error_result(f"LLM error: {e}", tool_results, actions, trace)

            decision = extract_json(raw)
            if decision is None:
                trace.append(f"step{step}:parse_fail")
                # Prefer synthesizing from tools we already ran
                if tool_results:
                    prose = (raw or "").strip()
                    reply = prose if prose and len(prose) > 40 and "{" not in prose[:20] else (
                        "Tool results ready; summarizing."
                    )
                    return self._finish(
                        reply,
                        tool_results,
                        actions,
                        trace,
                        tools_used,
                        pending_approval,
                        status="needs_synth",
                    )
                if on_parse_fail:
                    fb = on_parse_fail()
                    fb["used_fallback"] = True
                    fb.setdefault("trace", []).extend(trace)
                    fb["status"] = "fallback_parse"
                    return fb
                reply = raw.strip() if raw else "I could not parse a tool decision."
                return self._finish(reply, tool_results, actions, trace, tools_used, pending_approval)

            action = str(decision.get("action", "")).lower().strip()
            # If model forgot action=final but returned an answer after tools
            if action not in ("tool", "final") and tool_results and (
                decision.get("answer") or decision.get("reply")
            ):
                action = "final"
            trace.append(f"step{step}:{action}")

            if action == "final":
                answer = decision.get("answer") or decision.get("reply") or ""
                if not answer and tool_results:
                    return self._finish(
                        "Tool results ready; summarizing.",
                        tool_results,
                        actions,
                        trace,
                        tools_used,
                        pending_approval,
                        status="needs_synth",
                    )
                return self._finish(
                    str(answer).strip(),
                    tool_results,
                    actions,
                    trace,
                    tools_used,
                    pending_approval,
                )

            if force_final:
                # Ignore further tool calls — we already hit the tool budget
                trace.append(f"step{step}:forced_final_block")
                return self._finish(
                    "Tool budget reached; summarizing.",
                    tool_results,
                    actions,
                    trace,
                    tools_used,
                    pending_approval,
                    status="needs_synth",
                )

            if action != "tool":
                trace.append(f"step{step}:unknown_action")
                if tool_results:
                    return self._finish(
                        "Tool results ready; summarizing.",
                        tool_results,
                        actions,
                        trace,
                        tools_used,
                        pending_approval,
                        status="needs_synth",
                    )
                if on_parse_fail:
                    fb = on_parse_fail()
                    fb["used_fallback"] = True
                    fb.setdefault("trace", []).extend(trace)
                    fb["status"] = "fallback_bad_action"
                    return fb
                continue

            tool_name = str(decision.get("tool") or decision.get("name") or "").strip()
            args = decision.get("args") or decision.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            for k, v in self.default_args.items():
                args.setdefault(k, v)

            if tool_name not in self.registry:
                observations.append(f"Error: unknown tool '{tool_name}'. Choose from the list.")
                trace.append(f"step{step}:unknown_tool:{tool_name}")
                continue

            fp = _tool_fingerprint(tool_name, args)
            if fp in seen_calls:
                observations.append(
                    f"Duplicate call blocked for {tool_name}. Finish with action=final now."
                )
                trace.append(f"step{step}:duplicate:{tool_name}")
                continue
            seen_calls.add(fp)

            result = self.registry.call(tool_name, args, approved=approved)

            if result.get("pending_approval"):
                pending_approval = result
                actions.append(
                    AgentAction(
                        step=step,
                        tool=tool_name,
                        args=args,
                        result="pending_approval",
                    )
                )
                tools_used.append(tool_name)
                reply = (
                    f"Action '{tool_name}' requires human approval. "
                    "Re-submit with approved=true to execute."
                )
                return self._finish(
                    reply,
                    tool_results,
                    actions,
                    trace,
                    tools_used,
                    pending_approval,
                    status="pending_approval",
                )

            obs = truncate_observation(result)
            observations.append(f"Tool {tool_name} → {obs}")
            tool_results.append(result if isinstance(result, dict) else {"tool": tool_name, "result": result})
            tools_used.append(tool_name)
            actions.append(
                AgentAction(
                    step=step,
                    tool=tool_name,
                    args=args,
                    result=obs[:500],
                )
            )

            if isinstance(result, dict) and result.get("error"):
                error_replans += 1
                trace.append(f"step{step}:tool_error")
                if error_replans > MAX_REPLAN_ON_ERROR:
                    if tool_results:
                        return self._finish(
                            "Tool error; summarizing available results.",
                            tool_results,
                            actions,
                            trace,
                            tools_used,
                            pending_approval,
                            status="needs_synth",
                        )
                    if on_parse_fail:
                        fb = on_parse_fail()
                        fb["used_fallback"] = True
                        fb.setdefault("trace", []).extend(trace)
                        fb["status"] = "fallback_tool_error"
                        return fb
                    break
                observations.append("Previous tool failed. Try a different tool or finish with action=final.")

        # Max steps — ask caller to synthesize from tool results
        return self._finish(
            "Reached step limit; summarizing.",
            tool_results,
            actions,
            trace,
            tools_used,
            pending_approval,
            status="needs_synth" if tool_results else "max_steps",
        )

    def _build_prompt(
        self,
        system: str,
        message: str,
        history: str,
        observations: list[str],
        *,
        force_final: bool = False,
    ) -> str:
        parts = [system, ""]
        if history:
            parts.append(f"Conversation history:\n{history}\n")
        parts.append(f"User question: {message}\n")
        if observations:
            parts.append("Observations so far:")
            for i, obs in enumerate(observations, 1):
                parts.append(f"{i}. {obs}")
            parts.append("")
        if force_final:
            parts.append(
                "CRITICAL: You already have enough tool results. "
                'Reply ONLY with {"action":"final","answer":"<concise reply grounded in observations>"}.'
            )
        elif observations:
            parts.append(
                "You have tool observations. Prefer finishing now with action=final "
                "unless one more distinct tool is clearly required."
            )
        parts.append("Respond with JSON only:")
        return "\n".join(parts)

    def _finish(
        self,
        reply: str,
        tool_results: list[dict[str, Any]],
        actions: list[AgentAction],
        trace: list[str],
        tools_used: list[str],
        pending_approval: Optional[dict[str, Any]],
        status: str = "success",
    ) -> dict[str, Any]:
        return {
            "reply": reply,
            "tool_results": tool_results,
            "actions": [a.model_dump() if hasattr(a, "model_dump") else a.dict() for a in actions],
            "trace": trace,
            "step_count": len(actions),
            "tools_used": tools_used,
            "pending_approval": pending_approval,
            "status": status,
            "used_fallback": False,
        }

    def _error_result(
        self,
        msg: str,
        tool_results: list[dict[str, Any]],
        actions: list[AgentAction],
        trace: list[str],
    ) -> dict[str, Any]:
        return {
            "reply": msg,
            "tool_results": tool_results,
            "actions": [a.model_dump() if hasattr(a, "model_dump") else a.dict() for a in actions],
            "trace": trace,
            "step_count": len(actions),
            "tools_used": [],
            "pending_approval": None,
            "status": "error",
            "used_fallback": False,
        }
