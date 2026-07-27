"""
Tool registry for Veriscan agentic workflows.

Registers named tools with JSON-ish schemas so an LLM can select and call them.
Supports HITL gating via `requires_hitl`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# Tools that must not run without an explicit approved=True from the API/caller.
REQUIRE_HITL = frozenset({"challenge_auth"})


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    requires_hitl: bool = False


@dataclass
class ToolRegistry:
    """Name → ToolSpec map with schema export and safe execution."""

    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(
        self,
        name: str,
        description: str,
        fn: Callable[..., Any],
        parameters: Optional[dict[str, Any]] = None,
        requires_hitl: bool = False,
    ) -> None:
        hitl = requires_hitl or name in REQUIRE_HITL
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            fn=fn,
            requires_hitl=hitl,
        )

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def schemas_for_prompt(self) -> str:
        """Compact tool catalog for the LLM system prompt."""
        lines = []
        for spec in self._tools.values():
            hitl = " [REQUIRES_APPROVAL]" if spec.requires_hitl else ""
            props = spec.parameters.get("properties", {})
            arg_bits = ", ".join(
                f"{k}: {v.get('type', 'any')}" for k, v in props.items()
            )
            lines.append(f"- {spec.name}({arg_bits}){hitl}: {spec.description}")
        return "\n".join(lines)

    def call(
        self,
        name: str,
        args: Optional[dict[str, Any]] = None,
        *,
        approved: bool = False,
    ) -> dict[str, Any]:
        """
        Execute a registered tool.

        Returns a dict. On HITL block:
          {"pending_approval": True, "tool": name, "args": ...}
        """
        spec = self._tools.get(name)
        if spec is None:
            return {"error": f"Unknown tool: {name}", "tool": name}

        args = dict(args or {})
        if spec.requires_hitl and not approved:
            return {
                "pending_approval": True,
                "tool": name,
                "args": args,
                "message": f"Tool '{name}' requires human approval (set approved=true).",
            }

        try:
            result = spec.fn(**args)
            if isinstance(result, dict):
                return result
            if isinstance(result, list):
                return {"tool": name, "items": result}
            return {"tool": name, "result": result}
        except TypeError as e:
            # Retry with only declared parameters if LLM invents extras.
            props = set(spec.parameters.get("properties", {}).keys())
            filtered = {k: v for k, v in args.items() if k in props}
            try:
                result = spec.fn(**filtered)
                if isinstance(result, dict):
                    return result
                if isinstance(result, list):
                    return {"tool": name, "items": result}
                return {"tool": name, "result": result}
            except Exception as inner:
                return {"error": str(inner), "tool": name, "args": filtered, "type_error": str(e)}
        except Exception as e:
            return {"error": str(e), "tool": name, "args": args}

    def to_json_schema_list(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
                "requires_hitl": s.requires_hitl,
            }
            for s in self._tools.values()
        ]

    def subset(self, names: list[str]) -> "ToolRegistry":
        """Return a new registry containing only the named tools (order preserved)."""
        out = ToolRegistry()
        for name in names:
            spec = self._tools.get(name)
            if spec is None:
                continue
            out._tools[name] = spec
        return out

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def truncate_observation(obj: Any, max_chars: int = 1200) -> str:
    """Serialize tool output for the LLM observation window."""
    try:
        text = json.dumps(obj, default=str)
    except TypeError:
        text = str(obj)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text
