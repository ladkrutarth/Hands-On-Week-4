"""
Optional JSONL trajectory logger for agent runs (portfolio / eval observability).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "trajectories.jsonl"


def _enabled() -> bool:
    v = os.environ.get("VERISCAN_TRAJECTORY_LOG", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def log_trajectory(
    *,
    agent: str,
    message: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tools_used: Optional[list[str]] = None,
    step_count: int = 0,
    status: str = "success",
    trace: Optional[list[str]] = None,
    reply_preview: str = "",
    extra: Optional[dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> None:
    """Append one trajectory record. No-ops if logging disabled or write fails."""
    if not _enabled():
        return
    log_path = path or DEFAULT_LOG_PATH
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "user_id": user_id,
        "session_id": session_id,
        "message": message[:500],
        "tools_used": tools_used or [],
        "step_count": step_count,
        "status": status,
        "trace": trace or [],
        "reply_preview": (reply_preview or "")[:300],
    }
    if extra:
        record["extra"] = extra
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass
