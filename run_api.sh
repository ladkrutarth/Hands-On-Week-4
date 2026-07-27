#!/usr/bin/env bash
# Launch FastAPI backend (src-layout aware)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"
# Reload only when src/ changes — watching the whole repo restarts the API
# on Streamlit edits and makes the dashboard report "Backend API is not available".
exec python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir "${ROOT}/src"
