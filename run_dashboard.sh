#!/usr/bin/env bash
# Launch Streamlit dashboard (src-layout aware)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"
exec streamlit run app/streamlit_app.py "$@"
