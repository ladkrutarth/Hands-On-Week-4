"""Backward-compatible Streamlit entrypoint.

Prefer:  ./run_dashboard.sh
Or:      streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import runpy
from pathlib import Path

APP = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
runpy.run_path(str(APP), run_name="__main__")
