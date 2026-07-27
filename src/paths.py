"""Central project paths for Veriscan-Cortex.

All packages under ``src/`` should import from here so relocating
files does not break data/log resolution.
"""

from __future__ import annotations

from pathlib import Path

# src/ → project root
SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
CSV_DATA_DIR = DATA_DIR / "csv_data"
IC3_DATA_DIR = DATA_DIR / "ic3_2024_csvs"
PDF_DATA_DIR = DATA_DIR / "pdf_data"
IMAGE_DATA_DIR = DATA_DIR / "image_data"
UPLOADS_DIR = DATA_DIR / "user_uploads"

CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"
APP_DIR = PROJECT_ROOT / "app"

CHROMA_LOCAL_DIR = PROJECT_ROOT / ".chroma_db_local"
CHROMA_MULTIMODAL_DIR = PROJECT_ROOT / ".chroma_db_multimodal"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline_logs.csv"
TRAJECTORY_LOG_PATH = LOGS_DIR / "trajectories.jsonl"
