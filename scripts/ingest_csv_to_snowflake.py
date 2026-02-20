"""
GraphGuard — CSV to Snowflake Ingestion Script
Loads transaction CSV data into Snowflake warehouse.
Supports --dry-run mode for testing without credentials.
"""

import argparse
import csv
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "ingest_config.yaml"
PIPELINE_LOG_PATH = PROJECT_ROOT / "pipeline_logs.csv"

PIPELINE_LOG_COLUMNS = [
    "run_id", "timestamp", "stage", "status",
    "records_processed", "duration_ms", "error_message",
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Pipeline Logging
# ---------------------------------------------------------------------------

def log_pipeline_event(
    stage: str,
    status: str,
    records: int = 0,
    duration_ms: float = 0.0,
    error_message: str = "",
):
    """Append a row to pipeline_logs.csv."""
    file_exists = PIPELINE_LOG_PATH.exists()
    with open(PIPELINE_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PIPELINE_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "stage": stage,
            "status": status,
            "records_processed": records,
            "duration_ms": round(duration_ms, 2),
            "error_message": error_message,
        })


# ---------------------------------------------------------------------------
# Snowflake helpers
# ---------------------------------------------------------------------------

def get_snowflake_connection(cfg: dict):
    """Create a Snowflake connection from config dict."""
    try:
        import snowflake.connector
    except ImportError:
        print("ERROR: snowflake-connector-python is not installed.")
        print("  pip install snowflake-connector-python")
        sys.exit(1)

    sf = cfg["snowflake"]
    return snowflake.connector.connect(
        user=sf["user"],
        password=sf["password"],
        account=sf["account"],
        warehouse=sf["warehouse"],
        database=sf["database"],
        schema=sf["schema"],
        role=sf.get("role", "SYSADMIN"),
    )


def create_raw_table(cursor, cfg):
    """Create database, schema, and RAW_TRANSACTIONS table if they do not exist."""
    db = cfg["snowflake"]["database"]
    schema = cfg["snowflake"]["schema"]
    wh = cfg["snowflake"]["warehouse"]

    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    cursor.execute(f"USE DATABASE {db}")
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    cursor.execute(f"USE SCHEMA {schema}")
    cursor.execute(f"CREATE WAREHOUSE IF NOT EXISTS {wh} WITH WAREHOUSE_SIZE='XSMALL' AUTO_SUSPEND=60 AUTO_RESUME=TRUE")
    cursor.execute(f"USE WAREHOUSE {wh}")
    print(f"✅ Database {db}, schema {schema}, warehouse {wh} ready.")

    ddl = """
    CREATE TABLE IF NOT EXISTS RAW_TRANSACTIONS (
        TRANSACTION_ID  VARCHAR(50)  PRIMARY KEY,
        USER_ID         VARCHAR(20)  NOT NULL,
        MERCHANT_NAME   VARCHAR(100),
        CATEGORY        VARCHAR(50),
        AMOUNT          FLOAT,
        LOCATION        VARCHAR(100),
        TRANSACTION_DATE TIMESTAMP_NTZ,
        STATUS          VARCHAR(20),
        INGESTED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
    );
    """
    cursor.execute(ddl)
    print("✅ RAW_TRANSACTIONS table ready.")


def load_data_to_snowflake(df: pd.DataFrame, conn):
    """Write DataFrame into Snowflake RAW_TRANSACTIONS using write_pandas."""
    from snowflake.connector.pandas_tools import write_pandas

    success, num_chunks, num_rows, _ = write_pandas(
        conn, df, "RAW_TRANSACTIONS", quote_identifiers=False,
    )
    print(f"✅ Loaded {num_rows} rows in {num_chunks} chunk(s). Success={success}")
    return num_rows


# ---------------------------------------------------------------------------
# Dry-Run mode
# ---------------------------------------------------------------------------

def dry_run_ingest(csv_path: Path):
    """Simulate ingestion by reading CSV and validating schema."""
    print("🔄 [DRY-RUN] Reading CSV …")
    t0 = time.time()

    df = pd.read_csv(csv_path)
    elapsed = (time.time() - t0) * 1000

    expected_cols = {
        "TRANSACTION_ID", "USER_ID", "MERCHANT_NAME",
        "CATEGORY", "AMOUNT", "LOCATION", "TRANSACTION_DATE", "STATUS",
    }
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"✅ [DRY-RUN] Schema validated — {len(df)} rows, {len(df.columns)} cols")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Users  : {df['USER_ID'].nunique()}")
    print(f"   Date range: {df['TRANSACTION_DATE'].min()} → {df['TRANSACTION_DATE'].max()}")
    print(f"   Total spend: ${df['AMOUNT'].sum():,.2f}")

    log_pipeline_event("ingestion", "success_dry_run", len(df), elapsed)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest CSV → Snowflake")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate CSV without loading to Snowflake",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV file (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_config()
    csv_path = Path(args.csv) if args.csv else PROJECT_ROOT / cfg["source"]["csv_path"]

    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        log_pipeline_event("ingestion", "failed", error_message=f"CSV not found: {csv_path}")
        sys.exit(1)

    if args.dry_run:
        dry_run_ingest(csv_path)
        return

    # --- Live Snowflake ingestion ---
    t0 = time.time()
    try:
        conn = get_snowflake_connection(cfg)
        cur = conn.cursor()

        create_raw_table(cur, cfg)

        df = pd.read_csv(csv_path)
        nrows = load_data_to_snowflake(df, conn)
        elapsed = (time.time() - t0) * 1000

        log_pipeline_event("ingestion", "success", nrows, elapsed)
        print(f"🎉 Ingestion complete in {elapsed:.0f}ms")

    except Exception as exc:
        elapsed = (time.time() - t0) * 1000
        log_pipeline_event("ingestion", "failed", error_message=str(exc))
        print(f"❌ Ingestion failed: {exc}")
        sys.exit(1)
    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    main()
