"""
GraphGuard — Upload All Datasets to Snowflake
Uploads: raw transactions, features, fraud scores, auth profiles, and pipeline logs.
Also creates all schema tables from sql/create_tables.sql.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "ingest_config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_connection(cfg):
    import snowflake.connector
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


def setup_database(cursor, cfg):
    """Create database, schema, warehouse if needed."""
    db = cfg["snowflake"]["database"]
    schema = cfg["snowflake"]["schema"]
    wh = cfg["snowflake"]["warehouse"]
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    cursor.execute(f"USE DATABASE {db}")
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    cursor.execute(f"USE SCHEMA {schema}")
    cursor.execute(f"CREATE WAREHOUSE IF NOT EXISTS {wh} WITH WAREHOUSE_SIZE='XSMALL' AUTO_SUSPEND=60 AUTO_RESUME=TRUE")
    cursor.execute(f"USE WAREHOUSE {wh}")
    print(f"✅ Database {db}, schema {schema}, warehouse {wh} ready.\n")


def create_all_tables(cursor):
    """Create all tables from the SQL schema."""
    tables_sql = {
        "RAW_TRANSACTIONS": """
            CREATE TABLE IF NOT EXISTS RAW_TRANSACTIONS (
                TRANSACTION_ID    VARCHAR(50)  PRIMARY KEY,
                USER_ID           VARCHAR(20)  NOT NULL,
                MERCHANT_NAME     VARCHAR(100),
                CATEGORY          VARCHAR(50),
                AMOUNT            FLOAT,
                LOCATION          VARCHAR(100),
                TRANSACTION_DATE  TIMESTAMP_NTZ,
                STATUS            VARCHAR(20),
                INGESTED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """,
        "TRANSACTION_FEATURES": """
            CREATE TABLE IF NOT EXISTS TRANSACTION_FEATURES (
                TRANSACTION_ID         VARCHAR(50)   PRIMARY KEY,
                USER_ID                VARCHAR(20)   NOT NULL,
                AMOUNT                 FLOAT,
                AMOUNT_ZSCORE          FLOAT,
                IS_HIGH_VALUE          BOOLEAN,
                USER_AVG_AMOUNT        FLOAT,
                USER_STD_AMOUNT        FLOAT,
                USER_TOTAL_TRANSACTIONS INT,
                TXN_COUNT_1H           INT,
                TXN_COUNT_24H          INT,
                TXN_COUNT_7D           INT,
                CATEGORY               VARCHAR(50),
                CATEGORY_RISK_WEIGHT   FLOAT,
                HOUR_OF_DAY            INT,
                DAY_OF_WEEK            INT,
                IS_WEEKEND             BOOLEAN,
                LOCATION               VARCHAR(100),
                IS_NEW_LOCATION        BOOLEAN,
                LOCATION_ENTROPY       FLOAT,
                COMPUTED_AT            TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """,
        "FRAUD_SCORES": """
            CREATE TABLE IF NOT EXISTS FRAUD_SCORES (
                SCORE_ID               VARCHAR(50)   PRIMARY KEY,
                TRANSACTION_ID         VARCHAR(50)   NOT NULL,
                USER_ID                VARCHAR(20)   NOT NULL,
                ZSCORE_FLAG            FLOAT,
                VELOCITY_FLAG          FLOAT,
                CATEGORY_RISK_SCORE    FLOAT,
                GEOGRAPHIC_RISK_SCORE  FLOAT,
                ISOLATION_FOREST_SCORE FLOAT,
                COMBINED_RISK_SCORE    FLOAT,
                RISK_LEVEL             VARCHAR(20),
                RECOMMENDATION         VARCHAR(200),
                SCORED_AT              TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """,
        "AUTH_PROFILES": """
            CREATE TABLE IF NOT EXISTS AUTH_PROFILES (
                USER_ID                     VARCHAR(20)  PRIMARY KEY,
                AVG_RISK                    FLOAT,
                MAX_RISK                    FLOAT,
                HIGH_RISK_COUNT             INT,
                RECOMMENDED_SECURITY_LEVEL  VARCHAR(20),
                NUM_QUESTIONS               INT,
                UPLOADED_AT                 TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """,
        "PIPELINE_RUNS": """
            CREATE TABLE IF NOT EXISTS PIPELINE_RUNS (
                RUN_ID              VARCHAR(50),
                TIMESTAMP           VARCHAR(100),
                STAGE               VARCHAR(50),
                STATUS              VARCHAR(20),
                RECORDS_PROCESSED   INT,
                DURATION_MS         FLOAT,
                ERROR_MESSAGE       VARCHAR(500)
            )
        """,
    }

    for name, ddl in tables_sql.items():
        cursor.execute(ddl)
        print(f"  ✅ Table {name} ready")
    print()


def upload_csv(conn, csv_path, table_name, truncate_first=True):
    """Upload a CSV to a Snowflake table using write_pandas."""
    from snowflake.connector.pandas_tools import write_pandas

    if not csv_path.exists():
        print(f"  ⚠️  Skipping {table_name}: {csv_path.name} not found")
        return 0

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  ⚠️  Skipping {table_name}: {csv_path.name} is empty")
        return 0

    # Uppercase column names to match Snowflake
    df.columns = [c.upper() for c in df.columns]

    if truncate_first:
        conn.cursor().execute(f"TRUNCATE TABLE IF EXISTS {table_name}")

    success, num_chunks, num_rows, _ = write_pandas(
        conn, df, table_name, quote_identifiers=False,
    )
    print(f"  ✅ {table_name}: {num_rows} rows loaded from {csv_path.name}")
    return num_rows


def main():
    cfg = load_config()
    print("=" * 60)
    print("  GraphGuard — Upload All Datasets to Snowflake")
    print("=" * 60)
    print()

    t0 = time.time()
    conn = get_connection(cfg)
    cur = conn.cursor()

    # 1. Setup
    print("📦 Setting up database …")
    setup_database(cur, cfg)

    # 2. Create all tables
    print("📋 Creating tables …")
    create_all_tables(cur)

    # 3. Upload all datasets
    uploads = [
        (PROJECT_ROOT / "Week-4_Hand-on-Session" / "transactions_export_20260213_000349.csv",
         "RAW_TRANSACTIONS"),
        (PROJECT_ROOT / "features_output.csv",
         "TRANSACTION_FEATURES"),
        (PROJECT_ROOT / "fraud_scores_output.csv",
         "FRAUD_SCORES"),
        (PROJECT_ROOT / "auth_profiles_output.csv",
         "AUTH_PROFILES"),
        (PROJECT_ROOT / "pipeline_logs.csv",
         "PIPELINE_RUNS"),
    ]

    print("📤 Uploading datasets …")
    total_rows = 0
    for csv_path, table_name in uploads:
        rows = upload_csv(conn, csv_path, table_name)
        total_rows += rows

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"  🎉 Done! {total_rows} total rows uploaded in {elapsed:.1f}s")
    print("=" * 60)

    # 4. Verify by querying row counts
    print("\n📊 Verification — Row Counts:")
    for _, table_name in uploads:
        result = cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        print(f"  {table_name}: {result[0]} rows")

    conn.close()


if __name__ == "__main__":
    main()
