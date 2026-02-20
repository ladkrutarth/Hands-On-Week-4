"""
GraphGuard — Upload All Python & SQL Code to Snowflake Internal Stage
Stores all project scripts in a Snowflake stage for reproducibility.
"""

import sys
from pathlib import Path

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


def main():
    cfg = load_config()
    db = cfg["snowflake"]["database"]
    schema = cfg["snowflake"]["schema"]
    wh = cfg["snowflake"]["warehouse"]

    print("=" * 60)
    print("  GraphGuard — Upload Code to Snowflake Stage")
    print("=" * 60)

    conn = get_connection(cfg)
    cur = conn.cursor()

    # Setup
    cur.execute(f"USE DATABASE {db}")
    cur.execute(f"USE SCHEMA {schema}")
    cur.execute(f"USE WAREHOUSE {wh}")

    # Create internal stage for code
    stage_name = "GRAPHGUARD_CODE_STAGE"
    cur.execute(f"CREATE STAGE IF NOT EXISTS {stage_name} COMMENT='GraphGuard project source code'")
    print(f"\n✅ Stage {stage_name} ready.\n")

    # Collect all files to upload
    files_to_upload = []

    # Python scripts
    for py_file in sorted((PROJECT_ROOT / "scripts").glob("*.py")):
        files_to_upload.append((py_file, "scripts"))
    for py_file in sorted((PROJECT_ROOT / "models").glob("*.py")):
        files_to_upload.append((py_file, "models"))

    # SQL files
    for sql_file in sorted((PROJECT_ROOT / "sql").glob("*.sql")):
        files_to_upload.append((sql_file, "sql"))

    # Config files
    for yaml_file in sorted((PROJECT_ROOT / "scripts").glob("*.yaml")):
        files_to_upload.append((yaml_file, "config"))

    # Streamlit app
    streamlit_app = PROJECT_ROOT / "Week-4_Hand-on-Session" / "streamlit_app.py"
    if streamlit_app.exists():
        files_to_upload.append((streamlit_app, "app"))

    # Requirements
    req_file = PROJECT_ROOT / "Week-4_Hand-on-Session" / "requirements.txt"
    if req_file.exists():
        files_to_upload.append((req_file, "app"))

    # Upload each file
    print("📤 Uploading files …")
    for file_path, subfolder in files_to_upload:
        put_cmd = f"PUT 'file://{file_path}' @{stage_name}/{subfolder}/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        cur.execute(put_cmd)
        result = cur.fetchone()
        status = result[6] if len(result) > 6 else "UPLOADED"
        print(f"  ✅ {subfolder}/{file_path.name} — {status}")

    # Verify: list all staged files
    print(f"\n📋 Files in @{stage_name}:")
    cur.execute(f"LIST @{stage_name}")
    rows = cur.fetchall()
    for row in rows:
        name = row[0]
        size = row[1]
        print(f"  📄 {name}  ({size} bytes)")

    print(f"\n🎉 Done! {len(files_to_upload)} files uploaded to @{stage_name}")
    conn.close()


if __name__ == "__main__":
    main()
