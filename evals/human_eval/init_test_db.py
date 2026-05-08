"""Initialize test DB by copying videos & comparison_groups from production DB.

Usage:
    python -m human_eval.init_test_db          # default paths from config
    python -m human_eval.init_test_db --reset   # drop & recreate test DB
"""
import argparse
import sqlite3
from pathlib import Path

from human_eval.config import DB_PATH, TEST_DB_PATH
from human_eval.db import init_db, get_db


# Tables whose rows are copied verbatim from prod → test
_COPY_TABLES = ["videos", "comparison_groups"]


def init_test_db(*, reset: bool = False):
    if reset and TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
        print(f"removed old test DB: {TEST_DB_PATH}")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"production DB not found: {DB_PATH}")

    # Open production (read-only) and test DBs
    prod = sqlite3.connect(str(DB_PATH))
    prod.row_factory = sqlite3.Row
    prod.execute("PRAGMA query_only = ON")

    test = get_db(TEST_DB_PATH)
    init_db(test)

    for table in _COPY_TABLES:
        existing = test.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        if existing > 0 and not reset:
            print(f"  {table}: already has {existing} rows, skipping (use --reset to overwrite)")
            continue

        if reset:
            test.execute(f"DELETE FROM {table}")

        cols_info = prod.execute(f"PRAGMA table_info({table})").fetchall()
        col_names = [c["name"] for c in cols_info]
        cols_csv = ", ".join(col_names)
        placeholders = ", ".join("?" for _ in col_names)

        rows = prod.execute(f"SELECT {cols_csv} FROM {table}").fetchall()
        test.executemany(
            f"INSERT OR IGNORE INTO {table} ({cols_csv}) VALUES ({placeholders})",
            [tuple(r) for r in rows],
        )
        test.commit()
        print(f"  {table}: copied {len(rows)} rows")

    prod.close()
    test.close()
    print(f"test DB ready: {TEST_DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop & recreate test DB")
    args = parser.parse_args()
    init_test_db(reset=args.reset)
