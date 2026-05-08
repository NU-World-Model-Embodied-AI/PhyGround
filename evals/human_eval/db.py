import sqlite3
from pathlib import Path

from human_eval.config import STATUS_ASSIGNED, STATUS_COMPLETED, STATUS_PARTIAL, STATUS_SKIPPED

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS annotators (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    dataset TEXT NOT NULL,
    prompt TEXT NOT NULL,
    physical_laws TEXT NOT NULL,
    difficulty_score REAL,
    metadata_version INTEGER NOT NULL DEFAULT 1,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    import_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id),
    annotator_id INTEGER NOT NULL REFERENCES annotators(id),
    status TEXT NOT NULL DEFAULT 'assigned',
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    skip_reason TEXT,
    group_id TEXT REFERENCES comparison_groups(id),
    UNIQUE(video_id, annotator_id)
);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY,
    assignment_id INTEGER NOT NULL UNIQUE REFERENCES assignments(id),
    scores_json TEXT NOT NULL,
    metadata_version INTEGER,
    note TEXT,
    play_count INTEGER DEFAULT 0,
    stay_seconds REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS annotation_items (
    id INTEGER PRIMARY KEY,
    annotation_id INTEGER NOT NULL REFERENCES annotations(id),
    dimension TEXT NOT NULL,
    law TEXT,
    score INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS comparison_groups (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    physical_laws TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS excluded_annotators (
    annotator_id INTEGER PRIMARY KEY REFERENCES annotators(id),
    reason TEXT NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


_MIGRATIONS = [
    "ALTER TABLE videos ADD COLUMN metadata_version INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE annotations ADD COLUMN metadata_version INTEGER",
    "ALTER TABLE assignments ADD COLUMN group_id TEXT",
    # comparison_groups table is created in SCHEMA_SQL above; this migration
    # is a no-op for new DBs but needed for existing ones.
    ("CREATE TABLE IF NOT EXISTS comparison_groups ("
     "id TEXT PRIMARY KEY, prompt TEXT NOT NULL, physical_laws TEXT NOT NULL, "
     "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"),
    "ALTER TABLE annotations ADD COLUMN play_count INTEGER DEFAULT 0",
    "ALTER TABLE annotations ADD COLUMN stay_seconds REAL DEFAULT 0",
    "ALTER TABLE annotations ADD COLUMN na_laws TEXT",
    "ALTER TABLE annotators ADD COLUMN gender TEXT",
    "ALTER TABLE annotators ADD COLUMN age TEXT",
    "ALTER TABLE annotators ADD COLUMN major TEXT",
    "ALTER TABLE annotators ADD COLUMN education TEXT",
    "ALTER TABLE annotators ADD COLUMN cohort TEXT DEFAULT 'others'",
    ("CREATE TABLE IF NOT EXISTS excluded_annotators ("
     "annotator_id INTEGER PRIMARY KEY REFERENCES annotators(id), "
     "reason TEXT NOT NULL, "
     "generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"),
]


def init_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)
    conn.execute("PRAGMA foreign_keys = ON")
    # Apply migrations for existing DBs (silently skip if column already exists)
    for sql in _MIGRATIONS:
        try:
            conn.execute(sql)
            conn.commit()
        except sqlite3.OperationalError:
            pass


def get_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn



def pending_assignment_sql(alias: str = "a") -> str:
    """Assignment is still actionable: not finished AND not expired."""
    return (
        f"({alias}.status NOT IN ('{STATUS_COMPLETED}', '{STATUS_PARTIAL}', '{STATUS_SKIPPED}') "
        f"AND {alias}.expires_at > datetime('now'))"
    )
