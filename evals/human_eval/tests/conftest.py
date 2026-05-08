import sqlite3
import pytest
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.append(str(EVAL_ROOT))


@pytest.fixture
def db():
    from human_eval.db import init_db
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def app(tmp_path):
    pytest.importorskip("flask")
    from human_eval.app import create_app
    test_app = create_app(
        db_path=":memory:",
        video_data_dir=tmp_path,
        skip_import=True,
    )
    test_app.config["TESTING"] = True
    yield test_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def app_db(app):
    from human_eval.app import get_app_db
    with app.app_context():
        yield get_app_db()
