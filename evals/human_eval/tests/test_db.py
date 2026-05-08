import sqlite3
import pytest


def test_tables_exist(db):
    cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    assert "annotators" in tables
    assert "videos" in tables
    assert "assignments" in tables
    assert "annotations" in tables
    assert "annotation_items" in tables


def test_annotator_unique_name(db):
    db.execute("INSERT INTO annotators (name) VALUES ('alice')")
    db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO annotators (name) VALUES ('alice')")
        db.commit()


def test_assignment_unique_video_annotator(db):
    db.execute("INSERT INTO annotators (name) VALUES ('alice')")
    db.execute(
        "INSERT INTO videos (filename, dataset, prompt, physical_laws, import_hash) "
        "VALUES ('v.mp4', 'ds', 'prompt', '[\"fluid\"]', 'abc')"
    )
    db.commit()
    db.execute(
        "INSERT INTO assignments (video_id, annotator_id, status, expires_at) "
        "VALUES (1, 1, 'assigned', datetime('now', '+1 hour'))"
    )
    db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO assignments (video_id, annotator_id, status, expires_at) "
            "VALUES (1, 1, 'assigned', datetime('now', '+1 hour'))"
        )
        db.commit()


def test_annotation_unique_assignment(db):
    db.execute("INSERT INTO annotators (name) VALUES ('alice')")
    db.execute(
        "INSERT INTO videos (filename, dataset, prompt, physical_laws, import_hash) "
        "VALUES ('v.mp4', 'ds', 'prompt', '[\"fluid\"]', 'abc')"
    )
    db.execute(
        "INSERT INTO assignments (video_id, annotator_id, status, expires_at) "
        "VALUES (1, 1, 'assigned', datetime('now', '+1 hour'))"
    )
    db.execute(
        "INSERT INTO annotations (assignment_id, scores_json) VALUES (1, '{}')"
    )
    db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO annotations (assignment_id, scores_json) VALUES (1, '{}')"
        )
        db.commit()


