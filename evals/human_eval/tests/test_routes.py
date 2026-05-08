import json
import pytest

from human_eval.config import STATUS_COMPLETED


def _setup_user_and_video(app_db):
    app_db.execute("INSERT INTO annotators (name) VALUES ('alice')")
    app_db.execute(
        "INSERT INTO videos (filename, dataset, prompt, physical_laws, import_hash) "
        """VALUES ('v.mp4', 'ds', 'test prompt', '["flow_dynamics", "boundary_interaction", "fluid_continuity"]', 'hash')"""
    )
    app_db.commit()
    return 1, 1



class TestLogin:
    def test_get_login_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_login_creates_annotator(self, client, app_db):
        resp = client.post("/login", data={"username": "alice"}, follow_redirects=False)
        assert resp.status_code == 302
        row = app_db.execute("SELECT * FROM annotators WHERE name='alice'").fetchone()
        assert row is not None

    def test_login_empty_name_rejected(self, client):
        resp = client.post("/login", data={"username": ""}, follow_redirects=True)
        assert resp.status_code == 200


def _setup_comparison_group(app_db, video_id, annotator_id, group_id="test-group"):
    """Create a comparison group and a grouped assignment."""
    app_db.execute(
        "INSERT OR IGNORE INTO comparison_groups (id, prompt, physical_laws) "
        """VALUES (?, 'test prompt', '["flow_dynamics"]')""",
        (group_id,),
    )
    app_db.execute(
        "INSERT INTO assignments (video_id, annotator_id, status, expires_at, group_id) "
        "VALUES (?, ?, 'assigned', datetime('now', '+24 hours'), ?)",
        (video_id, annotator_id, group_id),
    )
    app_db.commit()


class TestTaskList:
    def test_requires_login(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 302

    def test_shows_assignments(self, client, app_db):
        _setup_user_and_video(app_db)
        with client.session_transaction() as sess:
            sess["annotator_id"] = 1
            sess["annotator_name"] = "alice"
        _setup_comparison_group(app_db, 1, 1)
        resp = client.get("/tasks", follow_redirects=True)
        assert resp.status_code == 200


class TestStats:
    def test_stats_returns_json(self, client, app_db):
        with client.session_transaction() as sess:
            sess["annotator_id"] = 1
            sess["annotator_name"] = "alice"
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_videos" in data
        assert "coverage" in data
        assert "per_annotator" in data


class TestEndToEnd:
    def test_full_flow_comparison(self, client, app_db):
        """Login → create comparison group → rate group → verify annotations."""
        # Insert two test videos (same prompt, different models)
        app_db.execute(
            "INSERT INTO videos (filename, dataset, prompt, physical_laws, import_hash) "
            """VALUES ('v1.mp4', 'model-a-wmb', 'water flows', '["flow_dynamics", "fluid_continuity"]', 'hash')"""
        )
        app_db.execute(
            "INSERT INTO videos (filename, dataset, prompt, physical_laws, import_hash) "
            """VALUES ('v1.mp4', 'model-b-wmb', 'water flows', '["flow_dynamics", "fluid_continuity"]', 'hash')"""
        )
        app_db.commit()

        # Login
        resp = client.post("/login", data={"username": "tester"}, follow_redirects=True)
        assert resp.status_code == 200

        # Manually create a comparison group (auto-assign won't match test models)
        group_id = "test-e2e-group"
        app_db.execute(
            "INSERT INTO comparison_groups (id, prompt, physical_laws) VALUES (?, ?, ?)",
            (group_id, "water flows", '["flow_dynamics", "fluid_continuity"]'),
        )
        annotator_id = app_db.execute("SELECT id FROM annotators WHERE name='tester'").fetchone()["id"]
        for vid in [1, 2]:
            app_db.execute(
                "INSERT INTO assignments (video_id, annotator_id, status, expires_at, group_id) "
                "VALUES (?, ?, 'assigned', datetime('now', '+24 hours'), ?)",
                (vid, annotator_id, group_id),
            )
        app_db.commit()

        # Get rate group page
        resp = client.get(f"/rate_group/{group_id}")
        assert resp.status_code == 200

        # Submit scores for both videos
        assignments = app_db.execute(
            "SELECT id FROM assignments WHERE group_id=? ORDER BY id",
            (group_id,),
        ).fetchall()

        scores = {}
        for a in assignments:
            aid = a["id"]
            scores[f"v{aid}_SA"] = "4"
            scores[f"v{aid}_PTV"] = "4"
            scores[f"v{aid}_persistence"] = "3"
            scores[f"v{aid}_flow_dynamics"] = "4"
            scores[f"v{aid}_fluid_continuity"] = "2"

        scores["action"] = "next"
        resp = client.post(f"/rate_group/{group_id}", data=scores, follow_redirects=False)
        assert resp.status_code == 200
        resp_data = resp.get_json()
        assert resp_data["ok"] is True

        # Verify annotations created for both
        for a in assignments:
            ann = app_db.execute(
                "SELECT * FROM annotations WHERE assignment_id=?", (a["id"],)
            ).fetchone()
            assert ann is not None
            parsed = json.loads(ann["scores_json"])
            assert parsed["general"]["SA"] == 4
            assert parsed["physical"]["flow_dynamics"] == 4

        # Verify all assignments marked completed
        statuses = app_db.execute(
            "SELECT status FROM assignments WHERE group_id=?", (group_id,)
        ).fetchall()
        assert all(r["status"] == STATUS_COMPLETED for r in statuses)
