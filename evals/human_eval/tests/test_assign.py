import sqlite3
import pytest


def _insert_annotator(db, name):
    db.execute("INSERT INTO annotators (name) VALUES (?)", (name,))
    db.commit()
    return db.execute("SELECT id FROM annotators WHERE name=?", (name,)).fetchone()["id"]


def _insert_video(db, filename, dataset="ds", prompt="p", physical_laws='["fluid"]', difficulty=None):
    db.execute(
        "INSERT INTO videos (filename, dataset, prompt, physical_laws, difficulty_score, import_hash) "
        "VALUES (?, ?, ?, ?, ?, 'hash')",
        (filename, dataset, prompt, physical_laws, difficulty),
    )
    db.commit()
    return db.execute(
        "SELECT id FROM videos WHERE filename=?", (filename,)
    ).fetchone()["id"]


def _insert_assignment(db, video_id, annotator_id, status="assigned", hours_offset=1):
    db.execute(
        "INSERT INTO assignments (video_id, annotator_id, status, expires_at) "
        "VALUES (?, ?, ?, datetime('now', ? || ' hours'))",
        (video_id, annotator_id, status, f"+{hours_offset}" if hours_offset >= 0 else str(hours_offset)),
    )
    db.commit()


def _insert_comparison_prompt(
    db,
    prompt,
    models=("model-a", "model-b"),
    physical_laws='["fluid"]',
):
    video_ids = []
    for model in models:
        video_ids.append(
            _insert_video(
                db,
                f"{prompt}-{model}.mp4",
                dataset=f"{model}-wmb",
                prompt=prompt,
                physical_laws=physical_laws,
            )
        )
    return video_ids


def _insert_comparison_group(db, group_id, prompt, physical_laws, video_ids, annotator_id, hours_offset=24):
    db.execute(
        "INSERT INTO comparison_groups (id, prompt, physical_laws) VALUES (?, ?, ?)",
        (group_id, prompt, physical_laws),
    )
    for video_id in video_ids:
        db.execute(
            "INSERT INTO assignments (video_id, annotator_id, status, expires_at, group_id) "
            "VALUES (?, ?, 'assigned', datetime('now', ? || ' hours'), ?)",
            (video_id, annotator_id, f"+{hours_offset}" if hours_offset >= 0 else str(hours_offset), group_id),
        )
    db.commit()



class TestAssignComparisonBatch:
    def test_expired_group_cleanup_deletes_saved_drafts_before_reassignment(self, db):
        from human_eval.assign import assign_comparison_batch

        uid = _insert_annotator(db, "alice")
        video_ids = _insert_comparison_prompt(db, "prompt-1")

        _insert_comparison_group(
            db,
            "expired-group",
            "prompt-1",
            '["fluid"]',
            video_ids,
            uid,
            hours_offset=-1,
        )

        expired_aid = db.execute(
            "SELECT id FROM assignments WHERE group_id = 'expired-group' ORDER BY id LIMIT 1"
        ).fetchone()["id"]
        db.execute(
            "INSERT INTO annotations (assignment_id, scores_json) VALUES (?, '{}')",
            (expired_aid,),
        )
        annotation_id = db.execute(
            "SELECT id FROM annotations WHERE assignment_id = ?",
            (expired_aid,),
        ).fetchone()["id"]
        db.execute(
            "INSERT INTO annotation_items (annotation_id, dimension, law, score) VALUES (?, ?, ?, ?)",
            (annotation_id, "SA", None, 4),
        )
        db.commit()

        new_groups = assign_comparison_batch(
            db,
            uid,
            n_annotators=3,
            batch_size=1,
            ttl_hours=24,
            models=["model-a", "model-b"],
            models_per_group=2,
        )

        assert len(new_groups) == 1
        assert db.execute(
            "SELECT 1 FROM comparison_groups WHERE id = 'expired-group'",
        ).fetchone() is None
        assert db.execute(
            "SELECT COUNT(*) AS c FROM annotations WHERE assignment_id = ?",
            (expired_aid,),
        ).fetchone()["c"] == 0
        assert db.execute(
            "SELECT COUNT(*) AS c FROM annotation_items WHERE annotation_id = ?",
            (annotation_id,),
        ).fetchone()["c"] == 0

    def test_prompt_stays_assignable_until_each_model_reaches_target_coverage(self, db):
        from human_eval.assign import assign_comparison_batch

        uid_a = _insert_annotator(db, "alice")
        uid_b = _insert_annotator(db, "bob")
        uid_c = _insert_annotator(db, "carol")
        uid_d = _insert_annotator(db, "dave")

        video_ids = _insert_comparison_prompt(
            db,
            "prompt-1",
            models=("model-a", "model-b", "model-c", "model-d"),
        )
        covered_video_ids = video_ids[:3]

        for idx, uid in enumerate((uid_a, uid_b, uid_c), start=1):
            group_id = f"completed-{idx}"
            _insert_comparison_group(
                db,
                group_id,
                "prompt-1",
                '["fluid"]',
                covered_video_ids,
                uid,
            )
            db.execute(
                "UPDATE assignments SET status = 'completed' WHERE group_id = ?",
                (group_id,),
            )
        db.commit()

        new_groups = assign_comparison_batch(
            db,
            uid_d,
            n_annotators=3,
            batch_size=1,
            ttl_hours=24,
            models=["model-a", "model-b", "model-c", "model-d"],
            models_per_group=3,
        )

        assert len(new_groups) == 1
        datasets = {
            row["dataset"]
            for row in db.execute(
                "SELECT v.dataset FROM assignments a "
                "JOIN videos v ON a.video_id = v.id "
                "WHERE a.group_id = ?",
                (new_groups[0],),
            ).fetchall()
        }
        assert "model-d-wmb" in datasets

    def test_repeated_call_keeps_pending_group_when_no_new_candidates(self, db):
        from human_eval.assign import assign_comparison_batch

        uid = _insert_annotator(db, "alice")
        video_ids = _insert_comparison_prompt(db, "prompt-1")

        first_groups = assign_comparison_batch(
            db,
            uid,
            n_annotators=3,
            batch_size=1,
            ttl_hours=24,
            models=["model-a", "model-b"],
            models_per_group=2,
        )

        assert len(first_groups) == 1
        group_id = first_groups[0]

        second_groups = assign_comparison_batch(
            db,
            uid,
            n_annotators=3,
            batch_size=1,
            ttl_hours=24,
            models=["model-a", "model-b"],
            models_per_group=2,
        )

        assert second_groups == []
        assert db.execute(
            "SELECT COUNT(*) AS c FROM assignments WHERE group_id = ?",
            (group_id,),
        ).fetchone()["c"] == len(video_ids)
        assert db.execute(
            "SELECT 1 FROM comparison_groups WHERE id = ?",
            (group_id,),
        ).fetchone() is not None

    def test_new_group_assignment_does_not_delete_existing_pending_group(self, db):
        from human_eval.assign import assign_comparison_batch

        uid = _insert_annotator(db, "alice")
        prompt_one_videos = _insert_comparison_prompt(db, "prompt-1")
        _insert_comparison_prompt(db, "prompt-2")

        _insert_comparison_group(
            db,
            "existing-group",
            "prompt-1",
            '["fluid"]',
            prompt_one_videos,
            uid,
        )

        new_groups = assign_comparison_batch(
            db,
            uid,
            n_annotators=3,
            batch_size=1,
            ttl_hours=24,
            models=["model-a", "model-b"],
            models_per_group=2,
        )

        assert len(new_groups) == 1
        assert new_groups[0] != "existing-group"
        assert db.execute(
            "SELECT COUNT(*) AS c FROM assignments WHERE group_id = 'existing-group'",
        ).fetchone()["c"] == len(prompt_one_videos)
