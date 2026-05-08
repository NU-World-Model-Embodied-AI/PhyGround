import pytest


def _make_eval_json(results, judge="gemini"):
    return {
        "schema_version": "v2.0",
        "judge": judge,
        "results": results,
    }


def _make_result(video, domain, prompt, scores=None):
    base = {"video": video, "domain": domain, "prompt": prompt}
    if scores:
        base.update(scores)
    return base


class TestImportHash:
    def test_deterministic(self, tmp_path):
        from human_eval.import_videos import compute_import_hash
        (tmp_path / "ds1").mkdir()
        (tmp_path / "ds1" / "a.mp4").touch()
        (tmp_path / "ds1" / "b.mp4").touch()
        h1 = compute_import_hash(tmp_path)
        h2 = compute_import_hash(tmp_path)
        assert h1 == h2

    def test_changes_with_files(self, tmp_path):
        from human_eval.import_videos import compute_import_hash
        (tmp_path / "ds1").mkdir()
        (tmp_path / "ds1" / "a.mp4").touch()
        h1 = compute_import_hash(tmp_path)
        (tmp_path / "ds1" / "b.mp4").touch()
        h2 = compute_import_hash(tmp_path)
        assert h1 != h2


class TestDifficultyScore:
    def test_computes_mean_abs_diff(self):
        from human_eval.import_videos import compute_difficulty_score
        gemini = {"SA": 4, "PTV": 3, "persistence": 4}
        qwen = {"SA": 2, "PTV": 1, "persistence": 2}
        score = compute_difficulty_score(gemini, qwen)
        assert abs(score - 6 / 3) < 1e-6

    def test_returns_none_when_missing_keys(self):
        from human_eval.import_videos import compute_difficulty_score
        assert compute_difficulty_score({"SA": 1}, {}) is None


class TestImportVideos:
    def test_importer_disabled_without_prompt_selection_json(self, db, tmp_path):
        from human_eval.import_videos import import_videos
        with pytest.raises(RuntimeError, match="prompt-selection JSON"):
            import_videos(db, tmp_path / "videos")
