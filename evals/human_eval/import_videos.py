import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from human_eval.config import COMPARISON_MODELS, STATUS_SKIPPED

GENERAL_KEYS = ["SA", "PTV", "persistence"]


def compute_import_hash(video_data_dir: Path) -> str:
    """Global hash across all datasets (used by initial import guard)."""
    paths = sorted(str(p.relative_to(video_data_dir)) for p in video_data_dir.rglob("*.mp4"))
    return hashlib.md5("\n".join(paths).encode()).hexdigest()


def compute_dataset_hash(dataset_dir: Path) -> str:
    """Per-dataset hash — only includes mp4 filenames within one dataset."""
    paths = sorted(p.name for p in dataset_dir.glob("*.mp4"))
    return hashlib.md5("\n".join(paths).encode()).hexdigest()


def compute_difficulty_score(gemini_scores: dict, qwen_scores: dict) -> float | None:
    diffs = []
    for key in GENERAL_KEYS:
        g, q = gemini_scores.get(key), qwen_scores.get(key)
        if g is None or q is None:
            return None
        diffs.append(abs(g - q))
    return sum(diffs) / len(diffs) if diffs else None


def _load_latest_eval(dataset_dir: Path, prefix: str) -> dict | None:
    files = sorted(dataset_dir.glob(f"eval_{prefix}_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def _build_scores_lookup(eval_data: dict | None) -> dict:
    if not eval_data or "results" not in eval_data:
        return {}
    lookup = {}
    for r in eval_data["results"]:
        scores = {k: r[k] for k in GENERAL_KEYS if k in r}
        lookup[r["video"]] = scores
    return lookup




_DATASET_SUFFIXES = ("openvid", "video_phy_2", "physics_iq", "wmb")


def _ds_suffix(db_dataset: str) -> str:
    """Extract source dataset suffix from DB dataset name, e.g. 'veo-3.1-video_phy_2' -> 'video_phy_2'."""
    for suffix in _DATASET_SUFFIXES:
        if db_dataset.endswith(suffix):
            return suffix
    return db_dataset


class _EvalLookupCache:
    """Caches per-dataset eval score lookups and dataset hashes."""

    def __init__(self):
        self._eval: dict[str, tuple[dict, dict]] = {}
        self._hash: dict[str, str] = {}

    def get_scores(self, ds_name: str, ds_dir: Path) -> tuple[dict, dict]:
        if ds_name not in self._eval:
            self._eval[ds_name] = (
                _build_scores_lookup(_load_latest_eval(ds_dir, "gemini")),
                _build_scores_lookup(_load_latest_eval(ds_dir, "qwen")),
            )
        return self._eval[ds_name]

    def get_hash(self, ds_name: str, ds_dir: Path) -> str:
        if ds_name not in self._hash:
            self._hash[ds_name] = compute_dataset_hash(ds_dir) if ds_dir.exists() else ""
        return self._hash[ds_name]


def import_videos(conn, video_data_dir: Path):
    """Import videos into the human-eval DB.

    This release omits the prompt-selection JSON consumed by the original
    importer, so the importer entry point is intentionally disabled.
    """
    raise RuntimeError(
        "import_videos is not included in this release because the prompt-selection "
        "JSON is omitted. Use the companion dataset metadata to build a DB import."
    )

