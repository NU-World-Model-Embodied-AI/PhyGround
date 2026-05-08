import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

N_ANNOTATORS_PER_VIDEO = 3
BATCH_SIZE_PER_USER = 4  # default fallback

# Per-cohort batch size (number of comparison pages per user)
COHORT_BATCH_SIZE: dict[str, int] = {
    "cohort_a":      4,
    "cohort_b":     11,
    "cohort_c":      4,
    "cohort_d":      4,
    "test":          2,   # Test cohort — excluded from dashboard/stats
    "others":      100,   # Default for unspecified cohort
}

# Expected headcount per cohort (for dashboard display)
COHORT_EXPECTED: dict[str, int] = {
    "cohort_a": 285,
    "cohort_b":  35,
    "cohort_d":  80,
}

# Fixed completion codes per cohort (not randomly generated)
COHORT_COMPLETION_CODE: dict[str, str] = {
    "cohort_c": "COMPLETION",
    "cohort_d": "COMPLETION",
}

# Cohorts to exclude from dashboard statistics and exports
TEST_COHORTS = {"test"}

VALID_COHORTS = set(COHORT_BATCH_SIZE.keys())

def get_batch_size(cohort: str | None) -> int:
    """Return batch size for a given cohort, falling back to default."""
    return COHORT_BATCH_SIZE.get(cohort or "others", BATCH_SIZE_PER_USER)
ASSIGNMENT_TTL_HOURS = 24
VIDEO_DATA_DIR = BASE_DIR / "../../data/videos"
DB_PATH = BASE_DIR / "human_eval.db"
TEST_DB_PATH = BASE_DIR / "human_eval_test.db"
DISAGREEMENT_TOP_K = 20
SECRET_KEY = os.environ.get("HUMAN_EVAL_SECRET_KEY", "human-eval-local-dev-key")

# Assignment status constants
STATUS_ASSIGNED = "assigned"
STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_SKIPPED = "skipped"

# ---------------------------------------------------------------------------
# Comparison mode: multi-model side-by-side evaluation
# ---------------------------------------------------------------------------
# Model prefixes — must match the prefix part of dataset directory names
# e.g. "ltx-2-19b-distilled-fp8" matches "ltx-2-19b-distilled-fp8-openvid", etc.
COMPARISON_MODELS = [
    "wan2.2-ti2v-5b",
    "ltx-2-19b-dev",
    "cosmos-predict2.5-2b",
    "cosmos-predict2.5-14b",
    "veo-3.1",
    "wan2.2-i2v-a14b",
    "omniweaving",
    "ltx-2.3-22b-dev",
]

# How many models to show per comparison group (randomly sampled from COMPARISON_MODELS)
MODELS_PER_GROUP = 3

SOURCE_DATASETS = {"wmb", "openvid", "physics_iq", "video_phy_2"}

# Redirect URL shown after user completes all assigned groups
COMPLETION_SURVEY_URL = "https://example.com/survey"


def extract_model(dataset_name: str) -> str:
    """Extract model prefix from a dataset directory name like 'ltx-2-19b-distilled-fp8-openvid' → 'ltx-2-19b-distilled-fp8'."""
    for src in sorted(SOURCE_DATASETS, key=len, reverse=True):
        suffix = f"-{src}"
        if dataset_name.endswith(suffix):
            return dataset_name[: -len(suffix)]
    return dataset_name
