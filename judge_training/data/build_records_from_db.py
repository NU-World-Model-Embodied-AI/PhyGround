"""Extract human-labeled annotations from SQLite into structured records.

Reads evals/human_eval/human_eval_filtered.db, aggregates per-dimension and
per-law scores via median (ties rounded down), excludes N/A physical laws.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

from evals.human_eval.config import COMPARISON_MODELS, SOURCE_DATASETS
from evals.prompts import GENERAL_DIMS

_COMPARISON_MODEL_SET = set(COMPARISON_MODELS)
_SOURCE_DATASETS_BY_LEN = sorted(SOURCE_DATASETS, key=len, reverse=True)


def extract_model_and_source(dataset: str) -> tuple[str, str]:
    """Extract (video_model, source_dataset) from the DB dataset field."""
    for suffix in _SOURCE_DATASETS_BY_LEN:
        if dataset.endswith("-" + suffix):
            model = dataset[: -(len(suffix) + 1)]
            return model, suffix
    return dataset, "unknown"


def median_round_down(values: list[int]) -> int:
    """Compute median, rounding .5 values down (floor)."""
    if not values:
        raise ValueError("Cannot compute median of empty list")
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) // 2


def _parse_na_laws(raw: str | None) -> set[str]:
    if not raw:
        return set()
    try:
        return set(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return set()


def build_records(
    db_path: str,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    """Build training records from human_eval_filtered.db.

    Returns one record per video with aggregated scores across annotators.
    Scores are aggregated via median (ties rounded down).
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    rows = conn.execute("""
        SELECT
            v.id AS video_id,
            v.filename,
            v.dataset,
            v.prompt,
            v.physical_laws,
            a.group_id,
            a2.id AS annotation_id,
            a2.na_laws,
            ai.dimension,
            ai.law,
            ai.score
        FROM videos v
        JOIN assignments a ON a.video_id = v.id AND a.status = 'completed'
        JOIN annotations a2 ON a2.assignment_id = a.id
        JOIN annotation_items ai ON ai.annotation_id = a2.id
        ORDER BY v.id, a2.id
    """).fetchall()

    if not rows:
        logger.warning("No completed annotations found in %s", db_path)
        return []

    videos: dict[int, dict[str, Any]] = {}
    annotation_na_laws: dict[int, set[str]] = {}
    skipped_videos: set[int] = set()

    for row in rows:
        vid = row["video_id"]
        if vid in skipped_videos:
            continue

        ann_id = row["annotation_id"]

        if vid not in videos:
            model, source = extract_model_and_source(row["dataset"])
            if active_only and model not in _COMPARISON_MODEL_SET:
                skipped_videos.add(vid)
                continue
            videos[vid] = {
                "video_id": vid,
                "filename": row["filename"],
                "dataset": row["dataset"],
                "video_model": model,
                "source_dataset": source,
                "prompt": row["prompt"],
                "physical_laws": json.loads(row["physical_laws"]) if row["physical_laws"] else [],
                "annotation_ids": set(),
                "general_scores": defaultdict(list),
                "physical_scores": defaultdict(list),
            }
        vdata = videos[vid]

        vdata["annotation_ids"].add(ann_id)

        if ann_id not in annotation_na_laws:
            annotation_na_laws[ann_id] = _parse_na_laws(row["na_laws"])

        dim = row["dimension"]
        law = row["law"]
        score = row["score"]

        if dim in GENERAL_DIMS and law is None:
            vdata["general_scores"][dim].append(score)
        elif law is not None:
            if law not in annotation_na_laws.get(ann_id, set()):
                vdata["physical_scores"][law].append(score)

    conn.close()

    records = []
    for vid, vdata in videos.items():
        vote_count = len(vdata["annotation_ids"])
        if vote_count == 0:
            continue

        general = {}
        for dim in GENERAL_DIMS:
            scores = vdata["general_scores"].get(dim, [])
            if scores:
                general[dim] = median_round_down(scores)

        physical = {}
        for law, scores in vdata["physical_scores"].items():
            if scores:
                physical[law] = median_round_down(scores)

        video_path = os.path.join("data", "videos",
                                  vdata["video_model"], vdata["filename"])

        record = {
            "video_path": video_path,
            "prompt": vdata["prompt"],
            "video_model": vdata["video_model"],
            "source_dataset": vdata["source_dataset"],
            "dataset": vdata["dataset"],
            "physical_laws": vdata["physical_laws"],
            "vote_count": vote_count,
            "general_scores": general,
            "physical_scores": physical,
        }
        records.append(record)

    return records


def split_by_prompt_and_model(
    records: list[dict[str, Any]],
    holdout_model: str,
    holdout_prompt_ratio: float,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Double holdout: leave out prompts AND a model.

    Returns dict with keys: train, test_prompt, test_model, test_both.
    """
    if not 0 < holdout_prompt_ratio < 1:
        raise ValueError(
            f"--holdout-prompt-ratio must be between 0 and 1, got: {holdout_prompt_ratio}"
        )

    prompts = sorted({r["prompt"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(prompts)
    n_holdout = max(1, int(len(prompts) * holdout_prompt_ratio))
    holdout_prompts = set(prompts[:n_holdout])

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "test_prompt": [],
        "test_model": [],
        "test_both": [],
    }
    for rec in records:
        p_held = rec["prompt"] in holdout_prompts
        m_held = rec["video_model"] == holdout_model
        if p_held and m_held:
            splits["test_both"].append(rec)
        elif p_held:
            splits["test_prompt"].append(rec)
        elif m_held:
            splits["test_model"].append(rec)
        else:
            splits["train"].append(rec)

    return splits
