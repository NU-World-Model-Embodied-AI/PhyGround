import random
import sqlite3
import uuid

from human_eval.config import (
    STATUS_ASSIGNED, STATUS_COMPLETED, STATUS_PARTIAL, STATUS_SKIPPED,
    COMPARISON_MODELS, MODELS_PER_GROUP, extract_model,
)
from human_eval.db import pending_assignment_sql


def _delete_assignments_with_annotations(
    conn: sqlite3.Connection,
    where_sql: str,
    params: tuple,
) -> list[int]:
    """Delete assignments plus any saved draft rows that reference them.

    Also cleans up comparison_groups left with no remaining assignments.
    """
    rows = conn.execute(
        f"SELECT id, group_id FROM assignments WHERE {where_sql}",
        params,
    ).fetchall()
    assignment_ids = [row["id"] for row in rows]
    if not assignment_ids:
        return []

    affected_group_ids = list({
        row["group_id"] for row in rows if row["group_id"] is not None
    })

    placeholders = ",".join("?" * len(assignment_ids))
    conn.execute(
        f"""
        DELETE FROM annotation_items
        WHERE annotation_id IN (
            SELECT id FROM annotations WHERE assignment_id IN ({placeholders})
        )
        """,
        assignment_ids,
    )
    conn.execute(
        f"DELETE FROM annotations WHERE assignment_id IN ({placeholders})",
        assignment_ids,
    )
    conn.execute(
        f"DELETE FROM assignments WHERE id IN ({placeholders})",
        assignment_ids,
    )

    # Clean up comparison_groups that lost all their assignments
    if affected_group_ids:
        gph = ",".join("?" * len(affected_group_ids))
        conn.execute(
            f"""
            DELETE FROM comparison_groups
            WHERE id IN ({gph})
              AND NOT EXISTS (
                  SELECT 1 FROM assignments WHERE group_id = comparison_groups.id
              )
            """,
            affected_group_ids,
        )

    return assignment_ids


def _select_group_videos(
    model_videos: dict[str, dict],
    models: list[str],
    video_coverage: dict[int, int],
    k: int,
    n_annotators: int,
) -> tuple[list[dict], tuple[int, ...]] | None:
    """Pick the least-covered videos for a prompt, randomizing only on ties."""
    ranked = []
    for model in models:
        row = model_videos.get(model)
        if row is None:
            continue
        ranked.append((video_coverage.get(row["id"], 0), row))

    if len(ranked) < 2:
        return None
    if all(coverage >= n_annotators for coverage, _ in ranked):
        return None

    random.shuffle(ranked)
    ranked.sort(key=lambda item: item[0])
    picked = ranked[: min(k, len(ranked))]
    return [row for _, row in picked], tuple(coverage for coverage, _ in picked)



def _build_prompt_video_map(conn, models: list[str]) -> dict:
    """Build {prompt_text: {model: video_row}} mapping for comparison models.

    Only includes prompts that have at least 2 models available.
    """
    rows = conn.execute(
        "SELECT id, filename, dataset, prompt, physical_laws FROM videos"
    ).fetchall()

    model_set = set(models)
    # prompt → {model → video_row}
    prompt_map: dict[str, dict[str, dict]] = {}
    for r in rows:
        model = extract_model(r["dataset"])
        if model not in model_set:
            continue
        prompt = r["prompt"]
        if prompt not in prompt_map:
            prompt_map[prompt] = {}
        existing = prompt_map[prompt].get(model)
        if existing is None:
            prompt_map[prompt][model] = dict(r)
        elif "perspective-center" in r["filename"]:
            prompt_map[prompt][model] = dict(r)

    # Filter: keep only prompts with >= 2 models
    return {p: m for p, m in prompt_map.items() if len(m) >= 2}


def assign_comparison_batch(
    conn: sqlite3.Connection,
    annotator_id: int,
    n_annotators: int,
    batch_size: int,
    ttl_hours: int,
    models: list[str] | None = None,
    models_per_group: int | None = None,
) -> list[str]:
    """Assign comparison groups to an annotator.

    Each group = 1 prompt × K randomly-sampled models.
    Returns list of new group_ids.
    """
    models = models or COMPARISON_MODELS
    k = models_per_group or MODELS_PER_GROUP

    prompt_map = _build_prompt_video_map(conn, models)
    if not prompt_map:
        return []

    # Find prompts with active (completed/skipped or pending) groups for this user
    pending_sql = pending_assignment_sql("a")
    active_groups = conn.execute(
        f"""
        SELECT DISTINCT cg.prompt
        FROM comparison_groups cg
        JOIN assignments a ON a.group_id = cg.id
        WHERE a.annotator_id = ?
          AND (a.status IN ('{STATUS_COMPLETED}', '{STATUS_PARTIAL}', '{STATUS_SKIPPED}')
               OR {pending_sql})
        """,
        (annotator_id,),
    ).fetchall()
    active_prompts = {r["prompt"] for r in active_groups}

    # Count how many non-excluded annotators have completed each video.
    # excluded_annotators is populated by filter_db.py; if the table is
    # empty or missing, all annotators count (original behavior).
    has_exclusion_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='excluded_annotators'"
    ).fetchone() is not None

    if has_exclusion_table:
        coverage = conn.execute(
            f"""
            SELECT a.video_id, COUNT(DISTINCT a.annotator_id) AS cnt
            FROM assignments a
            WHERE a.status IN ('{STATUS_COMPLETED}', '{STATUS_PARTIAL}')
              AND a.annotator_id NOT IN (SELECT annotator_id FROM excluded_annotators)
            GROUP BY a.video_id
            """,
        ).fetchall()
    else:
        coverage = conn.execute(
            f"""
            SELECT a.video_id, COUNT(DISTINCT a.annotator_id) AS cnt
            FROM assignments a
            WHERE a.status IN ('{STATUS_COMPLETED}', '{STATUS_PARTIAL}')
            GROUP BY a.video_id
            """,
        ).fetchall()
    video_coverage = {r["video_id"]: r["cnt"] for r in coverage}

    # Filter to assignable prompts: not already active for this user and still
    # containing at least one under-covered video.
    candidates = []
    for prompt, model_videos in prompt_map.items():
        if prompt in active_prompts:
            continue

        selected = _select_group_videos(
            model_videos,
            models,
            video_coverage,
            k,
            n_annotators,
        )
        if selected is None:
            continue
        video_rows, coverage_signature = selected
        candidates.append((coverage_signature, random.random(), prompt, video_rows))

    # Sort: prompts with the lowest-covered selected videos first.
    candidates.sort(key=lambda x: (x[0], x[1]))

    conn.execute("BEGIN IMMEDIATE")
    try:
        # Clean up non-completed, non-skipped group assignments (expired/abandoned)
        # and their orphaned comparison_groups
        deleted_ids = _delete_assignments_with_annotations(
            conn,
            (
                "annotator_id = ? AND group_id IS NOT NULL "
                "AND status NOT IN (?, ?, ?) "
                "AND expires_at <= datetime('now')"
            ),
            (annotator_id, STATUS_COMPLETED, STATUS_PARTIAL, STATUS_SKIPPED),
        )
        new_group_ids = []
        for _, _, prompt, video_rows in candidates[:batch_size]:

            # Get physical_laws from the first video (same prompt = same laws)
            physical_laws = video_rows[0]["physical_laws"]

            # Create comparison group
            group_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO comparison_groups (id, prompt, physical_laws) VALUES (?, ?, ?)",
                (group_id, prompt, physical_laws),
            )

            # Create assignments for each video in the group
            for vr in video_rows:
                conn.execute(
                    "INSERT OR IGNORE INTO assignments (video_id, annotator_id, status, expires_at, group_id) "
                    f"VALUES (?, ?, '{STATUS_ASSIGNED}', datetime('now', '+' || ? || ' hours'), ?)",
                    (vr["id"], annotator_id, ttl_hours, group_id),
                )

            new_group_ids.append(group_id)

        conn.commit()
        return new_group_ids

    except Exception:
        conn.rollback()
        raise
