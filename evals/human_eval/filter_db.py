"""Filter human_eval.db → human_eval_filtered.db by removing low-quality annotators.

Implements the cleaning rules documented in:
  docs/exp-results/eval/humaneval/cleandata.md

Rules (require multiple signals to converge):
  1. Constant scores (std=0): all dimensions get the same score regardless of video
  2. Near-constant (std<0.3): almost all scores identical
  3. 100% copy-paste: every video has all dimensions scored identically
  4. High copy-paste (≥75%) + behavioral anomaly (median stay <30s or never plays)
  5. Tiny volume: <5 annotation items or median stay <10s
  6. Extreme Peer MAE (>1.8) + behavioral anomaly (median stay <30s or never plays
     or copy-paste ≥50%)

Usage:
  python -m eval.human_eval.filter_db
  python -m eval.human_eval.filter_db --dry-run
  python -m eval.human_eval.filter_db --input evals/human_eval/human_eval.db \
      --output evals/human_eval/human_eval_filtered.db
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_INPUT = "evals/human_eval/human_eval.db"
DEFAULT_OUTPUT = "evals/human_eval/human_eval_filtered.db"


def _compute_annotator_metrics(conn: sqlite3.Connection) -> dict[int, dict]:
    """Compute per-annotator quality metrics."""
    rows = conn.execute("""
        SELECT
            a.annotator_id,
            ann.id AS annotation_id,
            ann.play_count,
            ann.stay_seconds,
            ai.dimension,
            ai.law,
            ai.score,
            a.video_id
        FROM assignments a
        JOIN annotations ann ON ann.assignment_id = a.id
        JOIN annotation_items ai ON ai.annotation_id = ann.id
        WHERE a.status = 'completed'
        ORDER BY a.annotator_id, a.video_id
    """).fetchall()

    # Group by annotator
    annotators: dict[int, dict] = {}
    for row in rows:
        aid = row[0]
        if aid not in annotators:
            annotators[aid] = {
                "all_scores": [],
                "videos": defaultdict(list),  # video_id -> [scores]
                "stay_seconds": [],
                "play_counts": [],
                "annotation_ids": set(),
                "item_count": 0,
            }
        m = annotators[aid]
        m["all_scores"].append(row[6])
        m["videos"][row[7]].append(row[6])
        m["item_count"] += 1
        ann_id = row[1]
        if ann_id not in m["annotation_ids"]:
            m["annotation_ids"].add(ann_id)
            if row[2] is not None:
                m["play_counts"].append(row[2])
            if row[3] is not None:
                m["stay_seconds"].append(row[3])

    # Compute metrics
    for aid, m in annotators.items():
        scores = m["all_scores"]
        m["overall_std"] = statistics.pstdev(scores) if len(scores) > 1 else 0.0

        # Copy-paste rate: fraction of videos where all dims got the same score
        cp_count = 0
        total_videos = 0
        for vid, vscores in m["videos"].items():
            if len(vscores) > 1:
                total_videos += 1
                if len(set(vscores)) == 1:
                    cp_count += 1
        m["copy_paste_rate"] = cp_count / total_videos if total_videos > 0 else 0.0
        m["video_count"] = total_videos

        m["median_stay"] = statistics.median(m["stay_seconds"]) if m["stay_seconds"] else 0.0
        m["median_play"] = statistics.median(m["play_counts"]) if m["play_counts"] else 0.0
        m["max_play"] = max(m["play_counts"]) if m["play_counts"] else 0
        n_annotations = len(m["annotation_ids"])
        zero_play = sum(1 for p in m["play_counts"] if p == 0)
        m["zero_play_rate"] = zero_play / n_annotations if n_annotations > 0 else 0.0

    return annotators


def _compute_peer_mae(conn: sqlite3.Connection, annotator_ids: set[int]) -> dict[int, float]:
    """Compute per-annotator Peer MAE: mean |score - peer_score| on shared video-dimensions."""
    rows = conn.execute("""
        SELECT
            a.annotator_id,
            a.video_id,
            ai.dimension,
            ai.law,
            ai.score
        FROM assignments a
        JOIN annotations ann ON ann.assignment_id = a.id
        JOIN annotation_items ai ON ai.annotation_id = ann.id
        WHERE a.status = 'completed'
        ORDER BY a.video_id, ai.dimension, ai.law
    """).fetchall()

    # Group scores by (video_id, dimension, law)
    dim_groups: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        aid, vid, dim, law, score = row
        if aid not in annotator_ids:
            continue
        key = (vid, dim, law)
        dim_groups[key].append((aid, score))

    # Compute pairwise MAE per annotator
    annotator_diffs: dict[int, list[int]] = defaultdict(list)
    for key, entries in dim_groups.items():
        if len(entries) < 2:
            continue
        for i, (aid_i, score_i) in enumerate(entries):
            for j, (aid_j, score_j) in enumerate(entries):
                if i < j:
                    diff = abs(score_i - score_j)
                    annotator_diffs[aid_i].append(diff)
                    annotator_diffs[aid_j].append(diff)

    return {
        aid: statistics.mean(diffs) if diffs else 0.0
        for aid, diffs in annotator_diffs.items()
    }


def identify_removals(
    metrics: dict[int, dict],
    peer_mae: dict[int, float],
) -> dict[int, str]:
    """Return {annotator_id: reason} for annotators to remove."""
    removals: dict[int, str] = {}

    for aid, m in metrics.items():
        # Rule 1: constant scores (std=0)
        if m["item_count"] >= 5 and m["overall_std"] == 0.0:
            removals[aid] = "constant_scores (std=0)"
            continue

        # Rule 2: near-constant (std<0.3)
        if m["item_count"] >= 5 and m["overall_std"] < 0.3:
            removals[aid] = f"near_constant (std={m['overall_std']:.2f})"
            continue

        # Rule 5: tiny volume (<5 items or median stay <10s)
        if m["item_count"] < 5:
            removals[aid] = f"tiny_volume ({m['item_count']} items)"
            continue
        if m["median_stay"] < 10.0 and m["item_count"] < 10:
            removals[aid] = f"tiny_volume (stay={m['median_stay']:.0f}s, {m['item_count']} items)"
            continue

        # Rule 3: 100% copy-paste (with enough videos)
        if m["video_count"] >= 3 and m["copy_paste_rate"] >= 1.0:
            removals[aid] = "100%_copy_paste"
            continue

        # Rule 4: high copy-paste + behavioral anomaly
        behavioral_anomaly = m["median_stay"] < 30.0 or m["max_play"] == 0
        if m["copy_paste_rate"] >= 0.75 and behavioral_anomaly:
            removals[aid] = (
                f"high_copy_paste ({m['copy_paste_rate']:.0%}) "
                f"+ behavioral (stay={m['median_stay']:.0f}s, max_play={m['max_play']})"
            )
            continue

        # Rule 6: extreme Peer MAE + behavioral anomaly
        pmae = peer_mae.get(aid, 0.0)
        if pmae > 1.8 and (behavioral_anomaly or m["copy_paste_rate"] >= 0.50):
            removals[aid] = (
                f"high_peer_mae ({pmae:.2f}) "
                f"+ behavioral (stay={m['median_stay']:.0f}s, cp={m['copy_paste_rate']:.0%})"
            )
            continue

        # Rule 7: never played video + short stay
        if m["zero_play_rate"] >= 1.0 and m["median_stay"] < 30.0:
            removals[aid] = (
                f"never_played (zero_play=100%) "
                f"+ short_stay (stay={m['median_stay']:.0f}s)"
            )
            continue

    return removals


def apply_filter(input_db: str, output_db: str, removals: dict[int, str]) -> dict:
    """Copy input_db to output_db and delete data for removed annotators."""
    shutil.copy2(input_db, output_db)
    conn = sqlite3.connect(output_db)
    conn.execute("PRAGMA foreign_keys = OFF")

    removed_ids = list(removals.keys())
    if not removed_ids:
        conn.close()
        return {"removed": 0}

    placeholders = ",".join("?" * len(removed_ids))

    # Get assignment IDs to remove
    assignment_ids = [
        r[0] for r in conn.execute(
            f"SELECT id FROM assignments WHERE annotator_id IN ({placeholders})",
            removed_ids,
        ).fetchall()
    ]

    if assignment_ids:
        a_ph = ",".join("?" * len(assignment_ids))

        # Get annotation IDs
        annotation_ids = [
            r[0] for r in conn.execute(
                f"SELECT id FROM annotations WHERE assignment_id IN ({a_ph})",
                assignment_ids,
            ).fetchall()
        ]

        if annotation_ids:
            ann_ph = ",".join("?" * len(annotation_ids))
            conn.execute(
                f"DELETE FROM annotation_items WHERE annotation_id IN ({ann_ph})",
                annotation_ids,
            )
            conn.execute(
                f"DELETE FROM annotations WHERE id IN ({ann_ph})",
                annotation_ids,
            )

        conn.execute(
            f"DELETE FROM assignments WHERE id IN ({a_ph})",
            assignment_ids,
        )

    conn.execute(
        f"DELETE FROM annotators WHERE id IN ({placeholders})",
        removed_ids,
    )

    # Clean up orphaned comparison_groups
    conn.execute("""
        DELETE FROM comparison_groups
        WHERE id NOT IN (SELECT DISTINCT group_id FROM assignments WHERE group_id IS NOT NULL)
    """)

    conn.commit()

    # Collect stats
    stats = {}
    for table in ["annotators", "videos", "assignments", "annotations", "annotation_items", "comparison_groups"]:
        stats[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()

    return stats


def _print_removal_summary(removals: dict[int, str], conn: sqlite3.Connection) -> None:
    """Print summary of annotators to be removed."""
    if not removals:
        print("No annotators to remove.")
        return

    # Get names
    ids = list(removals.keys())
    ph = ",".join("?" * len(ids))
    name_map = {
        r[0]: r[1]
        for r in conn.execute(
            f"SELECT id, name FROM annotators WHERE id IN ({ph})", ids
        ).fetchall()
    }

    # Group by reason category
    by_category: dict[str, list[str]] = defaultdict(list)
    for aid, reason in removals.items():
        cat = reason.split("(")[0].strip().split("+")[0].strip()
        by_category[cat].append(f"  {name_map.get(aid, '?')} (id={aid}): {reason}")

    print(f"\n=== Annotators to remove: {len(removals)} ===")
    for cat, entries in sorted(by_category.items()):
        print(f"\n{cat} ({len(entries)}):")
        for e in sorted(entries):
            print(e)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Filter human_eval.db")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input DB path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output DB path")
    parser.add_argument("--dry-run", action="store_true", help="Show removals without writing")
    args = parser.parse_args(argv)

    if not Path(args.input).exists():
        logger.error("Input DB not found: %s", args.input)
        return 1

    conn = sqlite3.connect(args.input, timeout=30)

    print("Computing annotator metrics...")
    metrics = _compute_annotator_metrics(conn)
    print(f"  {len(metrics)} annotators with completed annotations")

    print("Computing Peer MAE...")
    peer_mae = _compute_peer_mae(conn, set(metrics.keys()))

    removals = identify_removals(metrics, peer_mae)
    _print_removal_summary(removals, conn)

    if args.dry_run:
        conn.close()
        print("\n[DRY RUN] No output written.")
        return 0

    # Sync excluded_annotators table in the main DB so assignment can
    # skip bad annotators without needing the filtered DB.
    conn.execute("CREATE TABLE IF NOT EXISTS excluded_annotators ("
                 "annotator_id INTEGER PRIMARY KEY, reason TEXT NOT NULL, "
                 "generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("DELETE FROM excluded_annotators")
    for aid, reason in removals.items():
        conn.execute(
            "INSERT INTO excluded_annotators (annotator_id, reason) VALUES (?, ?)",
            (aid, reason),
        )
    conn.commit()
    print(f"\nUpdated excluded_annotators in main DB: {len(removals)} entries")
    conn.close()

    print(f"Writing filtered DB to {args.output}...")
    stats = apply_filter(args.input, args.output, removals)
    print("Filtered DB stats:")
    for table, count in stats.items():
        print(f"  {table}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
