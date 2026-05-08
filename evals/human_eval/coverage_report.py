"""Report video coverage and how many people are needed to reach 3x.

Usage:
  python -m evals.human_eval.coverage_report
  python -m evals.human_eval.coverage_report --batch-size 11
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter

from evals.human_eval.config import (
    COMPARISON_MODELS,
    DB_PATH,
    N_ANNOTATORS_PER_VIDEO,
)


def _active_video_sql() -> str:
    likes = " OR ".join(f"v.dataset LIKE '{m}%'" for m in COMPARISON_MODELS)
    return f"({likes})"


def _query_coverage(conn: sqlite3.Connection, exclude_bad: bool) -> list[tuple[int, int]]:
    vf = _active_video_sql()
    ea = ""
    if exclude_bad:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='excluded_annotators'"
        ).fetchone()
        if has_table:
            ea = "AND a.annotator_id NOT IN (SELECT annotator_id FROM excluded_annotators)"

    return conn.execute(f"""
        SELECT v.id, COUNT(a.id) AS cnt
        FROM videos v
        LEFT JOIN assignments a ON v.id = a.video_id AND a.status IN ('completed', 'partial')
            AND a.annotator_id NOT IN (
                SELECT id FROM annotators WHERE COALESCE(cohort, 'others') IN ('test')
            )
            {ea}
        WHERE {vf}
        GROUP BY v.id
    """).fetchall()


def _print_report(label: str, rows: list[tuple[int, int]], target: int, batch_size: int) -> None:
    total = len(rows)
    dist: Counter[int] = Counter()
    deficit = 0
    for _, cnt in rows:
        bucket = min(cnt, target)
        dist[bucket] += 1
        if cnt < target:
            deficit += target - cnt

    print(f"=== {label} ({total} active videos) ===")
    for k in sorted(dist.keys()):
        print(f"  {k}x: {dist[k]} videos ({dist[k] / total * 100:.1f}%)")
    print(f"  缺少 annotations: {deficit}")
    print(f"  需要补人 (batch={batch_size}): {-(-deficit // batch_size)}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video coverage report")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to human_eval.db")
    parser.add_argument("--batch-size", type=int, default=12, help="Videos per person (default: 4 groups × 3 models)")
    parser.add_argument("--target", type=int, default=N_ANNOTATORS_PER_VIDEO, help="Target annotations per video")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db, timeout=30)

    rows_all = _query_coverage(conn, exclude_bad=False)
    rows_filt = _query_coverage(conn, exclude_bad=True)
    conn.close()

    _print_report("不 filter", rows_all, args.target, args.batch_size)
    _print_report("Filter 后", rows_filt, args.target, args.batch_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
