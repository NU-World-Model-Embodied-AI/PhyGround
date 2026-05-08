"""Export per-annotator human evaluation scores from the filtered SQLite DB.

Reads a `human_eval_filtered.db` (path supplied via --db or $PHYGROUND_DB) and
writes one JSON file per annotator into the dataset's `annotations/` folder.

No personal information is exported: each annotator is identified only by a
serial id (annotator_001, annotator_002, ...). Names, gender, age, major,
education and creation timestamps are dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = REPO_ROOT.parent / "datasets" / "annotations"
DEFAULT_PROMPTS_JSON = REPO_ROOT.parent / "datasets" / "prompts" / "phyground.json"


def load_prompt_id_map(prompts_json: Path) -> dict[str, int]:
    if not prompts_json.exists():
        return {}
    data = json.loads(prompts_json.read_text())
    return {entry["video"]: entry["id"] for entry in data}


def parse_json(field: str | None, default):
    if field is None or field == "":
        return default
    try:
        return json.loads(field)
    except json.JSONDecodeError:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=os.environ.get("PHYGROUND_DB"),
        help="Path to human_eval_filtered.db (or set $PHYGROUND_DB).",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prompts_json", type=Path, default=DEFAULT_PROMPTS_JSON)
    args = parser.parse_args()

    if args.db is None:
        parser.error("--db is required (or set $PHYGROUND_DB).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_id_map = load_prompt_id_map(args.prompts_json)

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row

    annotator_ids = [
        row["annotator_id"]
        for row in conn.execute(
            "SELECT DISTINCT a.annotator_id AS annotator_id "
            "FROM annotations ann JOIN assignments a ON ann.assignment_id = a.id "
            "ORDER BY a.annotator_id"
        )
    ]
    width = max(3, len(str(len(annotator_ids))))
    serial_map = {real: f"annotator_{i+1:0{width}d}" for i, real in enumerate(annotator_ids)}

    rows = conn.execute(
        """
        SELECT a.annotator_id   AS annotator_id,
               v.dataset        AS model,
               v.filename       AS filename,
               v.physical_laws  AS physical_laws,
               ann.scores_json  AS scores_json,
               ann.na_laws      AS na_laws
        FROM annotations ann
        JOIN assignments a ON ann.assignment_id = a.id
        JOIN videos v      ON a.video_id        = v.id
        ORDER BY a.annotator_id, v.dataset, v.filename
        """
    ).fetchall()

    grouped: dict[int, list[dict]] = {}
    for r in rows:
        stem = Path(r["filename"]).stem
        scores = parse_json(r["scores_json"], {})
        item = {
            "model": r["model"],
            "video": stem,
            "physical_laws": parse_json(r["physical_laws"], []),
            "scores": {
                "general": scores.get("general", {}),
                "physical": scores.get("physical", {}),
            },
            "na_laws": parse_json(r["na_laws"], []),
        }
        if stem in prompt_id_map:
            item["prompt_id"] = prompt_id_map[stem]
        grouped.setdefault(r["annotator_id"], []).append(item)

    manifest = []
    for real_id, items in grouped.items():
        serial_id = serial_map[real_id]
        payload = {
            "annotator_id": serial_id,
            "num_annotations": len(items),
            "annotations": items,
        }
        (args.out_dir / f"{serial_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
        manifest.append({"annotator_id": serial_id, "num_annotations": len(items)})

    manifest.sort(key=lambda x: x["annotator_id"])
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {"num_annotators": len(manifest), "annotators": manifest},
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"Wrote {len(manifest)} annotator files to {args.out_dir}")
    print(f"Total annotations: {sum(m['num_annotations'] for m in manifest)}")


if __name__ == "__main__":
    main()
