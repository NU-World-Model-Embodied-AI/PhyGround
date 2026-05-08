"""Export per-annotator human evaluation scores from the filtered SQLite DB.

Reads /shared/user60/worldmodel/wmbench/evals/human_eval/human_eval_filtered.db
and writes one JSON file per annotator under
/shared/user60/workspace/worldmodel/wmbench/public/datasets/annotations/.

No personal information is exported: each annotator is identified only by a
serial id (annotator_001, annotator_002, ...). Names, gender, age,
major, education and creation timestamps are dropped.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

DB_PATH = Path("/shared/user60/worldmodel/wmbench/evals/human_eval/human_eval_filtered.db")
OUT_DIR = Path("/shared/user60/workspace/worldmodel/wmbench/public/datasets/annotations")
PROMPTS_JSON = Path("/shared/user60/workspace/worldmodel/wmbench/public/datasets/prompts/phyground.json")


def load_prompt_id_map() -> dict[str, int]:
    if not PROMPTS_JSON.exists():
        return {}
    data = json.loads(PROMPTS_JSON.read_text())
    return {entry["video"]: entry["id"] for entry in data}


def parse_json(field: str | None, default):
    if field is None or field == "":
        return default
    try:
        return json.loads(field)
    except json.JSONDecodeError:
        return default


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt_id_map = load_prompt_id_map()

    conn = sqlite3.connect(str(DB_PATH))
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
        (OUT_DIR / f"{serial_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
        manifest.append({"annotator_id": serial_id, "num_annotations": len(items)})

    manifest.sort(key=lambda x: x["annotator_id"])
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {"num_annotators": len(manifest), "annotators": manifest},
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"Wrote {len(manifest)} annotator files to {OUT_DIR}")
    print(f"Total annotations: {sum(m['num_annotations'] for m in manifest)}")


if __name__ == "__main__":
    main()
