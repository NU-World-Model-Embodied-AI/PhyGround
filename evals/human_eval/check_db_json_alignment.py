"""Check internal alignment for a human-eval DB.

The original source-JSON comparison depends on prompt-selection artifacts that
are omitted from this release. This release version only checks consistency
between DB videos and comparison groups.

Usage:
  python -m human_eval.check_db_json_alignment
"""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "eval" / "human_eval" / "human_eval_filtered.db"


def _norm_laws(laws_str: str) -> list[str]:
    try:
        return sorted(json.loads(laws_str))
    except (json.JSONDecodeError, TypeError):
        return []


def main():
    print("=" * 60)
    print("DB Internal Alignment Check")
    print("=" * 60)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    groups = [dict(r) for r in conn.execute(
        "SELECT id, prompt, physical_laws FROM comparison_groups"
    ).fetchall()]

    issues = []

    print(f"\n[1] comparison_groups ({len(groups)}) internal consistency with videos")
    rows = conn.execute(
        "SELECT DISTINCT cg.id as gid, cg.prompt as cg_prompt, cg.physical_laws as cg_laws, "
        "v.id as vid, v.prompt as v_prompt, v.physical_laws as v_laws "
        "FROM comparison_groups cg "
        "JOIN assignments a ON a.group_id = cg.id "
        "JOIN videos v ON a.video_id = v.id"
    ).fetchall()
    conn.close()

    cg_mismatches = 0
    for r in rows:
        if r["cg_prompt"] != r["v_prompt"]:
            cg_mismatches += 1
            if cg_mismatches <= 5:
                issues.append({"type": "prompt_mismatch", "layer": "group_vs_video",
                               "group_id": r["gid"][:12], "video_id": r["vid"],
                               "cg_prompt": r["cg_prompt"][:80], "v_prompt": r["v_prompt"][:80]})
        if _norm_laws(r["cg_laws"]) != _norm_laws(r["v_laws"]):
            cg_mismatches += 1
            if cg_mismatches <= 5:
                issues.append({"type": "laws_mismatch", "layer": "group_vs_video",
                               "group_id": r["gid"][:12], "video_id": r["vid"],
                               "cg_laws": _norm_laws(r["cg_laws"]), "v_laws": _norm_laws(r["v_laws"])})
    print(f"  checked {len(rows)} (group, video) pairs, {cg_mismatches} mismatches")

    print("\n" + "=" * 60)
    if not issues:
        print("ALL ALIGNED")
        return 0

    grouped = defaultdict(list)
    for iss in issues:
        key = (iss["layer"], iss["type"], iss.get("group_id", "?"))
        grouped[key].append(iss)

    print(f"FOUND {len(grouped)} unique issue(s) ({len(issues)} total across models):\n")
    display_keys = {
        "prompt_mismatch": ("cg_prompt", "v_prompt"),
        "laws_mismatch": ("cg_laws", "v_laws"),
    }
    for i, ((layer, typ, stem), group) in enumerate(sorted(grouped.items()), 1):
        iss = group[0]
        print(f"--- #{i} [{typ}] {layer} | stem={stem} | x{len(group)} models ---")
        for key in display_keys.get(typ, ()):
            if key in iss:
                print(f"  {key}: {iss[key]}")
        print()

    return len(grouped)


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
