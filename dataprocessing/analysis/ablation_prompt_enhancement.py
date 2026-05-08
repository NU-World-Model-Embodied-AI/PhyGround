#!/usr/bin/env python3
"""
Ablation study: Effect of prompt enhancement (adding physical law/phenomenon descriptions)
on VLM evaluation scores.

Compares backup (pre-enhancement) eval files with current (post-enhancement) eval files,
analyzing score deltas per video where the prompt actually changed.

Run from the repo root dir:
    python -m dataprocessing.analysis.ablation_prompt_enhancement
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = "data"
BACKUP_DIR = os.path.join(DATA_ROOT, "backup_before_laws_update")
VIDEOS_DIR = os.path.join(DATA_ROOT, "videos")

GENERAL_METRICS = ["SA", "PTV", "persistence"]
SCORE_BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

# Law-to-domain mapping
LAW_TO_DOMAIN = {
    # collision
    "collision": "collision",
    "impenetrability": "collision",
    "momentum_transfer": "collision",
    "momentum": "collision",
    "elastic_deformation": "collision",
    # gravity
    "gravity": "gravity",
    "free_fall": "gravity",
    "projectile_motion": "gravity",
    "buoyancy": "gravity",
    # fluid
    "fluid_continuity": "fluid",
    "flow_dynamics": "fluid",
    "flow": "fluid",
    "viscosity": "fluid",
    "surface_tension": "fluid",
    "pressure": "fluid",
    "continuity": "fluid",
    # temporal / motion
    "inertia": "temporal",
    "acceleration": "temporal",
    "velocity": "temporal",
    "displacement": "temporal",
    # lighting
    "reflection": "lighting",
    "refraction": "lighting",
    "light_absorption": "lighting",
    "shadow": "lighting",
    "illumination": "lighting",
    # deformation
    "deformation": "deformation",
    "plastic_deformation": "deformation",
    # material
    "material": "material",
    "rigidity": "material",
    "elasticity": "material",
    "phase_transition": "material",
    "melting": "material",
    "combustion": "material",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_judge_key(judge_str: str) -> str:
    """Extract a short judge key like 'gemini', 'qwen', 'gpt' from judge field."""
    if not judge_str:
        return "unknown"
    return judge_str.split(":")[0]


def parse_filename_judge(filename: str) -> str:
    """Extract judge key from filename like eval_gemini_..., eval_qwen_permetric_..."""
    m = re.match(r"eval_(gemini|qwen|gpt)", filename)
    return m.group(1) if m else "unknown"


def is_permetric(filename: str) -> bool:
    return "permetric" in filename


def get_physical_avg(result: dict) -> Optional[float]:
    """Extract physical average from a result entry."""
    phys = result.get("physical")
    if phys is None:
        return None
    if isinstance(phys, (int, float)):
        return float(phys)
    if isinstance(phys, dict):
        avg = phys.get("avg")
        if avg is not None:
            return float(avg)
    return None


def get_per_law_scores(result: dict) -> Dict[str, float]:
    """Extract per-law scores from a result entry."""
    phys = result.get("physical")
    if not isinstance(phys, dict):
        return {}
    laws = phys.get("laws", {})
    out = {}
    for law_name, law_data in laws.items():
        if isinstance(law_data, dict) and "score" in law_data and law_data["score"] is not None:
            out[law_name] = float(law_data["score"])
        elif isinstance(law_data, (int, float)) and law_data is not None:
            out[law_name] = float(law_data)
    return out


def extract_dataset_from_video_dir(video_dir: str) -> str:
    """Extract dataset name from video_dir field, e.g. 'data/videos/cosmos-predict2.5-2b-wmb/' -> 'wmb'."""
    vd = video_dir.rstrip("/")
    basename = os.path.basename(vd)
    # Try to find the dataset suffix: wmb, video_phy_2, physics_iq, openvid, video_phy_2
    for ds in ["wmb", "video_phy_2", "physics_iq", "openvid", "video_phy_2", "wmb"]:
        if basename.endswith(f"-{ds}"):
            return ds
    return basename


def extract_model_from_video_dir(video_dir: str) -> str:
    """Extract model name from video_dir."""
    vd = video_dir.rstrip("/")
    basename = os.path.basename(vd)
    for ds in ["wmb", "video_phy_2", "physics_iq", "openvid", "video_phy_2", "wmb"]:
        suffix = f"-{ds}"
        if basename.endswith(suffix):
            return basename[: -len(suffix)]
    return basename


def load_eval_file(filepath: str) -> Optional[dict]:
    """Load an eval JSON file, returning None on error."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if not data.get("results"):
            return None
        return data
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        return None


def get_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from filename like eval_gemini_20260322_200226.json."""
    m = re.search(r"(\d{8}_\d{6})", filename)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Core: find old/new pairs
# ---------------------------------------------------------------------------


def find_pairs() -> List[Dict[str, Any]]:
    """
    Find all (old_file, new_file) pairs for comparison.
    Returns list of dicts with keys: old_path, new_path, model, dataset, judge, mode (batched/permetric).
    """
    pairs = []

    if not os.path.isdir(BACKUP_DIR):
        print(f"ERROR: Backup directory not found: {BACKUP_DIR}", file=sys.stderr)
        return pairs

    for backup_model in sorted(os.listdir(BACKUP_DIR)):
        backup_model_dir = os.path.join(BACKUP_DIR, backup_model)
        if not os.path.isdir(backup_model_dir):
            continue

        for fname in sorted(os.listdir(backup_model_dir)):
            if not fname.startswith("eval_") or not fname.endswith(".json"):
                continue

            old_path = os.path.join(backup_model_dir, fname)
            old_data = load_eval_file(old_path)
            if old_data is None:
                continue

            video_dir = old_data.get("video_dir", "")
            if not video_dir:
                continue

            dataset = extract_dataset_from_video_dir(video_dir)
            model = extract_model_from_video_dir(video_dir)
            judge = parse_judge_key(old_data.get("judge", ""))
            mode = "permetric" if is_permetric(fname) else "batched"

            # Find corresponding current directory
            current_dir = os.path.join(VIDEOS_DIR, f"{model}-{dataset}")
            if not os.path.isdir(current_dir):
                continue

            # Find newest matching eval file in current dir
            old_timestamp = get_timestamp_from_filename(fname)
            best_new_path = None
            best_new_ts = ""

            for cur_fname in os.listdir(current_dir):
                if not cur_fname.startswith("eval_") or not cur_fname.endswith(".json"):
                    continue
                if ".old_pre_t26." in cur_fname:
                    continue

                cur_judge = parse_filename_judge(cur_fname)
                cur_mode = "permetric" if is_permetric(cur_fname) else "batched"

                if cur_judge != judge or cur_mode != mode:
                    continue

                cur_ts = get_timestamp_from_filename(cur_fname)

                # Skip files with same timestamp as the old file (these are copies)
                if cur_ts == old_timestamp:
                    continue

                # Must be newer than old
                if cur_ts <= old_timestamp:
                    continue

                # Pick the newest one
                if cur_ts > best_new_ts:
                    best_new_ts = cur_ts
                    best_new_path = os.path.join(current_dir, cur_fname)

            if best_new_path:
                pairs.append(
                    {
                        "old_path": old_path,
                        "new_path": best_new_path,
                        "model": model,
                        "dataset": dataset,
                        "judge": judge,
                        "mode": mode,
                    }
                )

    return pairs


# ---------------------------------------------------------------------------
# Core: compute deltas
# ---------------------------------------------------------------------------


def compute_deltas(
    old_data: dict, new_data: dict
) -> List[Dict[str, Any]]:
    """
    For each video present in both old and new where the prompt changed,
    compute score deltas.
    Returns list of per-video delta records.
    """
    old_by_video = {r["video"]: r for r in old_data["results"]}
    new_by_video = {r["video"]: r for r in new_data["results"]}

    deltas = []
    common_videos = set(old_by_video.keys()) & set(new_by_video.keys())

    for vid in sorted(common_videos):
        old_r = old_by_video[vid]
        new_r = new_by_video[vid]

        # Only analyze videos where the prompt actually changed
        if old_r.get("prompt", "") == new_r.get("prompt", ""):
            continue

        rec: Dict[str, Any] = {
            "video": vid,
            "old_prompt": old_r.get("prompt", ""),
            "new_prompt": new_r.get("prompt", ""),
            "physical_laws": new_r.get("physical_laws", old_r.get("physical_laws", [])),
        }

        # General metrics deltas
        for m in GENERAL_METRICS:
            old_val = old_r.get(m)
            new_val = new_r.get(m)
            if old_val is not None and new_val is not None:
                rec[f"{m}_old"] = float(old_val)
                rec[f"{m}_new"] = float(new_val)
                rec[f"{m}_delta"] = float(new_val) - float(old_val)

        # general_avg delta
        old_ga = old_r.get("general_avg")
        new_ga = new_r.get("general_avg")
        if old_ga is not None and new_ga is not None:
            rec["general_avg_old"] = float(old_ga)
            rec["general_avg_new"] = float(new_ga)
            rec["general_avg_delta"] = float(new_ga) - float(old_ga)

        # physical_avg delta
        old_pa = get_physical_avg(old_r)
        new_pa = get_physical_avg(new_r)
        if old_pa is not None and new_pa is not None:
            rec["physical_avg_old"] = float(old_pa)
            rec["physical_avg_new"] = float(new_pa)
            rec["physical_avg_delta"] = float(new_pa) - float(old_pa)

        # Per-law score deltas
        old_laws = get_per_law_scores(old_r)
        new_laws = get_per_law_scores(new_r)
        law_deltas = {}
        for law in set(old_laws.keys()) & set(new_laws.keys()):
            law_deltas[law] = new_laws[law] - old_laws[law]
        rec["per_law_deltas"] = law_deltas

        deltas.append(rec)

    return deltas


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def safe_mean(values: list) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def format_delta(val: Optional[float], decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.{decimals}f}"


def format_float(val: Optional[float], decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


class AblationAnalysis:
    def __init__(self):
        # Each entry: (model, dataset, judge, mode, delta_record)
        self.all_records: List[Tuple[str, str, str, str, Dict[str, Any]]] = []

    def add(self, model: str, dataset: str, judge: str, mode: str, deltas: List[Dict[str, Any]]):
        for d in deltas:
            self.all_records.append((model, dataset, judge, mode, d))

    def _filter(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        judge: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        out = []
        for m, ds, j, mode, rec in self.all_records:
            if model and m != model:
                continue
            if dataset and ds != dataset:
                continue
            if judge and j != judge:
                continue
            out.append(rec)
        return out

    def _metric_deltas(self, records: List[Dict], metric_key: str) -> List[float]:
        key = f"{metric_key}_delta"
        return [r[key] for r in records if key in r]

    def overall_summary(self) -> str:
        lines = []
        records = self._filter()
        n = len(records)
        lines.append(f"**Total video comparisons (prompt changed):** {n}")
        lines.append("")

        if n == 0:
            lines.append("No data to analyze.")
            return "\n".join(lines)

        # Table header
        metrics = GENERAL_METRICS + ["general_avg", "physical_avg"]
        lines.append("| Metric | Mean Delta | Median Delta | Std Dev | N |")
        lines.append("|--------|-----------|-------------|---------|---|")
        for metric in metrics:
            vals = self._metric_deltas(records, metric)
            if not vals:
                lines.append(f"| {metric} | N/A | N/A | N/A | 0 |")
                continue
            import statistics

            mean = statistics.mean(vals)
            median = statistics.median(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
            lines.append(
                f"| {metric} | {format_delta(mean)} | {format_delta(median)} | {format_float(stdev)} | {len(vals)} |"
            )
        return "\n".join(lines)

    def per_group_table(self, group_key: str) -> str:
        """Group by model, dataset, or judge."""
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for m, ds, j, mode, rec in self.all_records:
            if group_key == "model":
                key = m
            elif group_key == "dataset":
                key = ds
            elif group_key == "judge":
                key = j
            else:
                key = "all"
            groups[key].append(rec)

        metrics = GENERAL_METRICS + ["general_avg", "physical_avg"]
        lines = []
        header = f"| {group_key.capitalize()} | N |"
        sep = "|---|---|"
        for metric in metrics:
            header += f" {metric} |"
            sep += "---|"
        lines.append(header)
        lines.append(sep)

        for gname in sorted(groups.keys()):
            recs = groups[gname]
            row = f"| {gname} | {len(recs)} |"
            for metric in metrics:
                vals = self._metric_deltas(recs, metric)
                mean = safe_mean(vals)
                row += f" {format_delta(mean)} |"
            lines.append(row)

        return "\n".join(lines)

    def per_domain_table(self) -> str:
        """Aggregate per-law deltas into domains."""
        domain_deltas: Dict[str, List[float]] = defaultdict(list)
        for _, _, _, _, rec in self.all_records:
            for law, delta in rec.get("per_law_deltas", {}).items():
                domain = LAW_TO_DOMAIN.get(law, "other")
                domain_deltas[domain].append(delta)

        lines = []
        lines.append("| Domain | Mean Delta | N | Improved% | Degraded% | Unchanged% |")
        lines.append("|--------|-----------|---|----------|----------|-----------|")

        for domain in sorted(domain_deltas.keys()):
            vals = domain_deltas[domain]
            n = len(vals)
            mean = safe_mean(vals)
            improved = sum(1 for v in vals if v > 0) / n * 100 if n else 0
            degraded = sum(1 for v in vals if v < 0) / n * 100 if n else 0
            unchanged = sum(1 for v in vals if v == 0) / n * 100 if n else 0
            lines.append(
                f"| {domain} | {format_delta(mean)} | {n} | {improved:.1f}% | {degraded:.1f}% | {unchanged:.1f}% |"
            )

        return "\n".join(lines)

    def per_law_table(self) -> str:
        """Per-law breakdown."""
        law_deltas: Dict[str, List[float]] = defaultdict(list)
        for _, _, _, _, rec in self.all_records:
            for law, delta in rec.get("per_law_deltas", {}).items():
                law_deltas[law].append(delta)

        lines = []
        lines.append("| Law | Domain | Mean Delta | N | Improved% | Degraded% |")
        lines.append("|-----|--------|-----------|---|----------|----------|")

        # Sort by mean delta descending
        sorted_laws = sorted(
            law_deltas.keys(), key=lambda l: safe_mean(law_deltas[l]) or 0, reverse=True
        )

        for law in sorted_laws:
            vals = law_deltas[law]
            n = len(vals)
            mean = safe_mean(vals)
            domain = LAW_TO_DOMAIN.get(law, "other")
            improved = sum(1 for v in vals if v > 0) / n * 100 if n else 0
            degraded = sum(1 for v in vals if v < 0) / n * 100 if n else 0
            lines.append(
                f"| {law} | {domain} | {format_delta(mean)} | {n} | {improved:.1f}% | {degraded:.1f}% |"
            )

        return "\n".join(lines)

    def score_bin_table(self) -> str:
        """Effect on low-score vs high-score videos, binned by old general_avg."""
        bin_deltas: Dict[str, Dict[str, List[float]]] = {}
        for lo, hi in SCORE_BINS:
            label = f"{lo}-{hi}"
            bin_deltas[label] = defaultdict(list)

        metrics = GENERAL_METRICS + ["general_avg", "physical_avg"]
        for _, _, _, _, rec in self.all_records:
            old_ga = rec.get("general_avg_old")
            if old_ga is None:
                continue
            for lo, hi in SCORE_BINS:
                if lo <= old_ga < hi or (hi == 5 and old_ga == 5):
                    label = f"{lo}-{hi}"
                    for metric in metrics:
                        key = f"{metric}_delta"
                        if key in rec:
                            bin_deltas[label][metric].append(rec[key])
                    break

        lines = []
        header = "| Old general_avg Bin | N |"
        sep = "|---|---|"
        for metric in metrics:
            header += f" {metric} |"
            sep += "---|"
        lines.append(header)
        lines.append(sep)

        for lo, hi in SCORE_BINS:
            label = f"{lo}-{hi}"
            bd = bin_deltas[label]
            # N = number of records in this bin
            n_vals = bd.get("general_avg", [])
            n = len(n_vals)
            row = f"| {label} | {n} |"
            for metric in metrics:
                vals = bd.get(metric, [])
                mean = safe_mean(vals)
                row += f" {format_delta(mean)} |"
            lines.append(row)

        return "\n".join(lines)

    def prompt_length_analysis(self) -> str:
        """Analyze whether longer prompt additions correlate with bigger deltas."""
        records_with_len = []
        for _, _, _, _, rec in self.all_records:
            old_len = len(rec.get("old_prompt", ""))
            new_len = len(rec.get("new_prompt", ""))
            delta_len = new_len - old_len
            ga_delta = rec.get("general_avg_delta")
            pa_delta = rec.get("physical_avg_delta")
            if ga_delta is not None:
                records_with_len.append((delta_len, ga_delta, pa_delta))

        if not records_with_len:
            return "No data for prompt length analysis."

        lines = []
        # Bin by prompt length increase
        bins = [(0, 30), (30, 60), (60, 90), (90, 200)]
        lines.append("| Prompt Length Increase | N | Mean general_avg Delta | Mean physical_avg Delta |")
        lines.append("|----------------------|---|----------------------|----------------------|")

        for lo, hi in bins:
            subset = [(d, g, p) for d, g, p in records_with_len if lo <= d < hi]
            n = len(subset)
            ga_mean = safe_mean([g for _, g, _ in subset])
            pa_mean = safe_mean([p for _, _, p in subset if p is not None])
            lines.append(
                f"| {lo}-{hi} chars | {n} | {format_delta(ga_mean)} | {format_delta(pa_mean)} |"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Ensure we're in repo root or adjust paths
    if not os.path.isdir(DATA_ROOT):
        # Try from the script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
        os.chdir(repo_root)
        if not os.path.isdir(DATA_ROOT):
            print(f"ERROR: Cannot find {DATA_ROOT}. Run from the repo root directory.", file=sys.stderr)
            sys.exit(1)

    print("# Ablation Study: Effect of Prompt Enhancement on VLM Evaluation Scores")
    print()
    print("Comparing backup (pre-enhancement) vs current (post-enhancement) evaluation files.")
    print("Only videos where the prompt actually changed between versions are included.")
    print()

    # Step 1: Find all old/new pairs
    pairs = find_pairs()
    print(f"Found **{len(pairs)}** old/new eval file pairs.")
    print()

    if not pairs:
        print("No pairs found. Exiting.")
        return

    # Print pairs summary
    print("## File Pairs Found")
    print()
    print("| # | Model | Dataset | Judge | Mode | Old File | New File |")
    print("|---|-------|---------|-------|------|----------|----------|")
    for i, p in enumerate(pairs, 1):
        old_base = os.path.basename(p["old_path"])
        new_base = os.path.basename(p["new_path"])
        print(
            f"| {i} | {p['model']} | {p['dataset']} | {p['judge']} | {p['mode']} | {old_base} | {new_base} |"
        )
    print()

    # Step 2: Compute deltas for each pair
    analysis = AblationAnalysis()
    total_videos = 0
    total_changed = 0
    skipped_pairs = 0

    for p in pairs:
        old_data = load_eval_file(p["old_path"])
        new_data = load_eval_file(p["new_path"])
        if old_data is None or new_data is None:
            skipped_pairs += 1
            continue

        old_by_video = {r["video"]: r for r in old_data["results"]}
        new_by_video = {r["video"]: r for r in new_data["results"]}
        common = set(old_by_video.keys()) & set(new_by_video.keys())
        total_videos += len(common)

        deltas = compute_deltas(old_data, new_data)
        total_changed += len(deltas)

        analysis.add(p["model"], p["dataset"], p["judge"], p["mode"], deltas)

    print(f"**Total matched videos across all pairs:** {total_videos}")
    print(f"**Videos with prompt changes:** {total_changed}")
    if skipped_pairs:
        print(f"**Skipped pairs (load errors):** {skipped_pairs}")
    print()

    if total_changed == 0:
        print("No videos with prompt changes found. Exiting.")
        return

    # Step 3: Reports
    print("---")
    print()
    print("## 1. Overall Score Deltas (All Models, Datasets, Judges)")
    print()
    print(analysis.overall_summary())
    print()

    print("---")
    print()
    print("## 2. Per-Model Breakdown")
    print()
    print(analysis.per_group_table("model"))
    print()

    print("---")
    print()
    print("## 3. Per-Dataset Breakdown")
    print()
    print(analysis.per_group_table("dataset"))
    print()

    print("---")
    print()
    print("## 4. Per-Judge Breakdown")
    print()
    print(analysis.per_group_table("judge"))
    print()

    print("---")
    print()
    print("## 5. Per-Domain Breakdown (Physical Laws)")
    print()
    print(analysis.per_domain_table())
    print()

    print("---")
    print()
    print("## 6. Per-Law Breakdown (sorted by mean delta, descending)")
    print()
    print(analysis.per_law_table())
    print()

    print("---")
    print()
    print("## 7. Effect by Old Score Bin")
    print()
    print("Videos binned by their old general_avg score:")
    print()
    print(analysis.score_bin_table())
    print()

    print("---")
    print()
    print("## 8. Prompt Length Increase Analysis")
    print()
    print(analysis.prompt_length_analysis())
    print()


if __name__ == "__main__":
    main()
