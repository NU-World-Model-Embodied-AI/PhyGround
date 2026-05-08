"""Convert Claude CoT eval results to ms-swift JSONL training data.

Reads Claude eval JSON files (produced with cotnosubq.yaml), extracts
reasoning + scores, and builds TrainingSample records.

Usage:
    python -m judge_training.data.build_from_claude_cot convert \
        --prompt-config cotnosubq.yaml \
        --eval-dir data/scores/claude \
        --pattern 'eval_claude_cot_*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any

from evals.physics_criteria import get_criteria_text
from evals.prompts import GENERAL_DIMS, PromptConfig
from judge_training.data.sample import TrainingSample
from judge_training.data.build_records_from_db import build_records
from judge_training.data.build_swift_data import (
    add_common_convert_args,
    append_val_command_args,
    run_convert,
    validate_cli,
    write_test_splits,
)
from judge_training.data.naming import (
    prompt_config_stem,
    swift_train_path,
    swift_val_path,
)

logger = logging.getLogger(__name__)

MODULE = "judge_training.data.build_from_claude_cot"


def _parse_rationale(raw: Any) -> dict[str, str]:
    """Parse a rationale field that may be a JSON string or dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _load_eval_results(eval_dir: str, pattern: str) -> list[dict[str, Any]]:
    """Load all eval result JSONs matching pattern, dedup by video stem."""
    files = sorted(glob.glob(str(Path(eval_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {eval_dir}")

    seen: dict[str, dict[str, Any]] = {}
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        for result in data.get("results", []):
            video = result.get("video", "")
            if video:
                seen[video] = result
    logger.info("Loaded %d unique videos from %d files", len(seen), len(files))
    return list(seen.values())


def _build_human_score_lookup(
    db_path: str,
) -> dict[tuple[str, str], int]:
    """Build a lookup from (video_path, dim_or_law) -> human score."""
    records = build_records(db_path, active_only=False)
    lookup: dict[tuple[str, str], int] = {}
    for rec in records:
        vp = rec["video_path"]
        for dim, score in rec["general_scores"].items():
            lookup[(vp, dim)] = score
        for law, score in rec["physical_scores"].items():
            lookup[(vp, law)] = score
    return lookup


def _extract_cot_samples(
    results: list[dict[str, Any]],
    prompt_cfg: PromptConfig,
    model_prefix: str,
    base_dir: str,
    human_scores: dict[tuple[str, str], int],
) -> list[TrainingSample]:
    """Extract CoT training samples, keeping only where Claude score == human score."""
    samples: list[TrainingSample] = []
    skipped_no_reasoning = 0
    skipped_score_mismatch = 0
    skipped_no_human = 0

    for result in results:
        video_stem = result.get("video", "")
        prompt_text = result.get("prompt", "")
        if not video_stem or not prompt_text:
            continue

        video_path = f"data/videos/{model_prefix}/{video_stem}.mp4"

        rationale_general = _parse_rationale(result.get("rationale_general", "{}"))
        rationale_physical = _parse_rationale(result.get("rationale_physical", "{}"))

        for dim in GENERAL_DIMS:
            score = result.get(dim)
            if score is None or not isinstance(score, (int, float)):
                continue
            score = int(score)
            reasoning = rationale_general.get(dim, "")
            if not reasoning:
                skipped_no_reasoning += 1
                continue

            human = human_scores.get((video_path, dim))
            if human is None:
                skipped_no_human += 1
                continue
            if score != human:
                skipped_score_mismatch += 1
                continue

            try:
                samples.append(
                    TrainingSample.cot(
                        system=prompt_cfg.system_prompt,
                        user=prompt_cfg.build_training_prompt(prompt_text, dim),
                        video_path=video_path,
                        key=dim,
                        score=score,
                        reasoning=reasoning,
                        base_dir=base_dir,
                    )
                )
            except ValueError as e:
                logger.warning("Skip %s/%s: %s", video_stem, dim, e)

        physical_laws = result.get("physical_laws", [])
        physical_data = result.get("physical", {})
        laws_dict = physical_data.get("laws", {}) if isinstance(physical_data, dict) else {}

        for law in physical_laws:
            law_info = laws_dict.get(law, {})
            score = law_info.get("score")
            if score is None or not isinstance(score, (int, float)):
                continue
            score = int(score)

            reasoning = rationale_physical.get(law, "")
            if not reasoning:
                skipped_no_reasoning += 1
                continue

            human = human_scores.get((video_path, law))
            if human is None:
                skipped_no_human += 1
                continue
            if score != human:
                skipped_score_mismatch += 1
                continue

            criteria_text = get_criteria_text(law)
            if not criteria_text:
                continue

            try:
                samples.append(
                    TrainingSample.cot(
                        system=prompt_cfg.system_prompt,
                        user=prompt_cfg.build_physical_prompt(
                            prompt_text, law, criteria_text,
                        ),
                        video_path=video_path,
                        key=law,
                        score=score,
                        reasoning=reasoning,
                        base_dir=base_dir,
                    )
                )
            except ValueError as e:
                logger.warning("Skip %s/%s: %s", video_stem, law, e)

    if skipped_no_reasoning:
        logger.info("Skipped %d samples with empty reasoning", skipped_no_reasoning)
    if skipped_no_human:
        logger.info("Skipped %d samples with no human score", skipped_no_human)
    if skipped_score_mismatch:
        logger.info("Skipped %d samples where Claude != human score", skipped_score_mismatch)
    return samples


def _build_eval_records(
    eval_dir: str,
    pattern: str,
    model_prefixes: list[str],
) -> list[dict[str, Any]]:
    """Load eval results into prompt/model records for split assignment."""
    records: list[dict[str, Any]] = []

    for prefix in model_prefixes:
        model_pattern = pattern.replace("*", f"*{prefix}*")
        try:
            results = _load_eval_results(eval_dir, model_pattern)
        except FileNotFoundError:
            logger.warning("No files for %s, skipping", prefix)
            continue

        n_records = 0
        for result in results:
            prompt_text = result.get("prompt", "")
            video_stem = result.get("video", "")
            if not prompt_text or not video_stem:
                continue

            records.append(
                {
                    "prompt": prompt_text,
                    "video_model": prefix,
                    "result": result,
                }
            )
            n_records += 1

        logger.info("  %s: %d eval records", prefix, n_records)

    return records


def _records_to_samples(
    records: list[dict[str, Any]],
    prompt_cfg: PromptConfig,
    base_dir: str,
    human_scores: dict[tuple[str, str], int],
) -> list[TrainingSample]:
    """Convert split-assigned eval records into training samples."""
    model_to_results: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        model_to_results.setdefault(rec["video_model"], []).append(rec["result"])

    samples: list[TrainingSample] = []
    for model_prefix in sorted(model_to_results):
        model_results = model_to_results[model_prefix]
        model_samples = _extract_cot_samples(
            model_results,
            prompt_cfg,
            model_prefix,
            base_dir,
            human_scores,
        )
        logger.info(
            "    %s: %d results -> %d samples",
            model_prefix,
            len(model_results),
            len(model_samples),
        )
        samples.extend(model_samples)

    return samples


def _metadata(prompt_config: str) -> dict[str, object]:
    return {
        "prompt_config": prompt_config,
        "prompt_config_source": "cli",
        "label_source": "claude_cot",
        "target_format": "cot",
        "dims": [*GENERAL_DIMS, "physical_laws"],
        "score_scale": "1-5",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Claude CoT eval results to ms-swift JSONL.",
    )
    subparsers = parser.add_subparsers(dest="command")

    convert = subparsers.add_parser("convert", help="Build JSONL from Claude CoT evals")
    convert.add_argument("--output", default=None)
    convert.add_argument("--eval-dir", default="data/scores/claude")
    convert.add_argument("--pattern", default="eval_claude_cot_*.json")
    convert.add_argument("--model-prefix", action="append", dest="model_prefixes",
                         help="Model prefix(es) to include (repeat for multiple). "
                              "Default: infer from filenames.")
    convert.add_argument("--prompt-config", dest="prompt_config", required=True)
    convert.add_argument("--db-path", default="evals/human_eval/human_eval_filtered.db",
                         help="Path to human eval DB for score filtering")
    convert.add_argument(
        "--holdout-prompt-ratio",
        dest="holdout_prompt_ratio",
        type=float,
        default=0.1,
        help="Fraction of prompts to hold out (default: 0.1).",
    )
    convert.add_argument(
        "--prompt-seed",
        dest="prompt_seed",
        type=int,
        default=42,
        help="Random seed for prompt holdout sampling (default: 42)",
    )
    add_common_convert_args(convert)

    validate = subparsers.add_parser("validate", help="Validate a JSONL file")
    validate.add_argument("jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return validate_cli(args.jsonl)

    if args.command == "convert":
        stem = prompt_config_stem(args.prompt_config)
        data_stem = f"{stem}_claude_cot"
        if args.output is None:
            args.output = swift_train_path(data_stem)
        if args.val_output is None:
            args.val_output = swift_val_path(data_stem)

        prompt_cfg = PromptConfig.load(args.prompt_config)

        all_files = sorted(glob.glob(str(Path(args.eval_dir) / args.pattern)))
        if not all_files:
            logger.error("No files matching %s in %s", args.pattern, args.eval_dir)
            return 1

        if args.model_prefixes:
            prefixes = args.model_prefixes
        else:
            prefixes = set()
            for fpath in all_files:
                name = Path(fpath).stem
                parts = name.split("_humaneval_set_")
                if len(parts) == 2:
                    model = parts[1].rsplit("_", 2)[0]
                    prefixes.add(model)
            prefixes = sorted(prefixes)
            logger.info("Auto-detected model prefixes: %s", prefixes)

        human_scores = _build_human_score_lookup(args.db_path)
        logger.info("Loaded %d human score entries", len(human_scores))
        eval_records = _build_eval_records(args.eval_dir, args.pattern, prefixes)
        logger.info("Loaded %d eval records for split assignment", len(eval_records))

        from judge_training.data.build_records_from_db import split_by_prompt_and_model

        record_splits = split_by_prompt_and_model(
            eval_records,
            args.holdout_model,
            args.holdout_prompt_ratio,
            args.prompt_seed,
        )

        sample_splits = {
            name: _records_to_samples(recs, prompt_cfg, args.base_dir, human_scores)
            for name, recs in record_splits.items()
        }
        write_test_splits(sample_splits, data_stem)

        train_samples = sample_splits["train"]

        command_args = [
            "--eval-dir", args.eval_dir,
            "--pattern", args.pattern,
            "--prompt-config", args.prompt_config,
            "--db-path", args.db_path,
            "--holdout-prompt-ratio", str(args.holdout_prompt_ratio),
            "--prompt-seed", str(args.prompt_seed),
            "--output", args.output,
        ]
        if args.model_prefixes:
            for prefix in args.model_prefixes:
                command_args.extend(["--model-prefix", prefix])
        append_val_command_args(command_args, args)

        return run_convert(
            samples=train_samples,
            args=args,
            metadata=_metadata(args.prompt_config),
            module=MODULE,
            command_args=command_args,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
