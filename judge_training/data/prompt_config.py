"""Build ms-swift JSONL from human DB labels using a prompt config.

This flow regenerates json_only training samples from:
  human_eval_filtered.db + evals/prompts/{prompt_config}

It does not accept prebuilt conversations.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from evals.physics_criteria import get_criteria_text
from evals.prompts import GENERAL_DIMS, PromptConfig
from judge_training.data.sample import TrainingSample
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

MODULE = "judge_training.data.prompt_config"


def _records_to_samples(
    records: list[dict[str, Any]],
    prompt_cfg: PromptConfig,
    base_dir: str,
) -> list[TrainingSample]:
    """Convert human-label records into json_only TrainingSamples."""
    samples: list[TrainingSample] = []

    for rec in records:
        prompt_text = rec["prompt"]

        for dim in GENERAL_DIMS:
            score = rec["general_scores"].get(dim)
            if score is None:
                continue

            samples.append(
                TrainingSample.json_only(
                    system=prompt_cfg.system_prompt,
                    user=prompt_cfg.build_training_prompt(prompt_text, dim),
                    video_path=rec["video_path"],
                    key=dim,
                    score=score,
                    base_dir=base_dir,
                )
            )

        for law_name, score in rec["physical_scores"].items():
            criteria_text = get_criteria_text(law_name)
            if not criteria_text:
                continue

            samples.append(
                TrainingSample.json_only(
                    system=prompt_cfg.system_prompt,
                    user=prompt_cfg.build_physical_prompt(
                        prompt_text, law_name, criteria_text
                    ),
                    video_path=rec["video_path"],
                    key=law_name,
                    score=score,
                    base_dir=base_dir,
                )
            )

    return samples



def build_prompt_config_splits(
    db_path: str,
    base_dir: str,
    holdout_model: str,
    holdout_prompt_ratio: float,
    prompt_config: str,
    prompt_seed: int = 42,
) -> dict[str, list[TrainingSample]]:
    """Build samples with double holdout (model + prompt).

    Returns dict with keys: train, test_prompt, test_model, test_both.
    """
    from judge_training.data.build_records_from_db import (
        build_records,
        split_by_prompt_and_model,
    )

    prompt_cfg = PromptConfig.load(prompt_config)
    all_records = build_records(db_path)
    logger.info("Built %d human records", len(all_records))

    record_splits = split_by_prompt_and_model(
        all_records, holdout_model, holdout_prompt_ratio, prompt_seed,
    )

    sample_splits: dict[str, list[TrainingSample]] = {}
    for name, recs in record_splits.items():
        samples = _records_to_samples(recs, prompt_cfg, base_dir)
        sample_splits[name] = samples
        logger.info("  %s: %d records -> %d samples", name, len(recs), len(samples))

    return sample_splits


def _metadata(prompt_config: str) -> dict[str, object]:
    return {
        "prompt_config": prompt_config,
        "prompt_config_source": "cli",
        "label_source": "human",
        "target_format": "json_only",
        "dims": [*GENERAL_DIMS, "physical_laws"],
        "score_scale": "1-5",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert human DB labels plus --prompt-config to ms-swift JSONL."
    )
    subparsers = parser.add_subparsers(dest="command")

    convert = subparsers.add_parser("convert", help="Build JSONL from human DB")
    convert.add_argument("--output", default=None, help="Output JSONL path (default: auto-timestamped)")
    convert.add_argument(
        "--db",
        default="evals/human_eval/human_eval_filtered.db",
        help="Human eval DB path",
    )
    convert.add_argument(
        "--prompt-config",
        dest="prompt_config",
        required=True,
        help="YAML filename under evals/prompts",
    )
    convert.add_argument(
        "--holdout-prompt-ratio",
        dest="holdout_prompt_ratio",
        type=float,
        default=0.1,
        help="Fraction of prompts to hold out (default: 0.1). "
             "Requires --holdout_model.",
    )
    convert.add_argument(
        "--prompt-seed",
        dest="prompt_seed",
        type=int,
        default=42,
        help="Random seed for prompt holdout sampling (default: 42)",
    )
    add_common_convert_args(convert)

    validate = subparsers.add_parser("validate", help="Validate a json_only JSONL file")
    validate.add_argument("jsonl", help="JSONL file to validate")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return validate_cli(args.jsonl)

    if args.command == "convert":
        stem = prompt_config_stem(args.prompt_config)
        if args.output is None:
            args.output = swift_train_path(stem)
        if args.val_output is None:
            args.val_output = swift_val_path(stem)

        splits = build_prompt_config_splits(
            args.db,
            args.base_dir,
            args.holdout_model,
            args.holdout_prompt_ratio,
            args.prompt_config,
            args.prompt_seed,
        )

        write_test_splits(splits, stem)

        command_args = [
            "--db", args.db,
            "--prompt-config", args.prompt_config,
            "--output", args.output,
            "--holdout-prompt-ratio", str(args.holdout_prompt_ratio),
            "--prompt-seed", str(args.prompt_seed),
        ]
        append_val_command_args(command_args, args)

        return run_convert(
            samples=splits["train"],
            args=args,
            metadata=_metadata(args.prompt_config),
            module=MODULE,
            command_args=command_args,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
