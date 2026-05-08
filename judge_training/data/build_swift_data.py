"""ms-swift JSONL conversion pipeline: write, split, validate, register."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from judge_training.data.build_records_from_db import extract_model_and_source
from judge_training.data.naming import REGISTRY_PATH, swift_test_path
from judge_training.data.sample import ALLOWED_KEYS, TrainingSample
from judge_training.data.schema import resolve_schema

logger = logging.getLogger(__name__)


def write_swift_jsonl(samples: list[TrainingSample], output_path: str) -> int:
    """Write validated samples to a Swift JSONL file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.to_jsonl() + "\n")

    logger.info("Wrote %d lines to %s", len(samples), output_path)
    return len(samples)


def split_samples_by_video(
    samples: list[TrainingSample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    """Split samples by video path so all prompts for one video stay together."""
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    video_to_samples: dict[str, list[TrainingSample]] = {}
    for sample in samples:
        video_to_samples.setdefault(sample.video_path, []).append(sample)

    video_paths = sorted(video_to_samples)
    if not video_paths:
        return [], []

    rng = random.Random(seed)
    rng.shuffle(video_paths)
    n_val = max(1, int(len(video_paths) * val_ratio))
    if len(video_paths) > 1:
        n_val = min(n_val, len(video_paths) - 1)

    val_videos = set(video_paths[:n_val])
    train_samples: list[TrainingSample] = []
    val_samples: list[TrainingSample] = []

    for video_path in video_paths:
        if video_path in val_videos:
            val_samples.extend(video_to_samples[video_path])
        else:
            train_samples.extend(video_to_samples[video_path])

    logger.info(
        "Train/val split: %d videos train, %d videos val (%d/%d samples)",
        len(video_paths) - n_val,
        n_val,
        len(train_samples),
        len(val_samples),
    )
    return train_samples, val_samples


def validate_jsonl(jsonl_path: str) -> dict[str, Any]:
    """Validate a Swift JSONL file by parsing each line into TrainingSample."""
    errors: list[str] = []
    total = 0
    valid = 0

    score_dist: dict[int, int] = {}
    key_dist: dict[str, int] = {}
    video_model_dist: dict[str, int] = {}

    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: invalid JSON: {e}")
                continue

            try:
                sample = TrainingSample.from_jsonl_row(row)
            except ValueError as e:
                errors.append(f"Line {line_num}: {e}")
                continue

            valid += 1

            parsed = json.loads(sample.assistant)
            for k, v in parsed.items():
                if k in ALLOWED_KEYS:
                    key_dist[k] = key_dist.get(k, 0) + 1
                    score_dist[v] = score_dist.get(v, 0) + 1

            parts = Path(sample.video_path).parts
            for i, part in enumerate(parts):
                if part == "videos" and i + 1 < len(parts):
                    model, _ = extract_model_and_source(parts[i + 1])
                    video_model_dist[model] = video_model_dist.get(model, 0) + 1
                    break

    summary: dict[str, Any] = {
        "file": jsonl_path,
        "total_lines": total,
        "valid_lines": valid,
        "errors": errors[:20],
    }

    print(f"\n=== Validation: {jsonl_path} ===")
    print(f"  Total lines: {total}")
    print(f"  Valid: {valid}")

    if valid > 0:
        print("\n  Score distribution:")
        for score in sorted(score_dist):
            print(f"    {score}: {score_dist[score]}")
        print("\n  Key distribution:")
        for key in sorted(key_dist):
            print(f"    {key}: {key_dist[key]}")

    if video_model_dist:
        print("\n  Video model distribution:")
        for model in sorted(video_model_dist):
            print(f"    {model}: {video_model_dist[model]}")

    if errors:
        print(f"  Errors ({len(errors)} total, showing first 10):")
        for error in errors[:10]:
            print(f"    {error}")
    else:
        print("  No errors found.")

    return summary


def _is_ephemeral_output(output: str) -> bool:
    """Detect debug/temp output paths that shouldn't be registered."""
    resolved = str(Path(output).resolve())
    if resolved.startswith("/tmp") or resolved.startswith("/var/tmp"):
        return True
    return False


def write_test_splits(
    splits: dict[str, list[TrainingSample]],
    stem: str,
) -> None:
    """Write and validate test-split JSONL files."""
    for split_name in ("test_prompt", "test_model", "test_both"):
        samples = splits.get(split_name, [])
        if not samples:
            logger.info("  %s: empty, skipping", split_name)
            continue
        path = swift_test_path(stem, split_name)
        write_swift_jsonl(samples, path)
        validate_jsonl(path)


def append_training_registry(
    *,
    module: str,
    output: str,
    holdout_model: str | None,
    val_output: str | None,
    val_ratio: float | None,
    seed: int | None,
    n_train: int,
    n_val: int | None,
    metadata: dict[str, Any],
    command_args: list[str],
) -> None:
    """Append one run entry to training_registry.json.

    Auto-skips /tmp outputs.  If output_train already exists in the
    registry, the old entry is replaced instead of duplicated.
    """
    if _is_ephemeral_output(output):
        logger.info("Output is in /tmp — skipping training registry")
        return

    if REGISTRY_PATH.is_file():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        registry = []

    prompt_config = metadata.get("prompt_config") or "default.yaml"
    schema_info = resolve_schema(prompt_config)

    entry = {
        "scheme": schema_info.scheme,
        "schema": schema_info.version,
        "schema_fingerprint": schema_info.fingerprint,
        "prompt_config": prompt_config,
        "prompt_config_source": metadata.get("prompt_config_source", "unknown"),
        "label_source": metadata.get("label_source", "unknown"),
        "target_format": metadata.get("target_format", "unknown"),
        "dims": metadata.get("dims", []),
        "score_scale": metadata.get("score_scale", "unknown"),
        "holdout_model": holdout_model,
        "val_ratio": val_ratio if val_output else None,
        "seed": seed if val_output else None,
        "n_train": n_train,
        "n_val": n_val,
        "output_train": output,
        "output_val": val_output,
        "datetime": datetime.datetime.now().strftime("%m-%d %H:%M:%S"),
        "gen_command": " ".join(
            ["python", "-m", module, "convert", *command_args]
        ),
    }

    replaced = False
    for i, existing in enumerate(registry):
        if existing.get("output_train") == output:
            registry[i] = entry
            replaced = True
            break

    if not replaced:
        registry.append(entry)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")
    action = "Replaced" if replaced else "Appended"
    logger.info("%s training registry: scheme=%s, schema=%s, output=%s", action, schema_info.scheme, schema_info.version, output)


def validate_cli(jsonl_path: str) -> int:
    """Shared implementation for validate subcommands."""
    if not os.path.isfile(jsonl_path):
        logger.error("File not found: %s", jsonl_path)
        return 1
    summary = validate_jsonl(jsonl_path)
    if summary["total_lines"] == 0:
        logger.error("Empty file: %s", jsonl_path)
        return 1
    return 0 if not summary["errors"] else 1


def add_common_convert_args(parser: argparse.ArgumentParser) -> None:
    """Add shared convert subparser arguments (base_dir, holdout, val split)."""
    parser.add_argument(
        "--base_dir",
        default=".",
        help="Base directory for resolving video paths",
    )
    parser.add_argument(
        "--holdout_model",
        default="veo-3.1",
        help="Video model to hold out for testing",
    )
    parser.add_argument(
        "--val-output",
        default=None,
        help="Output path for validation split JSONL",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of videos to hold out for validation (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="Write/validate JSONL but do not append docs/exp-results/training/training_registry.json",
    )


def append_val_command_args(parts: list[str], args: argparse.Namespace) -> None:
    """Append shared val-split CLI args to a command_args list."""
    if args.base_dir != ".":
        parts.extend(["--base_dir", args.base_dir])
    if args.holdout_model:
        parts.extend(["--holdout_model", args.holdout_model])
    if args.val_output:
        parts.extend([
            "--val-output", args.val_output,
            "--val-ratio", str(args.val_ratio),
            "--seed", str(args.seed),
        ])


def run_convert(
    *,
    samples: list[TrainingSample],
    args: argparse.Namespace,
    metadata: dict[str, Any],
    module: str,
    command_args: list[str],
) -> int:
    """Shared convert flow: split, write, validate, register."""
    if not samples:
        raise ValueError("No valid samples; check input paths and --base_dir")

    n_val_written = None
    if args.val_output:
        train_samples, val_samples = split_samples_by_video(
            samples,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        n_val_written = write_swift_jsonl(val_samples, args.val_output)
        samples = train_samples

    n_train_written = write_swift_jsonl(samples, args.output)

    summary = validate_jsonl(args.output)
    if n_val_written is not None:
        validate_jsonl(args.val_output)

    if args.skip_registry:
        logger.info("Skipping training registry append")
    else:
        append_training_registry(
            module=module,
            output=args.output,
            holdout_model=args.holdout_model,
            val_output=args.val_output,
            val_ratio=args.val_ratio,
            seed=args.seed,
            n_train=n_train_written,
            n_val=n_val_written,
            metadata=metadata,
            command_args=command_args,
        )
    return 0 if not summary["errors"] else 1
