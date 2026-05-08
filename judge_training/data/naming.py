"""Centralized path-derivation rules for the judge training pipeline.

Every script — Python or bash — that needs to resolve a path from a
prompt-config name should go through this module.  Bash callers can
``eval "$(python -m judge_training.data.naming <prompt_config>)"``
to set the shell variables.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SWIFT_DIR = Path("data/training/swift")
_PROMPTS_DIR = Path("evals/prompts")


def prompt_config_stem(prompt_config: str) -> str:
    """Derive the canonical stem from a prompt config filename.

    Accepts ``'subq+human.yaml'``, ``'subq+human'``, or
    ``'evals/prompts/subq+human.yaml'``.
    """
    return Path(prompt_config).stem


def swift_train_path(stem: str) -> str:
    return str(_SWIFT_DIR / f"{stem}_train.jsonl")


def swift_val_path(stem: str) -> str:
    return str(_SWIFT_DIR / f"{stem}_val.jsonl")


def swift_test_path(stem: str, split: str) -> str:
    return str(_SWIFT_DIR / f"{stem}_{split}.jsonl")


def prompt_yaml_path(prompt_config: str) -> str:
    name = Path(prompt_config).name
    if not name.endswith(".yaml"):
        name += ".yaml"
    return str(_PROMPTS_DIR / name)


def output_dir(stem: str, model: str = "Qwen3.5-9B-swift-judge") -> str:
    return f"output/{model}/{stem}"


REGISTRY_PATH = (
    _PROJECT_ROOT / "docs" / "exp-results" / "training" / "training_registry.json"
)

VERSIONS_PATH = (
    _PROJECT_ROOT / "docs" / "exp-results" / "training" / "schemas" / "versions.json"
)

REGISTRY_REL = str(REGISTRY_PATH.relative_to(_PROJECT_ROOT))
VERSIONS_REL = str(VERSIONS_PATH.relative_to(_PROJECT_ROOT))


def _main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python -m {__name__} <prompt_config>", file=sys.stderr)
        sys.exit(1)
    stem = prompt_config_stem(sys.argv[1])
    pairs = {
        "STEM": stem,
        "TRAIN_JSONL": swift_train_path(stem),
        "VAL_JSONL": swift_val_path(stem),
        "OUTPUT_DIR": output_dir(stem),
        "PROMPT_YAML": prompt_yaml_path(sys.argv[1]),
        "SCHEMA_VERSIONS": VERSIONS_REL,
        "TRAINING_REGISTRY": REGISTRY_REL,
    }
    for key, val in pairs.items():
        print(f"{key}={shlex.quote(val)}")


if __name__ == "__main__":
    _main()
