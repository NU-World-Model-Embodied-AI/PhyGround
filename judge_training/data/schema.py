"""Training data schema versioning.

Auto-computes a schema fingerprint from scheme name, prompt templates,
target format, dimensions, and score scales.  Versions are scoped per
scheme — each scheme has its own independent version counter.

Schema history is persisted in
``docs/exp-results/training/schemas/versions.json``.

Usage
-----
    from judge_training.data.schema import resolve_schema

    schema = resolve_schema("subq+human.yaml")
    schema.scheme       # "subq_hint"
    schema.version      # 1
    schema.fingerprint  # "a1b2c3d4"
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from judge_training.data.naming import VERSIONS_PATH
from evals.prompts import (
    PromptConfig,
    GENERAL_DIMS,
)

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 80


def _compute_fingerprint(cfg: PromptConfig) -> str:
    """Hash all schema-defining elements into an 8-char hex fingerprint."""
    parts: list[str] = [
        cfg.scheme,
        cfg.system_prompt,
        json.dumps(cfg.training_prompts, sort_keys=True, ensure_ascii=False),
        cfg.physical_template,
        json.dumps(GENERAL_DIMS, ensure_ascii=False),
    ]
    if cfg.sub_questions:
        parts.append(json.dumps(cfg.sub_questions, sort_keys=True, ensure_ascii=False))
    blob = "\n---\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def _load_versions() -> dict[str, list[dict[str, Any]]]:
    if VERSIONS_PATH.is_file():
        with open(VERSIONS_PATH) as f:
            data = json.load(f)
        if isinstance(data, list):
            return {"_legacy": data}
        return data
    return {}


def _save_versions(versions: dict[str, list[dict[str, Any]]]) -> None:
    VERSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VERSIONS_PATH, "w") as f:
        json.dump(versions, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _snapshot(cfg: PromptConfig, prompt_config: str) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "prompt_config": prompt_config,
        "prompt_config_name": cfg.name,
        "scheme": cfg.scheme,
        "system_prompt": cfg.system_prompt,
        "general_dims": list(GENERAL_DIMS),
        "perdim_templates": {k: v[:_PREVIEW_LEN] + "..." for k, v in cfg.training_prompts.items()},
        "physical_perdim_template": cfg.physical_template[:_PREVIEW_LEN] + "...",
    }
    if cfg.sub_questions:
        snap["sub_questions"] = cfg.sub_questions
    return snap


@dataclass
class SchemaInfo:
    scheme: str
    version: int
    fingerprint: str


def resolve_schema(prompt_config: str = "default.yaml") -> SchemaInfo:
    """Resolve current schema version, auto-incrementing if fingerprint is new.

    Versions are scoped per scheme — different schemes have independent
    version counters.
    """
    cfg = PromptConfig.load(prompt_config)
    fp = _compute_fingerprint(cfg)
    scheme = cfg.scheme
    all_versions = _load_versions()
    scheme_versions = all_versions.get(scheme, [])

    for entry in scheme_versions:
        if entry["fingerprint"] == fp:
            return SchemaInfo(scheme=scheme, version=entry["version"], fingerprint=fp)

    if scheme_versions:
        next_v = max(e["version"] for e in scheme_versions) + 1
    else:
        next_v = 1

    new_entry = {
        "version": next_v,
        "fingerprint": fp,
        "snapshot": _snapshot(cfg, prompt_config),
    }
    scheme_versions.append(new_entry)
    all_versions[scheme] = scheme_versions
    _save_versions(all_versions)
    logger.info(
        "New training data schema: %s/v%d (fingerprint=%s)", scheme, next_v, fp,
    )

    return SchemaInfo(scheme=scheme, version=next_v, fingerprint=fp)
