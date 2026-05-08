"""Prompt templates and configuration for VLM evaluation and judge training.

Supports loading prompt configs from YAML files for A/B comparison.
Default config loaded at import for backward compatibility.

Usage:
    from evals.prompts import PromptConfig
    cfg = PromptConfig.load("subq+human.yaml")
    prompt = cfg.build_eval_prompt(caption, "SA")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evals.physics_criteria import SUB_QUESTIONS, SubQuestion

_PROMPTS_DIR = Path(__file__).parent


# ──────────────────────────────────────────────────────────────────────────────
# PromptConfig
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PromptConfig:
    """A prompt template set loaded from YAML."""

    name: str
    scheme: str
    system_prompt: str
    general_keys: list[str]
    eval_prompts: dict[str, str]
    training_prompts: dict[str, str]
    physical_template: str
    physical_sub_questions: bool = False
    sub_questions: dict[str, str] | None = None

    @classmethod
    def load(cls, name: str) -> PromptConfig:
        """Load evals/prompts/{name}; name must be a YAML filename."""
        raw_path = Path(name)
        if raw_path.name != name or raw_path.suffix != ".yaml":
            raise ValueError(
                "prompt config must be a YAML filename under evals/prompts, "
                f"got {name!r}"
            )
        path = _PROMPTS_DIR / name
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> PromptConfig:
        def _resolve(raw: str | dict) -> str:
            if isinstance(raw, str):
                return raw
            template = raw["template"]
            considerations = raw.get("considerations")
            if considerations and "{considerations}" in template:
                block = "\n".join(
                    f"{i}. {c}" for i, c in enumerate(considerations, 1)
                )
                template = template.replace("{considerations}", block)
            return template

        return cls(
            name=data.get("name", ""),
            scheme=data.get("scheme", "plain"),
            system_prompt=data.get(
                "system_prompt", "You are a strict video evaluation model."
            ),
            general_keys=data.get("general_keys", ["SA", "PTV", "persistence"]),
            eval_prompts={
                k: _resolve(v) for k, v in data.get("eval_prompts", {}).items()
            },
            training_prompts={
                k: _resolve(v) for k, v in data.get("training_prompts", {}).items()
            },
            physical_template=data.get("physical_template", data.get("physical_training_template", "")),
            physical_sub_questions=bool(data.get("physical_sub_questions", False)),
            sub_questions=data.get("sub_questions"),
        )

    @property
    def _answer_format(self) -> str | None:
        if self.sub_questions:
            return self.sub_questions.get("answer_format")
        return None

    def build_eval_prompt(
        self, prompt_text: str, metric: str, *, answers_block: str = "",
    ) -> str:
        questions_block, question_keys_str = build_general_questions_block(
            metric, self._answer_format,
        )
        return self.eval_prompts[metric].format(
            prompt=prompt_text,
            questions_block=questions_block,
            question_keys_str=question_keys_str,
            answers_block=answers_block,
        )

    def build_training_prompt(
        self, prompt_text: str, dim: str, *, answers_block: str = "",
    ) -> str:
        prompts = self.training_prompts or self.eval_prompts
        questions_block, question_keys_str = build_general_questions_block(
            dim, self._answer_format,
        )
        return prompts[dim].format(
            prompt=prompt_text,
            questions_block=questions_block,
            question_keys_str=question_keys_str,
            answers_block=answers_block,
        )

    def build_physical_prompt(
        self, prompt_text: str, law: str, criteria: str,
        *, answers_block: str = "",
    ) -> str:
        if self.physical_sub_questions:
            questions_block, question_keys_str = build_physical_questions_block(
                law, criteria, self._answer_format,
            )
        else:
            questions_block = ""
            question_keys_str = ""
        return self.physical_template.format(
            prompt=prompt_text,
            law=law,
            criteria=criteria,
            questions_block=questions_block,
            question_keys_str=question_keys_str,
            answers_block=answers_block,
        )




# ──────────────────────────────────────────────────────────────────────────────
# Scoring dimension keys
# ──────────────────────────────────────────────────────────────────────────────

GENERAL_KEYS = ["SA", "PTV", "persistence"]
GENERAL_DIMS = GENERAL_KEYS

GENERAL_SUB_QUESTIONS: dict[str, list[str]] = {
    "SA": [
        "Are the main objects in the caption present in the video?",
        "Are the key actions or interactions from the caption visible?",
        "Are important scene attributes and relationships preserved?",
        "Does the video avoid major contradictions to the caption?",
    ],
    "PTV": [
        "Do causes appear before their effects?",
        "Do physical events unfold in a plausible temporal order?",
        "Are motion transitions continuous rather than abrupt jumps or loops?",
        "Does the sequence avoid impossible reversals or repeated resets?",
    ],
    "persistence": [
        "Do objects maintain consistent existence throughout the video?",
        "Do objects keep a stable shape, size, color, and texture?",
        "Do objects avoid disappearing, appearing, or transforming unexpectedly?",
        "Do objects preserve identity through motion and brief occlusion?",
    ],
}


_ANSWER_FORMAT_SUFFIX = {
    "answer": "Answer each with yes/no/uncertain.",
}


def _format_questions_block(
    questions: list[str], answer_format: str | None = None,
) -> tuple[str, str]:
    keys: list[str] = []
    lines: list[str] = []
    for i, question in enumerate(questions, 1):
        key = f"q{i}"
        keys.append(key)
        lines.append(f"{key}: {question}")
    suffix = _ANSWER_FORMAT_SUFFIX.get(answer_format or "")
    if suffix:
        lines.append(suffix)
    return "\n".join(lines), ", ".join(keys)


def build_general_questions_block(
    dim: str, answer_format: str | None = None,
) -> tuple[str, str]:
    questions = GENERAL_SUB_QUESTIONS.get(dim)
    if not questions:
        questions = [f"Does the video satisfy the {dim} criterion?"]
    return _format_questions_block(questions, answer_format)


def build_physical_questions_block(
    law: str, criteria: str, answer_format: str | None = None,
) -> tuple[str, str]:
    sub_qs = SUB_QUESTIONS.get(law)
    if not sub_qs:
        sub_qs = [SubQuestion(f"{law}_q1", criteria, violation="no")]
    return _format_questions_block([sq.question for sq in sub_qs], answer_format)