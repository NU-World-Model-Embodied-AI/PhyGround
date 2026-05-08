"""Validated training sample type.

A TrainingSample that exists is valid — all invariants are checked at
construction time.  Downstream code can serialize without re-checking.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from evals.prompts import GENERAL_DIMS
from evals.physics_criteria import ALL_LAWS

ALLOWED_KEYS: frozenset[str] = frozenset(GENERAL_DIMS) | frozenset(ALL_LAWS)


def _resolve_path(video_path: str, base_dir: str | None) -> str:
    if base_dir and not os.path.isabs(video_path):
        video_path = os.path.join(base_dir, video_path)
    return os.path.normpath(video_path)


@dataclass(frozen=True)
class TrainingSample:
    system: str
    user: str
    assistant: str
    video_path: str

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not self.system:
            errors.append("empty system prompt")
        if not self.user:
            errors.append("empty user prompt")
        if not self.assistant:
            errors.append("empty assistant response")
        video_count = self.user.count("<video>") if self.user else 0
        if self.user and video_count != 1:
            errors.append(
                f"user must have exactly 1 <video>, got {video_count}"
            )
        if not self.video_path:
            errors.append("empty video_path")
        elif not os.path.isfile(self.video_path):
            errors.append(f"video not found: {self.video_path}")
        if errors:
            raise ValueError("; ".join(errors))

    # ── serialization ────────────────────────────────────────────────

    def to_swift_dict(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ],
            "videos": [self.video_path],
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_swift_dict(), ensure_ascii=False)

    # ── factory methods ──────────────────────────────────────────────

    @classmethod
    def json_only(
        cls,
        *,
        system: str,
        user: str,
        video_path: str,
        key: str,
        score: int,
        base_dir: str | None = None,
    ) -> TrainingSample:
        """Single-key JSON target, no CoT."""
        if key not in ALLOWED_KEYS:
            raise ValueError(f"unknown key: {key}")
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"score must be int 1-5, got {score!r}")
        if not user.startswith("<video>"):
            user = "<video>" + user
        assistant = json.dumps({key: score}, ensure_ascii=False)
        return cls(
            system=system,
            user=user,
            assistant=assistant,
            video_path=_resolve_path(video_path, base_dir),
        )

    @classmethod
    def cot(
        cls,
        *,
        system: str,
        user: str,
        video_path: str,
        key: str,
        score: int,
        reasoning: str,
        base_dir: str | None = None,
    ) -> TrainingSample:
        """JSON target with reasoning, for CoT distillation."""
        if key not in ALLOWED_KEYS:
            raise ValueError(f"unknown key: {key}")
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"score must be int 1-5, got {score!r}")
        if not reasoning:
            raise ValueError("reasoning must be non-empty for CoT samples")
        if not user.startswith("<video>"):
            user = "<video>" + user
        assistant = json.dumps(
            {"reasoning": reasoning, key: score}, ensure_ascii=False,
        )
        return cls(
            system=system,
            user=user,
            assistant=assistant,
            video_path=_resolve_path(video_path, base_dir),
        )

    @classmethod
    def from_jsonl_row(
        cls,
        row: dict[str, Any],
    ) -> TrainingSample:
        """Parse and validate from a JSONL row dict (json_only format)."""
        messages = row.get("messages")
        if not messages or not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("missing or invalid 'messages'")

        by_role: dict[str, str] = {}
        for m in messages:
            role = m.get("role")
            if role:
                by_role[role] = m.get("content", "")

        if "user" not in by_role or "assistant" not in by_role:
            raise ValueError("missing user or assistant role")
        if "system" not in by_role:
            raise ValueError("missing system role")

        system = by_role["system"]
        user = by_role["user"]
        assistant = by_role["assistant"]

        videos = row.get("videos", [])
        if not videos:
            raise ValueError("missing videos array")
        if len(videos) != 1:
            raise ValueError(f"videos must have length 1, got {len(videos)}")

        if "<video>" in assistant or "<think>" in assistant:
            raise ValueError("assistant contains forbidden tags")
        try:
            parsed = json.loads(assistant)
        except json.JSONDecodeError:
            raise ValueError("assistant is not valid JSON")
        if not isinstance(parsed, dict):
            raise ValueError("assistant must be a JSON object")
        score_keys = {k for k in parsed if k in ALLOWED_KEYS}
        if len(score_keys) != 1:
            raise ValueError(f"assistant must have exactly 1 score key, got {score_keys}")
        k = next(iter(score_keys))
        v = parsed[k]
        if not isinstance(v, int) or not 1 <= v <= 5:
            raise ValueError(f"score must be int 1-5, got {v!r}")

        return cls(system=system, user=user, assistant=assistant, video_path=videos[0])
