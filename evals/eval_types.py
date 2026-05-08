"""Typed data structures for VLM evaluation pipeline — parse, don't validate.

Every constructor enforces types and invariants.  Once you have a PromptEntry
or LawScore, the data is guaranteed clean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PromptEntry — parsed from prompts JSON at the boundary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptEntry:
    """A parsed, validated prompt entry for evaluation.

    Required: prompt must be non-empty.
    physical_laws: None means the field was absent (skip physical eval),
                   [] means explicitly empty.
    """
    prompt: str
    physical_laws: list[str] | None = None
    domain: str = ""
    first_frame_image: str = ""
    dataset: str = ""
    video: str = ""

    def __post_init__(self):
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError(f"PromptEntry.prompt must be non-empty str, got {self.prompt!r}")

    @property
    def has_physical_laws(self) -> bool:
        return self.physical_laws is not None


def parse_prompt_entry(raw: dict, *, key: str = "") -> PromptEntry | None:
    """Parse a raw JSON dict into PromptEntry.

    Returns None on bad data (tolerant — logs warning instead of raising).
    """
    prompt = raw.get("prompt") or raw.get("description") or ""
    if not isinstance(prompt, str) or not prompt.strip():
        logger.warning("Skipping entry %r: missing or empty prompt", key)
        return None

    physical_laws = raw.get("physical_laws")
    if physical_laws is not None and not isinstance(physical_laws, list):
        logger.warning("Entry %r: physical_laws is not a list, ignoring", key)
        physical_laws = None

    domain = raw.get("_domain") or raw.get("domain") or raw.get("our_domain") or ""

    try:
        return PromptEntry(
            prompt=prompt.strip().strip('"'),
            physical_laws=physical_laws,
            domain=str(domain),
            first_frame_image=str(raw.get("first_frame_image") or ""),
            dataset=str(raw.get("dataset") or ""),
            video=str(raw.get("video") or ""),
        )
    except (ValueError, TypeError) as e:
        logger.warning("Skipping entry %r: %s", key, e)
        return None


# ---------------------------------------------------------------------------
# LawScore — per-law evaluation result
# ---------------------------------------------------------------------------

SCORED = "scored"
NOT_OBSERVED = "not_observed"
FAILED = "failed"
_VALID_STATUSES = frozenset({SCORED, NOT_OBSERVED, FAILED})


@dataclass(frozen=True)
class LawScore:
    """Result of scoring one physical law on one video."""
    law: str
    score: int | None
    status: str  # SCORED | NOT_OBSERVED | FAILED
    sub_answers: dict[str, str] | None = None  # raw yes/no/na per sub-question

    def __post_init__(self):
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"LawScore.status must be one of {_VALID_STATUSES}, got {self.status!r}"
            )
        if self.status == SCORED and self.score is None:
            raise ValueError(f"LawScore({self.law!r}): status='scored' but score is None")

    @classmethod
    def scored(cls, law: str, score: int, sub_answers: dict[str, str] | None = None) -> LawScore:
        return cls(law=law, score=score, status=SCORED, sub_answers=sub_answers)

    @classmethod
    def not_observed(cls, law: str, sub_answers: dict[str, str] | None = None) -> LawScore:
        return cls(law=law, score=None, status=NOT_OBSERVED, sub_answers=sub_answers)

    @classmethod
    def failed(cls, law: str) -> LawScore:
        return cls(law=law, score=None, status=FAILED)

    def to_dict(self) -> dict:
        d: dict = {"score": self.score, "status": self.status}
        if self.sub_answers is not None:
            d["sub_answers"] = self.sub_answers
        return d


# ---------------------------------------------------------------------------
# PhysicalSummary — aggregated physical results for one video
# ---------------------------------------------------------------------------

@dataclass
class PhysicalSummary:
    """Aggregated physical evaluation for one video."""
    laws: dict[str, LawScore]
    missing_laws: list[str]
    coverage: float
    avg: float | None

    @classmethod
    def from_law_scores(
        cls, law_scores: dict[str, LawScore], total_laws: int,
    ) -> PhysicalSummary:
        scored_values = [
            ls.score for ls in law_scores.values() if ls.status == SCORED
        ]
        missing = [name for name, ls in law_scores.items() if ls.status == FAILED]
        coverage = len(scored_values) / total_laws if total_laws else 0
        avg = sum(scored_values) / len(scored_values) if scored_values else None
        return cls(
            laws=law_scores,
            missing_laws=missing,
            coverage=round(coverage, 4),
            avg=round(avg, 4) if avg is not None else None,
        )

    def to_dict(self) -> dict:
        return {
            "laws": {name: ls.to_dict() for name, ls in self.laws.items()},
            "missing_laws": self.missing_laws,
            "coverage": self.coverage,
            "avg": self.avg,
        }
