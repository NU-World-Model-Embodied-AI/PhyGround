"""Canonical video ID normalization and source-law loading.

Every prompt in the pipeline gets a single canonical vid of the form
``{dataset}_{original_key}`` (no domain prefix, no file extension).
This module is the single source of truth for that mapping.

The canonical vid is computed at runtime — it is NOT stored in source JSONs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# Ordered: first match wins for overlapping prompts (video_phy_2 before video_phy_2).
PROMPT_SOURCES: list[tuple[str, Path]] = [
    ("wmb", ROOT / "data/prompts/wmb/wmb.json"),
    ("physics_iq", ROOT / "data/prompts/physics_iq/physics_iq.json"),
    ("video_phy_2", ROOT / "data/prompts/video_phy_2/video_phy_2.json"),
    ("openvid", ROOT / "data/prompts/openvid/openvid.json"),
]


class SourceEntry(TypedDict):
    laws: list[str]
    dataset: str
    prompt: str
    source_key: str


# ---------------------------------------------------------------------------
# normalize_vid: the single rule for canonical IDs
# ---------------------------------------------------------------------------

def normalize_vid(dataset: str, key: str) -> str:
    """Pure function.  ``(dataset, key) -> canonical vid``.

    Rules:
    - Strip ``.mp4`` suffix from key
    - Prefix with ``{dataset}_``
    - For physics_iq perspectives, key is the full generated_video_name (without .mp4)

    Examples::

        normalize_vid("wmb", "195")                          -> "wmb_195"
        normalize_vid("video_phy_2", "119")                       -> "video_phy_2_119"
        normalize_vid("openvid", "abc.mp4")                  -> "openvid_abc"
        normalize_vid("physics_iq", "0052_...-double-cradle") -> "physics_iq_0052_...-double-cradle"
    """
    key = key.removesuffix(".mp4")
    return f"{dataset}_{key}"


# ---------------------------------------------------------------------------
# resolve_eval_vid: map eval-side video names back to canonical vids
# ---------------------------------------------------------------------------

def resolve_eval_vid(
    eval_video: str,
    eval_dataset_suffix: str,
    reverse_map: dict[str, str],
) -> str | None:
    """Map an eval-side video name to its canonical vid.

    ``reverse_map`` is built by :func:`load_source_laws` — it maps every
    legacy identifier (first_frame_image stem, generated_video_name, numeric
    key) to the canonical vid.

    Falls back to numeric-suffix extraction for video_phy_2/video_phy_2 domain-prefix
    mismatches (e.g. ``buoyancy_119`` -> key ``119``).
    """
    # Direct hit (covers physics_iq perspective names, openvid, exact wmb stems)
    if eval_video in reverse_map:
        return reverse_map[eval_video]

    # Fallback: strip domain prefix, try numeric key
    if "_" in eval_video:
        numeric = eval_video.rsplit("_", 1)[-1]
        if numeric in reverse_map:
            return reverse_map[numeric]

    return None


# ---------------------------------------------------------------------------
# load_source_laws: single loader for all source datasets
# ---------------------------------------------------------------------------

@dataclass
class SourceLawsResult:
    """Result of loading all source prompt JSONs."""
    entries: dict[str, SourceEntry]           # canonical_vid -> SourceEntry
    reverse_map: dict[str, str]               # legacy_id -> canonical_vid
    stats: dict[str, int] = field(default_factory=dict)

    def get(self, canonical_vid: str) -> SourceEntry | None:
        return self.entries.get(canonical_vid)

    def resolve_eval(self, eval_video: str, eval_ds_suffix: str) -> tuple[str, SourceEntry] | None:
        """Resolve eval-side video name -> (canonical_vid, entry)."""
        cvid = resolve_eval_vid(eval_video, eval_ds_suffix, self.reverse_map)
        if cvid and cvid in self.entries:
            return cvid, self.entries[cvid]
        return None

    @property
    def cvid_to_legacies(self) -> dict[str, set[str]]:
        """Inverse of reverse_map: canonical_vid -> set of legacy IDs."""
        if not hasattr(self, "_cvid_to_legacies"):
            inv: dict[str, set[str]] = {}
            for lid, cvid in self.reverse_map.items():
                inv.setdefault(cvid, set()).add(lid)
            self._cvid_to_legacies = inv
        return self._cvid_to_legacies


def load_source_laws(
    sources: list[tuple[str, Path]] | None = None,
) -> SourceLawsResult:
    """Load physical_laws from all canonical prompt JSONs.

    Returns a :class:`SourceLawsResult` with:
    - ``entries``: canonical_vid -> SourceEntry
    - ``reverse_map``: legacy identifiers -> canonical_vid (for eval matching)
    """
    if sources is None:
        sources = PROMPT_SOURCES

    entries: dict[str, SourceEntry] = {}
    reverse_map: dict[str, str] = {}
    stats: dict[str, int] = {}

    def _register(cvid: str, legacy_ids: list[str],
                  laws: list[str], dataset: str,
                  prompt: str, source_key: str) -> None:
        """Register a canonical vid + its legacy aliases (first-match-wins)."""
        if cvid in entries:
            return
        entries[cvid] = SourceEntry(
            laws=laws, dataset=dataset, prompt=prompt, source_key=source_key,
        )
        # Register all legacy identifiers for reverse lookup
        for lid in legacy_ids:
            if lid and lid not in reverse_map:
                reverse_map[lid] = cvid

    for ds_name, path in sources:
        if not path.exists():
            logger.warning("Source not found: %s", path)
            continue
        with open(path) as f:
            data = json.load(f)

        prompts = data.get("prompts", data)
        if not isinstance(prompts, dict):
            continue

        count = 0
        for key, item in prompts.items():
            if not isinstance(item, dict):
                continue
            if item.get("status") != "kept":
                continue

            laws = item.get("physical_laws", [])
            if not laws:
                continue

            prompt_text = item.get("prompt", item.get("description", ""))

            if ds_name == "physics_iq":
                # Each perspective is a separate vid
                for persp in item.get("perspectives", []):
                    gvn = persp.get("generated_video_name", "")
                    gvn_bare = gvn.removesuffix(".mp4")
                    if not gvn_bare:
                        continue
                    cvid = normalize_vid(ds_name, gvn_bare)
                    _register(
                        cvid,
                        legacy_ids=[gvn_bare],
                        laws=laws, dataset=ds_name,
                        prompt=prompt_text, source_key=key,
                    )
                    count += 1
            elif ds_name == "openvid":
                key_bare = key.removesuffix(".mp4")
                cvid = normalize_vid(ds_name, key_bare)
                _register(
                    cvid,
                    legacy_ids=[key_bare],
                    laws=laws, dataset=ds_name,
                    prompt=prompt_text, source_key=key,
                )
                count += 1
            else:
                # wmb, video_phy_2: first_frame_image stem is the legacy ID
                # video_phy_2 entries with subset=video_phy_2 get dataset="video_phy_2"
                effective_ds = item.get("subset", ds_name)
                ff = item.get("first_frame_image", "")
                ff_stem = Path(ff).stem if ff else key
                cvid = normalize_vid(effective_ds, key)
                _register(
                    cvid,
                    legacy_ids=[ff_stem, key],  # both stem and numeric key
                    laws=laws, dataset=effective_ds,
                    prompt=prompt_text, source_key=key,
                )
                count += 1

        stats[ds_name] = count
        logger.info("Loaded %d prompts from %s", count, path.name)

    return SourceLawsResult(entries=entries, reverse_map=reverse_map, stats=stats)
