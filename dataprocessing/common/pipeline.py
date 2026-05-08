"""Pipeline quality checks: staleness, missing data, empty laws.

Default behavior is warn-and-continue.  Pass ``--strict`` to fail on
issues that would otherwise only warn (for CI / batch runs).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Datasets where every prompt is expected to have physical_laws.
# Edge cases (empty perspectives, removed status) are the only exceptions.
LAWS_REQUIRED_DATASETS = frozenset([
    "wmb", "video_phy_2", "physics_iq", "video_phy_2", "openvid",
])


class PipelineCheck:
    """Collect warnings and errors during a pipeline run, report at the end.

    Usage::

        checker = PipelineCheck(strict=args.strict)
        # ... pipeline logic ...
        checker.check_staleness(source_path, eval_path)
        checker.check_missing_ratio(missing=5, total=279)
        checker.check_empty_laws("wmb_195", [], "wmb")
        # ... at the end ...
        score = checker.report()
        checker.finalize()   # raises if any FAIL-level issues
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._missing_count = 0
        self._total_count = 0
        self._empty_laws_count = 0
        self._stale_count = 0

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_staleness(self, source_path: Path, eval_path: Path) -> None:
        """Warn/fail if source JSON is older than eval JSON."""
        try:
            src_mtime = os.path.getmtime(source_path)
            eval_mtime = os.path.getmtime(eval_path)
        except OSError:
            return

        if src_mtime < eval_mtime:
            self._stale_count += 1
            msg = (f"Source {source_path.name} (mtime {src_mtime:.0f}) "
                   f"is older than eval {eval_path.name} (mtime {eval_mtime:.0f})")
            if self.strict:
                self._errors.append(msg)
            else:
                self._warnings.append(msg)

    def check_missing_ratio(self, missing: int, total: int) -> None:
        """Check ratio of unmatched vids.  >10% always fails."""
        self._missing_count = missing
        self._total_count = total

        if total == 0:
            return

        ratio = missing / total
        if ratio > 0.10:
            self._errors.append(
                f"Vid mismatch too high: {missing}/{total} ({ratio:.1%})")
        elif ratio > 0.01:
            msg = f"Moderate vid mismatch: {missing}/{total} ({ratio:.1%})"
            if self.strict:
                self._errors.append(msg)
            else:
                self._warnings.append(msg)
        elif missing > 0:
            self._warnings.append(
                f"Minor vid mismatch: {missing}/{total} ({ratio:.1%})")

    def check_empty_laws(self, vid: str, laws: list, dataset: str,
                         resolved: bool = True) -> None:
        """Check if a prompt is missing physical_laws that it should have.

        Args:
            resolved: Whether this vid was successfully matched to a source
                entry.  Unresolved vids get a warn (already counted by
                check_missing_ratio); only resolved-but-empty is an error.
        """
        if laws:
            return
        self._empty_laws_count += 1

        if not resolved:
            # Already captured by check_missing_ratio — just warn here
            self._warnings.append(
                f"No laws for {vid} (unresolved, dataset={dataset})")
        elif dataset in LAWS_REQUIRED_DATASETS:
            self._errors.append(f"Missing laws for {vid} (dataset={dataset})")
        else:
            self._warnings.append(
                f"No laws for {vid} (dataset={dataset}, allowed)")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> float:
        """Log summary and return quality score (0.0 - 1.0)."""
        total = self._total_count or 1
        missing_ratio = self._missing_count / total
        empty_ratio = self._empty_laws_count / total

        score = 1.0 - (
            0.5 * missing_ratio          # unmatched vids are most critical
            + 0.3 * empty_ratio           # missing laws degrade downstream use
            + 0.2 * min(self._stale_count / max(total, 1), 1.0)  # stale sources
        )
        score = max(0.0, min(1.0, score))

        logger.info("Pipeline Quality Score: %.2f", score)
        if self._warnings:
            logger.info("Warnings (%d):", len(self._warnings))
            # Show first 10, summarize rest
            for w in self._warnings[:10]:
                logger.warning("  %s", w)
            if len(self._warnings) > 10:
                logger.warning("  ... and %d more",
                               len(self._warnings) - 10)
        if self._errors:
            logger.error("Errors (%d):", len(self._errors))
            for e in self._errors[:10]:
                logger.error("  %s", e)
            if len(self._errors) > 10:
                logger.error("  ... and %d more", len(self._errors) - 10)

        return score

    def finalize(self) -> None:
        """Raise if any FAIL-level issues were recorded."""
        if self._errors:
            raise RuntimeError(
                f"Pipeline failed with {len(self._errors)} error(s). "
                f"First: {self._errors[0]}"
            )
