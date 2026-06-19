"""Rule-based prompt refinement (no LLM APIs)."""

from __future__ import annotations

from typing import Any, Dict


class PromptRefiner:
    """
    Append simple constraints when detection or count scores indicate failure.

    Future phases may replace this with VLM-guided refinement while keeping
    the same public ``refine`` interface.
    """

    def refine(
        self,
        original_instruction: str,
        detection_result: Dict[str, Any],
        count_result: Dict[str, Any],
    ) -> str:
        instruction = original_instruction.strip()
        suffixes = []

        lost_objects = [
            label
            for label, kept in detection_result.get("preserved", {}).items()
            if not kept
        ]
        if lost_objects:
            suffixes.append("while preserving all foreground objects")

        counts_before = count_result.get("counts_before", {})
        counts_after = count_result.get("counts_after", {})
        if self._counts_changed(counts_before, counts_after):
            suffixes.append("without creating additional objects")

        if not suffixes:
            return instruction

        if len(suffixes) == 1:
            return f"{instruction} {suffixes[0]}"

        return f"{instruction} {suffixes[0]} and {suffixes[1]}"

    @staticmethod
    def _counts_changed(
        counts_before: Dict[str, int],
        counts_after: Dict[str, int],
    ) -> bool:
        labels = set(counts_before) | set(counts_after)
        return any(counts_before.get(label, 0) != counts_after.get(label, 0) for label in labels)
