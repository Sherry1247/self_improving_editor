"""Count critic: penalize object duplication or disappearance."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from .base_critic import BaseCritic


class CountCritic(BaseCritic):
    """
    Compare per-label object counts before and after editing.

    Normalized score:
        count_score = max(0, 1 - penalty / max_penalty)
    where
        penalty = sum(|count_before[l] - count_after[l]|) over all labels
        max_penalty = 2 * sum(count_before)
    """

    def evaluate(
        self,
        detections_before: Dict[str, Any],
        detections_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        counts_before = self._label_counts(detections_before)
        counts_after = self._label_counts(detections_after)

        all_labels = set(counts_before) | set(counts_after)
        penalty = sum(
            abs(counts_before[label] - counts_after[label])
            for label in all_labels
        )

        total_before = sum(counts_before.values())
        if total_before == 0:
            count_score = 1.0 if sum(counts_after.values()) == 0 else 0.0
        else:
            max_penalty = 2 * total_before
            count_score = max(0.0, 1.0 - (penalty / max_penalty))

        return {
            "counts_before": dict(counts_before),
            "counts_after": dict(counts_after),
            "penalty": penalty,
            "count_score": round(count_score, 4),
        }

    def get_name(self) -> str:
        return "count"

    @staticmethod
    def _label_counts(detections: Dict[str, Any]) -> Counter:
        labels = [
            str(obj.get("label", "")).strip().lower()
            for obj in detections.get("objects", [])
            if str(obj.get("label", "")).strip()
        ]
        return Counter(labels)
