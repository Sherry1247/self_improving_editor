"""Detection critic: object preservation after editing."""

from __future__ import annotations

from typing import Any, Dict, Set

from .base_critic import BaseCritic


class DetectionCritic(BaseCritic):
    """
    Measures whether objects detected in the original image still appear
    after editing.

    Score = (# preserved labels) / (# unique labels in before image)
    """

    def evaluate(
        self,
        detections_before: Dict[str, Any],
        detections_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        labels_before = self._unique_labels(detections_before)
        labels_after = self._unique_labels(detections_after)

        preserved = {
            label: label in labels_after
            for label in sorted(labels_before)
        }

        if not labels_before:
            detection_score = 1.0
        else:
            preserved_count = sum(1 for kept in preserved.values() if kept)
            detection_score = preserved_count / len(labels_before)

        return {
            "preserved": preserved,
            "labels_before": sorted(labels_before),
            "labels_after": sorted(labels_after),
            "detection_score": round(detection_score, 4),
        }

    def get_name(self) -> str:
        return "detection"

    @staticmethod
    def _unique_labels(detections: Dict[str, Any]) -> Set[str]:
        labels: Set[str] = set()
        for obj in detections.get("objects", []):
            label = str(obj.get("label", "")).strip().lower()
            if label:
                labels.add(label)
        return labels
