"""Score aggregation for MVP critics."""

from __future__ import annotations

from typing import Dict


def compute_final_score(
    detection_score: float,
    count_score: float,
    weights: Dict[str, float],
) -> float:
    """
    Weighted combination of detection and count scores.

    Args:
        detection_score: Detection critic score in [0, 1].
        count_score: Count critic score in [0, 1].
        weights: Mapping with ``detection`` and ``count`` keys summing to 1.

    Returns:
        Final score in [0, 1].
    """
    w_det = weights.get("detection", 0.7)
    w_cnt = weights.get("count", 0.3)
    return round(w_det * detection_score + w_cnt * count_score, 4)
