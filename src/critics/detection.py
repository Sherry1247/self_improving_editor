"""Detection Critic implementation."""

import logging
from typing import Any, Dict, Tuple

from .utils import match_objects

logger = logging.getLogger("closed_loop_editor")


class DetectionCritic:
    """
    Detection Critic evaluating whether original objects still exist after editing.
    
    Metrics:
        - object_recall: Fraction of original objects preserved in edited image.
        - object_precision: Fraction of edited objects that originated in original image.
    """

    def evaluate(
        self,
        scene_graph_before: Dict[str, Any],
        scene_graph_after: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate object consistency.

        Args:
            scene_graph_before: Scene graph of pre-edit image.
            scene_graph_after: Scene graph of post-edit image.

        Returns:
            Tuple of (score, metrics_dict) where score is object recall.
        """
        objs_bef = scene_graph_before.get("objects", [])
        objs_aft = scene_graph_after.get("objects", [])

        if not objs_bef:
            # If there were no objects to begin with, recall is trivially 1.0.
            recall = 1.0
            precision = 1.0 if not objs_aft else 0.0
        else:
            matched, unmatched_bef, unmatched_aft = match_objects(objs_bef, objs_aft)
            recall = len(matched) / len(objs_bef)
            precision = len(matched) / len(objs_aft) if objs_aft else 0.0

        # Score is primarily based on recall (preserving the original subject)
        score = recall

        metrics = {
            "object_recall": recall,
            "object_precision": precision,
        }

        logger.info("DetectionCritic: recall=%.3f, precision=%.3f", recall, precision)
        return score, metrics
