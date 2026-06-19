"""Count Critic implementation."""

import logging
from collections import Counter
from typing import Any, Dict, Tuple

logger = logging.getLogger("closed_loop_editor")


class CountCritic:
    """
    Count Critic comparing object counts before and after editing.
    
    Metrics:
        - count_difference: Sum of absolute differences in object counts per category.
    """

    def evaluate(
        self,
        scene_graph_before: Dict[str, Any],
        scene_graph_after: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate count consistency.

        Args:
            scene_graph_before: Scene graph of pre-edit image.
            scene_graph_after: Scene graph of post-edit image.

        Returns:
            Tuple of (score, metrics_dict) where score is a normalized count similarity.
        """
        objs_bef = scene_graph_before.get("objects", [])
        objs_aft = scene_graph_after.get("objects", [])

        counts_bef = Counter([obj["label"] for obj in objs_bef])
        counts_aft = Counter([obj["label"] for obj in objs_aft])

        all_categories = set(counts_bef.keys()) | set(counts_aft.keys())
        total_diff = 0

        for cat in all_categories:
            total_diff += abs(counts_bef[cat] - counts_aft[cat])

        # Normalize score into [0, 1]. Perfect matches get 1.0.
        total_objects = len(objs_bef) + len(objs_aft)
        if total_objects == 0:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (total_diff / float(max(1, len(objs_bef)))))

        metrics = {
            "count_difference": total_diff,
            "counts_before": dict(counts_bef),
            "counts_after": dict(counts_aft),
        }

        logger.info("CountCritic: diff=%d, score=%.3f", total_diff, score)
        return score, metrics
