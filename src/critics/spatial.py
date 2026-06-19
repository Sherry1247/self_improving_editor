"""Spatial Critic implementation."""

import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .utils import match_objects

logger = logging.getLogger("closed_loop_editor")


class SpatialCritic:
    """
    Spatial Critic evaluating whether object locations/sizes are preserved.
    
    Metrics:
        - centroid_displacement: Euclidean distance between centroids normalized by image diagonal.
        - area_ratio: Ratio of the smaller area to the larger area of the matched object.
    """

    def evaluate(
        self,
        scene_graph_before: Dict[str, Any],
        scene_graph_after: Dict[str, Any],
        masks_before: List[np.ndarray],
        masks_after: List[np.ndarray],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate spatial alignment.

        Args:
            scene_graph_before: Pre-edit scene graph.
            scene_graph_after: Post-edit scene graph.
            masks_before: List of pre-edit binary masks.
            masks_after: List of post-edit binary masks.

        Returns:
            Tuple of (score, metrics_dict).
        """
        objs_bef = scene_graph_before.get("objects", [])
        objs_aft = scene_graph_after.get("objects", [])

        if not objs_bef:
            score = 1.0 if not objs_aft else 0.0
            return score, {"centroid_displacement": 0.0, "area_ratio": score}

        matched_pairs, _, _ = match_objects(objs_bef, objs_aft)

        if not matched_pairs:
            return 0.0, {"centroid_displacement": 1.0, "area_ratio": 0.0}

        # Calculate image diagonal from mask shape if available
        h, w = 384, 384  # Default
        for mask in masks_before:
            h, w = mask.shape
            break
        img_diagonal = math.sqrt(h**2 + w**2)

        displacements = []
        area_ratios = []

        for obj_bef, obj_aft in matched_pairs:
            cb = obj_bef["centroid"]
            ca = obj_aft["centroid"]

            # Centroid distance
            dist = math.sqrt((cb[0] - ca[0]) ** 2 + (cb[1] - ca[1]) ** 2)
            norm_dist = dist / img_diagonal
            displacements.append(norm_dist)

            # Area ratio
            ab = float(obj_bef["area"])
            aa = float(obj_aft["area"])
            if ab > 0 and aa > 0:
                ratio = min(ab, aa) / max(ab, aa)
            else:
                ratio = 0.0
            area_ratios.append(ratio)

        avg_disp = float(np.mean(displacements)) if displacements else 1.0
        avg_ratio = float(np.mean(area_ratios)) if area_ratios else 0.0

        # High score means low displacement and preserved area
        disp_score = max(0.0, 1.0 - avg_disp)
        score = (disp_score + avg_ratio) / 2.0

        metrics = {
            "centroid_displacement": avg_disp,
            "area_ratio": avg_ratio,
        }

        logger.info(
            "SpatialCritic: displacement=%.3f, area_ratio=%.3f, score=%.3f",
            avg_disp,
            avg_ratio,
            score,
        )

        return score, metrics
