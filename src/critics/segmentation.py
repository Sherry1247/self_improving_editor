"""Segmentation Critic implementation."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from .utils import match_objects

logger = logging.getLogger("closed_loop_editor")


class SegmentationCritic:
    """
    Segmentation Critic evaluating mask preservation.
    
    Metrics:
        - mask_iou: Average Intersection over Union (IoU) of binary masks for matched objects.
    """

    def evaluate(
        self,
        scene_graph_before: Dict[str, Any],
        scene_graph_after: Dict[str, Any],
        masks_before: List[np.ndarray],
        masks_after: List[np.ndarray],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate pixel-wise segmentation mask preservation.

        Args:
            scene_graph_before: Pre-edit scene graph.
            scene_graph_after: Post-edit scene graph.
            masks_before: List of pre-edit binary masks.
            masks_after: List of post-edit binary masks.

        Returns:
            Tuple of (score, metrics_dict) where score is average mask IoU.
        """
        objs_bef = scene_graph_before.get("objects", [])
        objs_aft = scene_graph_after.get("objects", [])

        if not objs_bef:
            # If no objects before, score is 1.0 if also none after, else 0.0
            score = 1.0 if not objs_aft else 0.0
            return score, {"mask_iou": score}

        matched_pairs, _, _ = match_objects(objs_bef, objs_aft)

        if not matched_pairs:
            return 0.0, {"mask_iou": 0.0}

        ious = []
        for obj_bef, obj_aft in matched_pairs:
            # Extract index from ID e.g. "obj_3_dog" -> 3
            try:
                idx_bef = int(obj_bef["id"].split("_")[1])
                idx_aft = int(obj_aft["id"].split("_")[1])

                mask_bef = masks_before[idx_bef]
                mask_aft = masks_after[idx_aft]

                if mask_bef.shape != mask_aft.shape:
                    import cv2
                    # Resize mask_aft to match mask_bef shape
                    mask_aft_uint8 = mask_aft.astype(np.uint8) * 255
                    resized_aft = cv2.resize(
                        mask_aft_uint8,
                        (mask_bef.shape[1], mask_bef.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    mask_aft = resized_aft > 0

                intersection = np.logical_and(mask_bef, mask_aft).sum()
                union = np.logical_or(mask_bef, mask_aft).sum()

                iou = float(intersection / union) if union > 0 else 0.0
                ious.append(iou)
            except Exception as e:
                logger.warning("Failed to calculate mask IoU for matched pair: %s", e)
                ious.append(0.0)

        avg_iou = float(np.mean(ious)) if ious else 0.0
        logger.info("SegmentationCritic: mask_iou=%.3f", avg_iou)

        return avg_iou, {"mask_iou": avg_iou}
