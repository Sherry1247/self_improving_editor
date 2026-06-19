"""Scene Graph Builder implementation."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("closed_loop_editor")


class SceneGraphBuilder:
    """
    Builds a Scene Graph from object detections and segmentation masks.
    
    Identifies objects and infers relationships (near, standing_on, floating_on, inside)
    based on bounding box and mask geometries.
    """

    def __init__(
        self,
        near_threshold_px: float = 150.0,
        standing_on_overlap: float = 0.2,
        standing_on_height_diff: float = 50.0,
        inside_iou_threshold: float = 0.6,
    ):
        self.near_threshold_px = near_threshold_px
        self.standing_on_overlap = standing_on_overlap
        self.standing_on_height_diff = standing_on_height_diff
        self.inside_iou_threshold = inside_iou_threshold

    def build(
        self,
        detections: List[Dict[str, Any]],
        masks: List[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Build a scene graph from detections and masks.

        Args:
            detections: List of detections from Grounding DINO:
                        [{"label": str, "score": float, "box": (x1, y1, x2, y2)}]
            masks: List of binary masks (boolean numpy arrays) matching the detections.

        Returns:
            Dict representing the scene graph:
            {
                "objects": [
                    {"id": str, "label": str, "score": float, "box": (x1, y1, x2, y2), "centroid": (x, y), "area": int}
                ],
                "relationships": [
                    {"subject": str, "relation": str, "object": str}
                ]
            }
        """
        if len(detections) != len(masks):
            logger.warning(
                "Mismatch between detections (%d) and masks (%d). Truncating.",
                len(detections),
                len(masks),
            )
            min_len = min(len(detections), len(masks))
            detections = detections[:min_len]
            masks = masks[:min_len]

        objects = []
        for i, (det, mask) in enumerate(zip(detections, masks)):
            label = det["label"]
            score = det["score"]
            box = det["box"]
            x1, y1, x2, y2 = box

            # Calculate centroid from mask
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                cx = float(np.mean(x_indices))
                cy = float(np.mean(y_indices))
                area = int(len(y_indices))
            else:
                # Fallback to bbox center
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                area = int((x2 - x1) * (y2 - y1))

            objects.append({
                "id": f"obj_{i}_{label}",
                "label": label,
                "score": score,
                "box": box,
                "centroid": (cx, cy),
                "area": area,
            })

        relationships = []
        num_objs = len(objects)

        for i in range(num_objs):
            obj_a = objects[i]
            mask_a = masks[i]
            box_a = obj_a["box"]
            ax1, ay1, ax2, ay2 = box_a
            acx, acy = obj_a["centroid"]

            for j in range(num_objs):
                if i == j:
                    continue
                obj_b = objects[j]
                mask_b = masks[j]
                box_b = obj_b["box"]
                bx1, by1, bx2, by2 = box_b
                bcx, bcy = obj_b["centroid"]

                # 1. RELATION: inside
                # Calculate what fraction of A's mask overlaps with B
                overlap_mask = np.logical_and(mask_a, mask_b)
                overlap_area = np.sum(overlap_mask)
                area_a = np.sum(mask_a)
                containment_ratio = float(overlap_area / area_a) if area_a > 0 else 0.0

                if containment_ratio >= self.inside_iou_threshold:
                    relationships.append({
                        "subject": obj_a["id"],
                        "relation": "inside",
                        "object": obj_b["id"],
                    })
                    continue

                # 2. RELATION: standing_on / floating_on
                # Object A is vertically above B, and A's bottom touches B's top
                # And there's horizontal overlap
                h_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
                width_a = ax2 - ax1
                h_overlap_ratio = h_overlap / float(width_a) if width_a > 0 else 0.0

                # A's bottom y (ay2) is near B's top y (by1), and A is above B
                y_diff = by1 - ay2
                is_above = acy < bcy

                if (
                    is_above
                    and -self.standing_on_height_diff <= y_diff <= self.standing_on_height_diff
                    and h_overlap_ratio >= self.standing_on_overlap
                ):
                    # Determine relation type based on target object class name
                    dest_label = obj_b["label"].lower()
                    if "river" in dest_label or "water" in dest_label or "lake" in dest_label:
                        relation = "floating_on"
                    else:
                        relation = "standing_on"

                    relationships.append({
                        "subject": obj_a["id"],
                        "relation": relation,
                        "object": obj_b["id"],
                    })
                    continue

                # 3. RELATION: near
                # Euclidean distance between centroids is small
                dist = np.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)
                if dist < self.near_threshold_px:
                    # To prevent duplicated "near" relationships from A to B and B to A,
                    # we can output them unidirectionally (e.g. from smaller index to larger index)
                    if i < j:
                        relationships.append({
                            "subject": obj_a["id"],
                            "relation": "near",
                            "object": obj_b["id"],
                        })

        return {
            "objects": objects,
            "relationships": relationships,
        }
