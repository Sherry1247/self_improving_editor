"""Utility functions for critics matching."""

from typing import Any, Dict, List, Tuple


def match_objects(
    objects_before: List[Dict[str, Any]],
    objects_after: List[Dict[str, Any]],
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Match objects before and after editing based on category labels.
    Uses greedy matching based on bounding box Intersection over Union (IoU) 
    or centroid distance as a fallback.

    Args:
        objects_before: List of objects from the pre-edit image.
        objects_after: List of objects from the post-edit image.

    Returns:
        matched_pairs: List of tuples (obj_before, obj_after)
        unmatched_before: List of unmatched objects from objects_before
        unmatched_after: List of unmatched objects from objects_after
    """
    matched_pairs = []
    unmatched_before = list(objects_before)
    unmatched_after = list(objects_after)

    # Group by labels to match objects of the same category
    unique_labels = set(obj["label"] for obj in objects_before) | set(obj["label"] for obj in objects_after)

    for label in unique_labels:
        objs_bef = [obj for obj in unmatched_before if obj["label"] == label]
        objs_aft = [obj for obj in unmatched_after if obj["label"] == label]

        # Greedy matching based on bbox IoU
        # If IoU is 0, we fallback to centroid distance
        cost_matrix = []
        for b in objs_bef:
            row = []
            for a in objs_aft:
                # Bbox IoU
                iou_val = _bbox_iou(b["box"], a["box"])
                # Centroid distance cost (negative distance, normalized by image dimensions)
                dist = _centroid_dist(b["centroid"], a["centroid"])
                # Combined score: IoU is primary, distance is secondary fallback
                score = iou_val + (1.0 / (1.0 + dist))
                row.append(score)
            cost_matrix.append(row)

        # Match greedily
        while objs_bef and objs_aft:
            # Find the highest similarity score in the matrix
            max_val = -1.0
            max_idx = (-1, -1)
            for r_idx, row in enumerate(cost_matrix):
                for c_idx, val in enumerate(row):
                    if val > max_val:
                        max_val = val
                        max_idx = (r_idx, c_idx)

            if max_val == -1.0:
                break

            r, c = max_idx
            matched_bef = objs_bef.pop(r)
            matched_aft = objs_aft.pop(c)

            matched_pairs.append((matched_bef, matched_aft))
            unmatched_before.remove(matched_bef)
            unmatched_after.remove(matched_aft)

            # Rebuild cost matrix for remaining
            cost_matrix = []
            for b in objs_bef:
                row = []
                for a in objs_aft:
                    iou_val = _bbox_iou(b["box"], a["box"])
                    dist = _centroid_dist(b["centroid"], a["centroid"])
                    score = iou_val + (1.0 / (1.0 + dist))
                    row.append(score)
                cost_matrix.append(row)

    return matched_pairs, unmatched_before, unmatched_after


def _bbox_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / float(union_area) if union_area > 0 else 0.0


def _centroid_dist(
    c1: Tuple[float, float],
    c2: Tuple[float, float],
) -> float:
    """Euclidean distance between two centroids."""
    import math
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
