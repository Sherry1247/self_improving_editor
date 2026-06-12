"""Object consistency critic implementation (IoU-based)."""

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from src.detectors.base import Detector

from .base import Critic


class ObjectConsistencyCritic(Critic):
    """
    Evaluates structural consistency of detected objects.

    Compares bounding boxes of target objects in original and edited images
    using Intersection over Union (IoU).
    """

    def __init__(self, detector: Detector):
        """
        Initialize object consistency critic.

        Args:
            detector: Detector instance for object localization.
        """
        self.detector = detector

    def score(
        self,
        original_image: np.ndarray,
        edited_image: np.ndarray,
        prompt: str,
        **kwargs: Any,
    ) -> float:
        """
        Score based on IoU between detected objects in original and edited images.

        Args:
            original_image: Original image (BGR format).
            edited_image: Edited image (BGR format).
            prompt: Text prompt (unused for this critic).
            **kwargs: Additional parameters (unused).

        Returns:
            IoU score in range [0, 1].
        """
        orig_cls, orig_box = self.detector.detect(original_image)
        edit_cls, edit_box = self.detector.detect(edited_image)

        if orig_cls is None or edit_cls is None:
            return 0.0

        if orig_cls != edit_cls:
            return 0.0

        iou = self._bbox_iou(orig_box, edit_box)
        return float(iou)

    def get_name(self) -> str:
        """Get critic name."""
        return "object_consistency"

    @staticmethod
    def _bbox_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """
        Calculate IoU between two bounding boxes.

        Args:
            box1: Bounding box as (x1, y1, x2, y2).
            box2: Bounding box as (x1, y1, x2, y2).

        Returns:
            IoU in range [0, 1].
        """
        if box1 is None or box2 is None:
            return 0.0

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

        return inter_area / float(union_area)
