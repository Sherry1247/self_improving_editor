"""YOLO-based object detector implementation."""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .base import Detector


class YOLODetector(Detector):
    """
    YOLO object detector for person and dog detection.

    Uses YOLOv8 to return the highest-confidence bounding box among
    target classes (person=0, dog=16 in COCO).
    """

    def __init__(
        self,
        model_path: str = "yolov8l.pt",
        target_classes: Optional[list] = None,
    ):
        """
        Args:
            model_path: YOLO weights path or model name (auto-downloaded).
            target_classes: COCO class IDs to consider. Defaults to [0, 16].
        """
        self.model = YOLO(model_path)
        self.target_classes = target_classes or [0, 16]

    def detect(
        self, image: np.ndarray
    ) -> Tuple[Optional[int], Optional[Tuple[int, int, int, int]]]:
        """Detect highest-confidence target object in a BGR image."""
        results = self.model(image)[0]

        best_box = None
        best_conf = -1.0
        best_cls = None

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self.target_classes:
                continue

            conf = float(box.conf[0].item())
            if conf > best_conf:
                best_conf = conf
                best_cls = cls_id
                best_box = box

        if best_box is None:
            return None, None

        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        return best_cls, (x1, y1, x2, y2)

    def detect_from_path(
        self,
        image_path: Union[str, Path],
        resize_to: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[Tuple[int, int, int, int]]]:
        """
        Load image from path, optionally resize, and run detection.

        Resizing ensures original and edited images share the same coordinate
        system for IoU comparison.
        """
        pil_image = Image.open(str(image_path)).convert("RGB")
        if resize_to is not None:
            pil_image = pil_image.resize((resize_to, resize_to), Image.LANCZOS)

        bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return self.detect(bgr)

    def get_class_name(self, class_id: int) -> str:
        return self.model.names.get(int(class_id), f"class_{class_id}")


def bbox_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
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
