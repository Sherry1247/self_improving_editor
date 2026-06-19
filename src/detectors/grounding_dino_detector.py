"""Grounding DINO open-vocabulary object detector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils.device_utils import get_device

from .base_detector import BaseDetector

logger = logging.getLogger("mvp_grounding_dino")


class GroundingDinoDetector(BaseDetector):
    """
    Grounding DINO detector for zero-shot object detection.

  Supports HuggingFace ``AutoModelForZeroShotObjectDetection`` and a
    deterministic mock mode for offline testing on CHTC login nodes.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        use_mock: bool = False,
        text_query: str = "person . dog . mountain . river",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        if device is None or device == "auto":
            device = get_device()

        self.model_id = model_id
        self.device = device
        self.use_mock = use_mock
        self.text_query = text_query
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = None
        self.processor = None

        if not self.use_mock:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )

            logger.info("Loading Grounding DINO: %s on %s", self.model_id, self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id
            ).to(self.device)
            self.model.eval()
        except Exception as exc:
            logger.warning(
                "Failed to load Grounding DINO (%s). Falling back to mock mode.",
                exc,
            )
            self.use_mock = True

    def detect(self, image: Union[Path, str, np.ndarray]) -> Dict[str, Any]:
        """Detect objects and return standardized JSON."""
        pil_image, width, height = self._load_image(image)
        query = self._format_query(self.text_query)

        if self.use_mock:
            raw_objects = self._mock_detect(width, height, query)
        else:
            raw_objects = self._run_model(pil_image, query, height, width)

        return {"objects": [self._to_object_dict(obj) for obj in raw_objects]}

    def _run_model(
        self,
        pil_image: Image.Image,
        query: str,
        height: int,
        width: int,
    ) -> List[Dict[str, Any]]:
        inputs = self.processor(
            images=pil_image,
            text=query,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )

        detections: List[Dict[str, Any]] = []
        if results:
            result = results[0]
            for box, score, label in zip(
                result["boxes"], result["scores"], result["labels"]
            ):
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append(
                    {
                        "label": str(label).strip().lower(),
                        "confidence": float(score.item()),
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        return detections

    def _mock_detect(
        self,
        width: int,
        height: int,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Deterministic mock detections for CPU-only testing."""
        targets = [t.strip() for t in query.replace(".", " ").split() if t.strip()]
        mock_db = {
            "person": [int(width * 0.15), int(height * 0.2), int(width * 0.5), int(height * 0.85)],
            "dog": [int(width * 0.55), int(height * 0.5), int(width * 0.85), int(height * 0.88)],
            "mountain": [0, 0, width, int(height * 0.55)],
            "river": [0, int(height * 0.7), width, height],
            "tree": [int(width * 0.05), int(height * 0.1), int(width * 0.25), int(height * 0.6)],
            "boat": [int(width * 0.3), int(height * 0.72), int(width * 0.7), int(height * 0.88)],
        }

        detections: List[Dict[str, Any]] = []
        for target in targets:
            key = target.lower()
            if key not in mock_db:
                continue
            detections.append(
                {
                    "label": key,
                    "confidence": 0.9,
                    "bbox": mock_db[key],
                }
            )
        return detections

    @staticmethod
    def _format_query(text_query: str) -> str:
        query = text_query.lower().strip()
        if not query.endswith("."):
            query += "."
        return query

    @staticmethod
    def _load_image(image: Union[Path, str, np.ndarray]) -> tuple[Image.Image, int, int]:
        if isinstance(image, np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
        else:
            pil_image = Image.open(str(image)).convert("RGB")
        width, height = pil_image.size
        return pil_image, width, height

    @staticmethod
    def _to_object_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
        bbox = obj.get("bbox", obj.get("box"))
        if isinstance(bbox, tuple):
            bbox = list(bbox)
        return {
            "label": str(obj["label"]).strip().lower(),
            "confidence": float(obj.get("confidence", obj.get("score", 0.0))),
            "bbox": [int(v) for v in bbox],
        }
