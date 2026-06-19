"""Grounding DINO zero-shot object detector wrapper."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("closed_loop_editor")


class GroundingDinoDetector:
    """
    Wrapper for Grounding DINO zero-shot object detector.
    
    Supports both real Hugging Face model execution and a deterministic mock
    fallback mode for CPU/offline testing.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cpu",
        use_mock: bool = False,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.model_id = model_id
        self.device = device
        self.use_mock = use_mock
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.model = None
        self.processor = None

        if not self.use_mock:
            try:
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
                logger.info("Loading Grounding DINO model: %s on %s", model_id, device)
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
                self.model.eval()
            except Exception as e:
                logger.warning(
                    "Failed to load real Grounding DINO model (%s). Falling back to MOCK mode.",
                    e
                )
                self.use_mock = True

    def detect(
        self,
        image: Union[np.ndarray, Image.Image],
        text_query: str,
    ) -> List[Dict[str, Union[str, float, Tuple[int, int, int, int]]]]:
        """
        Detect objects matching the text query in the image.

        Args:
            image: OpenCV BGR array or PIL Image.
            text_query: Dot-separated lowercase search items (e.g. "person . dog . river").

        Returns:
            List of detections, each a dict:
            {
                "label": str,
                "score": float,
                "box": (x1, y1, x2, y2)
            }
        """
        # Convert to PIL Image and get size
        if isinstance(image, np.ndarray):
            pil_image = self._to_pil(image)
            w, h = image.shape[1], image.shape[0]
        else:
            pil_image = image
            w, h = image.size

        # Format query: Grounding DINO expects lowercased dot-separated classes ending with dot.
        query = text_query.lower().strip()
        if not query.endswith("."):
            query += "."

        if self.use_mock:
            return self._mock_detect(w, h, query)

        try:
            inputs = self.processor(images=pil_image, text=query, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )

            detections = []
            if len(results) > 0:
                result = results[0]
                for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    detections.append({
                        "label": str(label).strip(),
                        "score": float(score.item()),
                        "box": (x1, y1, x2, y2),
                    })
            return detections

        except Exception as e:
            logger.error("Error running real Grounding DINO: %s. Falling back to mock.", e)
            return self._mock_detect(w, h, query)

    def _mock_detect(
        self,
        width: int,
        height: int,
        query: str,
    ) -> List[Dict[str, Union[str, float, Tuple[int, int, int, int]]]]:
        """Generate mock detections based on standard classes for testing."""
        # Parse targets from query e.g. "person . dog . river ."
        targets = [t.strip().replace(".", "") for t in query.split(".") if t.strip()]
        detections = []

        # Standard simulated objects at plausible coordinates
        mock_database = {
            "person": {"box": (int(width * 0.15), int(height * 0.2), int(width * 0.5), int(height * 0.85)), "score": 0.89},
            "adult_person": {"box": (int(width * 0.15), int(height * 0.2), int(width * 0.5), int(height * 0.85)), "score": 0.89},
            "dog": {"box": (int(width * 0.55), int(height * 0.5), int(width * 0.85), int(height * 0.88)), "score": 0.92},
            "river": {"box": (0, int(height * 0.7), width, height), "score": 0.78},
            "mountain": {"box": (0, 0, width, int(height * 0.55)), "score": 0.81},
            "boat": {"box": (int(width * 0.3), int(height * 0.72), int(width * 0.7), int(height * 0.88)), "score": 0.85},
            "fish": {"box": (int(width * 0.4), int(height * 0.8), int(width * 0.5), int(height * 0.9)), "score": 0.75},
            "car": {"box": (int(width * 0.2), int(height * 0.6), int(width * 0.8), int(height * 0.9)), "score": 0.88},
        }

        for target in targets:
            # Match directly or by substring (e.g. "adult_person" or "person")
            matched_key = None
            for key in mock_database:
                if key in target or target in key:
                    matched_key = key
                    break
            
            if matched_key:
                info = mock_database[matched_key]
                detections.append({
                    "label": target,
                    "score": info["score"],
                    "box": info["box"],
                })

        return detections

    def _to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert standard OpenCV BGR image to PIL RGB."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
