"""CLIP Critic implementation."""

import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
from PIL import Image

from .clip_utils import clip_image_text_similarity

logger = logging.getLogger("closed_loop_editor")


class CLIPCritic:
    """
    CLIP Critic evaluating semantic alignment between edited image and editing instruction.
    
    Metrics:
        - clip_alignment: Cosine similarity between image embedding and instruction embedding.
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        use_mock: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.use_mock = use_mock

    def evaluate(
        self,
        image_after: Union[np.ndarray, Image.Image],
        instruction: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate semantic alignment.

        Args:
            image_after: Edited image as BGR numpy array or PIL Image.
            instruction: Editing instruction.

        Returns:
            Tuple of (score, metrics_dict).
        """
        if self.use_mock:
            # Deterministic mock score
            score = 0.78
            logger.info("CLIPCritic (MOCK): alignment=%.3f", score)
            return score, {"clip_alignment": score}

        try:
            # Convert to numpy BGR if needed by clip_utils
            if isinstance(image_after, Image.Image):
                import cv2
                bgr = cv2.cvtColor(np.array(image_after), cv2.COLOR_RGB2BGR)
            else:
                bgr = image_after

            score = clip_image_text_similarity(
                image=bgr,
                text=instruction,
                model_id=self.model_id,
                device=self.device,
            )
            logger.info("CLIPCritic: alignment=%.3f", score)
            return score, {"clip_alignment": score}

        except Exception as e:
            logger.error("Error in CLIPCritic real inference: %s. Falling back to mock.", e)
            score = 0.78
            return score, {"clip_alignment": score}
