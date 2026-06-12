"""CLIP-based semantic similarity critic."""

from typing import Any, Optional

import numpy as np

from .base import Critic
from .clip_utils import clip_image_image_similarity


class CLIPSimilarityCritic(Critic):
    """
    Evaluates semantic preservation between original and edited images.

    Uses CLIP image embeddings to measure how similar the edited result
    remains to the original content.
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_id: HuggingFace CLIP model identifier.
            device: Inference device. Auto-detected when None.
        """
        self.model_id = model_id
        self.device = device

    def score(
        self,
        original_image: np.ndarray,
        edited_image: np.ndarray,
        prompt: str,
        **kwargs: Any,
    ) -> float:
        """Return CLIP image-image similarity in [0, 1]."""
        return clip_image_image_similarity(
            original_image,
            edited_image,
            model_id=self.model_id,
            device=self.device,
        )

    def get_name(self) -> str:
        return "clip_similarity"
