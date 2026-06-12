"""Instruction alignment critic using CLIP text-image similarity."""

from typing import Any, Optional

import numpy as np

from .base import Critic
from .clip_utils import clip_image_text_similarity


class InstructionAlignmentCritic(Critic):
    """
    Evaluates how well the edited image aligns with the editing instruction.

    Uses CLIP text-image similarity between the prompt and edited result.
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
        """Return CLIP text-image alignment score in [0, 1]."""
        return clip_image_text_similarity(
            edited_image,
            prompt,
            model_id=self.model_id,
            device=self.device,
        )

    def get_name(self) -> str:
        return "instruction_alignment"
