"""Base classes for editors and critics."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np


class Critic(ABC):
    """
    Abstract base class for evaluation critics.

    Critics score edited images on various dimensions:
    - Object consistency (IoU of bounding boxes)
    - CLIP similarity (semantic alignment to prompt)
    - Instruction alignment (prompt specificity)
    """

    @abstractmethod
    def score(
        self,
        original_image: np.ndarray,
        edited_image: np.ndarray,
        prompt: str,
        **kwargs: Any,
    ) -> float:
        """
        Score edited image on a specific dimension.

        Args:
            original_image: Original unedited image (BGR format).
            edited_image: Edited result from Editor (BGR format).
            prompt: Text prompt that was used for editing.
            **kwargs: Critic-specific parameters.

        Returns:
            Score in range [0, 1] where 1.0 = perfect.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get unique name for this critic.

        Returns:
            Short name string (e.g., "object_consistency").
        """
        pass
