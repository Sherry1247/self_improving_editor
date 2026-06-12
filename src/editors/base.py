"""Abstract base class for image editors."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Editor(ABC):
    """
    Abstract base class for image editors.

    Subclasses implement text-guided image editing and generation.
    """

    @abstractmethod
    def edit(self, image: np.ndarray, prompt: str, **kwargs: Any) -> np.ndarray:
        """
        Edit or generate an image based on text prompt.

        Args:
            image: Input image as numpy array (BGR format).
            prompt: Text instruction describing desired edits.
            **kwargs: Editor-specific parameters.

        Returns:
            Edited image as numpy array (BGR format).
        """
        ...
