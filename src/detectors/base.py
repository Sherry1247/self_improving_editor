"""Abstract base class for object detectors."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class Detector(ABC):
    """
    Abstract base class for object detectors.

    Subclasses implement detection logic for extracting bounding boxes
    and class IDs from images.
    """

    @abstractmethod
    def detect(
        self, image: np.ndarray
    ) -> Tuple[Optional[int], Optional[Tuple[int, int, int, int]]]:
        """
        Detect primary object in image.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Tuple of (class_id, bounding_box) where bounding_box is
            (x1, y1, x2, y2). Returns (None, None) if no object detected.
        """
        ...

    @abstractmethod
    def get_class_name(self, class_id: int) -> str:
        """Return human-readable name for a class ID."""
        ...
