"""Abstract detector interface for the Grounding DINO MVP."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


class BaseDetector(ABC):
    """Abstract base class for open-vocabulary object detectors."""

    @abstractmethod
    def detect(self, image: Union[Path, str, np.ndarray]) -> Dict[str, Any]:
        """
        Run detection on an image.

        Returns:
            Detection dict with an ``objects`` list. Each object contains
            ``label``, ``confidence``, and ``bbox`` [x1, y1, x2, y2].
        """
        ...
