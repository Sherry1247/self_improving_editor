"""Abstract detector interface for the Grounding DINO MVP."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for open-vocabulary object detectors.

    Future phases may add SAM2-guided detectors or multi-model ensembles
    without changing critic interfaces.
    """

    @abstractmethod
    def detect(self, image: Union[Path, str, np.ndarray]) -> Dict[str, Any]:
        """
        Run detection on an image.

        Args:
            image: File path or BGR numpy array.

        Returns:
            Detection dict with an ``objects`` list. Each object contains
            ``label``, ``confidence``, and ``bbox`` [x1, y1, x2, y2].
        """
        ...
