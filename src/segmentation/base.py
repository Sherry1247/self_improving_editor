"""Abstract base class for object segmentors."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
from PIL import Image


class Segmentor(ABC):
    """
    Abstract base class for image segmentation models.
    
    Conforms to a modular architecture to allow swapping SAM2 with other
    segmentation models in the future.
    """

    @abstractmethod
    def segment(
        self,
        image: Union[np.ndarray, Image.Image],
        bounding_boxes: List[Tuple[int, int, int, int]],
    ) -> List[np.ndarray]:
        """
        Generate binary segmentation masks for provided bounding boxes.

        Args:
            image: OpenCV BGR array or PIL Image.
            bounding_boxes: List of bounding boxes as (x1, y1, x2, y2).

        Returns:
            List of binary masks as numpy boolean arrays of shape (H, W).
        """
        pass
