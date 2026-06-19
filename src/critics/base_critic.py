"""Abstract critic interface for the Grounding DINO MVP."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseCritic(ABC):
    """
    Abstract base class for detection-based critics.

    Critics consume structured detection JSON and return a score plus
    diagnostic metadata. Future phases (CLIP, VLM, spatial, etc.) can
    subclass this interface or a parallel multimodal base.
    """

    @abstractmethod
    def evaluate(
        self,
        detections_before: Dict[str, Any],
        detections_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare before/after detections.

        Returns:
            Dictionary containing at least a ``*_score`` float in [0, 1].
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return a stable critic identifier."""
        ...
