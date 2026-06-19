"""Abstract critic interface for the Grounding DINO MVP."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseCritic(ABC):
    """Abstract base class for detection-based critics."""

    @abstractmethod
    def evaluate(
        self,
        detections_before: Dict[str, Any],
        detections_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare before/after detections and return score metadata."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return a stable critic identifier."""
        ...
