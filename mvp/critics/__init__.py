"""MVP critics package."""

from .count_critic import CountCritic
from .detection_critic import DetectionCritic

__all__ = ["DetectionCritic", "CountCritic"]
