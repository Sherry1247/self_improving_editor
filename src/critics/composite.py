"""Composite critic combining multiple scoring metrics."""

from typing import Any, Dict, Optional

import numpy as np

from .base import Critic


class CompositeCritic(Critic):
    """
    Combines multiple critics with configurable weights.

    Default weights: 0.4 object_consistency + 0.3 clip_similarity + 0.3 instruction_alignment
    """

    def __init__(
        self,
        critics: list,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize composite critic.

        Args:
            critics: List of Critic instances.
            weights: Dictionary mapping critic name → weight.
                    If None, uses equal weights.
                    Example: {"object_consistency": 0.4, "clip_similarity": 0.3, ...}

        Raises:
            ValueError: If weights don't sum to 1.0.
        """
        self.critics = {c.get_name(): c for c in critics}

        if weights is None:
            n = len(self.critics)
            weights = {name: 1.0 / n for name in self.critics.keys()}

        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights.values())}")

        if set(weights.keys()) != set(self.critics.keys()):
            raise ValueError(
                f"Weight keys {set(weights.keys())} don't match critic names {set(self.critics.keys())}"
            )

        self.weights = weights

    def score(
        self,
        original_image: np.ndarray,
        edited_image: np.ndarray,
        prompt: str,
        **kwargs: Any,
    ) -> float:
        """
        Compute weighted composite score from all critics.

        Args:
            original_image: Original image (BGR).
            edited_image: Edited image (BGR).
            prompt: Text prompt used for editing.
            **kwargs: Additional parameters passed to critics.

        Returns:
            Weighted composite score in range [0, 1].
        """
        individual_scores = {}
        for name, critic in self.critics.items():
            score = critic.score(original_image, edited_image, prompt, **kwargs)
            individual_scores[name] = score

        composite = sum(
            individual_scores[name] * self.weights[name]
            for name in self.critics.keys()
        )

        return float(composite)

    def get_name(self) -> str:
        """Get critic name."""
        return "composite"

    def get_individual_scores(
        self,
        original_image: np.ndarray,
        edited_image: np.ndarray,
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Get individual scores from all critics (for logging).

        Args:
            original_image: Original image (BGR).
            edited_image: Edited image (BGR).
            prompt: Text prompt used for editing.
            **kwargs: Additional parameters passed to critics.

        Returns:
            Dictionary mapping critic name → individual score.
        """
        scores = {}
        for name, critic in self.critics.items():
            scores[name] = critic.score(original_image, edited_image, prompt, **kwargs)
        return scores

    def get_weights(self) -> Dict[str, float]:
        """Get weight configuration."""
        return self.weights.copy()
