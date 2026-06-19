"""Reward Aggregation implementation."""

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger("closed_loop_editor")


class RewardAggregator:
    """
    Reward Aggregator that combines scores from all critics.
    
    Formula:
        FinalScore = w1 * DetectionCritic +
                     w2 * CountCritic +
                     w3 * SegmentationCritic +
                     w4 * SpatialCritic +
                     w5 * CLIPCritic +
                     w6 * VLMCritic +
                     w7 * PhysicsCritic
    """

    def __init__(self, weights: Dict[str, float]):
        """
        Args:
            weights: Dictionary of critic weights.
        """
        self.weights = weights
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Ensure all required weights are present and log their values."""
        required = [
            "detection",
            "count",
            "segmentation",
            "spatial",
            "clip",
            "vlm",
            "physics",
        ]
        missing = [r for r in required if r not in self.weights]
        if missing:
            raise ValueError(f"Missing critic weights in configuration: {missing}")

        total_weight = sum(self.weights[r] for r in required)
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(
                "Critic weights sum to %.3f (expected ~1.0). Normalizing weights.",
                total_weight,
            )
            # Normalize weights to sum exactly to 1.0
            for r in required:
                self.weights[r] /= total_weight

        logger.info("RewardAggregator initialized with weights: %s", self.weights)

    def aggregate(
        self,
        scores: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the final aggregated score.

        Args:
            scores: Dictionary containing individual critic scores.

        Returns:
            Tuple of (final_score, weighted_scores_dict).
        """
        final_score = 0.0
        weighted_scores = {}

        for critic_name, weight in self.weights.items():
            score = scores.get(critic_name, 0.0)
            weighted = score * weight
            final_score += weighted
            weighted_scores[f"{critic_name}_weighted"] = weighted

        logger.info("Aggregated Final Score: %.4f", final_score)
        return final_score, weighted_scores
