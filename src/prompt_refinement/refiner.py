"""Prompt Refiner implementation."""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("closed_loop_editor")


class PromptRefiner:
    """
    Prompt Refiner module.
    
    Generates targeted editing suggestions and refines the editing instruction
    based on individual critic scores and explanations.
    """

    def __init__(self, score_threshold: float = 0.75):
        self.score_threshold = score_threshold

    def refine(
        self,
        original_instruction: str,
        critic_scores: Dict[str, float],
        critic_explanations: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        """
        Refine the original instruction by appending targeted guidelines.

        Args:
            original_instruction: The original editing instruction.
            critic_scores: Dictionary of critic names to float scores in [0, 1].
            critic_explanations: Dictionary of critic explanations and logs.

        Returns:
            Tuple of:
            - refined_instruction: The updated editing instruction string.
            - suggestions: List of specific suggestion strings generated.
        """
        suggestions = []

        # 1. Check Detection & Count Critics (Foreground preservation)
        det_score = critic_scores.get("detection", 1.0)
        cnt_score = critic_scores.get("count", 1.0)
        if det_score < self.score_threshold or cnt_score < self.score_threshold:
            suggestions.append("Do not alter foreground objects.")

        # 2. Check Segmentation Critic (Shape/Identity preservation)
        seg_score = critic_scores.get("segmentation", 1.0)
        if seg_score < self.score_threshold:
            suggestions.append("Preserve the subject's shape and appearance.")

        # 3. Check Spatial Critic (Location preservation)
        spa_score = critic_scores.get("spatial", 1.0)
        if spa_score < self.score_threshold:
            suggestions.append("Maintain object positions.")

        # 4. Check Physics Critic (Physical plausibility rules)
        phy_score = critic_scores.get("physics", 1.0)
        if phy_score < self.score_threshold:
            triggered_rules = critic_explanations.get("physics", {}).get("triggered_rules", [])
            for rule in triggered_rules:
                if rule.startswith("-"):
                    rule_clean = rule[1:]  # Remove "-" prefix
                    # Map rule keys to user-friendly corrections
                    if "person_standing_on_river" in rule_clean:
                        suggestions.append("A person cannot stand directly on water; position them on the ground or mountain.")
                    elif "car_floating_on_river" in rule_clean:
                        suggestions.append("A car cannot float on a river; place it on a road or land.")
                    else:
                        suggestions.append(f"Avoid physically implausible relationship: {rule_clean.replace('_', ' ')}.")

        # 5. Check Semantic Alignment (Instruction Following)
        clip_score = critic_scores.get("clip", 1.0)
        vlm_score = critic_scores.get("vlm", 1.0)
        if clip_score < self.score_threshold or vlm_score < self.score_threshold:
            suggestions.append("Make sure the background edit is clearly visible and follows the instruction.")

        # Assemble refined prompt
        refined_instruction = original_instruction
        if suggestions:
            # Join suggestions with commas
            guidelines = ", ".join(suggestions)
            refined_instruction = f"{original_instruction}. Note: {guidelines}"

        logger.info("PromptRefiner suggestions generated: %s", suggestions)
        logger.info("Refined instruction: %s", refined_instruction)

        return refined_instruction, suggestions
