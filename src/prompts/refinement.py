"""Prompt refinement engine."""

import logging
from typing import Dict

logger = logging.getLogger("closed_loop_editor")


def refine_prompt_rule_based(base_prompt: str, scores: Dict[str, float]) -> str:
    """
    Refine prompt using rule-based heuristics.

    When object consistency is low, emphasize keeping the subject unchanged.
    When instruction alignment is low, reinforce the editing goal.
    """
    prompt = base_prompt

    if scores.get("object_consistency", 1.0) < 0.5:
        if "keep the main subject" not in prompt.lower():
            prompt = prompt + ", keep the main subject exactly the same"

    if scores.get("instruction_alignment", 1.0) < 0.5:
        if "follow the instruction precisely" not in prompt.lower():
            prompt = prompt + ", follow the instruction precisely"

    if scores.get("clip_similarity", 1.0) < 0.5:
        if "preserve the original appearance" not in prompt.lower():
            prompt = prompt + ", preserve the original appearance of the subject"

    return prompt


def refine_prompt(
    base_prompt: str,
    scores: Dict[str, float],
    iteration: int,
    max_iterations: int,
) -> str:
    """
    Refine prompt based on critic scores for the next iteration.

    Args:
        base_prompt: Original base prompt.
        scores: Individual critic scores from the current iteration.
        iteration: Current iteration index (0-based).
        max_iterations: Maximum planned iterations.

    Returns:
        Refined prompt for the next editing step.
    """
    refined = refine_prompt_rule_based(base_prompt, scores)

    logger.debug(
        "Iteration %d/%d: scores=%s refined_prompt=%s",
        iteration,
        max_iterations,
        scores,
        refined,
    )

    return refined
