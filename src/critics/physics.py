"""Physics Critic implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("closed_loop_editor")


def normalize_label(label: str) -> str:
    """Normalize object class names for rule matching."""
    label = label.lower().strip()
    if "person" in label:
        return "person"
    if "dog" in label:
        return "dog"
    if "river" in label or "water" in label:
        return "river"
    if "mountain" in label or "hill" in label:
        return "mountain"
    if "boat" in label:
        return "boat"
    if "fish" in label:
        return "fish"
    if "car" in label:
        return "car"
    return label


class PhysicsCritic:
    """
    Physics Critic evaluating the physical plausibility of scene graphs.
    
    Uses rule-based scoring (penalties/rewards) based on relationships.
    Designed for future replacement by GNNs, World Models, or VLMs.
    """

    def __init__(
        self,
        rewards: Optional[Dict[str, float]] = None,
        penalties: Optional[Dict[str, float]] = None,
    ):
        # Default physical rules
        self.rewards = rewards or {
            "boat_floating_on_river": 1.0,
            "fish_inside_river": 1.0,
        }
        self.penalties = penalties or {
            "person_standing_on_river": 0.8,
            "car_floating_on_river": 0.9,
        }

    def evaluate(
        self,
        scene_graph: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate physical plausibility based on scene graph relationships.

        Args:
            scene_graph: Dictionary representing objects and relationships.

        Returns:
            Tuple of (plausibility_score, metrics_dict).
        """
        objects = scene_graph.get("objects", [])
        relationships = scene_graph.get("relationships", [])

        # Create lookup map for object labels from object IDs
        id_to_label = {obj["id"]: obj["label"] for obj in objects}

        score = 0.5  # Neutral base score
        triggered_rules = []
        rule_logs = []

        for rel in relationships:
            subj_id = rel["subject"]
            relation = rel["relation"]
            obj_id = rel["object"]

            subj_label = normalize_label(id_to_label.get(subj_id, ""))
            obj_label = normalize_label(id_to_label.get(obj_id, ""))

            # Construct rule key, e.g., "person_standing_on_river"
            rule_key = f"{subj_label}_{relation}_{obj_label}"

            # Check rewards
            matched_reward = None
            for key, val in self.rewards.items():
                if key in rule_key or rule_key in key:
                    matched_reward = (key, val)
                    break
            
            if matched_reward:
                key, val = matched_reward
                score += val * 0.25
                triggered_rules.append(f"+{key}")
                rule_logs.append(f"Reward: {subj_label} {relation} {obj_label} (+{val})")
                continue

            # Check penalties
            matched_penalty = None
            for key, val in self.penalties.items():
                if key in rule_key or rule_key in key:
                    matched_penalty = (key, val)
                    break

            if matched_penalty:
                key, val = matched_penalty
                score -= val * 0.4
                triggered_rules.append(f"-{key}")
                rule_logs.append(f"Penalty: {subj_label} {relation} {obj_label} (-{val})")

        # Bounding score in [0.0, 1.0]
        score = max(0.0, min(1.0, score))

        metrics = {
            "physics_score": score,
            "triggered_rules": triggered_rules,
            "rule_logs": rule_logs,
        }

        logger.info(
            "PhysicsCritic: score=%.3f, triggered=%s",
            score,
            triggered_rules,
        )
        return score, metrics
