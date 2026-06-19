#!/usr/bin/env python3
"""
Orchestrates evaluation of an edited image against its original counterpart
and an editing instruction.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import yaml

# Insert repository root to PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detectors.grounding_dino import GroundingDinoDetector
from src.segmentation.sam2_segmentor import SAM2Segmentor
from src.scene_graph.builder import SceneGraphBuilder
from src.critics.detection import DetectionCritic
from src.critics.count import CountCritic
from src.critics.segmentation import SegmentationCritic
from src.critics.spatial import SpatialCritic
from src.critics.clip import CLIPCritic
from src.critics.vlm import VLMCritic
from src.critics.physics import PhysicsCritic
from src.reward_aggregation.aggregator import RewardAggregator
from src.prompt_refinement.refiner import PromptRefiner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate edited image quality and instruction adherence")
    parser.add_argument("--image-before", required=True, type=Path, help="Path to original image")
    parser.add_argument("--image-after", required=True, type=Path, help="Path to edited image")
    parser.add_argument("--instruction", required=True, type=str, help="Editing instruction")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "src" / "configs" / "evaluation.yaml",
        help="Path to evaluation YAML configuration",
    )
    parser.add_argument("--job-id", required=True, type=str, help="Unique job/experiment ID")
    return parser.parse_args()


def draw_mask_overlay(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    """Overlays a binary mask on an image for visualization."""
    overlay = image.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def main() -> int:
    args = parse_args()
    job_id = args.job_id

    # 1. Load config
    if not args.config.exists():
        logger.error("Configuration file not found: %s", args.config)
        return 1

    with args.config.open() as f:
        config = yaml.safe_load(f)

    # Resolve device and mock setting
    device = config.get("device", "cpu")
    if device == "auto":
        from src.utils.device_utils import get_device
        device = get_device()
    use_mock = config.get("use_mock", True)

    logger.info("Initializing models on device=%s, use_mock=%s", device, use_mock)

    # 2. Initialize components
    models_cfg = config.get("models", {})
    target_classes = config.get("target_classes", ["person", "dog"])
    
    # Dot-separated query for Grounding DINO
    dino_query = " . ".join(target_classes)

    detector = GroundingDinoDetector(
        model_id=models_cfg.get("grounding_dino", "IDEA-Research/grounding-dino-tiny"),
        device=device,
        use_mock=use_mock,
    )
    
    segmentor = SAM2Segmentor(
        model_id=models_cfg.get("sam2", "facebook/sam2.1-hiera-small"),
        device=device,
        use_mock=use_mock,
    )

    sg_cfg = config.get("scene_graph", {})
    scene_graph_builder = SceneGraphBuilder(
        near_threshold_px=sg_cfg.get("near_threshold_px", 150.0),
        standing_on_overlap=sg_cfg.get("standing_on_overlap", 0.2),
        standing_on_height_diff=sg_cfg.get("standing_on_height_diff", 50.0),
        inside_iou_threshold=sg_cfg.get("inside_iou_threshold", 0.6),
    )

    # Critics
    det_critic = DetectionCritic()
    cnt_critic = CountCritic()
    seg_critic = SegmentationCritic()
    spa_critic = SpatialCritic()
    clip_critic = CLIPCritic(
        model_id=models_cfg.get("clip", "openai/clip-vit-base-patch32"),
        device=device,
        use_mock=use_mock,
    )
    vlm_critic = VLMCritic(provider="mock")
    
    rules_cfg = config.get("physics_rules", {})
    phy_critic = PhysicsCritic(
        rewards=rules_cfg.get("rewards"),
        penalties=rules_cfg.get("penalties"),
    )

    # Aggregator & Refiner
    aggregator = RewardAggregator(weights=config.get("critics", {}).get("weights", {}))
    refiner = PromptRefiner()

    # 3. Read input images
    img_before = cv2.imread(str(args.image_before))
    img_after = cv2.imread(str(args.image_after))

    if img_before is None:
        logger.error("Failed to load image_before: %s", args.image_before)
        return 1
    if img_after is None:
        logger.error("Failed to load image_after: %s", args.image_after)
        return 1

    logger.info("Loaded images successfully. Running detection and segmentation...")

    # 4. Run Detection Branch (Grounding DINO)
    det_before = detector.detect(img_before, dino_query)
    det_after = detector.detect(img_after, dino_query)

    # 5. Run Segmentation Branch (SAM2)
    boxes_before = [d["box"] for d in det_before]
    boxes_after = [d["box"] for d in det_after]

    masks_before = segmentor.segment(img_before, boxes_before)
    masks_after = segmentor.segment(img_after, boxes_after)

    # 6. Build Scene Graphs
    sg_before = scene_graph_builder.build(det_before, masks_before)
    sg_after = scene_graph_builder.build(det_after, masks_after)

    # 7. Evaluate Critics
    c_det_score, c_det_meta = det_critic.evaluate(sg_before, sg_after)
    c_cnt_score, c_cnt_meta = cnt_critic.evaluate(sg_before, sg_after)
    c_seg_score, c_seg_meta = seg_critic.evaluate(sg_before, sg_after, masks_before, masks_after)
    c_spa_score, c_spa_meta = spa_critic.evaluate(sg_before, sg_after, masks_before, masks_after)
    c_clip_score, c_clip_meta = clip_critic.evaluate(img_after, args.instruction)
    c_vlm_score, c_vlm_reasoning = vlm_critic.evaluate(img_before, img_after, args.instruction)
    c_phy_score, c_phy_meta = phy_critic.evaluate(sg_after)

    # Compile scores for aggregator
    critic_scores = {
        "detection": c_det_score,
        "count": c_cnt_score,
        "segmentation": c_seg_score,
        "spatial": c_spa_score,
        "clip": c_clip_score,
        "vlm": c_vlm_score,
        "physics": c_phy_score,
    }

    # Compile details for explanations
    critic_explanations = {
        "detection": c_det_meta,
        "count": c_cnt_meta,
        "segmentation": c_seg_meta,
        "spatial": c_spa_meta,
        "clip": c_clip_meta,
        "vlm": {"reasoning": c_vlm_reasoning},
        "physics": c_phy_meta,
    }

    # 8. Aggregate Scores
    final_score, weighted_scores = aggregator.aggregate(critic_scores)

    # 9. Prompt Refinement
    refined_instruction, suggestions = refiner.refine(args.instruction, critic_scores, critic_explanations)

    # 10. Save results to experiment directory structure
    out_dir = Path(config.get("output", {}).get("base_dir", "results"))
    logger.info("Saving results to directory structure under: %s", out_dir)

    # Create directories
    subdirs = ["image_before", "image_after", "detections", "masks", "scene_graphs", "scores", "prompts"]
    for sub in subdirs:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # Copy / Save images
    cv2.imwrite(str(out_dir / "image_before" / f"{job_id}_before.jpg"), img_before)
    cv2.imwrite(str(out_dir / "image_after" / f"{job_id}_after.jpg"), img_after)

    # Save detections
    with (out_dir / "detections" / f"{job_id}_detections_before.json").open("w") as f:
        json.dump(det_before, f, indent=2)
    with (out_dir / "detections" / f"{job_id}_detections_after.json").open("w") as f:
        json.dump(det_after, f, indent=2)

    # Save masks arrays and visual mask overlays
    np.savez_compressed(str(out_dir / "masks" / f"{job_id}_masks_before.npz"), *masks_before)
    np.savez_compressed(str(out_dir / "masks" / f"{job_id}_masks_after.npz"), *masks_after)

    vis_before = img_before.copy()
    for m in masks_before:
        vis_before = draw_mask_overlay(vis_before, m)
    cv2.imwrite(str(out_dir / "masks" / f"{job_id}_masks_before_vis.jpg"), vis_before)

    vis_after = img_after.copy()
    for m in masks_after:
        vis_after = draw_mask_overlay(vis_after, m)
    cv2.imwrite(str(out_dir / "masks" / f"{job_id}_masks_after_vis.jpg"), vis_after)

    # Save scene graphs
    with (out_dir / "scene_graphs" / f"{job_id}_scene_graph_before.json").open("w") as f:
        json.dump(sg_before, f, indent=2)
    with (out_dir / "scene_graphs" / f"{job_id}_scene_graph_after.json").open("w") as f:
        json.dump(sg_after, f, indent=2)

    # Save prompts refinement report
    prompt_report = {
        "original_instruction": args.instruction,
        "refined_instruction": refined_instruction,
        "suggestions": suggestions,
    }
    with (out_dir / "prompts" / f"{job_id}_prompt_refinement.json").open("w") as f:
        json.dump(prompt_report, f, indent=2)

    # Save final report and scores
    final_report = {
        "job_id": job_id,
        "instruction": args.instruction,
        "final_score": final_score,
        "critic_scores": critic_scores,
        "weighted_scores": weighted_scores,
        "critic_explanations": critic_explanations,
        "prompt_refinement": prompt_report,
    }
    with (out_dir / "scores" / f"{job_id}_report.json").open("w") as f:
        json.dump(final_report, f, indent=2)

    logger.info("Evaluation complete! Job ID: %s. Final Score: %.4f", job_id, final_score)
    print(json.dumps({"job_id": job_id, "final_score": final_score}, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
