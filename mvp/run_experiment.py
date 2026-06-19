#!/usr/bin/env python3
"""
Standalone MVP experiment runner for Grounding DINO object-preservation evaluation.

This module only imports from the ``mvp/`` package and never touches ``src/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Ensure imports resolve to mvp/ subpackages only (not src/).
MVP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MVP_ROOT.parent
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

from critics.count_critic import CountCritic
from critics.detection_critic import DetectionCritic
from detectors.grounding_dino_detector import GroundingDinoDetector
from refinement.prompt_refiner import PromptRefiner
from utils.io import (
    discover_samples,
    load_detector_config,
    load_scoring_config,
    load_text,
    save_json,
    save_text,
)
from utils.scoring import compute_final_score

logger = logging.getLogger("mvp_grounding_dino")


def setup_logging(log_file: Path | None = None) -> None:
    """Configure structured console logging for the MVP runner."""
    root = logging.getLogger("mvp_grounding_dino")
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def _load_rgb(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _draw_boxes(
    ax,
    image: np.ndarray,
    objects: List[Dict[str, Any]],
    title: str,
) -> None:
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
        ax.add_patch(rect)
        label = obj.get("label", "object")
        conf = obj.get("confidence", 0.0)
        ax.text(
            x1,
            max(y1 - 5, 10),
            f"{label} ({conf:.2f})",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="green", alpha=0.6, pad=1),
        )


def save_summary_png(
    before_path: Path,
    after_path: Path,
    detections_before: Dict[str, Any],
    detections_after: Dict[str, Any],
    scores: Dict[str, float],
    output_path: Path,
) -> None:
    """Create a side-by-side summary image with Grounding DINO boxes."""
    before_rgb = _load_rgb(before_path)
    after_rgb = _load_rgb(after_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    _draw_boxes(axes[0], before_rgb, detections_before.get("objects", []), "Before")
    _draw_boxes(axes[1], after_rgb, detections_after.get("objects", []), "After")

    score_text = (
        f"detection={scores.get('detection_score', 0):.2f} | "
        f"count={scores.get('count_score', 0):.2f} | "
        f"final={scores.get('final_score', 0):.2f}"
    )
    fig.suptitle(score_text, fontsize=12)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone Grounding DINO MVP preservation experiment"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing sample_*/ folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory for experiment outputs",
    )
    parser.add_argument(
        "--scoring-config",
        type=Path,
        default=None,
        help="Path to scoring YAML (default: mvp/config/scoring.yaml)",
    )
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=None,
        help="Path to detector YAML (default: mvp/config/detector.yaml)",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Run a single sample id (e.g. sample_001)",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Force mock Grounding DINO (no model download)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path",
    )
    return parser.parse_args()


def run_sample(
    sample_dir: Path,
    output_dir: Path,
    detector: GroundingDinoDetector,
    detection_critic: DetectionCritic,
    count_critic: CountCritic,
    refiner: PromptRefiner,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """Run the full MVP pipeline on one sample directory."""
    sample_id = sample_dir.name
    before_path = sample_dir / "before.jpg"
    after_path = sample_dir / "after.jpg"
    instruction_path = sample_dir / "instruction.txt"

    instruction = load_text(instruction_path)
    logger.info("Processing %s", sample_id)

    detections_before = detector.detect(before_path)
    detections_after = detector.detect(after_path)

    detection_result = detection_critic.evaluate(detections_before, detections_after)
    count_result = count_critic.evaluate(detections_before, detections_after)

    detection_score = detection_result["detection_score"]
    count_score = count_result["count_score"]
    final_score = compute_final_score(detection_score, count_score, weights)

    scores = {
        "detection_score": detection_score,
        "count_score": count_score,
        "final_score": final_score,
    }

    refined_prompt = refiner.refine(instruction, detection_result, count_result)

    sample_out = output_dir / sample_id
    save_json(detections_before, sample_out / "detections_before.json")
    save_json(detections_after, sample_out / "detections_after.json")
    save_json(
        {
            **scores,
            "detection_details": detection_result,
            "count_details": count_result,
            "instruction": instruction,
        },
        sample_out / "scores.json",
    )
    save_text(refined_prompt, sample_out / "refined_prompt.txt")
    save_summary_png(
        before_path,
        after_path,
        detections_before,
        detections_after,
        scores,
        sample_out / "summary.png",
    )

    logger.info(
        "%s done: detection=%.3f count=%.3f final=%.3f",
        sample_id,
        detection_score,
        count_score,
        final_score,
    )

    return {
        "sample_id": sample_id,
        "scores": scores,
        "refined_prompt": refined_prompt,
    }


def build_detector(
    detector_cfg: Dict[str, Any],
    force_mock: bool,
) -> GroundingDinoDetector:
    use_mock = force_mock or bool(detector_cfg.get("use_mock", False))
    return GroundingDinoDetector(
        model_id=detector_cfg.get("model_id", "IDEA-Research/grounding-dino-tiny"),
        device=detector_cfg.get("device", "auto"),
        use_mock=use_mock,
        text_query=detector_cfg.get(
            "text_query",
            "person . dog . mountain . river",
        ),
        box_threshold=float(detector_cfg.get("box_threshold", 0.35)),
        text_threshold=float(detector_cfg.get("text_threshold", 0.25)),
    )


def main() -> int:
    args = parse_args()
    setup_logging(log_file=args.log_file)

    scoring_cfg = load_scoring_config(args.scoring_config)
    detector_cfg = load_detector_config(args.detector_config)
    weights = scoring_cfg.get("weights", {"detection": 0.7, "count": 0.3})

    detector = build_detector(detector_cfg, force_mock=args.use_mock)
    detection_critic = DetectionCritic()
    count_critic = CountCritic()
    refiner = PromptRefiner()

    samples = discover_samples(args.data_dir)
    if args.sample:
        samples = [p for p in samples if p.name == args.sample]
        if not samples:
            logger.error("Sample not found: %s", args.sample)
            return 1

    if not samples:
        logger.error("No valid samples found in %s", args.data_dir)
        return 1

    logger.info("Found %d sample(s)", len(samples))
    summaries = []

    for sample_dir in samples:
        summary = run_sample(
            sample_dir=sample_dir,
            output_dir=args.output_dir,
            detector=detector,
            detection_critic=detection_critic,
            count_critic=count_critic,
            refiner=refiner,
            weights=weights,
        )
        summaries.append(summary)

    save_json({"samples": summaries}, args.output_dir / "experiment_summary.json")
    logger.info("All samples complete. Results in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
