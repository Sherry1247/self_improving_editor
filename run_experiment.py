#!/usr/bin/env python3
"""
MVP experiment runner for Grounding DINO object-preservation evaluation.

Processes every sample under ``data/sample_*/`` and writes results to
``results/<sample_id>/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.critics.count_critic import CountCritic
from src.critics.detection_critic import DetectionCritic
from src.detectors.grounding_dino_detector import GroundingDinoDetector
from src.refinement.prompt_refiner import PromptRefiner
from src.utils.config_loader import load_detector_config, load_scoring_config
from src.utils.io import discover_samples, load_text, save_json, save_text
from src.utils.logging_utils import setup_logging
from src.utils.scoring import compute_final_score
from src.utils.visualization_mvp import save_summary_png

logger = logging.getLogger("mvp_grounding_dino")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO MVP preservation experiment"
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
        help="Path to scoring YAML (default: config/scoring.yaml)",
    )
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=None,
        help="Path to detector YAML (default: config/detector.yaml)",
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
