#!/usr/bin/env python3
"""
Single-job runner for HTCondor / CHTC batch execution.

Each job processes one image-background combination independently and
writes JSON results to a dedicated experiment directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.configs import build_pipeline, load_config
from src.utils.io_utils import build_prompt, load_image

logger = logging.getLogger("closed_loop_editor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single closed-loop experiment job")
    parser.add_argument("--job-id", required=True, help="Unique job identifier")
    parser.add_argument("--filename", required=True, help="Image filename")
    parser.add_argument("--object", required=True, help="Object label (e.g. adult_person)")
    parser.add_argument("--action", required=True, help="Action label (e.g. sit)")
    parser.add_argument("--background", required=True, help="Background label (e.g. river)")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=REPO_ROOT / "data" / "images" / "original",
        help="Directory containing source images",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override experiment output base directory from config",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max iterations from config",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Override score threshold from config",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.output_dir is not None:
        config.setdefault("output", {})["base_dir"] = str(args.output_dir)

    pipeline = build_pipeline(config)

    image_path = args.image_dir / args.filename
    if not image_path.exists():
        fallback = REPO_ROOT / "data" / "images" / args.filename
        if fallback.exists():
            image_path = fallback
        else:
            logger.error("Image not found: %s", args.filename)
            return 1

    editor_size = config.get("editor", {}).get("size", 384)
    original_image = load_image(image_path, size=editor_size)
    base_prompt = build_prompt(args.object, args.action, args.background)

    logger.info(
        "Job %s: %s | object=%s action=%s background=%s",
        args.job_id,
        args.filename,
        args.object,
        args.action,
        args.background,
    )

    result = pipeline.run(
        original_image=original_image,
        base_prompt=base_prompt,
        max_iterations=args.max_iterations,
        score_threshold=args.score_threshold,
        metadata={
            "job_id": args.job_id,
            "filename": args.filename,
            "object": args.object,
            "action": args.action,
            "background": args.background,
            "image_path": str(image_path),
        },
        experiment_id=args.job_id,
    )

    summary = {
        "job_id": args.job_id,
        "experiment_id": result["experiment_id"],
        "best_score": result["best_score"],
        "num_iterations": len(result["all_iterations"]),
        "metadata": result["metadata"],
    }
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
