#!/usr/bin/env python3
"""
Main entry point for the closed-loop image editing framework.

Runs the full dataset (or a subset) through the iterative pipeline:
Generate → Evaluate → Refine Prompt → Generate Again.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from src.configs import build_pipeline, load_config
from src.utils.io_utils import backup_originals, build_prompt, load_image, load_labels

logger = logging.getLogger("closed_loop_editor")

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_LABELS = DATA_DIR / "labels.csv"
DEFAULT_ORIGINALS = DATA_DIR / "images" / "original"
DEFAULT_IMAGES = DATA_DIR / "images"
DEFAULT_EDITED = DATA_DIR / "images" / "edited"
DEFAULT_METRICS = DATA_DIR / "metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run closed-loop image editing on labeled dataset"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: src/configs/default.yaml)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Path to labels CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up originals",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export aggregate metrics to data/metrics.csv",
    )
    return parser.parse_args()


def export_metrics_csv(results: list, output_path: Path) -> None:
    """Write per-image summary metrics compatible with the legacy CSV format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "object",
        "action",
        "background",
        "score",
        "object_consistency",
        "clip_similarity",
        "instruction_alignment",
        "experiment_id",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline = build_pipeline(config)

    if not args.no_backup:
        backup_originals(args.labels, DEFAULT_IMAGES, DEFAULT_ORIGINALS)

    rows = load_labels(args.labels)
    if args.limit is not None:
        rows = rows[: args.limit]

    logger.info("Processing %d images", len(rows))
    csv_rows = []

    for row in rows:
        filename = row["filename"]
        img_path = DEFAULT_ORIGINALS / filename
        if not img_path.exists():
            img_path = DEFAULT_IMAGES / filename

        editor_size = config.get("editor", {}).get("size", 384)
        original_image = load_image(img_path, size=editor_size)
        base_prompt = build_prompt(row["object"], row["action"], row["background"])

        logger.info("Processing %s", filename)
        result = pipeline.run(
            original_image=original_image,
            base_prompt=base_prompt,
            metadata={
                "filename": filename,
                "object": row["object"],
                "action": row["action"],
                "background": row["background"],
            },
        )

        best_iteration = max(
            (it for it in result["all_iterations"] if "individual_scores" in it),
            key=lambda it: it["score"],
            default={},
        )
        individual = best_iteration.get("individual_scores", {})

        if result["best_image"] is not None:
            DEFAULT_EDITED.mkdir(parents=True, exist_ok=True)
            out_name = filename.replace(".jpg", "_edited.jpg")
            import cv2

            cv2.imwrite(str(DEFAULT_EDITED / out_name), result["best_image"])

        csv_rows.append({
            "filename": filename,
            "object": row["object"],
            "action": row["action"],
            "background": row["background"],
            "score": f"{result['best_score']:.4f}",
            "object_consistency": f"{individual.get('object_consistency', 0):.4f}",
            "clip_similarity": f"{individual.get('clip_similarity', 0):.4f}",
            "instruction_alignment": f"{individual.get('instruction_alignment', 0):.4f}",
            "experiment_id": result["experiment_id"],
        })

    if args.export_csv:
        export_metrics_csv(csv_rows, DEFAULT_METRICS)
        logger.info("Exported metrics to %s", DEFAULT_METRICS)

    logger.info("Finished processing %d images", len(rows))


if __name__ == "__main__":
    main()
