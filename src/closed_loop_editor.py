"""
Backward-compatible entry point for the original closed-loop prototype.

Delegates to the modular framework while preserving legacy behavior:
- Processes first 15 labeled images
- Exports metrics to data/metrics.csv
- Saves edited images to data/images/edited/
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from src.configs import build_pipeline, load_config
from src.utils.io_utils import backup_originals, build_prompt, load_image, load_labels

logger = logging.getLogger("closed_loop_editor")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
LABELS_PATH = DATA_DIR / "labels.csv"
IMG_DIR = DATA_DIR / "images"
ORIG_IMG_DIR = IMG_DIR / "original"
EDITED_DIR = IMG_DIR / "edited"
METRICS_CSV = DATA_DIR / "metrics.csv"


def main(limit: int = 15, export_csv: bool = True) -> None:
    """Run legacy closed-loop workflow on a subset of labeled images."""
    config = load_config()
    pipeline_cfg = config.get("pipeline", {})
    pipeline_cfg.setdefault("max_iterations", 1)
    pipeline_cfg.setdefault("score_threshold", 0.5)
    config["pipeline"] = pipeline_cfg

    pipeline = build_pipeline(config)
    backup_originals(LABELS_PATH, IMG_DIR, ORIG_IMG_DIR)

    rows = load_labels(LABELS_PATH)[:limit]
    logger.info("Loaded %d labeled images (limit=%d)", len(rows), limit)

    if export_csv and METRICS_CSV.exists():
        METRICS_CSV.unlink()

    editor_size = config.get("editor", {}).get("size", 384)
    csv_rows = []

    for row in rows:
        filename = row["filename"]
        img_path = ORIG_IMG_DIR / filename
        original_image = load_image(img_path, size=editor_size)
        base_prompt = build_prompt(row["object"], row["action"], row["background"])

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
            (it for it in result["all_iterations"] if "detections" in it),
            key=lambda it: it.get("score", -1),
            default={},
        )
        detections = best_iteration.get("detections", {})
        individual = best_iteration.get("individual_scores", {})
        iou = individual.get("object_consistency", best_iteration.get("score", 0))

        if result["best_image"] is not None:
            EDITED_DIR.mkdir(parents=True, exist_ok=True)
            out_path = EDITED_DIR / filename.replace(".jpg", "_edited.jpg")
            import cv2

            cv2.imwrite(str(out_path), result["best_image"])
            logger.info("Saved best edit to %s, score=%.3f", out_path, result["best_score"])

        csv_rows.append({
            "filename": filename,
            "object": row["object"],
            "action": row["action"],
            "background": row["background"],
            "score": f"{result['best_score']:.4f}",
            "iou": f"{iou:.4f}",
            "orig_cls": detections.get("orig_cls"),
            "edit_cls": detections.get("edit_cls"),
        })

    if export_csv:
        with METRICS_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "filename",
                    "object",
                    "action",
                    "background",
                    "score",
                    "iou",
                    "orig_cls",
                    "edit_cls",
                ],
            )
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
        logger.info("Wrote metrics to %s", METRICS_CSV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legacy closed-loop editor")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()
    main(limit=args.limit, export_csv=not args.no_csv)
