"""Matplotlib visualization for experiment summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _load_rgb(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _draw_boxes(ax, image: np.ndarray, objects: List[Dict[str, Any]], title: str) -> None:
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
    """
    Create a side-by-side summary image with Grounding DINO boxes and scores.
    """
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
