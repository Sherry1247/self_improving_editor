"""Visualization utilities for bounding boxes and comparisons."""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def draw_bbox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box on image (modifies copy).

    Args:
        image: Input image as numpy array (BGR format expected).
        bbox: Bounding box as (x1, y1, x2, y2).
        label: Optional label text to display above bbox.
        color: BGR color tuple. Defaults to green (0, 255, 0).
        thickness: Line thickness in pixels.

    Returns:
        Image with bounding box drawn.
    """
    result = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x, text_y = x1, max(y1 - 5, text_size[1])
        cv2.rectangle(
            result,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0], text_y + 5),
            color,
            -1,
        )
        cv2.putText(
            result,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return result


def save_comparison(
    orig_image: np.ndarray,
    edited_image: np.ndarray,
    orig_bbox: Optional[Tuple[int, int, int, int]] = None,
    edit_bbox: Optional[Tuple[int, int, int, int]] = None,
    orig_label: str = "Original",
    edit_label: str = "Edited",
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Create side-by-side comparison image with optional bounding boxes.

    Args:
        orig_image: Original image (BGR or RGB, will infer).
        edited_image: Edited image (same format as orig_image).
        orig_bbox: Optional bounding box for original as (x1, y1, x2, y2).
        edit_bbox: Optional bounding box for edited image.
        orig_label: Label for original image.
        edit_label: Label for edited image.
        output_path: If provided, saves comparison to this path.

    Returns:
        Combined comparison image (side-by-side).
    """
    orig_copy = orig_image.copy()
    edit_copy = edited_image.copy()

    if orig_bbox is not None:
        orig_copy = draw_bbox(orig_copy, orig_bbox, label=orig_label)
    if edit_bbox is not None:
        edit_copy = draw_bbox(edit_copy, edit_bbox, label=edit_label)

    comparison = np.hstack([orig_copy, edit_copy])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), comparison)

    return comparison


def create_iteration_grid(
    images: list,
    titles: list,
    cols: int = 2,
) -> np.ndarray:
    """
    Create a grid of images for visualization.

    Args:
        images: List of images (BGR format).
        titles: List of title strings for each image.
        cols: Number of columns in grid.

    Returns:
        Combined grid image.
    """
    if len(images) != len(titles):
        raise ValueError("Number of images must match number of titles")

    if len(images) == 0:
        raise ValueError("At least one image required")

    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]

    grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255

    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // cols
        col = idx % cols
        y, x = row * h, col * w

        resized = cv2.resize(img, (w, h))
        grid[y : y + h, x : x + w] = resized

        cv2.putText(
            grid,
            title,
            (x + 10, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

    return grid
