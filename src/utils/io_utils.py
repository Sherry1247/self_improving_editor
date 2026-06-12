"""I/O utilities for JSON serialization and data handling."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("closed_loop_editor")


def load_image(
    path: Union[str, Path],
    size: Optional[int] = None,
    bgr: bool = True,
) -> np.ndarray:
    """
    Load image from disk as numpy array.

    Args:
        path: Image file path.
        size: Optional square resize dimension.
        bgr: If True, return BGR (OpenCV convention).

    Returns:
        Image as uint8 numpy array.
    """
    pil_image = Image.open(str(path)).convert("RGB")
    if size is not None:
        pil_image = pil_image.resize((size, size), Image.LANCZOS)

    array = np.array(pil_image)
    if bgr:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


def load_labels(labels_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Load labeled image metadata from CSV.

    Expected columns: filename, object, action, background.
    """
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels CSV not found at {path}")

    rows: List[Dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def backup_originals(
    labels_path: Union[str, Path],
    images_dir: Union[str, Path],
    original_dir: Union[str, Path],
) -> None:
    """
    Ensure all labeled images exist under original_dir.

    Copies from images_dir when missing in original_dir.
    """
    original_path = Path(original_dir)
    images_path = Path(images_dir)
    original_path.mkdir(parents=True, exist_ok=True)

    for row in load_labels(labels_path):
        filename = row["filename"]
        dst = original_path / filename
        if dst.exists():
            continue

        src = images_path / filename
        if src.exists():
            copy2(src, dst)
            logger.info("Backed up %s -> %s", src, dst)
        else:
            logger.warning("Cannot find original image for %s", filename)


def build_prompt(obj: str, action: str, background: str) -> str:
    """Build the default background-replacement editing prompt."""
    return (
        f"replace the current {background} background with a different scene, "
        f"while keeping the {obj} and its {action} pose unchanged, "
        f"natural lighting, realistic photo"
    )


def save_experiment(
    result: Dict[str, Any],
    exp_dir: Path,
) -> Path:
    """
    Save experiment results to metadata.json inside exp_dir.

    Args:
        result: Experiment result dictionary (must be JSON-serializable).
        exp_dir: Experiment output directory.

    Returns:
        Path to saved metadata.json.
    """
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "images").mkdir(exist_ok=True)
    (exp_dir / "iterations").mkdir(exist_ok=True)

    metadata_file = exp_dir / "metadata.json"
    with metadata_file.open("w") as f:
        json.dump(result, f, indent=2, default=str)

    return metadata_file


def save_iteration(
    iteration_data: Dict[str, Any],
    exp_dir: Path,
    iteration_idx: int,
) -> Path:
    """Save individual iteration details to JSON."""
    iter_dir = exp_dir / "iterations"
    iter_dir.mkdir(parents=True, exist_ok=True)
    iter_file = iter_dir / f"{iteration_idx:03d}.json"
    with iter_file.open("w") as f:
        json.dump(iteration_data, f, indent=2, default=str)
    return iter_file


def load_experiment(metadata_file: Path) -> Dict[str, Any]:
    """Load experiment results from metadata.json."""
    with metadata_file.open("r") as f:
        return json.load(f)


def combine_metrics(
    scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Combine multiple scores using provided weights.

    Raises:
        ValueError: If weights don't sum to 1.0 or keys mismatch.
    """
    if weights is None:
        weights = {k: 1.0 / len(scores) for k in scores.keys()}

    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights.values())}")

    if set(weights.keys()) != set(scores.keys()):
        raise ValueError(
            f"Weight keys {set(weights.keys())} don't match score keys {set(scores.keys())}"
        )

    return sum(scores[k] * weights[k] for k in scores.keys())


def make_experiment_id(job_id: Optional[str] = None) -> str:
    """Create a unique experiment directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if job_id:
        return f"{job_id}_{timestamp}"
    return timestamp
