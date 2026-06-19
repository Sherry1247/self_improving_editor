"""I/O and configuration helpers for the MVP experiment runner."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

MVP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORING_CONFIG = MVP_ROOT / "config" / "scoring.yaml"
DEFAULT_DETECTOR_CONFIG = MVP_ROOT / "config" / "detector.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_scoring_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return load_yaml(path or DEFAULT_SCORING_CONFIG)


def load_detector_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return load_yaml(path or DEFAULT_DETECTOR_CONFIG)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Write a dictionary to JSON with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def save_text(text: str, path: Path) -> None:
    """Write plain text to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")


def load_text(path: Path) -> str:
    """Read a text file and strip trailing whitespace."""
    return path.read_text().strip()


def discover_samples(data_dir: Path) -> List[Path]:
    """
    Find sample directories under ``data/``.

    A valid sample directory contains ``before.jpg``, ``after.jpg``, and
    ``instruction.txt``.
    """
    if not data_dir.exists():
        return []

    samples = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if _is_valid_sample(child):
            samples.append(child)
    return samples


def _is_valid_sample(sample_dir: Path) -> bool:
    required = ["before.jpg", "after.jpg", "instruction.txt"]
    return all((sample_dir / name).exists() for name in required)
