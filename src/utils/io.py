"""I/O helpers for the MVP experiment runner."""

import json
from pathlib import Path
from typing import Any, Dict, List


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
