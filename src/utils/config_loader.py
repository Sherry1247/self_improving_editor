"""YAML configuration loader."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORING_CONFIG = REPO_ROOT / "config" / "scoring.yaml"
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_scoring_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return load_yaml(path or DEFAULT_SCORING_CONFIG)


def load_detector_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return load_yaml(path or DEFAULT_DETECTOR_CONFIG)
