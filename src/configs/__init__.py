"""Configuration loading and component factory."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from src.critics import (
    CLIPSimilarityCritic,
    CompositeCritic,
    InstructionAlignmentCritic,
    ObjectConsistencyCritic,
)
from src.detectors import YOLODetector
from src.editors import Pix2PixEditor
from src.pipelines import ClosedLoopPipeline
from src.utils.device_utils import get_device, set_seed
from src.utils.logging_config import setup_logging

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to config YAML. Defaults to default.yaml.

    Returns:
        Configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_device(config: Dict[str, Any]) -> str:
    """Resolve device string from config ('auto' → detected device)."""
    device = config.get("device", "auto")
    if device == "auto":
        return get_device()
    return device


def resolve_dtype(config: Dict[str, Any], device: str) -> torch.dtype:
    """Resolve torch dtype from config and device."""
    precision = config.get("precision", "float16")
    if precision == "float16" and device == "cuda":
        return torch.float16
    return torch.float32


def build_pipeline(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> ClosedLoopPipeline:
    """
    Build a fully wired ClosedLoopPipeline from configuration.

    Args:
        config: Pre-loaded config dict. Loaded from config_path if None.
        config_path: Path to YAML config when config is None.

    Returns:
        Configured ClosedLoopPipeline instance.
    """
    if config is None:
        config = load_config(config_path)

    device = resolve_device(config)
    dtype = resolve_dtype(config, device)

    seed = config.get("seed", 42)
    set_seed(seed)

    log_cfg = config.get("logging", {})
    log_level_name = log_cfg.get("level", "INFO")
    log_file = log_cfg.get("file")
    setup_logging(
        log_level=getattr(logging, log_level_name, logging.INFO),
        log_file=Path(log_file) if log_file else None,
    )

    detector_cfg = config.get("detector", {})
    detector = YOLODetector(
        model_path=config.get("detector_model", "yolov8l.pt"),
        target_classes=detector_cfg.get("target_classes", [0, 16]),
    )

    editor_cfg = config.get("editor", {})
    editor = Pix2PixEditor(
        model_id=config.get("editor_model", "timbrooks/instruct-pix2pix"),
        device=device,
        dtype=dtype,
        default_size=editor_cfg.get("size", 384),
        num_inference_steps=editor_cfg.get("num_inference_steps", 30),
        image_guidance_scale=editor_cfg.get("image_guidance_scale", 1.5),
        guidance_scale=editor_cfg.get("guidance_scale", 7.5),
    )

    clip_model_id = config.get("clip_model", "openai/clip-vit-base-patch32")
    critics = [
        ObjectConsistencyCritic(detector),
        CLIPSimilarityCritic(model_id=clip_model_id, device=device),
        InstructionAlignmentCritic(model_id=clip_model_id, device=device),
    ]

    critic_weights = config.get("critics", {}).get("weights", {
        "object_consistency": 0.4,
        "clip_similarity": 0.3,
        "instruction_alignment": 0.3,
    })
    critic = CompositeCritic(critics, weights=critic_weights)

    output_dir = Path(config.get("output", {}).get("base_dir", "data/experiments"))

    return ClosedLoopPipeline(
        detector=detector,
        editor=editor,
        critic=critic,
        output_dir=output_dir,
        config=config,
    )
