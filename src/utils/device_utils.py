"""Device detection and configuration utilities."""

import random
from typing import Literal

import numpy as np
import torch


def get_device() -> Literal["cuda", "mps", "cpu"]:
    """
    Detect and return available device.

    Returns:
        Device string: "cuda" for NVIDIA GPU, "mps" for Apple Metal,
        or "cpu" as fallback.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
