"""Editor interface. Each editor declares its own tunable parameters so the router never hard-codes them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.spec import EditSpec


@dataclass(frozen=True)
class ParamRange:
    lo: float
    hi: float
    step: float

    def nudge(self, value: float, direction: int) -> float:
        return float(np.clip(value + direction * self.step, self.lo, self.hi))


class Editor(ABC):
    name = "editor"
    prompt_style = "instruction"  # "instruction" (IP2P / Kontext) or "scene" (inpainting)
    # knobs the router may turn; sign convention: +1 => preserve the input more, -1 => edit more strongly
    preserve_knobs: dict[str, int] = {}

    def __init__(self, cfg: dict):
        self.ecfg = cfg.get("editors", {}).get(self.name, {})

    @abstractmethod
    def default_params(self) -> dict[str, float]: ...

    @abstractmethod
    def param_ranges(self) -> dict[str, ParamRange]: ...

    @abstractmethod
    def edit(self, image: np.ndarray, prompt: str, params: dict[str, float], seed: int,
             spec: EditSpec, subject_mask: np.ndarray | None = None) -> np.ndarray:
        """Return an RGB uint8 image with exactly the same shape as ``image``."""
