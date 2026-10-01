"""Qwen-Image-Edit (open weights, Apache-2.0): a strong instruction editor that can replace whole backgrounds.

~20B MMDiT + Qwen2.5-VL text encoder: needs an 80 GB GPU in bf16 (set GPUMEM on CHTC), or cpu_offload.
It has no image-guidance knob, so the router works with prompt clauses, true_cfg_scale and reseeding.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.data.images import match_size
from src.editors.base import Editor, ParamRange
from src.models import ModelRegistry
from src.spec import EditSpec


class QwenImageEditor(Editor):
    name = "qwen_edit"
    prompt_style = "instruction"
    preserve_knobs = {"true_cfg_scale": -1}  # lower CFG = follows the text less = stays closer to the input

    def __init__(self, cfg: dict, registry: ModelRegistry):
        super().__init__(cfg)
        self.reg = registry

    def default_params(self) -> dict[str, float]:
        return {"true_cfg_scale": float(self.ecfg.get("true_cfg_scale", 4.0)),
                "num_inference_steps": float(self.ecfg.get("num_inference_steps", 40))}

    def param_ranges(self) -> dict[str, ParamRange]:
        return {"true_cfg_scale": ParamRange(2.5, 6.0, 0.75)}

    @torch.inference_mode()
    def edit(self, image: np.ndarray, prompt: str, params: dict[str, float], seed: int,
             spec: EditSpec, subject_mask: np.ndarray | None = None) -> np.ndarray:
        with self.reg.use("qwen_edit") as b:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            out = b.model(image=Image.fromarray(image), prompt=prompt, negative_prompt=" ",
                          true_cfg_scale=float(params["true_cfg_scale"]),
                          num_inference_steps=int(params["num_inference_steps"]), generator=gen).images[0]
        return match_size(np.asarray(out.convert("RGB")), image.shape[:2])
