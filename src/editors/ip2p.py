"""InstructPix2Pix editor (weak baseline). Runs at the working resolution — no more square squashing."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.data.images import match_size
from src.editors.base import Editor, ParamRange
from src.models import ModelRegistry
from src.spec import EditSpec


class InstructPix2PixEditor(Editor):
    name = "ip2p"
    prompt_style = "instruction"
    preserve_knobs = {"image_guidance_scale": +1, "guidance_scale": -1}

    def __init__(self, cfg: dict, registry: ModelRegistry):
        super().__init__(cfg)
        self.reg = registry

    def default_params(self) -> dict[str, float]:
        return {"guidance_scale": float(self.ecfg.get("guidance_scale", 7.5)),
                "image_guidance_scale": float(self.ecfg.get("image_guidance_scale", 1.5)),
                "num_inference_steps": float(self.ecfg.get("num_inference_steps", 30))}

    def param_ranges(self) -> dict[str, ParamRange]:
        return {"image_guidance_scale": ParamRange(1.0, 2.5, 0.25), "guidance_scale": ParamRange(5.0, 12.5, 1.5)}

    @torch.inference_mode()
    def edit(self, image: np.ndarray, prompt: str, params: dict[str, float], seed: int,
             spec: EditSpec, subject_mask: np.ndarray | None = None) -> np.ndarray:
        with self.reg.use("ip2p") as b:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            out = b.model(prompt, image=Image.fromarray(image),
                          num_inference_steps=int(params["num_inference_steps"]),
                          guidance_scale=float(params["guidance_scale"]),
                          image_guidance_scale=float(params["image_guidance_scale"]),
                          generator=gen).images[0]
        return match_size(np.asarray(out.convert("RGB")), image.shape[:2])
