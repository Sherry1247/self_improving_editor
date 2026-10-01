"""Compositing baseline: keep the subject pixels, regenerate everything else with SDXL inpainting.

By construction it preserves the subject perfectly, so it isolates the *physics* failure modes
(no contact shadow, lighting mismatch, halos) — the opposite end of the trade-off from IP2P.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.images import match_size, resize_long_side
from src.editors.base import Editor, ParamRange
from src.models import ModelRegistry
from src.spec import EditSpec
from src.utils.geometry import dilate

NEGATIVE = "blurry, deformed, extra limbs, duplicate, text, watermark, cartoon, painting"


class CompositingEditor(Editor):
    name = "compositing"
    prompt_style = "scene"
    preserve_knobs = {"mask_dilate": +1}

    def __init__(self, cfg: dict, registry: ModelRegistry):
        super().__init__(cfg)
        self.reg = registry

    def default_params(self) -> dict[str, float]:
        return {"guidance_scale": float(self.ecfg.get("guidance_scale", 7.0)),
                "strength": float(self.ecfg.get("strength", 0.99)),
                "mask_dilate": float(self.ecfg.get("mask_dilate", 6)),
                "num_inference_steps": float(self.ecfg.get("num_inference_steps", 30))}

    def param_ranges(self) -> dict[str, ParamRange]:
        return {"guidance_scale": ParamRange(4.0, 10.0, 1.0), "mask_dilate": ParamRange(2, 22, 4)}

    @torch.inference_mode()
    def edit(self, image: np.ndarray, prompt: str, params: dict[str, float], seed: int,
             spec: EditSpec, subject_mask: np.ndarray | None = None) -> np.ndarray:
        if subject_mask is None or not subject_mask.any():
            raise ValueError("CompositingEditor needs the before-image subject mask")
        work = resize_long_side(image, int(self.ecfg.get("long_side", 1024)))
        h, w = work.shape[:2]
        m = cv2.resize(subject_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        keep = dilate(m, int(params["mask_dilate"]))
        repaint = Image.fromarray(((~keep) * 255).astype(np.uint8))
        with self.reg.use("sdxl_inpaint") as b:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            out = b.model(prompt=prompt, negative_prompt=NEGATIVE, image=Image.fromarray(work), mask_image=repaint,
                          height=h, width=w, strength=float(params["strength"]),
                          guidance_scale=float(params["guidance_scale"]),
                          num_inference_steps=int(params["num_inference_steps"]), generator=gen).images[0]
        out = match_size(np.asarray(out.convert("RGB")), image.shape[:2])
        # paste the original subject back with a 1-px feather so it is preserved exactly
        alpha = cv2.GaussianBlur(subject_mask.astype(np.float32), (3, 3), 0)[..., None]
        return (alpha * image + (1 - alpha) * out).astype(np.uint8)
