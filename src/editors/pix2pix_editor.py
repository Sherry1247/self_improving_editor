"""InstructPix2Pix image editor implementation."""

from typing import Any, Optional

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image

from src.utils.device_utils import get_device

from .base import Editor


class Pix2PixEditor(Editor):
    """
    InstructPix2Pix-based image editor.

    Wraps Stable Diffusion InstructPix2Pix for text-guided editing.
    """

    def __init__(
        self,
        model_id: str = "timbrooks/instruct-pix2pix",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        default_size: int = 384,
        num_inference_steps: int = 30,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
    ):
        if device is None:
            device = get_device()

        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32

        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.default_size = default_size
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.guidance_scale = guidance_scale

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        self.pipe.to(device)
        self.pipe.safety_checker = self._dummy_safety_checker
        self.pipe.enable_attention_slicing()

    @staticmethod
    def _dummy_safety_checker(images, clip_input):
        return images, [False] * len(images)

    def edit(
        self,
        image: np.ndarray,
        prompt: str,
        num_inference_steps: Optional[int] = None,
        image_guidance_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        size: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Edit image based on text instruction.

        Args:
            image: Input BGR numpy array.
            prompt: Editing instruction.
            num_inference_steps: Denoising steps (defaults to config value).
            image_guidance_scale: Input-image adherence strength.
            guidance_scale: Text prompt adherence strength.
            size: Square output size in pixels.

        Returns:
            Edited image as BGR numpy array.
        """
        pil_image = self._numpy_to_pil(image)
        target_size = size or self.default_size
        pil_image = pil_image.resize((target_size, target_size), Image.LANCZOS)

        with torch.inference_mode():
            result = self.pipe(
                prompt,
                image=pil_image,
                num_inference_steps=num_inference_steps or self.num_inference_steps,
                image_guidance_scale=image_guidance_scale or self.image_guidance_scale,
                guidance_scale=guidance_scale or self.guidance_scale,
            )

        edited_rgb = np.array(result.images[0])
        return cv2.cvtColor(edited_rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _numpy_to_pil(image: np.ndarray) -> Image.Image:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        return Image.fromarray(rgb, "RGB")

    def to(self, device: str) -> None:
        """Move model to specified device."""
        self.device = device
        self.pipe.to(device)
