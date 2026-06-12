"""Shared CLIP model utilities for critics."""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.utils.device_utils import get_device

_clip_bundle: Optional[Tuple[CLIPModel, CLIPProcessor, str]] = None


def get_clip_model(
    model_id: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
) -> Tuple[CLIPModel, CLIPProcessor, str]:
    """
    Lazily load and cache a CLIP model + processor.

    Args:
        model_id: HuggingFace model identifier.
        device: Target device. Auto-detected when None.

    Returns:
        Tuple of (model, processor, device_string).
    """
    global _clip_bundle

    if device is None:
        device = get_device()

    if _clip_bundle is None or _clip_bundle[2] != device:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        model.to(device)
        model.eval()
        _clip_bundle = (model, processor, device)

    return _clip_bundle


def clip_image_image_similarity(
    image_a: np.ndarray,
    image_b: np.ndarray,
    model_id: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
) -> float:
    """
    Cosine similarity between two images in CLIP embedding space.

    Returns a value in [0, 1] via (cosine + 1) / 2 normalization.
    """
    model, processor, device = get_clip_model(model_id, device)

    pil_a = _to_pil(image_a)
    pil_b = _to_pil(image_b)

    with torch.inference_mode():
        inputs = processor(images=[pil_a, pil_b], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        cosine = float((features[0] @ features[1]).item())

    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def clip_image_text_similarity(
    image: np.ndarray,
    text: str,
    model_id: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
) -> float:
    """
    Cosine similarity between an image and text in CLIP embedding space.

    Returns a value in [0, 1] via (cosine + 1) / 2 normalization.
    """
    model, processor, device = get_clip_model(model_id, device)

    pil_image = _to_pil(image)

    with torch.inference_mode():
        inputs = processor(
            text=[text],
            images=pil_image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image_features = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cosine = float((image_features @ text_features.T).item())

    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def _to_pil(image: np.ndarray) -> Image.Image:
    """Convert BGR or RGB numpy array to PIL RGB image."""
    import cv2

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image

    return Image.fromarray(rgb, "RGB")
