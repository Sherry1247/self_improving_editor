"""Image I/O. Fixes the legacy bugs: EXIF rotation ignored, portrait images squashed to squares."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(path: str | Path) -> np.ndarray:
    """Load as RGB uint8, honouring EXIF orientation."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im).copy()


def save_image(img: np.ndarray, path: str | Path, quality: int = 95) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, quality=quality)


def resize_long_side(img: np.ndarray, long_side: int, multiple: int = 8) -> np.ndarray:
    """Resize keeping aspect ratio so the long side is ~``long_side`` and both sides are multiples of ``multiple``."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    nh = max(multiple, int(round(h * s / multiple)) * multiple)
    nw = max(multiple, int(round(w * s / multiple)) * multiple)
    if (nh, nw) == (h, w):
        return img
    return np.asarray(Image.fromarray(img).resize((nw, nh), Image.LANCZOS))


def prepare_image(src: str | Path, cache_dir: str | Path, long_side: int = 1024) -> np.ndarray:
    """Load an original photo once, downscale (aspect preserved) and cache it as PNG."""
    src = Path(src)
    cached = Path(cache_dir) / f"{src.stem}_{long_side}.png"
    if cached.exists():
        return np.asarray(Image.open(cached).convert("RGB")).copy()
    im = Image.open(src)
    im.draft("RGB", (long_side * 2, long_side * 2))  # fast JPEG downscale for 6000px photos
    img = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    img = resize_long_side(img, long_side)
    cached.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(cached)
    return img


def match_size(img: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Resize ``img`` to exactly (H, W) — used when an editor returns a slightly different size."""
    if img.shape[:2] == tuple(shape_hw):
        return img
    return np.asarray(Image.fromarray(img).resize((shape_hw[1], shape_hw[0]), Image.LANCZOS))
