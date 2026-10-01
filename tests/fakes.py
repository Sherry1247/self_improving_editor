"""Synthetic scenes + a pixel-rule 'perceiver' so the whole pipeline runs on CPU without model weights.

The fake perceiver is deliberately simple and transparent: it finds the subject by its colour key,
uses colour statistics as 'embeddings', a ground-plane depth and palette matching for backgrounds.
It is only for testing control flow and critic maths — never for reporting results.
"""

from __future__ import annotations

import numpy as np

from src.spec import BACKGROUNDS, EditSpec
from src.types import Detection, Perception
from src.utils.geometry import mask_box

H, W = 192, 128
SUBJECT_RGB = np.array([200, 40, 40], np.uint8)
BG_PALETTE = {
    "river": ((60, 110, 160), (90, 80, 60)),  # (sky/upper colour, ground colour)
    "mountain": ((140, 150, 170), (110, 90, 70)),
    "snow": ((210, 220, 235), (240, 240, 245)),
    "beach": ((120, 180, 230), (225, 205, 150)),
    "city": ((150, 150, 155), (100, 100, 105)),
    "forest": ((40, 100, 50), (70, 60, 40)),
    "indoor": ((190, 170, 140), (130, 90, 60)),
}


def make_background(key: str, h: int = H, w: int = W, horizon: float = 0.55) -> np.ndarray:
    top, ground = (np.array(c, np.float32) for c in BG_PALETTE[key])
    img = np.zeros((h, w, 3), np.float32)
    hy = int(horizon * h)
    img[:hy] = top
    img[hy:] = ground
    rng = np.random.default_rng(abs(hash(key)) % 2**32)
    img += rng.normal(0, 4, img.shape)  # texture so patch features are not degenerate
    return np.clip(img, 0, 255).astype(np.uint8)


def subject_mask(h: int = H, w: int = W, cx: float = 0.5, bottom: float = 0.85, sw: float = 0.25,
                 shh: float = 0.4) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    cy = bottom * h - shh * h / 2
    return ((xx - cx * w) / (sw * w / 2)) ** 2 + ((yy - cy) / (shh * h / 2)) ** 2 <= 1.0


def make_scene(bg: str, mask: np.ndarray | None = None, shadow: float = 0.25, subject_rgb=SUBJECT_RGB,
               stripes: bool = True) -> tuple[np.ndarray, np.ndarray]:
    img = make_background(bg).astype(np.float32)
    mask = subject_mask() if mask is None else mask
    if shadow > 0:
        box = mask_box(mask)
        y2 = int(box[3])
        x1, x2 = int(box[0]), int(box[2])
        img[y2:y2 + 6, x1:x2] *= (1 - shadow)
    subj = np.broadcast_to(np.asarray(subject_rgb, np.float32), img.shape).copy()
    if stripes:  # identity texture
        subj[::6] *= 0.75
    img[mask] = subj[mask]
    return np.clip(img, 0, 255).astype(np.uint8), mask


class FakePerceiver:
    def __init__(self, grid: int = 8, tol: int = 60):
        self.grid, self.tol = grid, tol

    def find_subjects(self, img: np.ndarray) -> list[np.ndarray]:
        import cv2

        d = np.abs(img.astype(int) - SUBJECT_RGB.astype(int)).sum(-1)
        m = (d < self.tol) | (np.abs(img.astype(int) - (SUBJECT_RGB * 0.75).astype(int)).sum(-1) < self.tol)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        return [lab == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 80]

    def perceive(self, image: np.ndarray, spec: EditSpec) -> Perception:
        h, w = image.shape[:2]
        masks = sorted(self.find_subjects(image), key=lambda m: -m.sum())
        dets = [Detection(spec.subject_query, mask_box(m), 0.9, m) for m in masks]
        sm = masks[0] if masks else np.zeros((h, w), bool)
        # palette-based "old background" mask: pixels close to the source palette
        src = [np.array(c) for c in BG_PALETTE[spec.source_bg.key]]
        old = np.zeros((h, w), bool)
        for c in src:
            old |= np.abs(image.astype(int) - c).sum(-1) < 30
        old &= ~sm
        depth = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))
        return Perception(image=image, subject_detections=dets, subject_mask=sm, old_bg_mask=old, depth=depth,
                          subject_embedding=self.embed(image, sm), patch_features=self.patches(image),
                          bg_probs=self.bg_probs(image, sm), mock=True)

    def embed(self, img, m):
        if not m.any():
            return None
        px = img[m].astype(np.float32)
        rows = np.nonzero(m)[0]
        stripe = img[rows[rows % 6 == 0], :][:, :].mean() if len(rows) else 0
        v = np.concatenate([px.mean(0) / 255, px.std(0) / 64, [stripe / 255]])
        v = v - 0.3
        return v / (np.linalg.norm(v) + 1e-9)

    def patches(self, img):
        g = self.grid
        gh, gw = img.shape[0] // g, img.shape[1] // g
        x = img[: gh * g, : gw * g].astype(np.float32).reshape(gh, g, gw, g, 3)
        f = np.concatenate([x.mean((1, 3)) / 255 - 0.5, x.std((1, 3)) / 64], -1)
        return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9)

    def bg_probs(self, img, m):
        bg = img[~m].astype(np.float32).mean(0)
        d = {k: -np.linalg.norm(bg - (np.array(a) * 0.55 + np.array(b) * 0.45)) / 10 for k, (a, b) in BG_PALETTE.items()
             if k in BACKGROUNDS}
        z = np.array(list(d.values()))
        p = np.exp(z - z.max())
        p /= p.sum()
        return dict(zip(d, map(float, p)))


class FakeEditor:
    """Background swapper whose failure modes depend on its parameters, like a real editor's knobs.

    image_guidance_scale (preserve knob):
      low  (<1.4)  -> background replaced but the subject loses its texture (identity drift)
      mid          -> clean swap
      high (>2.1)  -> background barely changes
    The 'contact' clause adds a contact shadow; without it the result has none.
    """

    name = "fake"
    prompt_style = "instruction"
    preserve_knobs = {"image_guidance_scale": +1}

    def __init__(self, start_igs: float = 2.25):
        self.start = start_igs
        self.calls = 0

    def default_params(self):
        return {"image_guidance_scale": self.start}

    def param_ranges(self):
        from src.editors.base import ParamRange

        return {"image_guidance_scale": ParamRange(1.0, 2.5, 0.25)}

    def edit(self, image, prompt, params, seed, spec, subject_mask=None):
        self.calls += 1
        igs = params["image_guidance_scale"]
        if igs > 2.1:
            return image.copy()
        shadow = 0.25 if "contact shadow" in prompt else 0.0
        out, _ = make_scene(spec.target_bg.key, subject_mask, shadow=shadow, stripes=igs >= 1.4)
        return out
