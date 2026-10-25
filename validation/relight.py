"""Constructive covariance test: pasted (original lighting) vs re-lit (new scene lighting) versions of the SAME result.

By construction the re-lit version is the better background replacement for lighting, and it keeps texture and
structure. A covariance-aware critic must therefore (1) not call re-lighting subject drift and (2) not prefer the
pasted version. Needs no human labels, so it is the first line of evidence about the 'preservation critics prefer
copy-paste' bias (critics v2 fail it, v3 is built to pass it).
"""

from __future__ import annotations

import cv2
import numpy as np


def relight_subject(img: np.ndarray, mask: np.ndarray, bg_mask: np.ndarray | None = None, strength: float = 0.3,
                    side: float = 0.18, direction: int = 1, feather: float = 2.0) -> np.ndarray:
    """Re-light the subject for the scene it now stands in.

    * tint:   blend the subject colour toward the mean colour of the new background (colour adaptation);
    * shading: a smooth gain gradient across the subject (side light), brighter on one side;
    * brightness: gain is anchored to the background's mean luminance relative to the subject's.
    Texture and structure are preserved (multiplicative gain + convex blend). ``direction`` = +1 / -1 flips the side.
    """
    out = img.astype(np.float32)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return img.copy()
    bg = out[(~mask) if bg_mask is None else bg_mask]
    bg_rgb = bg.mean(0) if len(bg) else np.array([128.0, 128.0, 128.0])
    x01 = (xs - xs.min()) / max(1, xs.max() - xs.min())
    gain = 1.0 + side * direction * (1 - 2 * x01)
    sub = out[mask]
    lum_s, lum_b = float(sub.mean()), float(bg_rgb.mean())
    scene_gain = np.clip(1.0 + 0.5 * strength * (lum_b - lum_s) / max(lum_s, 1.0), 0.8, 1.35)
    relit = ((1 - strength) * sub + strength * bg_rgb) * (gain * scene_gain)[:, None]
    full = out.copy()
    full[mask] = np.clip(relit, 0, 255)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather)[..., None] if feather > 0 else mask[..., None]
    soft = np.maximum(soft, mask[..., None].astype(np.float32)) if feather > 0 else soft
    return np.clip(out * (1 - soft) + full * soft, 0, 255).astype(np.uint8)
