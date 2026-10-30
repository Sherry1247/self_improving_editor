"""Constructive covariance test: pasted (original lighting) vs re-lit (new scene lighting) versions of the SAME result.

By construction the re-lit version is the better background replacement for lighting, and it keeps texture and
structure. A covariance-aware critic must therefore (1) not call re-lighting subject drift and (2) not prefer the
pasted version. Needs no human labels, so it is the first line of evidence about the 'preservation critics prefer
copy-paste' bias.

The operator is deliberately conservative and works in Lab: it only (a) scales luminance with a smooth
side-light gradient and a scene-brightness gain, and (b) shifts the subject's mean chroma a little toward the
scene's. Texture, edges and the spatial colour pattern are untouched (no blending of pixel content).
``strength=0, side=0`` is the NULL control: it runs the same Lab round trip and feathering and must score the
same as the input (any difference is pipeline noise, not lighting).
"""

from __future__ import annotations

import cv2
import numpy as np


def relight_subject(img: np.ndarray, mask: np.ndarray, bg_mask: np.ndarray | None = None, strength: float = 0.15,
                    side: float = 0.10, direction: int = 1, feather: float = 2.0) -> np.ndarray:
    """Re-light the subject for the scene it now stands in.

    * brightness: luminance gain toward the background's mean luminance (``strength`` of the gap, clipped);
    * side light: smooth gain gradient across the subject, brighter on one side (``direction`` = +1 / -1);
    * colour: the subject's mean (a, b) moves ``strength`` of the way toward the background's mean (a, b).
    """
    mask = mask.astype(bool)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return img.copy()
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    bg = lab[(~mask) if bg_mask is None else bg_mask]
    bg_l, bg_ab = (float(bg[:, 0].mean()), bg[:, 1:].mean(0)) if len(bg) else (50.0, np.zeros(2, np.float32))
    sub = lab[mask]
    s_l, s_ab = float(sub[:, 0].mean()), sub[:, 1:].mean(0)
    x01 = (xs - xs.min()) / max(1, xs.max() - xs.min())
    gain = 1.0 + side * direction * (1 - 2 * x01)
    scene_gain = float(np.clip(1.0 + strength * (bg_l - s_l) / max(s_l, 1.0), 0.85, 1.25))
    new = lab.copy()
    new[mask, 0] = np.clip(sub[:, 0] * gain * scene_gain, 0, 100)
    new[mask, 1:] = sub[:, 1:] + strength * (bg_ab - s_ab)
    out = np.clip(cv2.cvtColor(new, cv2.COLOR_LAB2RGB) * 255.0, 0, 255)
    base = img.astype(np.float32)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather)[..., None] if feather > 0 else mask[..., None].astype(np.float32)
    soft = np.maximum(soft, mask[..., None].astype(np.float32))
    return np.clip(base * (1 - soft) + out * soft, 0, 255).round().astype(np.uint8)


def degrade_image(img: np.ndarray, blur: float = 0.8, shift: tuple[float, float] = (0.6, 0.4), jpeg: int = 80) -> np.ndarray:
    """Identity-preserving NUISANCE control: slight blur, sub-pixel shift and JPEG re-compression of the whole image.

    It mimics what a generative editor's re-encoding does to a subject that is semantically unchanged. An appearance
    critic must keep scoring this version about as high as the original; if it does not, its tolerance is too tight
    (this is how the structure thresholds are calibrated without looking at which editor wins).
    """
    h, w = img.shape[:2]
    out = cv2.GaussianBlur(img, (0, 0), blur) if blur > 0 else img.copy()
    out = cv2.warpAffine(out, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else out
