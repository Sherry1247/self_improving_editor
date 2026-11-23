"""Controlled interventions on ONE photo, for testing what a Keep-type measure counts as 'the subject changed'.

Two families act ONLY inside the subject mask and leave the rest of the image untouched:

  LIGHTING (the paper's illumination term l; a good Keep measure should NOT penalise it)
      gain   multiply the subject's luminance (brighter / darker light)
      side   smooth left-right luminance gradient (light from one side)
      cast   shift the subject's mean chroma (a, b) (warmer / cooler light)
      combo  relight_subject(): scene gain + side light + chroma shift together
  STRUCTURE (the paper's reflectance term rho: texture, pattern, shape, colour of the material; SHOULD be penalised)
      blur     masked Gaussian blur (texture destroyed)
      shuffle  fine texture moved by a few pixels (same colours and shading, different pattern)
      hue      rotate the (a, b) chroma vector by an angle (the material has another colour)
      mirror   the subject flipped left-right in place (pose / shape change), on a clean plate

plus NULL (zero-strength relight: same Lab round trip and feathering) and NUISANCE (blur + sub-pixel shift + JPEG of the
whole image: what an editor's re-encoding does to an unchanged subject). Every variant has a ``level`` 1..3 (mild, moderate,
strong) so a dose-response can be read off.

``standardize`` / ``l_image`` implement the paper's light-normalisation operator (Eq. 5): per-channel (x - mean) / std inside the mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from validation.relight import degrade_image, relight_subject

LIGHT, STRUCT, NULL, NUISANCE = "light", "struct", "null", "nuisance"


@dataclass
class Variant:
    family: str      # light | struct | null | nuisance
    kind: str        # gain | side | cast | combo | blur | shuffle | hue | mirror | null | degrade
    name: str        # unique, e.g. gain_x0.70
    level: int       # 0 for null / nuisance, else 1..3
    image: np.ndarray
    params: dict


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _rgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0, 0, 255)


def _composite(img: np.ndarray, new_rgb: np.ndarray, mask: np.ndarray, feather: float = 1.5) -> np.ndarray:
    """new_rgb inside the mask, feathered at the edge, original outside (the same blending as relight_subject)."""
    m = mask.astype(np.float32)
    soft = np.maximum(cv2.GaussianBlur(m, (0, 0), feather), m)[..., None] if feather > 0 else m[..., None]
    return np.clip(img.astype(np.float32) * (1 - soft) + new_rgb * soft, 0, 255).round().astype(np.uint8)


def masked_blur(x: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that averages only pixels inside ``mask`` (no bleeding in from outside). x: HxW or HxWxC."""
    m = mask.astype(np.float32)
    x = x.astype(np.float32)
    if x.ndim == 3:
        num = np.stack([cv2.GaussianBlur(x[..., c] * m, (0, 0), sigma) for c in range(x.shape[2])], -1)
        return num / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-3)[..., None]
    return cv2.GaussianBlur(x * m, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-3)


# ------------------------------------------------------------------ lighting
def light_gain(img: np.ndarray, mask: np.ndarray, gain: float) -> np.ndarray:
    lab = _lab(img)
    lab[..., 0] = np.where(mask, np.clip(lab[..., 0] * gain, 0, 100), lab[..., 0])
    return _composite(img, _rgb(lab), mask)


def light_side(img: np.ndarray, mask: np.ndarray, amount: float, direction: int = 1) -> np.ndarray:
    """Luminance gain 1 + amount*direction*(1 - 2*x01) across the subject's width (brighter on the left if direction=+1)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return img.copy()
    lab = _lab(img)
    x01 = (np.arange(mask.shape[1], dtype=np.float32) - xs.min()) / max(1, xs.max() - xs.min())
    gain = 1.0 + amount * direction * (1 - 2 * np.clip(x01, 0, 1))[None, :]
    lab[..., 0] = np.where(mask, np.clip(lab[..., 0] * gain, 0, 100), lab[..., 0])
    return _composite(img, _rgb(lab), mask)


def light_cast(img: np.ndarray, mask: np.ndarray, da: float, db: float) -> np.ndarray:
    """Shift the subject's mean chroma by (da, db) Lab units: a pure colour-temperature change of the light."""
    lab = _lab(img)
    lab[..., 1] = np.where(mask, lab[..., 1] + da, lab[..., 1])
    lab[..., 2] = np.where(mask, lab[..., 2] + db, lab[..., 2])
    return _composite(img, _rgb(lab), mask)


# ------------------------------------------------------------------ structure
def texture_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    return _composite(img, masked_blur(img, mask, sigma), mask)


def texture_shuffle(img: np.ndarray, mask: np.ndarray, dx: int, dy: int, sigma: float = 3.0) -> np.ndarray:
    """Keep the low-frequency luminance (shading) and the colours, move the fine luminance layer by (dx, dy) pixels.

    The pattern (fur, cloth weave, edges inside the subject) is now at the wrong place, but the subject has exactly the
    same mean colour, shading and illumination. Pure texture change."""
    lab = _lab(img)
    low = masked_blur(lab[..., 0], mask, sigma)
    high = (lab[..., 0] - low) * mask
    high = np.roll(high, (dy, dx), axis=(0, 1))
    lab[..., 0] = np.where(mask, np.clip(low + high, 0, 100), lab[..., 0])
    return _composite(img, _rgb(lab), mask)


def hue_rotate(img: np.ndarray, mask: np.ndarray, deg: float) -> np.ndarray:
    """Rotate each pixel's (a, b) chroma vector by ``deg`` degrees: the material gets another colour (not a cast)."""
    lab = _lab(img)
    t = np.deg2rad(deg)
    a, b = lab[..., 1].copy(), lab[..., 2].copy()
    lab[..., 1] = np.where(mask, np.cos(t) * a - np.sin(t) * b, a)
    lab[..., 2] = np.where(mask, np.sin(t) * a + np.cos(t) * b, b)
    return _composite(img, _rgb(lab), mask)


def mirror_subject(img: np.ndarray, mask: np.ndarray, plate: np.ndarray) -> np.ndarray:
    """Flip the subject left-right inside its bounding box and paste it on the clean plate (a pose / shape change)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return img.copy()
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out = plate.copy()
    crop, mcrop = img[y1:y2, x1:x2][:, ::-1], mask[y1:y2, x1:x2][:, ::-1]
    alpha = cv2.GaussianBlur(mcrop.astype(np.float32), (3, 3), 0)[..., None]
    out[y1:y2, x1:x2] = (alpha * crop + (1 - alpha) * out[y1:y2, x1:x2]).astype(np.uint8)
    return out


# --------------------------------------------------------------------- suite
GAINS = {1: (0.85, 1.15), 2: (0.70, 1.30), 3: (0.50, 1.50)}
SIDES = {1: 0.15, 2: 0.30, 3: 0.50}
CASTS = {1: (4.0, 6.0), 2: (8.0, 12.0), 3: (14.0, 20.0)}          # warm; level 2 also gets the cool mirror image
COMBOS = {1: (0.15, 0.10), 2: (0.30, 0.20), 3: (0.60, 0.35)}      # (strength, side) of relight_subject
BLURS = {1: 1.5, 2: 3.0, 3: 6.0}
SHUFFLES = {1: (4, 3), 2: (10, 6), 3: (20, 12)}
HUES = {1: 25.0, 2: 60.0, 3: 120.0}


def build_variants(img: np.ndarray, mask: np.ndarray, plate: np.ndarray, bg_mask: np.ndarray | None = None) -> list[Variant]:
    V: list[Variant] = []

    def add(family, kind, name, level, image, **params):
        V.append(Variant(family, kind, name, level, image, params))

    add(NULL, "null", "null", 0, relight_subject(img, mask, bg_mask, strength=0.0, side=0.0))
    add(NUISANCE, "degrade", "degrade", 0, degrade_image(img))
    for lv, (lo, hi) in GAINS.items():
        add(LIGHT, "gain", f"gain_x{lo:.2f}", lv, light_gain(img, mask, lo), gain=lo)
        add(LIGHT, "gain", f"gain_x{hi:.2f}", lv, light_gain(img, mask, hi), gain=hi)
    for lv, amt in SIDES.items():
        add(LIGHT, "side", f"side_{amt:.2f}", lv, light_side(img, mask, amt), amount=amt)
    for lv, (da, db) in CASTS.items():
        add(LIGHT, "cast", f"cast_warm{lv}", lv, light_cast(img, mask, da, db), da=da, db=db)
    add(LIGHT, "cast", "cast_cool2", 2, light_cast(img, mask, -8.0, -12.0), da=-8.0, db=-12.0)
    for lv, (st, sd) in COMBOS.items():
        add(LIGHT, "combo", f"combo{lv}", lv, relight_subject(img, mask, bg_mask, strength=st, side=sd), strength=st, side=sd)
    for lv, s in BLURS.items():
        add(STRUCT, "blur", f"blur_s{s:g}", lv, texture_blur(img, mask, s), sigma=s)
    for lv, (dx, dy) in SHUFFLES.items():
        add(STRUCT, "shuffle", f"shuffle_{dx}_{dy}", lv, texture_shuffle(img, mask, dx, dy), dx=dx, dy=dy)
    for lv, d in HUES.items():
        add(STRUCT, "hue", f"hue_{d:g}", lv, hue_rotate(img, mask, d), deg=d)
    add(STRUCT, "mirror", "mirror", 3, mirror_subject(img, mask, plate))
    return V


# ----------------------------------------------- the paper's light normalisation
def standardize(img: np.ndarray, mask: np.ndarray, space: str = "lab") -> np.ndarray:
    """Eq. 5: per-channel (x - mean_S) / std_S inside the mask. Returns float32 HxWx3 (values outside the mask are 0)."""
    x = _lab(img) if space == "lab" else img.astype(np.float32) / 255.0
    px = x[mask]
    out = np.zeros_like(x, np.float32)
    if len(px) < 10:
        return out
    out[mask] = (px - px.mean(0)) / (px.std(0) + 1e-6)
    return out


def l_image(img: np.ndarray, mask: np.ndarray, space: str = "lab") -> np.ndarray:
    """The standardised subject mapped back to an RGB image with FIXED statistics, so a vision encoder can embed it.

    Every subject ends up with the same per-channel mean and spread (Lab: L 55 +- 18, a, b 0 +- 12; RGB: 0.5 +- 0.2), i.e. no
    information about its brightness, contrast or colour cast reaches the encoder; only the pattern of deviations does."""
    z = standardize(img, mask, space)
    if space == "lab":
        lab = z * np.array([18.0, 12.0, 12.0], np.float32) + np.array([55.0, 0.0, 0.0], np.float32)
        rgb = _rgb(lab)
    else:
        rgb = np.clip((z * 0.2 + 0.5) * 255.0, 0, 255)
    out = img.copy()
    out[mask] = rgb[mask].round().astype(np.uint8)
    return out


# ------------------------------------------- colour terms that ignore a global cast
def chroma_terms(before_warp: np.ndarray, after: np.ndarray, region: np.ndarray, floor: float = 1.0) -> dict:
    """Two colour measures inside ``region`` that do not count a uniform colour-temperature shift as a change.

    chroma_std  exp(-d), d = mean distance between the per-channel standardised (a, b) of the two images (std floored at ``floor`` Lab
                units so a nearly uniform subject does not amplify noise). Standardising removes mean and spread, i.e. a cast and a chroma gain.
    chroma_cov  exp(-d / 0.5), d = ||cov_after - cov_before||_F / trace(cov_before) of the (a, b) values: the SHAPE of the colour
                distribution. A uniform shift leaves it unchanged; rotating every pixel's hue rotates it (when the colours vary).
    Caveat: for a single-coloured subject a hue rotation and a cast are nearly the same edit, so neither term can tell them apart there.
    """
    if region.sum() < 50:
        return {}
    ab_b = cv2.cvtColor(before_warp.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)[..., 1:][region]
    ab_a = cv2.cvtColor(after.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)[..., 1:][region]
    zb = (ab_b - ab_b.mean(0)) / (ab_b.std(0) + floor)
    za = (ab_a - ab_a.mean(0)) / (ab_a.std(0) + floor)
    d_z = float(np.linalg.norm(za - zb, axis=-1).mean())
    sb, sa = np.cov(ab_b.T), np.cov(ab_a.T)
    d_cov = float(np.linalg.norm(sa - sb) / (np.trace(sb) + 1e-6))
    return {"chroma_std": float(np.exp(-d_z)), "chroma_cov": float(np.exp(-d_cov / 0.5))}
