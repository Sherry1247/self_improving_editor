"""Mask-derived regions (innovation B): no manual annotation needed.

S  subject             = after-mask
B  boundary band       = dilate(S, k) - erode(S, k), split into inner / outer halves
C  contact band        = strip just below the subject's lowest pixels (where support + shadow live)
G  far background      = not dilate(S, 2k)
"""

from __future__ import annotations

import numpy as np

from src.types import Regions
from src.utils.geometry import dilate, erode, mask_box, similarity_align


def partition(mask_after: np.ndarray, mask_before: np.ndarray, band_frac: float = 0.03,
              contact_h_frac: float = 0.15, contact_w_pad: float = 0.2, foot_frac: float = 0.12) -> Regions:
    h, w = mask_after.shape
    aligned, tf = similarity_align(mask_before, mask_after) if mask_before.any() else (mask_before.copy(), {})
    box = mask_box(mask_after)
    if box is None:
        empty = np.zeros((h, w), bool)
        return Regions(empty, empty, empty, empty, np.ones((h, w), bool), aligned, tf)

    x1, y1, x2, y2 = box
    short = min(x2 - x1, y2 - y1)
    k = max(2, int(round(band_frac * max(short, 1) * 2)))
    inner = mask_after & ~erode(mask_after, k)
    outer = dilate(mask_after, k) & ~mask_after
    background = ~dilate(mask_after, 2 * k)

    # contact band: below the lowest `foot_frac` of the subject's height
    sh = y2 - y1
    foot_top = int(y2 - foot_frac * sh)
    foot = mask_after.copy()
    foot[:foot_top] = False
    fbox = mask_box(foot) or box
    pad = contact_w_pad * (fbox[2] - fbox[0])
    cx1, cx2 = int(max(0, fbox[0] - pad)), int(min(w, fbox[2] + pad))
    cy1, cy2 = int(max(0, foot_top)), int(min(h, y2 + contact_h_frac * sh))
    contact = np.zeros((h, w), bool)
    contact[cy1:cy2, cx1:cx2] = True
    contact &= ~mask_after
    return Regions(subject=mask_after.astype(bool), boundary_inner=inner, boundary_outer=outer, contact=contact,
                   background=background, aligned_before_subject=aligned, transform=tf)
