"""Colour helpers (Lab conversion, masked statistics, Reinhard transfer)."""

from __future__ import annotations

import cv2
import numpy as np


def to_lab(img: np.ndarray) -> np.ndarray:
    """RGB uint8 -> Lab float32 with L in [0, 100], a/b roughly in [-128, 127]."""
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return lab


def masked_stats(lab: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px = lab[mask]
    if len(px) == 0:
        return np.zeros(3, np.float32), np.ones(3, np.float32)
    return px.mean(0), px.std(0) + 1e-6


def reinhard_match(src_lab: np.ndarray, src_mask: np.ndarray, ref_lab: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Match per-channel mean/std of ``src`` (inside src_mask) to ``ref`` (inside ref_mask)."""
    ms, ss = masked_stats(src_lab, src_mask)
    mr, sr = masked_stats(ref_lab, ref_mask)
    return (src_lab - ms) / ss * sr + mr


def luminance(img: np.ndarray) -> np.ndarray:
    return to_lab(img)[..., 0]
