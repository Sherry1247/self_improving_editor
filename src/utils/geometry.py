"""Single home for box / mask geometry (legacy code had bbox_iou implemented 3x)."""

from __future__ import annotations

import cv2
import numpy as np

from src.types import BBox


def box_iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def nms(boxes: list[BBox], scores: list[float], iou: float) -> list[int]:
    order = list(np.argsort(scores)[::-1])
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        order = [j for j in order if box_iou(boxes[i], boxes[j]) < iou]
    return keep


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def mask_box(mask: np.ndarray) -> BBox | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _kernel(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), _kernel(r)).astype(bool)


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    return cv2.erode(mask.astype(np.uint8), _kernel(r)).astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask.astype(bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == idx


def mask_moments(mask: np.ndarray) -> tuple[float, float, float] | None:
    """(cx, cy, sqrt(area))."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean()), float(np.sqrt(len(xs)))


def similarity_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Warp ``src`` mask onto ``dst`` with translation + isotropic scale estimated from mask moments."""
    ms, md = mask_moments(src), mask_moments(dst)
    if ms is None or md is None:
        return src.copy(), {"dx": 0.0, "dy": 0.0, "scale": 1.0}
    s = md[2] / ms[2]
    m = np.array([[s, 0, md[0] - s * ms[0]], [0, s, md[1] - s * ms[1]]], dtype=np.float32)
    h, w = dst.shape
    warped = cv2.warpAffine(src.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST) > 0
    return warped, {"dx": md[0] - ms[0], "dy": md[1] - ms[1], "scale": float(s)}


def warp_image(img: np.ndarray, transform: dict[str, float], src_mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Apply the same similarity transform used by :func:`similarity_align` to an image."""
    ms = mask_moments(src_mask)
    if ms is None:
        return cv2.resize(img, (out_hw[1], out_hw[0]))
    s = transform["scale"]
    cx_d, cy_d = ms[0] + transform["dx"], ms[1] + transform["dy"]
    m = np.array([[s, 0, cx_d - s * ms[0]], [0, s, cy_d - s * ms[1]]], dtype=np.float32)
    return cv2.warpAffine(img, m, (out_hw[1], out_hw[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
