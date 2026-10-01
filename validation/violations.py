"""Controlled physical-violation suite (innovation C).

Real photos are physically valid -> they are the positives. Each negative applies exactly ONE
controlled violation to the same photo, so a critic that separates them is measuring that violation
and nothing else. Used to (1) test whether critics see physics errors, (2) set reliability weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.color import to_lab
from src.utils.geometry import dilate, mask_box


@dataclass
class Violation:
    kind: str
    image: np.ndarray
    mask: np.ndarray  # where the subject now is
    params: dict


def clean_plate(img: np.ndarray, mask: np.ndarray, extra: np.ndarray | None = None, work: int = 512) -> np.ndarray:
    """Remove the subject (and optionally an extra region such as its shadow) by Telea inpainting."""
    h, w = img.shape[:2]
    hole = dilate(mask, max(2, int(0.01 * max(h, w))))
    if extra is not None:
        hole |= extra
    s = work / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    m = cv2.resize(hole.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    filled = cv2.resize(cv2.inpaint(small, m, 7, cv2.INPAINT_TELEA), (w, h), interpolation=cv2.INTER_CUBIC)
    out = img.copy()
    out[hole] = filled[hole]
    return out


def paste(plate: np.ndarray, src: np.ndarray, mask: np.ndarray, dx: int = 0, dy: int = 0, scale: float = 1.0,
          anchor: str = "bottom") -> tuple[np.ndarray, np.ndarray]:
    """Paste the masked subject from ``src`` onto ``plate`` with a shift / scale (scale anchored at the feet)."""
    box = mask_box(mask)
    ax, ay = (box[0] + box[2]) / 2, box[3] if anchor == "bottom" else (box[1] + box[3]) / 2
    m = np.array([[scale, 0, ax - scale * ax + dx], [0, scale, ay - scale * ay + dy]], np.float32)
    h, w = mask.shape
    src_w = cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR)
    mask_w = cv2.warpAffine(mask.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST) > 0
    alpha = cv2.GaussianBlur(mask_w.astype(np.float32), (3, 3), 0)[..., None]
    return (alpha * src_w + (1 - alpha) * plate).astype(np.uint8), mask_w


def contact_band(mask: np.ndarray, h_frac: float = 0.12) -> np.ndarray:
    box = mask_box(mask)
    sh = box[3] - box[1]
    band = np.zeros_like(mask)
    pad = 0.25 * (box[2] - box[0])
    band[int(box[3] - 0.05 * sh):int(box[3] + h_frac * sh), int(max(0, box[0] - pad)):int(box[2] + pad)] = True
    return band & ~mask


def make_violations(img: np.ndarray, mask: np.ndarray, foreign_plate: np.ndarray | None = None) -> list[Violation]:
    box = mask_box(mask)
    sh = box[3] - box[1]
    plate = clean_plate(img, mask)
    plate_noshadow = clean_plate(img, mask, extra=contact_band(mask))
    out: list[Violation] = []

    def add(kind, res, **params):
        out.append(Violation(kind, res[0], res[1], params))

    add("float", paste(plate, img, mask, dy=-int(0.10 * sh)), dy=-0.10)
    add("sink", paste(plate, img, mask, dy=int(0.06 * sh)), dy=0.06)
    add("scale_up", paste(plate, img, mask, scale=1.35), scale=1.35)
    add("scale_down", paste(plate, img, mask, scale=0.7), scale=0.7)
    add("no_shadow", paste(plate_noshadow, img, mask))

    # colour cast on the subject only (lighting mismatch)
    lab = to_lab(img)
    lab[..., 2] += 25.0  # strong yellow cast
    cast = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    cast = (np.clip(cast, 0, 1) * 255).astype(np.uint8)
    add("color_cast", paste(img, cast, mask), db=25)

    # halo: paste with a dilated mask (carries old-background pixels) onto a foreign background
    if foreign_plate is not None:
        fp = cv2.resize(foreign_plate, (img.shape[1], img.shape[0]))
        clean, m = paste(fp, img, mask)
        out.append(Violation("foreign_clean", clean, m, {}))  # reference for halo
        halo_mask = dilate(mask, max(3, int(0.03 * sh)))
        add("halo", paste(fp, img, halo_mask), dilate=0.03)
    return out


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """P(score_pos > score_neg); ties count half. None if either side is empty."""
    if not pos or not neg:
        return None
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    gt = (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
    return float(gt / (len(p) * len(n)))
