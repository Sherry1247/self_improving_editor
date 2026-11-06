"""Controlled physical-violation suite (innovation C).

Real photos are physically valid. Every variant -- the positive included -- goes through the SAME
pipeline (subject cut out, background plate inpainted, subject pasted back); the positive ("null") pastes
the subject back exactly where it was, each negative changes exactly ONE thing. A critic that separates
null from a negative is therefore measuring that violation, not the inpainting / pasting artefacts.
(The first E1 run compared against the untouched photo, which let any artefact detector look perfect.)

Two confounds of the first same-pipeline suite are handled here (docs/results/2026-11-05-reward-baselines-violation-suite.md):
  * smear: moving or shrinking the subject uncovers the Telea-inpainted hole, a clean paste hides it. Now EVERY variant
    (null included) uses one plate inpainted over the union of the masks the variants occupy (scale_up excluded: it only covers), so the smeared region
    is the same in all of them; the share of it left uncovered is stored per variant (``smear_visible``) as a covariate. It is NOT equalised across variants
    (moving up uncovers more than moving sideways); analyses should stratify on it or report it next to the AUROC.
  * "any displacement is detected": ``shift_side`` / ``shift_side_s`` move the subject sideways by the same amount as
    ``float`` / ``sink`` and carry the ground strip under the feet with it, so the subject still stands on the ground.
    A critic that scores float below shift_side is reacting to lost contact, not to "something moved".
Known residual: a cast shadow that lies beside the subject is not carried along (only the strip under the feet is).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.color import to_lab
from src.utils.geometry import dilate, fill_holes, mask_box


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


def _moved(mask: np.ndarray, dx: int = 0, dy: int = 0, scale: float = 1.0) -> np.ndarray:
    """The subject mask after paste(...) with the same shift / scale."""
    return paste(np.zeros(mask.shape + (3,), np.uint8), np.zeros(mask.shape + (3,), np.uint8), mask, dx, dy, scale)[1]


def side_shift(mask: np.ndarray, frac: float) -> int:
    """Horizontal shift of ``frac`` * subject height, towards the side of the image with more room (keeps it in frame)."""
    box = mask_box(mask)
    d = max(1, int(frac * (box[3] - box[1])))
    room_right, room_left = mask.shape[1] - box[2], box[0]
    return d if room_right >= room_left else -d


def make_violations(img: np.ndarray, mask: np.ndarray, foreign_plate: np.ndarray | None = None) -> list[Violation]:
    mask = fill_holes(mask)  # pin-holes would let the inpainted plate show through the subject
    box = mask_box(mask)
    sh = box[3] - box[1]
    dy_f, dy_s = -int(0.10 * sh), int(0.06 * sh)
    dx_f, dx_s = side_shift(mask, 0.10), side_shift(mask, 0.06)
    placed = {  # where the subject stands in each variant
        "null": mask, "float": _moved(mask, dy=dy_f), "sink": _moved(mask, dy=dy_s),
        "scale_up": _moved(mask, scale=1.35), "scale_down": _moved(mask, scale=0.7),
        "shift_side": _moved(mask, dx=dx_f), "shift_side_s": _moved(mask, dx=dx_s),
    }
    union = np.zeros_like(mask)
    for k_, m_ in placed.items():
        if k_ != "scale_up":  # growing the subject hides the plate instead of exposing it; keeping it out stops the null
            union |= m_       # from carrying a huge smeared ring (its smear_visible is just lower than the null's)
    plate = clean_plate(img, mask, extra=union)  # one plate for all variants -> the same smear everywhere
    plate_noshadow = clean_plate(img, mask, extra=union | contact_band(mask))
    out: list[Violation] = []
    union_area = max(1, int(union.sum()))

    def add(kind, res, **params):
        if kind in placed:  # share of the inpainted region (union of masks) NOT covered by the subject in this variant
            params["smear_visible"] = round(float((union & ~res[1]).sum()) / union_area, 3)
        out.append(Violation(kind, res[0], res[1], params))

    def shifted(dx: int):
        """Subject + the ground strip under its feet moved sideways by dx (benign: still standing on the ground)."""
        img_c, _ = paste(plate, img, mask | contact_band(mask), dx=dx)
        return paste(img_c, img, mask, dx=dx)

    add("null", paste(plate, img, mask))  # positive: same pipeline, no violation
    add("float", paste(plate, img, mask, dy=dy_f), dy=-0.10)
    add("sink", paste(plate, img, mask, dy=dy_s), dy=0.06)
    add("scale_up", paste(plate, img, mask, scale=1.35), scale=1.35)
    add("scale_down", paste(plate, img, mask, scale=0.7), scale=0.7)
    add("no_shadow", paste(plate_noshadow, img, mask))
    add("shift_side", shifted(dx_f), dx=dx_f / sh)    # benign control for float
    add("shift_side_s", shifted(dx_s), dx=dx_s / sh)  # benign control for sink

    # colour cast on the subject only (lighting mismatch)
    lab = to_lab(img)
    lab[..., 2] += 25.0  # strong yellow cast
    cast = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    cast = (np.clip(cast, 0, 1) * 255).astype(np.uint8)
    add("color_cast", paste(plate, cast, mask), db=25)

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
