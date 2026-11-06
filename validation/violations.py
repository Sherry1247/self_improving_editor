"""Controlled physical-violation suite (innovation C), version 4.

Real photos are physically valid. Every variant -- the positive included -- goes through the SAME pipeline (subject cut out,
background plate inpainted, subject pasted back); the positive ("null") pastes the subject back exactly where it was, each negative
changes exactly ONE thing, so a critic that separates null from a negative is measuring that violation and not the
inpainting / pasting artefacts. (The first E1 run compared against the untouched photo, which let any artefact detector look perfect.)

History of the confounds this file handles (docs/results/2026-11-05-reward-baselines-violation-suite.md, 2026-11-06-*.md):
  v2  null = same pipeline.                         -> moving / shrinking the subject uncovers inpainted pixels, a clean paste hides them.
  v3  one plate inpainted over the union of the masks, shift_side benign controls (same displacement, ground strip carried along).
      -> the union reached below the feet, so null's ground under the feet was smeared while shift_side's carried strip was real:
         signed-gap features scored shifts ABOVE null (benign AUROC 0.19-0.21).
  v4  (this file)
      * the ground strip under / beside the feet is PROTECTED from inpainting in every variant (real ground below the feet
        everywhere; the sunk / buried subject is pasted over it). Only shift_side* use a plate whose old strip is inpainted, because
        the strip (with its contact shadow) travels with the subject there.
      * ``buried``: the subject is moved down 10% and everything below the original ground line (a wavy line, 1% amplitude) is
        replaced by the real ground, i.e. the feet are hidden by the ground. ``sink`` (whole subject lower, nothing hidden) stays,
        but it is ambiguous with 'standing a bit nearer to the camera' (no depth feature changes), so do not read it as a contact violation.
      * ``no_shadow``: the dark low-frequency component of the strip is lifted towards the brightness of the ground just beyond it
        (texture kept, NO inpainting), instead of inpainting the strip. A cast shadow beside the subject is not touched.
      * ``smear_ctrl``: the subject pasted with an eroded mask that exposes as many pixels as scale_down does -- the same inpainted
        area, no scale change -- the matched control for scale_down (scale_up only covers, so it has less smear than null).
  ``params['smear_visible']`` = inpainted-but-visible area / subject area of that variant: equal for null / float / sink / buried-ish,
  different for shift_side* (their old strip is inpainted) and scale_*; report it next to the AUROC.
Known residuals: a cast shadow beside the subject is not carried along in shift_side*; floating leaves the real contact shadow on the
ground (intended: that is what a floating object looks like).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.color import to_lab
from src.utils.geometry import dilate, erode, fill_holes, mask_box


@dataclass
class Violation:
    kind: str
    image: np.ndarray
    mask: np.ndarray  # where the subject now is (visible part)
    params: dict


def clean_plate_hole(img: np.ndarray, mask: np.ndarray, extra: np.ndarray | None = None, protect: np.ndarray | None = None,
                     work: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Telea-inpaint the subject (+ ``extra``), but never the ``protect`` region. Returns (plate, hole)."""
    h, w = img.shape[:2]
    hole = dilate(mask, max(2, int(0.01 * max(h, w))))
    if extra is not None:
        hole = hole | extra
    if protect is not None:
        hole = hole & ~protect
    hole = hole | mask
    s = work / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    m = cv2.resize(hole.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    filled = cv2.resize(cv2.inpaint(small, m, 7, cv2.INPAINT_TELEA), (w, h), interpolation=cv2.INTER_CUBIC)
    out = img.copy()
    out[hole] = filled[hole]
    return out, hole


def clean_plate(img: np.ndarray, mask: np.ndarray, extra: np.ndarray | None = None, work: int = 512) -> np.ndarray:
    """Remove the subject (and optionally an extra region such as its shadow) by Telea inpainting."""
    return clean_plate_hole(img, mask, extra, None, work)[0]


def _warp(src: np.ndarray, mask: np.ndarray, dx: int = 0, dy: int = 0, scale: float = 1.0, anchor: str = "bottom"):
    box = mask_box(mask)
    ax = (box[0] + box[2]) / 2
    ay = box[3] if anchor == "bottom" else (box[1] + box[3]) / 2
    m = np.array([[scale, 0, ax - scale * ax + dx], [0, scale, ay - scale * ay + dy]], np.float32)
    h, w = mask.shape
    return (cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR),
            cv2.warpAffine(mask.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST) > 0)


def _blend(plate: np.ndarray, src_w: np.ndarray, mask_w: np.ndarray) -> np.ndarray:
    alpha = cv2.GaussianBlur(mask_w.astype(np.float32), (3, 3), 0)[..., None]
    return (alpha * src_w + (1 - alpha) * plate).astype(np.uint8)


def paste(plate: np.ndarray, src: np.ndarray, mask: np.ndarray, dx: int = 0, dy: int = 0, scale: float = 1.0,
          anchor: str = "bottom") -> tuple[np.ndarray, np.ndarray]:
    """Paste the masked subject from ``src`` onto ``plate`` with a shift / scale (scale anchored at the feet)."""
    src_w, mask_w = _warp(src, mask, dx, dy, scale, anchor)
    return _blend(plate, src_w, mask_w), mask_w


def contact_band(mask: np.ndarray, h_frac: float = 0.12) -> np.ndarray:
    box = mask_box(mask)
    sh = box[3] - box[1]
    band = np.zeros_like(mask)
    pad = 0.25 * (box[2] - box[0])
    band[int(box[3] - 0.05 * sh):int(box[3] + h_frac * sh), int(max(0, box[0] - pad)):int(box[2] + pad)] = True
    return band & ~mask


def _moved(mask: np.ndarray, dx: int = 0, dy: int = 0, scale: float = 1.0) -> np.ndarray:
    """The subject mask after paste(...) with the same shift / scale."""
    return _warp(np.zeros(mask.shape + (3,), np.uint8), mask, dx, dy, scale)[1]


def side_shift(mask: np.ndarray, frac: float) -> int:
    """Horizontal shift of ``frac`` * subject height, towards the side of the image with more room (keeps it in frame)."""
    box = mask_box(mask)
    d = max(1, int(frac * (box[3] - box[1])))
    room_right, room_left = mask.shape[1] - box[2], box[0]
    return d if room_right >= room_left else -d


def lift_shadow(img: np.ndarray, strip: np.ndarray, ref_region: np.ndarray, max_lift: float = 25.0) -> np.ndarray:
    """Remove the dark low-frequency component of ``strip`` without touching its texture or any pixel outside it:
    L += clip(L_ref - blur(L), 0, max_lift), L_ref = 60th percentile of L in ``ref_region`` (ground just beyond the strip)."""
    h, w = strip.shape
    sig = max(2.0, 0.02 * max(h, w))
    lab = to_lab(img).astype(np.float32)
    L = lab[..., 0].copy()
    ref = float(np.percentile(L[ref_region], 60)) if ref_region.sum() >= 30 else float(np.percentile(L[strip], 75))
    lift = np.clip(ref - cv2.GaussianBlur(L, (0, 0), sig), 0, max_lift)
    wgt = cv2.GaussianBlur(strip.astype(np.float32), (0, 0), sig / 2)
    lab[..., 0] = L + lift * wgt
    rgb = (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255).astype(np.uint8)
    out = img.copy()
    sel = wgt > 0.01
    out[sel] = rgb[sel]
    return out


def make_violations(img: np.ndarray, mask: np.ndarray, foreign_plate: np.ndarray | None = None) -> list[Violation]:
    mask = fill_holes(mask)  # pin-holes would let the inpainted plate show through the subject
    h, w = mask.shape
    box = mask_box(mask)
    sh = box[3] - box[1]
    subj_area = max(1, int(mask.sum()))
    dy_f, dy_s, dy_b = -int(0.10 * sh), int(0.06 * sh), int(0.10 * sh)
    dx_f, dx_s = side_shift(mask, 0.10), side_shift(mask, 0.06)
    strip = contact_band(mask, 0.15)  # real ground under / beside the feet; deeper than the sunk / buried subject
    sd_mask = _moved(mask, scale=0.7)
    exposed = int((mask & ~sd_mask).sum())  # what scale_down uncovers; smear_ctrl is built to uncover the same amount

    union = mask | _moved(mask, dy=dy_f) | _moved(mask, dy=dy_s) | sd_mask | _moved(mask, dx=dx_f) | _moved(mask, dx=dx_s)
    plate, hole = clean_plate_hole(img, mask, extra=union, protect=strip)  # shared by every variant except shift_side*
    ys, xs = np.nonzero(strip)
    ref = np.zeros_like(strip)
    ref[int(ys.max()) + 1:int(ys.max()) + 1 + max(3, int(0.10 * sh)), xs.min():xs.max() + 1] = True
    ref[:h] &= ~mask
    plate_ns = lift_shadow(plate, strip, ref)
    plate_sh, hole_sh = clean_plate_hole(img, mask, extra=union | strip)  # old strip inpainted: it travels with the subject
    out: list[Violation] = []

    def smear(hole_: np.ndarray, vis: np.ndarray) -> float:
        return round(float((hole_ & ~vis).sum()) / subj_area, 3)

    def add(kind, res, smear_=None, **params):
        if smear_ is not None:
            params["smear_visible"] = smear_
        out.append(Violation(kind, res[0], res[1], params))

    def shifted(dx: int):
        """Subject + the real strip under its feet (ground and contact shadow) moved sideways: still standing on the ground."""
        src_w, m_w = _warp(img, mask | strip, dx=dx)
        img_c = _blend(plate_sh, src_w, m_w)
        res = paste(img_c, img, mask, dx=dx)
        return res, smear(hole_sh, m_w | res[1])

    r = paste(plate, img, mask)
    add("null", r, smear(hole, r[1]))  # positive: same pipeline, no violation
    for kind, kw in (("float", {"dy": dy_f}), ("sink", {"dy": dy_s}), ("scale_up", {"scale": 1.35}),
                     ("scale_down", {"scale": 0.7})):
        r = paste(plate, img, mask, **kw)
        add(kind, r, smear(hole, r[1]), **({"dy": kw["dy"] / sh} if "dy" in kw else kw))
    r = paste(plate_ns, img, mask)
    add("no_shadow", r, smear(hole, r[1]))

    # buried: moved down, everything below the original ground line (wavy) hidden by the real ground
    src_w, m_w = _warp(img, mask, dy=dy_b)
    cut_row = box[3] + 0.01 * sh * np.sin(2 * np.pi * 2 * (np.arange(w)[None, :] - box[0]) / max(1.0, box[2] - box[0]))
    m_cut = m_w & (np.arange(h)[:, None] < cut_row)
    add("buried", (_blend(plate, src_w, m_cut), m_cut), smear(hole, m_cut), dy=dy_b / sh)

    # smear_ctrl: same exposed (inpainted) area as scale_down, subject not scaled -- eroded rim instead
    rad = 1
    while rad < 80 and int((mask & ~erode(mask, rad)).sum()) < exposed:
        rad += 1
    m_er = erode(mask, rad)
    if m_er.any():
        r = paste(plate, img, m_er)
        add("smear_ctrl", r, smear(hole, r[1]), erode=rad)

    res, sm = shifted(dx_f)
    add("shift_side", res, sm, dx=dx_f / sh)   # benign control for float
    res, sm = shifted(dx_s)
    add("shift_side_s", res, sm, dx=dx_s / sh)  # benign control for sink

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
