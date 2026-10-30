"""Keep branch — invariants inside the subject region S (innovation A, 'must not change').

Appearance is measured AFTER removing global illumination / colour-cast differences, because in a
background swap the subject's lighting SHOULD change. Only texture / structure changes are penalised.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.critics.base import Critic, CriticContext, ramp
from src.critics.gate import grid_mask
from src.types import CriticResult, IssueType
from src.utils.color import reinhard_match, to_lab
from src.utils.geometry import erode, mask_iou, warp_image


def revealed_allowed(ctx: CriticContext) -> np.ndarray:
    """After-subject pixels that were hidden by the old scene in the before image and may legitimately appear.

    Example: a dog standing in a river has its legs under water; after moving it to rocky ground the legs
    must be regenerated (a covariant change). Growth is allowed only into the before image's old-background
    region (water) BELOW the subject's centre, never sideways or above.
    """
    b, a = ctx.regions.aligned_before_subject, ctx.after.subject_mask
    excess = a & ~b
    if not excess.any() or not b.any():
        return np.zeros_like(a)
    ys = np.nonzero(b)[0]
    below = np.zeros_like(a)
    below[int(ys.mean()):] = True
    return excess & ctx.before.old_bg_mask & below


class SilhouetteCritic(Critic):
    """IoU of aligned before/after subject masks, not penalising parts revealed from behind the old scene."""

    name, branch = "silhouette", "keep"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        if not ctx.before.subject_mask.any() or not ctx.after.subject_mask.any():
            return self.not_applicable("missing subject mask")
        b, a = ctx.regions.aligned_before_subject, ctx.after.subject_mask
        allowed = revealed_allowed(ctx)
        inter = (a & b).sum()
        union = (b | (a & ~allowed)).sum()
        score = float(inter / union) if union else 0.0
        issues = [IssueType.SUBJECT_DRIFT] if score < self.p("issue_below", 0.75) else []
        return self.result(score, issues, iou=score, raw_iou=mask_iou(b, a), revealed_px=int(allowed.sum()),
                           transform=ctx.regions.transform)


def _masked_blur(x: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that only averages pixels inside ``mask`` (no bleeding from outside the subject)."""
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(x.astype(np.float32) * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def structure_zncc(l_before: np.ndarray, l_after: np.ndarray, region: np.ndarray, sigma: float) -> float:
    """Zero-normalised correlation of the HIGH-PASS luminance of two images inside ``region``.

    High-pass removes the slowly varying illumination (side light, gradients, global gain); the
    correlation is invariant to local gain and offset, so only fine structure (texture, edges, pattern)
    is compared. Identical structure -> 1, destroyed/changed structure -> ~0.
    """
    hb = l_before - _masked_blur(l_before, region, sigma)
    ha = l_after - _masked_blur(l_after, region, sigma)
    xb, xa = hb[region], ha[region]
    xb, xa = xb - xb.mean(), xa - xa.mean()
    nb, na = float(np.sqrt((xb ** 2).sum())), float(np.sqrt((xa ** 2).sum()))
    eps = 1e-3 * np.sqrt(len(xb))
    if nb < eps and na < eps:
        return 1.0  # both flat: nothing to compare
    if nb < eps or na < eps:
        return 0.0  # structure appeared or vanished
    return float((xb * xa).sum() / (nb * na))


def zncc_by_scale(l_before: np.ndarray, l_after: np.ndarray, region: np.ndarray, sigma: float,
                  scales: tuple[int, ...] = (1, 2, 4)) -> dict[int, float]:
    """High-pass ZNCC at each downsampling factor (1 = full resolution)."""
    out = {}
    for k in scales:
        if k == 1:
            out[1] = structure_zncc(l_before, l_after, region, sigma)
            continue
        h, w = region.shape
        size = (max(8, w // k), max(8, h // k))
        lb = cv2.resize(l_before.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        la = cv2.resize(l_after.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        rg = cv2.resize(region.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.5
        if rg.sum() >= 20:
            out[k] = structure_zncc(lb, la, rg, max(1.5, sigma / k))
    return out


def multiscale_zncc(l_before: np.ndarray, l_after: np.ndarray, region: np.ndarray, sigma: float,
                    scales: tuple[int, ...] = (1, 2)) -> float:
    """Mean high-pass ZNCC after downsampling by each factor in ``scales`` (1 = full resolution).

    A generative editor re-encodes the whole image (VAE, sampling noise), which changes the finest pixel detail of a
    subject that is semantically identical. Correlating only coarser bands (>= 2 px) ignores that nuisance while a
    real change of pattern, texture or pose still lowers the score.
    """
    zs = list(zncc_by_scale(l_before, l_after, region, sigma, scales).values())
    return float(np.mean(zs)) if zs else structure_zncc(l_before, l_after, region, sigma)


def refine_zncc(l_before: np.ndarray, l_after: np.ndarray, region: np.ndarray, sigma: float, max_shift: float,
                zncc0: float, scales: tuple[int, ...] = (1,)) -> tuple[float, tuple[float, float]]:
    """Best high-pass ZNCC after a small extra translation (phase correlation, |shift| <= max_shift px).

    The similarity alignment comes from mask moments and is a few pixels off for non-rigid subjects (a dog that
    lowers its head); without this, a perfectly preserved texture can score ~0 just because it is misregistered.
    Returns (zncc, (dx, dy)); (zncc0, (0, 0)) when no shift helps.
    """
    if max_shift < 1.0:
        return zncc0, (0.0, 0.0)
    hb = (l_before - _masked_blur(l_before, region, sigma)) * region
    ha = (l_after - _masked_blur(l_after, region, sigma)) * region
    (dx, dy), _ = cv2.phaseCorrelate(hb.astype(np.float32), ha.astype(np.float32))
    best, best_shift = zncc0, (0.0, 0.0)
    h, w = region.shape
    for sx, sy in ((dx, dy), (-dx, -dy)):
        sx, sy = float(np.clip(sx, -max_shift, max_shift)), float(np.clip(sy, -max_shift, max_shift))
        if abs(sx) < 0.5 and abs(sy) < 0.5:
            continue
        m = np.float32([[1, 0, sx], [0, 1, sy]])
        la = cv2.warpAffine(l_after.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ra = cv2.warpAffine(region.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
        reg = region & ra
        if reg.sum() < 50:
            continue
        z = multiscale_zncc(l_before, la, reg, sigma, scales)
        if z > best:
            best, best_shift = z, (sx, sy)
    return float(best), best_shift


class AppearanceCritic(Critic):
    """Subject appearance preserved? (identity / texture / structure — NOT lighting).

    ``variant="v3"`` (default) compares only illumination-invariant structure: zero-normalised correlation of
    the high-pass luminance (so side light, shading gradients and global gain are ignored) combined with a
    low-weight chroma term after removing the global colour cast. ``variant="v2"`` is the original Lab
    distance after global Reinhard matching, kept for ablations: it still counts local re-lighting
    (side light, new shadow side) as drift and therefore prefers copy-pasted subjects.
    """

    name, branch = "appearance", "keep"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        r = ctx.regions
        if not r.transform or not ctx.after.subject_mask.any():
            return self.not_applicable("missing subject mask")
        h, w = ctx.after.shape
        before_warp = warp_image(ctx.before.image, r.transform, ctx.before.subject_mask, (h, w))
        region = erode(r.aligned_before_subject & ctx.after.subject_mask, 2)
        if region.sum() < 50:
            return self.result(0.0, [IssueType.SUBJECT_DRIFT], overlap_px=int(region.sum()))
        lab_b, lab_a = to_lab(before_warp), to_lab(ctx.after.image)
        lab_a_norm = reinhard_match(lab_a, region, lab_b, region)  # removes global lighting / cast change
        diff = np.abs(lab_a_norm - lab_b)[region]
        d = float((diff * np.array([0.5, 1.0, 1.0])).sum(-1).mean())
        raw = float(np.abs(lab_a - lab_b)[region].sum(-1).mean())
        v2_score = float(np.exp(-d / self.p("sigma", 20.0)))
        if self.p("variant", "v3") == "v2":
            score = v2_score
            issues = [IssueType.SUBJECT_DRIFT] if score < self.p("issue_below", 0.4) else []
            return self.result(score, issues, normalized_dist=d, raw_dist=raw, variant="v2")

        # --- alignment quality: appearance is only meaningful if the two subjects were actually registered
        align_iou = mask_iou(r.aligned_before_subject, ctx.after.subject_mask)
        sigma = max(2.0, self.p("highpass_frac", 0.05) * float(np.sqrt(region.sum())))
        scales = tuple(self.p("scales", (1, 2)))
        by_scale = zncc_by_scale(lab_b[..., 0], lab_a[..., 0], region, sigma)  # diagnostics: pick scales offline
        zncc_fine = by_scale.get(1)
        zncc_raw = multiscale_zncc(lab_b[..., 0], lab_a[..., 0], region, sigma, scales)
        zncc, shift = refine_zncc(lab_b[..., 0], lab_a[..., 0], region, sigma,
                                  self.p("max_shift_frac", 0.08) * float(np.sqrt(region.sum())), zncc_raw, scales)
        if align_iou < self.p("unreliable_iou", 0.5) and self.p("unreliable_policy", "na") == "na":
            # The silhouette changed so much that a pixel comparison says nothing about texture. Do NOT score
            # this as 'zero appearance': flag it, the silhouette / identity / texture critics judge the change.
            return self.result(None, [], variant="v3", unreliable_alignment=True, align_iou=float(align_iou),
                               zncc=zncc, zncc_raw=zncc_raw, shift=shift)
        structure = ramp(zncc, self.p("zncc_lo", 0.2), self.p("zncc_hi", 0.8))
        # chroma: remove only the global mean shift (a cast is lighting), keep the spatial colour pattern
        ab_b, ab_a = lab_b[..., 1:], lab_a[..., 1:]
        cshift = ab_a[region].mean(0) - ab_b[region].mean(0)
        chroma_d = float(np.linalg.norm((ab_a - cshift - ab_b)[region], axis=-1).mean())
        chroma = float(np.exp(-chroma_d / self.p("chroma_sigma", 30.0)))
        score = structure * chroma
        issues = [IssueType.SUBJECT_DRIFT] if score < self.p("issue_below", 0.4) else []
        return self.result(score, issues, variant="v3", zncc=zncc, zncc_raw=zncc_raw, zncc_fine=zncc_fine, zncc_by_scale=by_scale, shift=shift,
                           align_iou=float(align_iou), unreliable_alignment=False, structure=structure,
                           chroma_dist=chroma_d, chroma=chroma, normalized_dist=d, raw_dist=raw, v2_score=v2_score)


class TextureCritic(Critic):
    """Mean DINOv2 patch cosine over subject cells (before grid shifted by the alignment translation)."""

    name, branch = "texture", "keep"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        pb, pa = ctx.before.patch_features, ctx.after.patch_features
        r = ctx.regions
        if pb is None or pa is None or pb.shape != pa.shape or not r.transform:
            return self.not_applicable("patch features missing")
        gh, gw = pa.shape[:2]
        h, w = ctx.after.shape
        sx, sy = int(round(r.transform["dx"] * gw / w)), int(round(r.transform["dy"] * gh / h))
        pb_shift = np.roll(pb, (sy, sx), axis=(0, 1))
        cells = grid_mask(r.aligned_before_subject & ctx.after.subject_mask, (gh, gw), 0.7)
        if cells.sum() < 2:
            return self.not_applicable("subject too small for the patch grid")
        cos = float((pb_shift[cells] * pa[cells]).sum(-1).mean())
        score = ramp(cos, self.p("lo", 0.4), self.p("hi", 0.85))
        issues = [IssueType.SUBJECT_DRIFT] if score < self.p("issue_below", 0.6) else []
        return self.result(score, issues, patch_cosine=cos, cells=int(cells.sum()))
