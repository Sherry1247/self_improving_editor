"""World branch — covariants (innovation A, 'must change together with the background').

All four critics are image-level physical-consistency heuristics, NOT physics simulation.
Where possible each one is *relative to the before image*: the original photo is physically
valid, so it calibrates what "normal" looks like for this subject / camera / depth model.
VLM region-QA versions of the same checks are added in the next milestone.
"""

from __future__ import annotations

import numpy as np

from src.critics.base import Critic, CriticContext
from src.types import CriticResult, IssueType, Regions
from src.utils.color import to_lab
from src.utils.geometry import mask_box


def _touches_bottom(mask: np.ndarray, margin: int = 3) -> bool:
    return bool(mask[-margin:].any())


def _support_gap(depth: np.ndarray, mask: np.ndarray, regions: Regions) -> float | None:
    """|depth(subject's lowest pixels) - depth(ground right below)|, both relative inverse depth in [0, 1]."""
    box = mask_box(mask)
    if box is None or not regions.contact.any():
        return None
    h, w = mask.shape
    sh = box[3] - box[1]
    band = max(2, int(0.03 * sh))
    cols = np.nonzero(mask.any(0))[0]
    bottoms = np.array([np.nonzero(mask[:, c])[0].max() for c in cols])
    low = bottoms >= np.percentile(bottoms, 85)  # columns that reach the lowest part (feet / paws)
    sub_d, gnd_d = [], []
    for c, b in zip(cols[low], bottoms[low]):
        sub_d.append(depth[max(0, b - band):b + 1, c].mean())
        below = depth[min(h - 1, b + 1):min(h, b + 1 + band), c]
        if below.size:
            gnd_d.append(below.mean())
    if not gnd_d:
        return None
    return float(abs(np.mean(sub_d) - np.mean(gnd_d)))


class SupportCritic(Critic):
    """Is the subject resting on something? Depth continuity between the lowest subject pixels and the ground below."""

    name, branch = "support", "world"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        a = ctx.after
        if a.depth is None:
            return self.not_applicable("depth disabled")
        if _touches_bottom(a.subject_mask):
            return self.not_applicable("subject is cut off by the bottom edge — contact not visible")
        gap_a = _support_gap(a.depth, a.subject_mask, ctx.regions)
        if gap_a is None:
            return self.not_applicable("no ground visible below the subject")
        gap_b = None
        if ctx.before.depth is not None and ctx.regions_before is not None and not _touches_bottom(ctx.before.subject_mask):
            gap_b = _support_gap(ctx.before.depth, ctx.before.subject_mask, ctx.regions_before)
        excess = max(0.0, gap_a - (gap_b or 0.0))
        score = float(np.exp(-excess / self.p("sigma", 0.08)))
        issues = [IssueType.FLOATING] if score < self.p("issue_below", 0.5) else []
        return self.result(score, issues, gap_after=gap_a, gap_before=gap_b, excess=excess)


def _shadow_darkness(img: np.ndarray, regions: Regions) -> float | None:
    """1 - L(near band under the feet) / L(rest of the contact band)."""
    c = regions.contact
    box = mask_box(regions.subject)
    if c.sum() < 30 or box is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    sh = y2 - y1
    near = np.zeros_like(c)
    near[max(0, y2 - int(0.04 * sh)): y2 + max(2, int(0.07 * sh)), x1:x2] = True
    near &= c
    far = c & ~near
    if near.sum() < 10 or far.sum() < 10:
        return None
    L = to_lab(img)[..., 0]
    return float(1.0 - L[near].mean() / (L[far].mean() + 1e-6))


class ContactShadowCritic(Critic):
    """Is the ground right under the subject darker than its surroundings, about as much as in the real photo?"""

    name, branch = "contact_shadow", "world"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        if _touches_bottom(ctx.after.subject_mask):
            return self.not_applicable("subject is cut off by the bottom edge")
        dark_a = _shadow_darkness(ctx.after.image, ctx.regions)
        if dark_a is None:
            return self.not_applicable("contact band too small")
        dark_b = None
        if ctx.regions_before is not None and not ctx.spec.submerged:
            dark_b = _shadow_darkness(ctx.before.image, ctx.regions_before)
        required = float(np.clip(dark_b if dark_b is not None else 0.08, 0.03, 0.15))
        score = float(np.clip(dark_a / required, 0.0, 1.0))
        issues = [IssueType.MISSING_SHADOW] if score < self.p("issue_below", 0.4) else []
        return self.result(score, issues, darkness_after=dark_a, darkness_before=dark_b, required=required)


def _highlight_ab(lab: np.ndarray, mask: np.ndarray, q: float = 95) -> np.ndarray | None:
    """Mean (a, b) of the brightest pixels in a region — a max-RGB style estimate of the illuminant colour."""
    if mask.sum() < 50:
        return None
    L = lab[..., 0][mask]
    sel = L >= np.percentile(L, q)
    return lab[..., 1:][mask][sel].mean(0)


def _illum_mismatch(img: np.ndarray, subject: np.ndarray, background: np.ndarray) -> float | None:
    lab = to_lab(img)
    s, g = _highlight_ab(lab, subject), _highlight_ab(lab, background)
    if s is None or g is None:
        return None
    return float(np.linalg.norm(s - g))


class LightHarmonyCritic(Critic):
    """Do subject highlights share the scene illuminant colour? Excess mismatch vs the real photo = pasted look."""

    name, branch = "light_harmony", "world"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        m_a = _illum_mismatch(ctx.after.image, ctx.after.subject_mask, ctx.regions.background)
        if m_a is None:
            return self.not_applicable("regions too small")
        m_b = None
        if ctx.regions_before is not None:
            m_b = _illum_mismatch(ctx.before.image, ctx.before.subject_mask, ctx.regions_before.background)
        excess = max(0.0, m_a - (m_b or 0.0))
        score = float(np.exp(-excess / self.p("sigma", 18.0)))
        issues = [IssueType.LIGHT_MISMATCH] if score < self.p("issue_below", 0.5) else []
        return self.result(score, issues, mismatch_after=m_a, mismatch_before=m_b, excess=excess)


class HaloCritic(Critic):
    """Old-background halo: the ring around the subject stayed like the old scene while the far background changed."""

    name, branch = "halo", "world"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        ring, far = ctx.regions.boundary_outer & ~ctx.before.subject_mask, ctx.regions.background
        if ring.sum() < 30 or far.sum() < 30:
            return self.not_applicable("regions too small")
        lab_a, lab_b = to_lab(ctx.after.image), to_lab(ctx.before.image)
        d = np.linalg.norm(lab_a - lab_b, axis=-1)
        d_ring, d_far = float(d[ring].mean()), float(d[far].mean())
        if d_far < 8.0:
            return self.not_applicable("far background barely changed (gate handles this)")
        ratio = d_ring / d_far
        score = float(np.clip(ratio / 0.6, 0.0, 1.0))
        issues = [IssueType.HALO] if score < self.p("issue_below", 0.5) else []
        return self.result(score, issues, ring_change=d_ring, far_change=d_far, ratio=ratio)
