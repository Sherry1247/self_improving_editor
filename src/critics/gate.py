"""Gate critics — hard constraints (innovation D). Failing any of them zeroes the overall score.

The gate is two-sided on purpose: the subject must survive (count, identity) AND the
background must actually change. Without the second half, "return the input unchanged"
scores perfectly on every preservation metric — the classic reward hack of a closed loop.
"""

from __future__ import annotations

import numpy as np

from src.critics.base import Critic, CriticContext, ramp
from src.types import CriticResult, IssueType


class SubjectCountCritic(Critic):
    name, branch = "subject_count", "gate"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        n_before, n_after = ctx.before.subject_count, ctx.after.subject_count
        expected = max(1, n_before)
        if n_after == 0:
            return self.result(0.0, [IssueType.SUBJECT_MISSING], True, n_before=n_before, n_after=n_after)
        if n_after > expected:
            return self.result(0.0, [IssueType.EXTRA_SUBJECT], True, n_before=n_before, n_after=n_after)
        return self.result(1.0, n_before=n_before, n_after=n_after)


class IdentityCritic(Critic):
    """Cosine of DINOv2 CLS embeddings of the masked subject crops (background removed)."""

    name, branch = "identity", "gate"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        eb, ea = ctx.before.subject_embedding, ctx.after.subject_embedding
        if eb is None:
            return self.not_applicable("no subject found in the before image")
        if ea is None:
            return self.result(0.0, [IssueType.IDENTITY_LOSS], True, cosine=None)
        cos = float(np.dot(eb, ea) / (np.linalg.norm(eb) * np.linalg.norm(ea) + 1e-9))
        cat = cos < self.p("catastrophic_below", 0.45)
        return self.result(ramp(cos, self.p("lo", 0.45), self.p("hi", 0.85)),
                           [IssueType.IDENTITY_LOSS] if cat else [], cat, cosine=cos)


class BackgroundChangedCritic(Critic):
    """1 - mean DINOv2 patch cosine over the far-background cells: did the background really change?"""

    name, branch = "bg_changed", "gate"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        pb, pa = ctx.before.patch_features, ctx.after.patch_features
        if pb is None or pa is None or pb.shape != pa.shape:
            return self.not_applicable("patch features missing or shape mismatch")
        cells = grid_mask(ctx.regions.background & ~ctx.before.subject_mask, pa.shape[:2])
        if cells.sum() < 4:
            return self.not_applicable("background region too small")
        change = float(1.0 - (pb[cells] * pa[cells]).sum(-1).mean())
        cat = change < self.p("catastrophic_below", 0.15)
        return self.result(ramp(change, self.p("lo", 0.15), self.p("hi", 0.45)),
                           [IssueType.BG_UNCHANGED] if cat else [], cat, change=change, cells=int(cells.sum()))


def grid_mask(mask: np.ndarray, grid_hw: tuple[int, int], min_frac: float = 0.6) -> np.ndarray:
    """Downsample a pixel mask to a patch grid: a cell is on if >= min_frac of its pixels are on."""
    import cv2

    frac = cv2.resize(mask.astype(np.float32), (grid_hw[1], grid_hw[0]), interpolation=cv2.INTER_AREA)
    return frac >= min_frac
