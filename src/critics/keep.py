"""Keep branch — invariants inside the subject region S (innovation A, 'must not change').

Appearance is measured AFTER removing global illumination / colour-cast differences, because in a
background swap the subject's lighting SHOULD change. Only texture / structure changes are penalised.
"""

from __future__ import annotations

import numpy as np

from src.critics.base import Critic, CriticContext, ramp
from src.critics.gate import grid_mask
from src.types import CriticResult, IssueType
from src.utils.color import reinhard_match, to_lab
from src.utils.geometry import erode, mask_iou, warp_image


class SilhouetteCritic(Critic):
    name, branch = "silhouette", "keep"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        if not ctx.before.subject_mask.any() or not ctx.after.subject_mask.any():
            return self.not_applicable("missing subject mask")
        iou = mask_iou(ctx.regions.aligned_before_subject, ctx.after.subject_mask)
        issues = [IssueType.SUBJECT_DRIFT] if iou < self.p("issue_below", 0.75) else []
        return self.result(iou, issues, iou=iou, transform=ctx.regions.transform)


class AppearanceCritic(Critic):
    """Illumination-normalised colour distance inside the overlap of aligned-before and after subject masks."""

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
        score = float(np.exp(-d / self.p("sigma", 12.0)))
        issues = [IssueType.SUBJECT_DRIFT] if score < self.p("issue_below", 0.6) else []
        return self.result(score, issues, normalized_dist=d, raw_dist=raw)


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
