"""Follow branch — the requested change happened in the background region G."""

from __future__ import annotations

from src.critics.base import Critic, CriticContext
from src.types import CriticResult, IssueType


class BackgroundSemanticCritic(Critic):
    """SigLIP zero-shot classification of the subject-removed image over all known backgrounds."""

    name, branch = "bg_semantic", "follow"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        probs = ctx.after.bg_probs
        if not probs:
            return self.not_applicable("no background probabilities")
        tgt, src = ctx.spec.target_bg.key, ctx.spec.source_bg.key
        top = max(probs, key=probs.get)
        p_t = probs.get(tgt, 0.0)
        # score: target prob relative to the best competitor (1.0 when target is top with a margin)
        competitor = max(v for k, v in probs.items() if k != tgt)
        score = p_t / (p_t + competitor + 1e-9)
        issues = []
        catastrophic = False
        if top == src:
            issues.append(IssueType.BG_UNCHANGED)
            # still recognised as the SOURCE scene: a restyle (e.g. green tint for "forest"), not a replacement.
            # Pixel-feature change alone cannot catch this, because heavy recolouring moves DINOv2 features too.
            # (not when the target scene may legitimately contain the source concepts, e.g. river -> forest)
            catastrophic = bool(self.p("catastrophic_if_source", True)) and not ctx.spec.old_bg_may_reappear
        elif top != tgt or score < self.p("issue_below", 0.5):
            issues.append(IssueType.BG_WRONG)
        return self.result(score, issues, catastrophic, p_target=p_t, top=top, p_top=probs[top],
                           p_source=probs.get(src, 0.0))


class OldBackgroundResidueCritic(Critic):
    """Fraction of the far background still covered by source-scene concepts (e.g. water), relative to before."""

    name, branch = "old_bg_residue", "follow"

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        if ctx.spec.old_bg_may_reappear:
            return self.not_applicable(f"{ctx.spec.target_bg.key} may legitimately contain {ctx.spec.source_bg.concepts}")
        g = ctx.regions.background
        if g.sum() == 0:
            return self.not_applicable("empty background region")
        frac_after = float((ctx.after.old_bg_mask & g).sum() / g.sum())
        frac_before = float((ctx.before.old_bg_mask & g).sum() / g.sum())
        if frac_before < 0.02:
            return self.not_applicable("source concepts not detected in the before image")
        ratio = frac_after / frac_before
        issues = [IssueType.OLD_BG_RESIDUE] if ratio > self.p("issue_above", 0.3) else []
        return self.result(1.0 - ratio, issues, frac_before=frac_before, frac_after=frac_after, ratio=ratio)
