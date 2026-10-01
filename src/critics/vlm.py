"""VLM region-QA critics. They read the P(yes) answers computed in perception (src/perception/vlm.py).

Physical questions are scored RELATIVE to the before image: the real photo is physically valid, so if the
VLM is unsure about it (e.g. no visible shadow for a dog standing in water), the requirement is relaxed.
    score = min(1, p_after / max(p_before, floor))
The background question is absolute (the before image is not supposed to show the target scene).
"""

from __future__ import annotations

from src.critics.base import Critic, CriticContext
from src.types import CriticResult, IssueType


class VLMQACritic(Critic):
    key: str = ""
    issue: IssueType = IssueType.IMPLAUSIBLE
    relative: bool = True

    def evaluate(self, ctx: CriticContext) -> CriticResult:
        va, vb = ctx.after.vlm or {}, ctx.before.vlm or {}
        if self.key not in va:
            return self.not_applicable("VLM answers not available (perception.use_vlm is off)")
        pa = va[self.key]
        if self.relative and self.key in vb:
            ref = max(vb[self.key], self.p("floor", 0.5))
            score = min(1.0, pa / ref)
        else:
            ref, score = None, pa
        issues = [self.issue] if score < self.p("issue_below", 0.5) else []
        return self.result(score, issues, p_after=pa, p_before=vb.get(self.key), reference=ref)


def _make(name: str, key: str, branch: str, issue: IssueType, relative: bool = True) -> type[VLMQACritic]:
    return type(f"VLM_{key}", (VLMQACritic,), {"name": name, "key": key, "branch": branch, "issue": issue,
                                               "relative": relative})


VLM_CRITICS = [
    _make("vlm_support", "support", "world", IssueType.FLOATING),
    _make("vlm_shadow", "shadow", "world", IssueType.MISSING_SHADOW),
    _make("vlm_surface", "surface", "world", IssueType.IMPLAUSIBLE),
    _make("vlm_integration", "integration", "world", IssueType.IMPLAUSIBLE),
    _make("vlm_lighting", "lighting", "world", IssueType.LIGHT_MISMATCH),
    _make("vlm_background", "background", "follow", IssueType.BG_WRONG, relative=False),
]
