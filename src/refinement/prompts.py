"""Prompt = base instruction + a small SET of corrective clauses.

The legacy refiner appended a sentence every iteration, so prompts grew without bound and
CLIP's 77-token window silently truncated the newest (most relevant) feedback. Here the clause
set is capped; adding a new clause evicts the oldest.
"""

from __future__ import annotations

from src.spec import EditSpec

CLAUSES: dict[str, str] = {
    "preserve": "keep the {subject} identical in identity, pose and colors, and do not add any other {subject}",
    "target": "the background must clearly be {scene}",
    "remove_old": "remove every trace of the {old}",
    "contact": "the {subject} is {pose} on the {support} with a soft contact shadow underneath",
    "light": "the lighting and color tone on the {subject} match the {target} scene",
    "edges": "clean natural edges around the {subject}",
    "natural": "the {subject} is naturally part of the scene, with nothing attached to or wrapped around its body, "
               "standing on a realistic {support}",
}

MAX_CLAUSES = 3


def fill(clause_key: str, spec: EditSpec) -> str:
    return CLAUSES[clause_key].format(subject=spec.subject, scene=spec.target_bg.scene, target=spec.target_bg.name,
                                      old=" and ".join(spec.source_bg.concepts), pose=spec.pose,
                                      support=spec.target_bg.support)


def add_clause(clauses: tuple[str, ...], key: str) -> tuple[str, ...]:
    if key in clauses:
        return clauses
    out = (*clauses, key)
    return out[-MAX_CLAUSES:]


def build_prompt(spec: EditSpec, style: str, clauses: tuple[str, ...] = ()) -> str:
    base = spec.instruction() if style == "instruction" else spec.scene_prompt()
    extra = [fill(k, spec) for k in clauses]
    return ", ".join([base, *extra])
