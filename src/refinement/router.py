"""Diagnosis -> action routing (innovation E) + cross-sample action memory.

A failure type selects a small set of candidate actions; the memory picks among them with UCB
using Δscore observed on earlier tasks, so the loop gets better at repairing as the dataset is processed.
Region-local repair tools (contact-band inpainting, relighting) plug in here as additional actions later.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from src.editors.base import Editor
from src.refinement.prompts import add_clause
from src.types import EvaluationResult, IssueType


@dataclass(frozen=True)
class EditState:
    params: dict = field(default_factory=dict)
    clauses: tuple[str, ...] = ()

    def key(self) -> str:
        return json.dumps({"p": self.params, "c": self.clauses}, sort_keys=True)


@dataclass(frozen=True)
class Action:
    name: str
    kind: str  # "preserve_more" | "edit_more" | "clause" | "reseed"
    clause: str | None = None


# issue -> ordered candidate actions (the order is the prior used before any memory exists)
ISSUE_ACTIONS: dict[IssueType, list[Action]] = {
    IssueType.SUBJECT_MISSING: [Action("preserve_more", "preserve_more"), Action("clause_preserve", "clause", "preserve")],
    IssueType.EXTRA_SUBJECT: [Action("clause_preserve", "clause", "preserve"), Action("preserve_more", "preserve_more")],
    IssueType.IDENTITY_LOSS: [Action("preserve_more", "preserve_more"), Action("clause_preserve", "clause", "preserve")],
    IssueType.SUBJECT_DRIFT: [Action("preserve_more", "preserve_more"), Action("clause_preserve", "clause", "preserve")],
    IssueType.BG_UNCHANGED: [Action("edit_more", "edit_more"), Action("clause_target", "clause", "target")],
    IssueType.BG_WRONG: [Action("clause_target", "clause", "target"), Action("edit_more", "edit_more")],
    IssueType.OLD_BG_RESIDUE: [Action("clause_remove_old", "clause", "remove_old"), Action("edit_more", "edit_more")],
    IssueType.FLOATING: [Action("clause_contact", "clause", "contact")],
    IssueType.MISSING_SHADOW: [Action("clause_contact", "clause", "contact")],
    IssueType.LIGHT_MISMATCH: [Action("clause_light", "clause", "light")],
    IssueType.HALO: [Action("clause_edges", "clause", "edges"), Action("edit_more", "edit_more")],
}
GATE_ISSUES = {IssueType.SUBJECT_MISSING, IssueType.EXTRA_SUBJECT, IssueType.IDENTITY_LOSS, IssueType.BG_UNCHANGED}
BRANCH_FALLBACK = {"keep": IssueType.SUBJECT_DRIFT, "follow": IssueType.BG_WRONG, "world": IssueType.MISSING_SHADOW}
RESEED = Action("reseed", "reseed")


class ActionMemory:
    """Per (issue, editor, action): count and mean Δscore. Persisted as JSON so it spans tasks and runs."""

    def __init__(self, path: str | Path | None = None, c: float = 0.3):
        self.path = Path(path) if path else None
        self.c = c
        self.stats: dict[str, dict[str, list[float]]] = {}
        if self.path and self.path.exists():
            self.stats = json.loads(self.path.read_text())

    def _k(self, issue: IssueType, editor: str) -> str:
        return f"{issue.value}|{editor}"

    def update(self, issue: IssueType, editor: str, action: str, delta: float) -> None:
        s = self.stats.setdefault(self._k(issue, editor), {}).setdefault(action, [0, 0.0])
        s[0] += 1
        s[1] += delta
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.stats, indent=1))

    def rank(self, issue: IssueType, editor: str, actions: list[Action]) -> list[Action]:
        table = self.stats.get(self._k(issue, editor), {})
        total = sum(v[0] for v in table.values()) or 1

        def ucb(i_a):
            i, a = i_a
            n, s = table.get(a.name, [0, 0.0])
            if n == 0:
                return (1, -i)  # untried actions first, in prior order
            return (0, s / n + self.c * math.sqrt(math.log(total + 1) / n))

        return [a for _, a in sorted(enumerate(actions), key=ucb, reverse=True)]


class ActionRouter:
    def __init__(self, editor: Editor, memory: ActionMemory | None = None):
        self.editor = editor
        self.memory = memory

    def primary_issue(self, ev: EvaluationResult) -> IssueType | None:
        issues = ev.issues
        for i in issues:
            if i in GATE_ISSUES:
                return i
        if issues:
            # pick an issue from the weakest branch
            weakest = min((b for b in ("keep", "follow", "world") if b in ev.branch_scores),
                          key=lambda b: ev.branch_scores[b], default=None)
            for name, c in ev.critics.items():
                if c.branch == weakest and c.issues:
                    return c.issues[0]
            return issues[0]
        if ev.branch_scores:
            weakest = min((b for b in ("keep", "follow", "world") if b in ev.branch_scores),
                          key=lambda b: ev.branch_scores[b], default=None)
            return BRANCH_FALLBACK.get(weakest)
        return None

    def propose(self, ev: EvaluationResult, state: EditState, tried: set[str]) -> tuple[IssueType | None, Action, EditState]:
        """Return (issue addressed, action, new state). Never returns a state already tried for this task."""
        issue = self.primary_issue(ev)
        candidates = list(ISSUE_ACTIONS.get(issue, [])) if issue else []
        if self.memory and issue:
            candidates = self.memory.rank(issue, self.editor.name, candidates)
        for action in candidates + [RESEED]:
            new = self.apply(action, state)
            if action.kind == "reseed" or new.key() not in tried:
                return issue, action, new
        return issue, RESEED, state

    def apply(self, action: Action, state: EditState) -> EditState:
        params = dict(state.params)
        if action.kind in ("preserve_more", "edit_more"):
            direction = 1 if action.kind == "preserve_more" else -1
            ranges = self.editor.param_ranges()
            for knob, sign in self.editor.preserve_knobs.items():
                if knob in ranges:
                    params[knob] = ranges[knob].nudge(params[knob], direction * sign)
            return EditState(params, state.clauses)
        if action.kind == "clause" and action.clause:
            return EditState(params, add_clause(state.clauses, action.clause))
        return EditState(params, state.clauses)  # reseed: same state, new seeds
