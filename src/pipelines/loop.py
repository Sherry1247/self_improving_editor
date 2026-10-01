"""RefinementLoop: edit -> evaluate -> diagnose -> act, under a fixed editor-call budget.

* best-of-N seeds per round
* accept-if-improves: a round that does not beat the best is reverted (state goes back to the best's)
* stops on threshold / plateau (patience) / budget; always returns the best candidate seen
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.editors.base import Editor
from src.pipelines.evaluator import Evaluator
from src.refinement import ActionRouter, EditState, build_prompt
from src.spec import EditSpec
from src.types import EvaluationResult, Perception, Regions

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    round: int
    index: int
    seed: int
    prompt: str
    state: EditState
    image: np.ndarray
    evaluation: EvaluationResult
    regions: Regions | None = None

    @property
    def score(self) -> float:
        return float(self.evaluation.overall or 0.0)

    def summary(self) -> dict:
        return {"round": self.round, "index": self.index, "seed": self.seed, "prompt": self.prompt,
                "params": self.state.params, "clauses": list(self.state.clauses), "score": self.score,
                "gate_passed": self.evaluation.gate_passed, "branch_scores": self.evaluation.branch_scores,
                "issues": [i.value for i in self.evaluation.issues]}


@dataclass
class LoopResult:
    best: Candidate
    candidates: list[Candidate]
    rounds: list[dict] = field(default_factory=list)
    stopped_reason: str = ""
    edits_used: int = 0


class RefinementLoop:
    def __init__(self, editor: Editor, evaluator: Evaluator, router: ActionRouter, cfg: dict):
        lcfg = cfg.get("loop", {})
        self.editor, self.evaluator, self.router = editor, evaluator, router
        self.max_edits = int(lcfg.get("max_edits", 8))
        self.n = int(lcfg.get("n_candidates", 2))
        self.threshold = float(lcfg.get("threshold", 0.75))
        self.patience = int(lcfg.get("patience", 2))
        self.full_budget = bool(lcfg.get("full_budget", False))
        self.seed = int(cfg.get("seed", 42))
        self.eps = 1e-3

    def run(self, before: Perception, spec: EditSpec, on_candidate=None) -> LoopResult:
        state = EditState(self.editor.default_params(), ())
        tried: set[str] = set()
        best: Candidate | None = None
        all_c: list[Candidate] = []
        rounds: list[dict] = []
        edits, stale, r = 0, 0, 0
        pending: tuple | None = None  # (issue, action, score_before) for the memory update
        reason = "budget"

        while edits + self.n <= self.max_edits:
            tried.add(state.key())
            prompt = build_prompt(spec, self.editor.prompt_style, state.clauses)
            round_c = []
            for i in range(self.n):
                seed = self.seed + 1000 * r + i
                img = self.editor.edit(before.image, prompt, state.params, seed, spec, before.subject_mask)
                ev, _, regions = self.evaluator.evaluate(before, img, spec)
                c = Candidate(r, i, seed, prompt, state, img, ev, regions)
                round_c.append(c)
                if on_candidate:
                    on_candidate(c)
            edits += self.n
            all_c.extend(round_c)
            rb = max(round_c, key=lambda c: c.score)
            prev = best.score if best else 0.0
            improved = best is None or rb.score > best.score + self.eps
            if pending and self.router.memory and pending[0] is not None:
                self.router.memory.update(pending[0], self.editor.name, pending[1].name, rb.score - pending[2])
            rounds.append({"round": r, "prompt": prompt, "params": state.params, "clauses": list(state.clauses),
                           "action": pending[1].name if pending else "initial",
                           "issue": pending[0].value if pending and pending[0] else None,
                           "round_best": rb.score, "best_before": prev, "accepted": improved})
            logger.info("[%s] round %d: best=%.3f (prev %.3f) %s", spec.task_id, r, rb.score, prev,
                        "accepted" if improved else "reverted")
            if improved:
                best, stale = rb, 0
            else:
                stale += 1
            if best.score >= self.threshold and not self.full_budget:
                reason = "threshold"
                break
            if stale >= self.patience:
                reason = "plateau"
                break
            issue, action, state = self.router.propose(best.evaluation, best.state, tried)
            pending = (issue, action, best.score)
            r += 1
        return LoopResult(best=best, candidates=all_c, rounds=rounds, stopped_reason=reason, edits_used=edits)
