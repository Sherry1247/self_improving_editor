"""Aggregation strategies (innovation D).

* ``GatedGeometricAggregator`` (ours): any catastrophic gate -> 0. Otherwise a geometric mean of the
  Keep / Follow / World branch scores, so one strong branch cannot buy back a failed one.
* ``WeightedSumAggregator`` (legacy baseline): flat weighted mean of every applicable critic.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from src.types import CriticResult, EvaluationResult

BRANCHES = ("keep", "follow", "world")


class Aggregator(ABC):
    name = "aggregator"

    def __init__(self, cfg: dict):
        acfg = cfg.get("aggregation", {})
        self.weights: dict[str, float] = acfg.get("weights", {})
        self.branch_weights: dict[str, float] = acfg.get("branch_weights", {})

    @abstractmethod
    def aggregate(self, critics: dict[str, CriticResult]) -> EvaluationResult: ...

    def branch_scores(self, critics: dict[str, CriticResult]) -> dict[str, float]:
        out = {}
        for b in BRANCHES:
            items = [(c.score, self.weights.get(n, 1.0)) for n, c in critics.items()
                     if c.branch == b and c.applicable]
            wsum = sum(w for _, w in items)
            if items and wsum > 0:
                out[b] = sum(s * w for s, w in items) / wsum
        return out


class GatedGeometricAggregator(Aggregator):
    name = "gated_geometric"

    def aggregate(self, critics: dict[str, CriticResult]) -> EvaluationResult:
        gates = [c for c in critics.values() if c.branch == "gate"]
        passed = not any(c.is_catastrophic for c in gates)
        branches = self.branch_scores(critics)
        overall = 0.0
        if passed and branches:
            num = sum(self.branch_weights.get(b, 1.0) * math.log(max(s, 1e-3)) for b, s in branches.items())
            den = sum(self.branch_weights.get(b, 1.0) for b in branches)
            overall = math.exp(num / den)
        return EvaluationResult(critics=critics, overall=overall, branch_scores=branches,
                                gate_passed=passed, aggregator=self.name)


class WeightedSumAggregator(Aggregator):
    name = "weighted_sum"

    def aggregate(self, critics: dict[str, CriticResult]) -> EvaluationResult:
        items = [(c.score, self.weights.get(n, 1.0)) for n, c in critics.items() if c.applicable]
        wsum = sum(w for _, w in items)
        overall = sum(s * w for s, w in items) / wsum if wsum else 0.0
        gates = [c for c in critics.values() if c.branch == "gate"]
        return EvaluationResult(critics=critics, overall=overall, branch_scores=self.branch_scores(critics),
                                gate_passed=not any(c.is_catastrophic for c in gates), aggregator=self.name)


AGGREGATORS = {a.name: a for a in (GatedGeometricAggregator, WeightedSumAggregator)}


def build_aggregator(cfg: dict, method: str | None = None) -> Aggregator:
    method = method or cfg.get("aggregation", {}).get("method", "gated_geometric")
    return AGGREGATORS[method](cfg)
