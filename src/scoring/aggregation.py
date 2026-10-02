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
        # per-branch reduction: "mean" (weighted) or "softmin<k>" (mean of the k lowest scores).
        # World defaults to softmin2: one physical error (floating, pasted look) ruins realism on its own,
        # so a branch average over seven checks would hide it.
        self.reduce: dict[str, str] = acfg.get("branch_reduce", {})

    @abstractmethod
    def aggregate(self, critics: dict[str, CriticResult]) -> EvaluationResult: ...

    def branch_scores(self, critics: dict[str, CriticResult]) -> dict[str, float]:
        out = {}
        for b in BRANCHES:
            items = [(c.score, self.weights.get(n, 1.0)) for n, c in critics.items()
                     if c.branch == b and c.applicable]
            wsum = sum(w for _, w in items)
            if not items or wsum <= 0:
                continue
            mode = self.reduce.get(b, "mean")
            if mode.startswith("softmin"):
                k = int(mode[len("softmin"):] or 2)
                low = sorted(s for s, _ in items)[:k]
                out[b] = sum(low) / len(low)
            else:
                out[b] = sum(s * w for s, w in items) / wsum
        return out


class GatedGeometricAggregator(Aggregator):
    name = "gated_geometric"

    def aggregate(self, critics: dict[str, CriticResult]) -> EvaluationResult:
        passed = not any(c.is_catastrophic for c in critics.values())  # any critic may act as a gate
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
        return EvaluationResult(critics=critics, overall=overall, branch_scores=self.branch_scores(critics),
                                gate_passed=not any(c.is_catastrophic for c in critics.values()), aggregator=self.name)


AGGREGATORS = {a.name: a for a in (GatedGeometricAggregator, WeightedSumAggregator)}


def build_aggregator(cfg: dict, method: str | None = None) -> Aggregator:
    method = method or cfg.get("aggregation", {}).get("method", "gated_geometric")
    return AGGREGATORS[method](cfg)
