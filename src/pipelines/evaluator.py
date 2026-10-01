"""Evaluator: (before perception, after image, spec) -> EvaluationResult. No editing, no iteration."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.critics import Critic, CriticContext
from src.regions import partition
from src.scoring import Aggregator, WeightedSumAggregator
from src.spec import EditSpec
from src.types import EvaluationResult, Perception, Regions


class PerceiverLike(Protocol):
    def perceive(self, image: np.ndarray, spec: EditSpec) -> Perception: ...


class Evaluator:
    def __init__(self, perceiver: PerceiverLike, critics: list[Critic], aggregator: Aggregator):
        self.perceiver = perceiver
        self.critics = critics
        self.aggregator = aggregator
        self.baseline = WeightedSumAggregator({"aggregation": {"weights": aggregator.weights}})

    def evaluate(self, before: Perception, after_image: np.ndarray, spec: EditSpec,
                 after: Perception | None = None) -> tuple[EvaluationResult, Perception, Regions]:
        if after_image.shape != before.image.shape:
            raise ValueError(f"after image {after_image.shape} != before image {before.image.shape}; "
                             "editors must return the working resolution")
        after = after or self.perceiver.perceive(after_image, spec)
        regions = partition(after.subject_mask, before.subject_mask)
        regions_before = partition(before.subject_mask, before.subject_mask)
        ctx = CriticContext(before, after, regions, spec, regions_before)
        results = {c.name: c.evaluate(ctx) for c in self.critics}
        ev = self.aggregator.aggregate(results)
        ev.mock = before.mock or after.mock
        ev.branch_scores["weighted_sum_baseline"] = self.baseline.aggregate(results).overall
        return ev, after, regions
