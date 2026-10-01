"""Critic interface. Critics never run models: they read the pre-computed Perceptions + Regions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.spec import EditSpec
from src.types import CriticResult, IssueType, Perception, Regions


@dataclass
class CriticContext:
    before: Perception
    after: Perception
    regions: Regions  # regions of the AFTER image
    spec: EditSpec
    regions_before: Regions | None = None  # same partition on the BEFORE image (natural-photo reference)


class Critic(ABC):
    name: str = "critic"
    branch: str = "keep"  # gate | keep | follow | world

    def __init__(self, **params):
        self.params = params

    def p(self, key: str, default):
        return self.params.get(key, default)

    @abstractmethod
    def evaluate(self, ctx: CriticContext) -> CriticResult: ...

    # helpers
    def result(self, score: float | None, issues: list[IssueType] | None = None, catastrophic: bool = False,
               **evidence) -> CriticResult:
        if score is not None:
            score = float(np.clip(score, 0.0, 1.0))
        return CriticResult(self.name, self.branch, score, issues or [], evidence, catastrophic)

    def not_applicable(self, reason: str) -> CriticResult:
        return CriticResult(self.name, self.branch, None, [], {"not_applicable": reason})


def ramp(x: float, lo: float, hi: float) -> float:
    """Linear map lo -> 0, hi -> 1, clipped."""
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))
