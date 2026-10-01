"""Core data structures shared across the pipeline.

Conventions
-----------
* Images are RGB ``uint8`` numpy arrays of shape (H, W, 3).
* Masks are ``bool`` numpy arrays of shape (H, W).
* Boxes are ``(x1, y1, x2, y2)`` in pixel coordinates of the image they belong to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

BBox = tuple[float, float, float, float]


class IssueType(str, Enum):
    """Typed failure categories produced by critics and consumed by the router."""

    # gate / invariant failures
    SUBJECT_MISSING = "subject_missing"
    EXTRA_SUBJECT = "extra_subject"
    IDENTITY_LOSS = "identity_loss"
    SUBJECT_DRIFT = "subject_drift"  # silhouette / appearance changed
    # target-change failures
    BG_UNCHANGED = "bg_unchanged"
    BG_WRONG = "bg_wrong"
    OLD_BG_RESIDUE = "old_bg_residue"
    # covariant (world) failures
    FLOATING = "floating"
    MISSING_SHADOW = "missing_shadow"
    LIGHT_MISMATCH = "light_mismatch"
    HALO = "halo"


@dataclass
class Detection:
    label: str
    box: BBox
    score: float
    mask: np.ndarray | None = None


@dataclass
class Perception:
    """Everything the critics need to know about ONE image. Computed once, cached."""

    image: np.ndarray
    subject_detections: list[Detection]
    subject_mask: np.ndarray  # primary subject (highest-scoring detection)
    old_bg_mask: np.ndarray  # union of old-background concept masks (e.g. water)
    depth: np.ndarray | None = None  # relative inverse depth, larger = closer
    subject_embedding: np.ndarray | None = None  # DINOv2 CLS of masked subject crop
    patch_features: np.ndarray | None = None  # DINOv2 patch grid (gh, gw, C), L2-normalised
    bg_probs: dict[str, float] | None = None  # SigLIP probs over candidate backgrounds (subject removed)
    mock: bool = False

    @property
    def subject_count(self) -> int:
        return len(self.subject_detections)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]


@dataclass
class Regions:
    """Mask-derived regions of the EDITED image (see docs: S / B / C / G)."""

    subject: np.ndarray  # S
    boundary_inner: np.ndarray  # inner half of B
    boundary_outer: np.ndarray  # outer half of B
    contact: np.ndarray  # C: band just below the subject's lowest pixels (excluding subject)
    background: np.ndarray  # G: far background
    aligned_before_subject: np.ndarray  # before-mask warped onto the after image
    transform: dict[str, float] = field(default_factory=dict)  # dx, dy, scale

    @property
    def boundary(self) -> np.ndarray:
        return self.boundary_inner | self.boundary_outer


@dataclass
class CriticResult:
    name: str
    branch: str  # "gate" | "keep" | "follow" | "world"
    score: float | None  # [0, 1]; None = not applicable for this sample
    issues: list[IssueType] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    is_catastrophic: bool = False

    @property
    def applicable(self) -> bool:
        return self.score is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "branch": self.branch,
            "score": self.score,
            "issues": [i.value for i in self.issues],
            "evidence": _jsonable(self.evidence),
            "is_catastrophic": self.is_catastrophic,
        }


@dataclass
class EvaluationResult:
    critics: dict[str, CriticResult]
    overall: float | None = None
    branch_scores: dict[str, float] = field(default_factory=dict)
    gate_passed: bool | None = None
    aggregator: str | None = None
    mock: bool = False

    @property
    def issues(self) -> list[IssueType]:
        out: list[IssueType] = []
        for c in self.critics.values():
            for i in c.issues:
                if i not in out:
                    out.append(i)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "mock": self.mock,
            "overall": self.overall,
            "gate_passed": self.gate_passed,
            "aggregator": self.aggregator,
            "branch_scores": self.branch_scores,
            "issues": [i.value for i in self.issues],
            "critics": {k: v.to_dict() for k, v in self.critics.items()},
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 16 else f"<array {obj.shape}>"
    if isinstance(obj, Enum):
        return obj.value
    return obj
