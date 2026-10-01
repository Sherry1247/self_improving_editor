"""Dataset loading: labels.csv -> samples -> (sample x target) edit tasks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.spec import DEFAULT_TARGETS, EditSpec, build_spec, targets_for


@dataclass(frozen=True)
class Sample:
    sample_id: str
    filename: str
    obj: str
    action: str
    background: str
    submerged: bool = False


def load_samples(labels_csv: str | Path) -> list[Sample]:
    samples = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["filename"].strip()
            samples.append(Sample(
                sample_id=Path(fn).stem,
                filename=fn,
                obj=row["object"].strip(),
                action=row["action"].strip(),
                background=row["background"].strip(),
                submerged=str(row.get("submerged", "0")).strip() in {"1", "true", "True", "yes"},
            ))
    return samples


def build_tasks(samples: list[Sample], targets: tuple[str, ...] = DEFAULT_TARGETS,
                include_swap: bool = True, only: set[str] | None = None) -> list[tuple[Sample, EditSpec]]:
    tasks = []
    for s in samples:
        if only and s.sample_id not in only:
            continue
        for t in targets_for(s.background, targets, include_swap):
            tasks.append((s, build_spec(s.sample_id, s.obj, s.action, s.background, t, s.submerged)))
    return tasks
