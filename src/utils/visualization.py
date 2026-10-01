"""Debug panels: before | after | region overlay, with the score breakdown printed underneath."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from src.types import EvaluationResult, Regions

REGION_COLORS = {
    "subject": (255, 64, 64),
    "boundary_outer": (255, 200, 0),
    "contact": (0, 160, 255),
    "background": (80, 200, 80),
}


def overlay_regions(img: np.ndarray, regions: Regions, alpha: float = 0.45) -> np.ndarray:
    out = img.astype(np.float32).copy()
    for name, color in REGION_COLORS.items():
        m = getattr(regions, name)
        if name == "background":
            a = alpha * 0.4
        else:
            a = alpha
        out[m] = (1 - a) * out[m] + a * np.array(color, np.float32)
    return out.astype(np.uint8)


def score_lines(ev: EvaluationResult) -> list[str]:
    head = f"overall={ev.overall:.3f}  gate={'PASS' if ev.gate_passed else 'FAIL'}  " + "  ".join(
        f"{k}={v:.2f}" for k, v in ev.branch_scores.items() if v is not None)
    lines = [head]
    row = []
    for name, c in ev.critics.items():
        s = "n/a" if c.score is None else f"{c.score:.2f}"
        flag = "!" if c.is_catastrophic else ("*" if c.issues else "")
        row.append(f"{name}={s}{flag}")
        if len(row) == 4:
            lines.append("  ".join(row))
            row = []
    if row:
        lines.append("  ".join(row))
    if ev.issues:
        lines.append("issues: " + ", ".join(i.value for i in ev.issues))
    return lines


def make_panel(before: np.ndarray, after: np.ndarray, regions: Regions, ev: EvaluationResult,
               title: str = "", height: int = 384) -> np.ndarray:
    def fit(x):
        h, w = x.shape[:2]
        return np.asarray(Image.fromarray(x).resize((int(w * height / h), height)))

    tiles = [fit(before), fit(after), fit(overlay_regions(after, regions))]
    strip = np.concatenate(tiles, axis=1)
    lines = ([title] if title else []) + score_lines(ev)
    text_h = 16 * len(lines) + 10
    canvas = Image.new("RGB", (strip.shape[1], height + text_h), "white")
    canvas.paste(Image.fromarray(strip), (0, 0))
    d = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        d.text((6, height + 5 + 16 * i), line, fill=(0, 0, 0))
    return np.asarray(canvas)
