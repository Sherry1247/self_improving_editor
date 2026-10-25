"""Constructive covariance test on REAL results: does a critic prefer the pasted subject over the re-lit one?

    python experiments/covariance_test.py --src runs/comp_v2 runs/qwen --name covtest_v3
    python experiments/covariance_test.py --src runs/comp_v2 --name covtest_v2 \
        --set critics.appearance.variant=v2 --set critics.light_harmony.variant=v2

For every task: score best.png ('pasted') and the same image with the subject re-lit for its new background
('relit', validation/relight.py), with the CURRENT critics. A covariance-aware critic gives relit >= pasted on the
world branch and does not lose it on keep. Writes runs/<name>/covtest.csv and prints the win rates.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from common import ROOT, base_parser, build_evaluation_stack, new_run_dir, setup


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--name", default="covtest")
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N tasks per run")
    args = ap.parse_args()
    cfg = setup(args)

    import json

    from PIL import Image

    from src.spec import build_spec
    from validation.relight import relight_subject

    out = new_run_dir(cfg, args.name)
    _, perceiver, evaluator = build_evaluation_stack(cfg)
    rows = []
    for src in args.src:
        src = Path(src) if Path(src).is_absolute() else ROOT / src
        reps = sorted(src.glob("*/report.json"))
        for rep_path in reps[: args.limit or None]:
            tdir = rep_path.parent
            if not (tdir / "best.png").exists():
                continue
            sp = json.loads(rep_path.read_text())["spec"]
            obj = "dog" if sp["subject"] == "dog" else "adult_person"
            spec = build_spec(sp["sample_id"], obj, "sit" if sp["pose"] == "sitting" else "stand",
                              sp["source_bg"]["key"], sp["target_bg"]["key"], sp["submerged"])
            before_img = np.asarray(Image.open(tdir / "before.png").convert("RGB"))
            pasted = np.asarray(Image.open(tdir / "best.png").convert("RGB"))
            before = perceiver.perceive_cached(before_img, spec, sp["sample_id"])
            after = perceiver.perceive(pasted, spec)
            if not after.subject_mask.any():
                continue
            direction = 1 if int(hashlib.md5(spec.task_id.encode()).hexdigest(), 16) % 2 else -1
            relit = relight_subject(pasted, after.subject_mask, strength=args.strength, direction=direction)
            row = {"run": src.name, "task_id": spec.task_id, "target": spec.target_bg.key}
            for tag, img in (("pasted", pasted), ("relit", relit)):
                ev, _, _ = evaluator.evaluate(before, img, spec)
                bs = ev.branch_scores
                row.update({f"{tag}_overall": _r(ev.overall), f"{tag}_keep": _r(bs.get("keep")),
                            f"{tag}_world": _r(bs.get("world")),
                            f"{tag}_appearance": _r(_s(ev, "appearance")),
                            f"{tag}_light_harmony": _r(_s(ev, "light_harmony")),
                            f"{tag}_gate": ev.gate_passed})
            rows.append(row)
            print(f"{src.name}/{spec.task_id}: world {row['pasted_world']} -> {row['relit_world']}  "
                  f"keep {row['pasted_keep']} -> {row['relit_keep']}")
    if not rows:
        raise SystemExit("no tasks scored")
    with open(out / "covtest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} tasks -> {out / 'covtest.csv'}")
    for key in ("world", "light_harmony", "keep", "appearance", "overall"):
        pairs = [(r[f"pasted_{key}"], r[f"relit_{key}"]) for r in rows
                 if r[f"pasted_{key}"] is not None and r[f"relit_{key}"] is not None]
        if pairs:
            a, b = np.array(pairs).T
            print(f"{key:14s} relit>pasted {int((b > a + 1e-3).sum()):3d}  tie {int((abs(b - a) <= 1e-3).sum()):3d}  "
                  f"relit<pasted {int((b < a - 1e-3).sum()):3d}   mean {a.mean():.3f} -> {b.mean():.3f}")


def _s(ev, name):
    c = ev.critics.get(name)
    return None if c is None else c.score


def _r(x):
    return None if x is None else round(float(x), 4)


if __name__ == "__main__":
    main()
