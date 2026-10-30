"""Does an editor leave big pixel-identical chunks of the OLD scene behind (railing, bench, wall)?

    python experiments/old_scene_residue_check.py --src runs/comp_v2 runs/qwen --name residue

Per task: share of the background that is an unchanged copy of the before image (UnchangedBackgroundCritic).
Writes runs/<name>/residue.csv, prints per-run statistics and saves overlay panels (before | after | red = unchanged
blobs) for the worst tasks of each run. Exploratory: motivated by Qwen results that kept the original railing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from common import ROOT, base_parser, build_evaluation_stack, new_run_dir, setup


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--name", default="residue")
    ap.add_argument("--panels", type=int, default=8, help="overlay panels per run (worst tasks)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = setup(args)
    if "unchanged_bg" not in cfg["critics"]["enabled"]:
        cfg["critics"]["enabled"] = list(cfg["critics"]["enabled"]) + ["unchanged_bg"]

    from PIL import Image

    from src.critics.follow import unchanged_blobs
    from src.data import save_image
    from src.spec import build_spec

    out = new_run_dir(cfg, args.name)
    _, perceiver, evaluator = build_evaluation_stack(cfg)
    rows, keep = [], {}
    for src in args.src:
        src = Path(src) if Path(src).is_absolute() else ROOT / src
        for rep_path in sorted(src.glob("*/report.json"))[: args.limit or None]:
            tdir = rep_path.parent
            if not (tdir / "best.png").exists():
                continue
            sp = json.loads(rep_path.read_text())["spec"]
            obj = "dog" if sp["subject"] == "dog" else "adult_person"
            spec = build_spec(sp["sample_id"], obj, "sit" if sp["pose"] == "sitting" else "stand",
                              sp["source_bg"]["key"], sp["target_bg"]["key"], sp["submerged"])
            before_img = np.asarray(Image.open(tdir / "before.png").convert("RGB"))
            after_img = np.asarray(Image.open(tdir / "best.png").convert("RGB"))
            before = perceiver.perceive_cached(before_img, spec, sp["sample_id"])
            ev, after, regions = evaluator.evaluate(before, after_img, spec)
            c = ev.critics["unchanged_bg"]
            frac = c.evidence.get("blob_frac")
            rows.append({"run": src.name, "task_id": spec.task_id, "target": spec.target_bg.key,
                         "blob_frac": None if frac is None else round(frac, 4),
                         "score": None if c.score is None else round(c.score, 4),
                         "overall": None if ev.overall is None else round(ev.overall, 4)})
            if frac is not None:
                g = regions.background & ~before.subject_mask
                keep.setdefault(src.name, []).append((frac, spec.task_id, before_img, after_img,
                                                      unchanged_blobs(before_img, after_img, g)))
    with open(out / "residue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for run, items in keep.items():
        v = np.array([i[0] for i in items])
        print(f"{run:10s} n={len(v)}  mean blob share {v.mean():.3f}  median {np.median(v):.3f}  "
              f">0.10: {int((v > 0.10).sum())}  >0.25: {int((v > 0.25).sum())}")
        for frac, tid, b, a, blobs in sorted(items, key=lambda x: -x[0])[: args.panels]:
            ov = a.copy()
            ov[blobs] = (0.45 * ov[blobs] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
            save_image(np.concatenate([b, a, ov], axis=1), out / "panels" / f"{run}__{tid}__{frac:.2f}.jpg")
    print(f"-> {out / 'residue.csv'}")


if __name__ == "__main__":
    main()
