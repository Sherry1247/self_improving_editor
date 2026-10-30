"""Contact/boundary RegionRepair ablation on finished results (uses before.png / best.png of source runs).

    python experiments/region_repair.py --src runs/comp_v3 --name rr --kinds rect footprint

Per task: variants A_none, B_contact, C_boundary, D_both, E_misplaced (equal-area control) for each contact
shape, all re-scored with the FROZEN critics (exploratory critics off). Writes runs/<name>/repair.csv
(one row per task x kind x variant, with the same columns as rescore.csv plus region_px) and images.
Full-image re-edit (F) and best-of-N reseed (G) are existing runs: compare them by rescoring those runs.
Analyse with: paired bootstrap over source images (cluster = sample_id up to the last '_').
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
    ap.add_argument("--name", default="region_repair")
    ap.add_argument("--kinds", nargs="+", default=["rect", "footprint"], choices=["rect", "footprint", "cast"])
    ap.add_argument("--light-dx", type=float, default=0.5, help="shear for kind=cast (px per px of depth)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = setup(args)

    from PIL import Image

    from src.data import save_image
    from src.refinement.region_repair import RegionRepairer, build_variants
    from src.spec import build_spec

    out = new_run_dir(cfg, args.name)
    registry, perceiver, evaluator = build_evaluation_stack(cfg)
    repairer = RegionRepairer(cfg, registry)
    rows = []
    n = 0
    for src in args.src:
        src = Path(src) if Path(src).is_absolute() else ROOT / src
        for rep_path in sorted(src.glob("*/report.json")):
            tdir = rep_path.parent
            if not (tdir / "best.png").exists() or not (tdir / "before.png").exists():
                continue
            if args.limit and n >= args.limit:
                break
            n += 1
            sp = json.loads(rep_path.read_text())["spec"]
            obj = "dog" if sp["subject"] == "dog" else "adult_person"
            spec = build_spec(sp["sample_id"], obj, "sit" if sp["pose"] == "sitting" else "stand",
                              sp["source_bg"]["key"], sp["target_bg"]["key"], sp["submerged"])
            before_img = np.asarray(Image.open(tdir / "before.png").convert("RGB"))
            best_img = np.asarray(Image.open(tdir / "best.png").convert("RGB"))
            before = perceiver.perceive_cached(before_img, spec, sp["sample_id"])
            _, _, regions = evaluator.evaluate(before, best_img, spec)
            for kind in args.kinds:
                for s in range(args.seeds):
                    variants = build_variants(best_img, regions, "dog" if sp["subject"] == "dog" else "person",
                                              spec.target_bg.key.replace("_", " "), repairer, kind,
                                              args.light_dx, seed=17 + s)
                    for vname, v in variants.items():
                        ev, _, _ = evaluator.evaluate(before, v["image"], spec)
                        bs = ev.branch_scores
                        row = {"src": src.name, "task_id": spec.task_id, "sample_id": sp["sample_id"], "kind": kind,
                               "seed": s, "variant": vname, "region_px": v["region_px"],
                               "overall": _r(ev.overall), "gate": ev.gate_passed,
                               "keep": _r(bs.get("keep")), "follow": _r(bs.get("follow")), "world": _r(bs.get("world")),
                               "issues": " ".join(i.value for i in ev.issues)}
                        for cname, c in ev.critics.items():
                            row["c_" + cname] = _r(c.score)
                        rows.append(row)
                        save_image(v["image"], out / "images" / f"{spec.task_id}__{kind}__s{s}__{vname}.jpg")
                    print(f"{spec.task_id} [{kind}] " + " ".join(f"{r['variant'][0]}={r['overall']}" for r in rows[-len(variants):]))
    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / "repair.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    print(f"\n{len(rows)} rows -> {out / 'repair.csv'}")


def _r(x):
    return None if x is None else round(float(x), 4)


if __name__ == "__main__":
    main()
