"""Re-score finished runs with the CURRENT critics (no editing): uses each task's before.png / best.png.

    python experiments/rescore.py --src runs/ip2p runs/comp --name rescored

Writes runs/<name>/rescore.csv (old vs new overall score + branch scores + issues) and per-task panels
for the tasks whose verdict changed most. Full runs only kept the best candidate, so this re-scores bests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from common import ROOT, base_parser, build_evaluation_stack, new_run_dir, setup


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--src", nargs="+", required=True, help="run directories to re-score")
    ap.add_argument("--name", default="rescored")
    ap.add_argument("--panels", type=int, default=12, help="save panels for the N largest score changes")
    args = ap.parse_args()
    cfg = setup(args)

    from PIL import Image

    from src.spec import build_spec
    from src.utils.visualization import make_panel
    from src.data import save_image

    out = new_run_dir(cfg, args.name)
    _, perceiver, evaluator = build_evaluation_stack(cfg)
    rows, changes = [], []
    for src in args.src:
        src = Path(src) if Path(src).is_absolute() else ROOT / src
        for rep_path in sorted(src.glob("*/report.json")):
            tdir = rep_path.parent
            if not (tdir / "best.png").exists() or not (tdir / "before.png").exists():
                continue
            rep = json.loads(rep_path.read_text())
            sp = rep["spec"]
            obj = "dog" if sp["subject"] == "dog" else "adult_person"
            spec = build_spec(sp["sample_id"], obj, "sit" if sp["pose"] == "sitting" else "stand",
                              sp["source_bg"]["key"], sp["target_bg"]["key"], sp["submerged"])
            before_img = np.asarray(Image.open(tdir / "before.png").convert("RGB"))
            best_img = np.asarray(Image.open(tdir / "best.png").convert("RGB"))
            before = perceiver.perceive_cached(before_img, spec, sp["sample_id"])
            ev, _, regions = evaluator.evaluate(before, best_img, spec)
            old = rep["best_evaluation"]
            bs = ev.branch_scores
            row = {"run": src.name, "task_id": spec.task_id, "target": spec.target_bg.key,
                   "submerged": int(spec.submerged), "old_score": round(old["overall"] or 0, 4),
                   "new_score": round(ev.overall or 0, 4), "old_gate": old["gate_passed"], "new_gate": ev.gate_passed,
                   "keep": _r(bs.get("keep")), "follow": _r(bs.get("follow")), "world": _r(bs.get("world")),
                   "weighted_sum_baseline": _r(bs.get("weighted_sum_baseline")),
                   "bg_top": ev.critics["bg_semantic"].evidence.get("top") if "bg_semantic" in ev.critics else None,
                   "issues": " ".join(i.value for i in ev.issues)}
            rows.append(row)
            changes.append((abs(row["new_score"] - row["old_score"]), spec.task_id, src.name, before_img, best_img,
                            regions, ev))
            (out / "evaluations").mkdir(exist_ok=True)
            (out / "evaluations" / f"{src.name}__{spec.task_id}.json").write_text(json.dumps(ev.to_dict(), indent=1))
            print(f"{src.name}/{spec.task_id}: {row['old_score']:.3f} -> {row['new_score']:.3f}")
    with open(out / "rescore.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for _, tid, run, b, a, reg, ev in sorted(changes, key=lambda x: -x[0])[: args.panels]:
        save_image(make_panel(b, a, reg, ev, f"{run}/{tid} (rescored)"), out / "panels" / f"{run}__{tid}.jpg")
    print(f"\n{len(rows)} tasks -> {out / 'rescore.csv'}")


def _r(x):
    return None if x is None else round(float(x), 4)


if __name__ == "__main__":
    main()
