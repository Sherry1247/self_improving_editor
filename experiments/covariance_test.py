"""Constructive covariance test on REAL results: does a critic prefer the pasted subject over the re-lit one?

    python experiments/covariance_test.py --src runs/comp_v2 runs/qwen --name covtest_v3
    python experiments/covariance_test.py --src runs/comp_v2 --name covtest_v2 \
        --set critics.appearance.variant=v2 --set critics.light_harmony.variant=v2

Every task's best.png is scored in three versions with the CURRENT critics:
  pasted : as produced by the editor
  null   : same Lab round trip as the re-lit one but with zero strength -> must score like 'pasted'
           (the largest |null - pasted| over all critics is the pipeline noise floor)
  relit  : the subject re-lit for its new background (validation/relight.py, conservative)
A covariance-aware critic gives relit >= pasted on the world branch and does not lose it on keep.

Writes runs/<name>/covtest.csv with EVERY critic's score per version, the appearance evidence (zncc, alignment IoU,
unreliable flag), whether the light co-change was actually testable, and the dev/test split (source image _03 = test).
Prints win rates per run and split. Tune only on dev; report test.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from common import ROOT, base_parser, build_evaluation_stack, new_run_dir, setup

VERSIONS = ("pasted", "null", "relit")


def split_of(sample_id: str) -> str:
    return "test" if sample_id.endswith("_03") else "dev"


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--name", default="covtest")
    ap.add_argument("--strength", type=float, default=0.15)
    ap.add_argument("--side", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N tasks per run")
    args = ap.parse_args()
    cfg = setup(args)

    from PIL import Image

    from src.spec import build_spec
    from validation.relight import relight_subject

    out = new_run_dir(cfg, args.name)
    _, perceiver, evaluator = build_evaluation_stack(cfg)
    rows = []
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
            pasted = np.asarray(Image.open(tdir / "best.png").convert("RGB"))
            before = perceiver.perceive_cached(before_img, spec, sp["sample_id"])
            after = perceiver.perceive(pasted, spec)
            if not after.subject_mask.any():
                continue
            direction = 1 if int(hashlib.md5(spec.task_id.encode()).hexdigest(), 16) % 2 else -1
            imgs = {"pasted": pasted,
                    "null": relight_subject(pasted, after.subject_mask, strength=0.0, side=0.0),
                    "relit": relight_subject(pasted, after.subject_mask, strength=args.strength, side=args.side,
                                             direction=direction)}
            row = {"run": src.name, "task_id": spec.task_id, "sample": sp["sample_id"], "split": split_of(sp["sample_id"]),
                   "target": spec.target_bg.key}
            for tag, img in imgs.items():
                ev, _, _ = evaluator.evaluate(before, img, spec)
                bs = ev.branch_scores
                row[f"{tag}_overall"], row[f"{tag}_gate"] = _r(ev.overall), ev.gate_passed
                for b in ("keep", "follow", "world"):
                    row[f"{tag}_{b}"] = _r(bs.get(b))
                for name, c in ev.critics.items():
                    row[f"{tag}_c_{name}"] = _r(c.score)
                a = ev.critics.get("appearance")
                if a is not None:
                    for k in ("zncc", "zncc_raw", "align_iou", "chroma_dist"):
                        row[f"{tag}_app_{k}"] = _r(a.evidence.get(k))
                    row[f"{tag}_app_unreliable"] = bool(a.evidence.get("unreliable_alignment"))
                lh = ev.critics.get("light_harmony")
                if lh is not None:
                    row[f"{tag}_lh_cochange_tested"] = bool(lh.evidence.get("cochange_tested"))
                    row[f"{tag}_lh_follow"] = _r(lh.evidence.get("follow"))
            rows.append(row)
            print(f"{src.name}/{spec.task_id} [{row['split']}]: world {row['pasted_world']} -> {row['relit_world']}  "
                  f"keep {row['pasted_keep']} -> {row['relit_keep']}")
    if not rows:
        raise SystemExit("no tasks scored")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "covtest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} tasks -> {out / 'covtest.csv'}")
    summarize(rows)


def summarize(rows):
    keys = ["overall", "keep", "world", "c_appearance", "c_light_harmony", "c_support", "c_halo", "c_vlm_lighting",
            "c_vlm_integration", "c_vlm_shadow"]
    noise = max((abs((r.get(f"null_{k}") or 0) - (r.get(f"pasted_{k}") or 0)) for r in rows for k in keys
                 if r.get(f"null_{k}") is not None and r.get(f"pasted_{k}") is not None), default=0.0)
    print(f"\nNULL control: max |null - pasted| over critics and tasks = {noise:.3f}  (pipeline noise floor)")
    for run in sorted({r["run"] for r in rows}):
        for split in ("dev", "test"):
            sel = [r for r in rows if r["run"] == run and r["split"] == split]
            if not sel:
                continue
            tested = np.mean([bool(r.get("pasted_lh_cochange_tested")) for r in sel])
            unrel = np.mean([bool(r.get("pasted_app_unreliable")) for r in sel])
            print(f"\n== {run} / {split}  n={len(sel)}  light co-change testable {tested:.0%}  appearance unreliable-alignment {unrel:.0%}")
            for k in keys:
                pairs = [(r[f"pasted_{k}"], r[f"relit_{k}"]) for r in sel
                         if r.get(f"pasted_{k}") is not None and r.get(f"relit_{k}") is not None]
                if pairs:
                    a, b = np.array(pairs).T
                    print(f"  {k:18s} relit>pasted {int((b > a + 1e-3).sum()):3d}  tie {int((abs(b - a) <= 1e-3).sum()):3d}  "
                          f"relit<pasted {int((b < a - 1e-3).sum()):3d}   mean {a.mean():.3f} -> {b.mean():.3f}")


def _r(x):
    return None if x is None else round(float(x), 4)


if __name__ == "__main__":
    main()
