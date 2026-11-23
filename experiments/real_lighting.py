"""Where do REAL editor outputs fall on the synthetic lighting / structure ladder?  (item 1, step 3; needs GPU, no VLM)

lint1/lint2 measured every Keep-type measure on synthetic edits of known size. This script scores the same measures on the finished runs of real
editors (before.png vs best.png of every task) and, for the SAME source photo, on the synthetic ladder (null, nuisance, LIGHT 1-3, STRUCT 1-3).
Each real output is then placed in one of four bands per measure (experiments/lighting_criteria.py::band_of):
  A  within re-encoding noise     score >= the photo's nuisance score
  B  within the lighting range    below the nuisance score but not below the weakest LIGHT level-3 variant of that photo
  C  beyond the lighting range    below every LIGHT level-3 variant of that photo (something other than lighting changed, or lighting changed more
                                  than the strongest synthetic edit)
  D  structure-like               below the median of the photo's STRUCT level >= 2 variants
D is a subset of C only when the measure separates the two; a measure whose C share is high for an editor that is known to paste the subject back is not
trustworthy. Bands say how big the change is, not whether it is lighting: a real edit can change lighting and structure together.
No threshold is tuned. The synthetic edits are uniform Lab edits; real relighting also moves shadows and highlights.

    bash chtc/pack_rescore.sh comp_v2 qwen ip2p_v2 && condor_submit chtc/job.sub MODE=reallight RUN=rl1 LIST=jobs/one.txt GPUMEM=24000M EXTRA_IN=,chtc/rescore_in.tar.gz
    python experiments/lighting_criteria.py --real runs/rl1          (offline, torch-free: band table per editor and measure)
Writes real.csv (one row per task and measure), ladder.csv (one row per photo and synthetic variant).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from common import ROOT, base_parser, new_run_dir, setup


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--name", default="real_lighting")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N tasks per run")
    args = ap.parse_args()
    cfg = setup(args)
    cfg["perception"]["use_vlm"] = False

    from lighting_intervention import score_pair, split_of
    from PIL import Image

    from src.critics import build_critics
    from src.critics.keep import AppearanceCritic
    from src.models import ModelRegistry
    from src.perception import Perceiver
    from src.regions import partition
    from src.spec import build_spec
    from validation.interventions import build_variants, l_image
    from validation.violations import clean_plate

    cfg["critics"]["enabled"] = ["identity", "silhouette", "appearance", "texture"]
    critics = {c.name: c for c in build_critics(cfg)}
    app_v2 = AppearanceCritic(**{**cfg["critics"].get("appearance", {}), "variant": "v2"})
    perceiver = Perceiver(ModelRegistry(cfg), cfg)
    out = new_run_dir(cfg, args.name)

    ladder_rows: list[dict] = []
    real_rows: list[dict] = []
    done_ladder: set[str] = set()
    before_cache: dict[str, tuple] = {}

    def prep(sp, before_img, spec):
        sid = sp["sample_id"]
        if sid in before_cache:
            return before_cache[sid]
        before = perceiver.perceive_cached(before_img, spec, sid)
        if before.subject_count == 0:
            before_cache[sid] = None
            return None
        rb = partition(before.subject_mask, before.subject_mask)
        emb = {s: perceiver.dino(l_image(before_img, before.subject_mask, s), before.subject_mask)[0] for s in ("lab", "rgb")}
        before_cache[sid] = (before, rb, emb)
        if sid not in done_ladder:                      # the synthetic ladder of this photo, scored exactly like a real output
            done_ladder.add(sid)
            plate = clean_plate(before_img, before.subject_mask)
            for v in build_variants(before_img, before.subject_mask, plate):
                r = {"sample": sid, "split": split_of(sid), "family": v.family, "kind": v.kind, "name": v.name, "level": v.level}
                r.update(score_pair(perceiver, critics, app_v2, before, v.image, spec, rb, emb))
                ladder_rows.append(r)
            print(f"ladder {sid}: {sum(r['sample'] == sid for r in ladder_rows)} variants")
        return before_cache[sid]

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
            best = np.asarray(Image.open(tdir / "best.png").convert("RGB"))
            pre = prep(sp, before_img, spec)
            if pre is None:
                continue
            before, rb, emb = pre
            r = {"run": src.name, "task_id": spec.task_id, "sample": sp["sample_id"], "split": split_of(sp["sample_id"]), "target": spec.target_bg.key}
            r.update(score_pair(perceiver, critics, app_v2, before, best, spec, rb, emb))
            real_rows.append(r)
            print(f"{src.name}/{spec.task_id}: found={r['subject_found']}")

    for name, rows in (("ladder.csv", ladder_rows), ("real.csv", real_rows)):
        cols = list(dict.fromkeys(c for r in rows for c in r))
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    print(f"\n{len(real_rows)} real outputs, {len(ladder_rows)} ladder variants -> {out}")


if __name__ == "__main__":
    main()
