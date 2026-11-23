"""Keep redesign, GPU part: record the raw result of EVERY critic (score, branch, catastrophic flag) on the violation suite v4 and on real editor
outputs, so the different Keep designs can be compared offline on exactly the same critic results (experiments/keep_redesign_report.py).

    --part suite      15 photos x the v4 violation suite (as experiments/critic_auroc.py; null = positive, one violation per negative)
    --part copypaste  finished editor runs (rescore bundle: before.png / best.png / report.json): pasted, relit (two strengths), null, degraded
                      versions of every best.png. The relit versions are better by construction (validation/relight.py); a Keep design that
                      prefers the pasted copy over the re-lit one is preferring copy-paste.

All critics of configs/chtc.yaml plus app_structure and app_chroma are evaluated; nothing is aggregated here. Writes raw.jsonl, one line per image:
  {"part", "sample", "kind", "task", "run", "critics": {name: {"s": score|null, "b": branch, "c": catastrophic}}}

    python experiments/keep_redesign.py --part suite --name kr_suite --config configs/chtc.yaml
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from common import ROOT, base_parser, build_evaluation_stack, new_run_dir, setup

EXTRA_CRITICS = ("app_structure", "app_chroma")


def record(ev) -> dict:
    return {n: {"s": c.score, "b": c.branch, "c": bool(c.is_catastrophic)} for n, c in ev.critics.items()}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--part", choices=["suite", "copypaste"], required=True)
    ap.add_argument("--name", default="keep_redesign")
    ap.add_argument("--samples", nargs="*")
    ap.add_argument("--src", nargs="*", default=[], help="copypaste: run directories with */report.json")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N samples / tasks per run")
    args = ap.parse_args()
    cfg = setup(args)
    for extra in EXTRA_CRITICS:
        if extra not in cfg["critics"]["enabled"]:
            cfg["critics"]["enabled"] = list(cfg["critics"]["enabled"]) + [extra]

    out = new_run_dir(cfg, args.name)
    _, perceiver, evaluator = build_evaluation_stack(cfg)
    fh = open(out / "raw.jsonl", "w", encoding="utf-8")

    def emit(**row):
        fh.write(json.dumps(row) + "\n")
        fh.flush()

    if args.part == "suite":
        from src.data import build_tasks, load_samples, prepare_image
        from validation.violations import clean_plate, make_violations

        samples = load_samples(cfg["paths"]["labels"])
        if args.samples:
            samples = [s for s in samples if s.sample_id in args.samples]
        samples = samples[: args.limit or None]
        imgs = {s.sample_id: prepare_image(Path(cfg["paths"]["images"]) / s.filename, cfg["paths"]["cache"], cfg["image"]["long_side"])
                for s in samples}
        for k, s in enumerate(samples):
            spec = build_tasks([s], tuple(cfg["targets"]), cfg["include_swap"])[0][1]
            img = imgs[s.sample_id]
            before = perceiver.perceive_cached(img, spec, s.sample_id)
            if before.subject_count == 0:
                continue
            other = samples[(k + 1) % len(samples)]
            o_per = perceiver.perceive_cached(imgs[other.sample_id], spec, other.sample_id)
            foreign = clean_plate(imgs[other.sample_id], o_per.subject_mask)
            variants = [("real", img)] + [(v.kind, v.image) for v in make_violations(img, before.subject_mask, foreign)]
            for kind, vimg in variants:
                if vimg.shape != img.shape:
                    continue
                ev, _, _ = evaluator.evaluate(before, vimg, spec)
                emit(part="suite", sample=s.sample_id, kind=kind, critics=record(ev))
            print(f"[{k + 1}/{len(samples)}] {s.sample_id}: {len(variants)} variants")
    else:
        from PIL import Image

        from src.spec import build_spec
        from validation.relight import degrade_image, relight_subject

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
                versions = {"pasted": pasted,
                            "null": relight_subject(pasted, after.subject_mask, strength=0.0, side=0.0),
                            "relit1": relight_subject(pasted, after.subject_mask, strength=0.15, side=0.10, direction=direction),
                            "relit2": relight_subject(pasted, after.subject_mask, strength=0.30, side=0.20, direction=direction),
                            "degraded": degrade_image(pasted)}
                for kind, vimg in versions.items():
                    ev, _, _ = evaluator.evaluate(before, vimg, spec)
                    emit(part="copypaste", run=src.name, task=spec.task_id, sample=sp["sample_id"], kind=kind, critics=record(ev))
                print(f"{src.name}/{spec.task_id}: {len(versions)} versions")
    fh.close()
    print(f"-> {out / 'raw.jsonl'}")


if __name__ == "__main__":
    main()
