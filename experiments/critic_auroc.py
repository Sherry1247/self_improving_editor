"""E1: do the critics detect controlled physical violations?

For each sample: positive = the real photo, negatives = one violation each (validation/violations.py).
Reports per-critic AUROC for every violation type and saves example images.

    python experiments/critic_auroc.py --name e1
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from common import base_parser, build_evaluation_stack, new_run_dir, setup, write_json


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--name", default="critic_auroc")
    ap.add_argument("--samples", nargs="*")
    args = ap.parse_args()
    cfg = setup(args)

    from src.data import build_tasks, load_samples, prepare_image, save_image
    from validation.violations import auroc, clean_plate, make_violations

    samples = load_samples(cfg["paths"]["labels"])
    if args.samples:
        samples = [s for s in samples if s.sample_id in args.samples]
    run = new_run_dir(cfg, args.name)
    _, perceiver, evaluator = build_evaluation_stack(cfg)

    imgs = {s.sample_id: prepare_image(Path(cfg["paths"]["images"]) / s.filename, cfg["paths"]["cache"],
                                       cfg["image"]["long_side"]) for s in samples}
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))  # critic -> kind -> scores
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
            save_image(vimg, run / "examples" / f"{s.sample_id}_{kind}.jpg")
            for name, c in ev.critics.items():
                if c.score is not None:
                    scores[name][kind].append(c.score)
        print(f"[{k + 1}/{len(samples)}] {s.sample_id}: {len(variants)} variants")

    kinds = sorted({k for d in scores.values() for k in d if k not in ("real", "foreign_clean")})
    table = []
    for name, d in scores.items():
        row = {"critic": name}
        for kind in kinds:
            ref = d.get("foreign_clean") if kind == "halo" else d.get("real")
            a = auroc(ref or [], d.get(kind, []))
            row[kind] = None if a is None else round(a, 3)
        table.append(row)
    with open(run / "auroc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["critic", *kinds])
        w.writeheader()
        w.writerows(table)
    write_json({k: {kk: vv for kk, vv in v.items()} for k, v in scores.items()}, run / "raw_scores.json")
    print("\nAUROC (1.0 = critic always scores the real photo higher than the violation; 0.5 = blind)")
    print("critic".ljust(16) + "".join(k[:10].rjust(11) for k in kinds))
    for row in table:
        print(row["critic"].ljust(16) + "".join(("-" if row[k] is None else f"{row[k]:.2f}").rjust(11) for k in kinds))
    print(f"\n-> {run}")


if __name__ == "__main__":
    main()
