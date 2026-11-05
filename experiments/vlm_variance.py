"""How stable are the VLM critics' P(Yes) answers, and which judge do they depend on? (GPU; scores only, no editing)

    python experiments/vlm_variance.py --src runs/comp_v2 runs/qwen --name vlmvar --vlm-kind vlm
    python experiments/vlm_variance.py --src runs/comp_v2 runs/qwen --name vlmvar2 --vlm-kind vlm2     # second VLM
    python experiments/vlm_variance.py --report-only runs/vlmvar/variance.csv --name vlmvar_report

The critics read P(Yes) from logits, so re-running the same input repeats the same number; 'sampling 3 times' would be
pointless. Instead every question is asked under several VIEWS (original wording, two paraphrases, tighter / looser crop,
lower resolution) plus an exact repeat (orig_rep = nondeterminism floor), on both the before and the best image.
Outputs: variance.csv (long format), variance_summary.json, variance_report.md. With >= 2 runs it also reports whether
'run A beats run B' on each question survives the perturbations.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

from common import ROOT, base_parser, build_evaluation_stack, new_run_dir, setup

FIELDS = ["run", "task", "sample_id", "image", "question", "view", "p_yes"]


def write_report(rows: list[dict], out: Path) -> None:
    from src.baselines.variance import ranking_stability, summarise

    s = summarise(rows)
    (out / "variance_summary.json").write_text(json.dumps(s, indent=1))
    f = lambda x: "-" if x is None else f"{x:.3f}"
    L = ["# VLM answer stability\n", f"Repeat floor (same input twice): mean {f(s['repeat_floor_mean'])}, "
         f"max {f(s['repeat_floor_max'])}\n", "| question | n | mean std over views | mean range | max range | mean |view-orig| |",
         "|---|---|---|---|---|---|"]
    for q, d in sorted(s["questions"].items()):
        L.append(f"| {q} | {d['n']} | {f(d['mean_std'])} | {f(d['mean_range'])} | {f(d['max_range'])} | {f(d['mean_dev_from_orig'])} |")
    runs = sorted({r["run"] for r in rows})
    for a, b in combinations(runs, 2):
        L += [f"\n## {a} vs {b} on the best images: wins of {a} per view (sign-stable = same winner in every perturbed view)\n",
              "| question | tasks | sign-stable tasks | " + " | ".join(["orig", "para1", "para2", "pad_small", "pad_large", "side512"]) + " |",
              "|---|---|---|---|---|---|---|---|---|"]
        for q, d in sorted(ranking_stability(rows, a, b).items()):
            cells = []
            for v in ["orig", "para1", "para2", "pad_small", "pad_large", "side512"]:
                x = d["views"].get(v)
                cells.append("-" if not x else f"{x['a_wins']}/{x['b_wins']}")
            L.append(f"| {q} | {d['tasks']} | {d['sign_stable_tasks']} | " + " | ".join(cells) + " |")
    (out / "variance_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--src", nargs="*", default=[], help="finished run directories (before.png / best.png / report.json)")
    ap.add_argument("--name", default="vlmvar")
    ap.add_argument("--vlm-kind", default="vlm", choices=["vlm", "vlm2"])
    ap.add_argument("--limit", type=int, default=0, help="tasks per run (smoke tests)")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--report-only", help="variance.csv from an earlier run")
    args = ap.parse_args()
    cfg = setup(args)
    out = new_run_dir(cfg, args.name)
    if args.report_only:
        rows = [{**r, "p_yes": float(r["p_yes"]) if r["p_yes"] not in ("", "None") else None}
                for r in csv.DictReader(open(args.report_only, newline="", encoding="utf-8"))]
        write_report(rows, out)
        return
    cfg["perception"]["use_vlm"] = False  # masks only; this script asks the questions itself

    import numpy as np
    from PIL import Image

    from src.baselines.variance import VIEWS, paraphrases, view_params
    from src.perception.vlm import VLMScorer, build_questions, question_images
    from src.spec import build_spec

    registry, perceiver, _ = build_evaluation_stack(cfg)
    rows: list[dict] = []
    for src in args.src:
        src = Path(src) if Path(src).is_absolute() else ROOT / src
        done = 0
        for rep_path in sorted(src.glob("*/report.json")):
            d = rep_path.parent
            if not (d / "best.png").exists() or not (d / "before.png").exists():
                continue
            sp = json.loads(rep_path.read_text())["spec"]
            if args.split != "all" and (("test" if sp["sample_id"].endswith("_03") else "dev") != args.split):
                continue
            if args.limit and done >= args.limit:
                break
            done += 1
            spec = build_spec(sp["sample_id"], "dog" if sp["subject"] == "dog" else "adult_person",
                              "sit" if sp["pose"] == "sitting" else "stand", sp["source_bg"]["key"],
                              sp["target_bg"]["key"], sp["submerged"])
            imgs = {"before": np.asarray(Image.open(d / "before.png").convert("RGB")),
                    "best": np.asarray(Image.open(d / "best.png").convert("RGB"))}
            masks = {"before": perceiver.perceive_cached(imgs["before"], spec, sp["sample_id"]).subject_mask,
                     "best": perceiver.perceive(imgs["best"], spec).subject_mask}
            texts, qs = paraphrases(spec), build_questions(spec)
            with registry.use(args.vlm_kind) as b:
                scorer = VLMScorer(b)
                for image, img in imgs.items():
                    for q in qs:
                        if q.region != "full" and not masks[image].any():
                            continue
                        for view in VIEWS:
                            prm = view_params(view)
                            p = scorer.p_yes(question_images(img, masks[image], q.region, prm["max_side"], prm["pad"]),
                                             texts[q.key][prm["para"]])
                            rows.append({"run": src.name, "task": d.name, "sample_id": sp["sample_id"], "image": image,
                                         "question": q.key, "view": view, "p_yes": p})
            print(f"{src.name}/{d.name}: {len(rows)} answers so far")
    with open(out / "variance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    write_report(rows, out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
