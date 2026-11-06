"""Matched-control comparison on the v3 violation suite: our critics vs reward baselines, with CIs and combinations. CPU only.

    python experiments/violation_compare.py --critics runs/e1v3/scores_by_sample.json --scoreset runs/rw_v3 \
        --scores runs/rw_v3_er/scores_editreward.jsonl runs/rw_v3_vie/scores_viescore_vlm.jsonl --name vio_compare

Writes compare.csv and compare.md: per model AUROC (95% cluster bootstrap over source images) for
  detect (null vs violation), benign (null vs shift_side*: should be ~0.5), contrast (shift_side vs float, shift_side_s vs sink),
and, for ``--combine-ref`` (default editreward), the z-score mean with each of our critics (exploratory, unfitted).
"""

from __future__ import annotations

import csv
import glob
import json

from common import base_parser, new_run_dir, setup


def _f(x):
    return "-" if x is None else f"{x:.2f}"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--critics", help="runs/<e1>/scores_by_sample.json")
    ap.add_argument("--scoreset")
    ap.add_argument("--scores", nargs="*", default=[], help="scores_*.jsonl (globs ok)")
    ap.add_argument("--combine-ref", default="editreward", help="reward column to combine with each of our critics ('' = none)")
    ap.add_argument("--name", default="vio_compare")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    cfg = setup(args)

    from src.baselines.compare import add_combinations, compare_table, load_all, pairs, smear_summary

    files = sorted({p for g in args.scores for p in glob.glob(g)})
    cj = json.load(open(args.critics, encoding="utf-8")) if args.critics else None
    T, smear = load_all(cj, args.scoreset, files)
    if args.combine_ref and args.combine_ref in T:
        T = add_combinations(T, args.combine_ref)
    rows = compare_table(T, args.n_boot)
    out = new_run_dir(cfg, args.name)
    with open(out / "compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "role", "pos", "neg", "n_images", "auroc", "lo", "hi"])
        w.writeheader()
        w.writerows(rows)

    cols = [(r, p, n) for r, p, n in pairs()]
    head = "| model | " + " | ".join(f"{r}:{n}" for r, p, n in cols) + " |"
    lines = ["# Violation suite v3: matched controls, CIs, combinations\n",
             "AUROC [95% cluster CI over source images]. detect = null vs violation; benign = null vs sideways shift "
             "(~0.5 expected); contrast = shift_side vs float / shift_side_s vs sink (same displacement, only contact differs).\n"]
    if smear:
        lines.append("Share of the inpainted region left uncovered (smear covariate): " +
                     ", ".join(f"{k} {v:.2f}" for k, v in sorted(smear_summary(smear).items())) + "\n")
    lines += [head, "|---|" + "---|" * len(cols)]
    by = {(r["model"], r["role"], r["neg"]): r for r in rows}
    for m in sorted({r["model"] for r in rows}):
        cells = []
        for r, p, n in cols:
            x = by.get((m, r, n))
            cells.append("-" if x is None else f"{_f(x['auroc'])} [{_f(x['lo'])},{_f(x['hi'])}]")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (out / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
