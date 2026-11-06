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
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all",
                    help="dev = source images _01/_02, test = _03. NOTE: for detectors chosen after looking at all 15 images "
                         "(contact_dres) the test split is NOT held out; see docs/specs/2026-11-06-contact-detector-freeze.md")
    ap.add_argument("--diff", nargs=2, action="append", metavar=("A", "B"), default=[],
                    help="paired AUROC difference A - B with a cluster-bootstrap CI on the float / buried / benign pairs "
                         "(model names as in the table, e.g. ours:support editreward); may be given several times")
    ap.add_argument("--name", default="vio_compare")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    cfg = setup(args)

    from src.baselines.compare import add_combinations, compare_table, filter_split, load_all, pairs, smear_summary

    files = sorted({p for g in args.scores for p in glob.glob(g)})
    cj = json.load(open(args.critics, encoding="utf-8")) if args.critics else None
    T, smear = load_all(cj, args.scoreset, files)
    T = filter_split(T, args.split)
    smear = {k: {s: v for s, v in d.items() if args.split == "all" or (s.endswith("_03") == (args.split == "test"))}
             for k, d in smear.items()}
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
    if args.diff:
        from src.baselines.compare import paired_difference

        lines += ["", "## Paired AUROC difference A - B (cluster bootstrap over source images; CI excluding 0 = they differ)", "",
                  "| A | B | pair (pos vs neg) | diff [95% CI] | n images |", "|---|---|---|---|---|"]
        for a_, b_ in args.diff:
            for role, pk, nk in [x for x in pairs() if x[0] != "benign"] + [x for x in pairs() if x[0] == "benign"]:
                if a_ in T and b_ in T and nk in T[a_] and nk in T[b_]:
                    d, lo, hi, n = paired_difference(T, a_, b_, pk, nk, args.n_boot)
                    if d is not None:
                        lines.append(f"| {a_} | {b_} | {role}: {pk} vs {nk} | {d:+.2f} [{_f(lo)},{_f(hi)}] | {n} |")
    (out / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
