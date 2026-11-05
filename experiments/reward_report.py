"""Tabulate reward-model baselines against our critics. CPU only.

    python experiments/reward_report.py --scoreset runs/rw_violations --name rw_violations_report \
        --scores runs/rw_violations_vie/scores_viescore_vlm.jsonl runs/rw_violations_er/scores_editreward.jsonl \
        --critics-auroc runs/e1v2/auroc.csv

Violation sets -> reward_violations.csv: AUROC (null vs violation; foreign_clean vs halo) and paired accuracy, with 95 %
cluster-bootstrap CIs over source images, next to our critics' E1 AUROCs. Run sets -> reward_runs.csv: mean per run,
pairwise wins and Spearman with our overall score. Also writes reward_report.md.
"""

from __future__ import annotations

import csv
import glob

from common import base_parser, new_run_dir, setup


def _w(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--scoreset", required=True)
    ap.add_argument("--scores", nargs="+", required=True, help="scores_*.jsonl files (globs ok)")
    ap.add_argument("--critics-auroc", help="runs/e1*/auroc.csv of our critics, appended to the violation table")
    ap.add_argument("--name", default="reward_report")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    cfg = setup(args)

    from src.baselines.report import load_manifest, load_scores, our_critics_rows, run_table, violation_table

    files = sorted({p for g in args.scores for p in glob.glob(g)})
    manifest = load_manifest(args.scoreset)
    scores = load_scores(files)
    out = new_run_dir(cfg, args.name)
    lines = [f"# Reward baselines vs our critics\n\nscore files: {', '.join(files)}\n"]

    vrows = violation_table(manifest, scores, args.n_boot)
    if vrows:
        _w(out / "reward_violations.csv", vrows)
        kinds = sorted({r["kind"] for r in vrows})
        lines += ["## Violation suite (AUROC, 0.5 = blind; paired accuracy in brackets, 95% cluster CI over source images)\n",
                  "| model | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
        for m in sorted({r["model"] for r in vrows}):
            cells = []
            for k in kinds:
                r = next((x for x in vrows if x["model"] == m and x["kind"] == k), None)
                cells.append("-" if r is None else f"{_f(r['auroc'])} [{_f(r['auroc_lo'])},{_f(r['auroc_hi'])}] ({_f(r['paired_acc'])})")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        if args.critics_auroc:
            ours = our_critics_rows(args.critics_auroc)
            _w(out / "ours_violations.csv", ours)
            lines += ["\n## Our critics (E1, cross-sample AUROC, no CI)\n", "| critic | " + " | ".join(kinds) + " |",
                      "|---|" + "---|" * len(kinds)]
            for m in sorted({r["model"] for r in ours}):
                lines.append(f"| {m} | " + " | ".join(
                    _f(next((x["auroc"] for x in ours if x["model"] == m and x["kind"] == k), None)) for k in kinds) + " |")
    rrows = run_table(manifest, scores)
    if rrows:
        _w(out / "reward_runs.csv", rrows)
        lines += ["\n## Editor runs (see reward_runs.csv: mean per run, pairwise wins, Spearman with our overall)\n"]
        for r in rrows:
            lines.append("- " + r["model"] + ": " + ", ".join(f"{k}={_f(v) if isinstance(v, float) else v}"
                                                              for k, v in r.items() if k != "model"))
    (out / "reward_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
