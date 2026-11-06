"""Figure for the VLM-stability study, read from a variance_report.md (runs/vlmvar/variance_report.md or the saved copy
in docs/results/data/). Left: per question, share of the 90 tasks whose comp-vs-qwen winner is the same in every perturbed
view, against the 80% rule set before the run. Right: qwen wins minus comp wins per view (aggregate direction).

    python experiments/plot_vlmvar_figure.py --report docs/results/data/2026-11-05-vlmvar-variance-report.md --out docs/img/vlmvar_fig1_stability.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VIEWS = ["orig", "para1", "para2", "pad_small", "pad_large", "side512"]


def parse(path: Path):
    stable, wins = {}, {}
    in_sec = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and "wins of" in line:
            in_sec = True
            continue
        if in_sec and line.startswith("|") and not line.startswith("|---") and "question" not in line:
            c = [x.strip() for x in line.strip("|").split("|")]
            q, tasks, st = c[0], int(c[1]), int(c[2])
            stable[q] = (st, tasks)
            wins[q] = [tuple(int(v) for v in x.split("/")) for x in c[3:3 + len(VIEWS)]]
    return stable, wins


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stable, wins = parse(Path(a.report))
    qs = list(stable)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    share = [100 * stable[q][0] / stable[q][1] for q in qs]
    ax1.bar(qs, share, color="#4b6a88")
    ax1.axhline(80, color="#c0392b", ls="--", lw=1.2)
    ax1.text(len(qs) - 0.5, 82, "80% rule", color="#c0392b", ha="right", fontsize=9)
    for i, (q, s) in enumerate(zip(qs, share)):
        ax1.text(i, s + 1.5, f"{stable[q][0]}/{stable[q][1]}", ha="center", fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("tasks with the same winner in every perturbed view (%)")
    ax1.set_title("Per-task winner stability (comp vs qwen)", fontsize=10.5)
    ax1.tick_params(axis="x", rotation=25)
    w = 0.8 / len(VIEWS)
    for k, v in enumerate(VIEWS):
        d = [wins[q][k][1] - wins[q][k][0] for q in qs]  # qwen wins - comp wins
        ax2.bar(np.arange(len(qs)) + (k - (len(VIEWS) - 1) / 2) * w, d, w, label=v)
    ax2.axhline(0, color="black", lw=.8)
    ax2.set_xticks(range(len(qs)), qs)
    ax2.set_ylabel("qwen wins - comp wins (tasks)")
    ax2.set_title("Aggregate direction per view (>0: qwen ahead)", fontsize=10.5)
    ax2.legend(ncol=3, fontsize=8, title="view")
    fig.suptitle("VLM critic answers under question / crop / resolution perturbations (Qwen2.5-VL-7B, 90 tasks)", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170)
    print({q: (stable[q], [x[1] - x[0] for x in wins[q]]) for q in qs})


if __name__ == "__main__":
    main()
