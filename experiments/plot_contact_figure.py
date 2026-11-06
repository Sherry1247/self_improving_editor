"""Forest plot for the contact question on violation suite v3: float / sink vs the matched benign shift, and the benign shifts.

Numbers are copied from CHTC runs/vio_er/compare.md (EditReward, existing critics) and runs/contactdiag_cmp/compare.md
(candidate contact detectors of experiments/support_diagnostics.py), 2026-11-06; AUROC with 95% cluster-bootstrap CI over 15 source images.

    python experiments/plot_contact_figure.py --out docs/img/contact_fig1_matched_controls.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# v4 (protected ground strip; docs/results/2026-11-06-violation-suite-v4-results.md): runs/vio4/compare.md and runs/cdiag4_cmp/compare.md
DATA_V4 = {
    "EditReward": ((.60, .44, .75), (.47, .31, .61), (.64, .49, .79), (.59, .47, .72)),
    "support (existing)": ((.84, .66, .98), (.54, .35, .72), (.50, .35, .65), (.46, .32, .59)),
    "vlm_support": ((.82, .68, .94), (.58, .46, .70), (.55, .43, .69), (.50, .39, .62)),
    "vlm_surface": ((.77, .61, .90), (.36, .18, .55), (.63, .53, .76), (.58, .47, .70)),
    "contact_dres (frozen)": ((.80, .60, .94), (.54, .37, .72), (.66, .49, .84), (.58, .42, .74)),
}
# v3 numbers, kept for the comparison. model: (contrast float, contrast sink, benign shift_side, benign shift_side_s)  each (auroc, lo, hi)
DATA = {
    "EditReward": ((.63, .46, .79), (.52, .39, .66), (.54, .40, .67), (.46, .33, .57)),
    "support (existing)": ((.77, .55, .94), (.57, .38, .74), (.37, .18, .56), (.32, .13, .52)),
    "vlm_support": ((.82, .66, .94), (.52, .42, .61), (.36, .20, .54), (.25, .11, .38)),
    "vlm_surface": ((.82, .66, .94), (.38, .18, .57), (.41, .21, .64), (.33, .13, .60)),
    "contact_dres (new)": ((.81, .62, .96), (.49, .31, .66), (.50, .34, .65), (.48, .32, .65)),
}
TITLES = ["float vs shift_side\n(want > 0.5)", "sink vs shift_side_s\n(want > 0.5)", "null vs shift_side\n(want = 0.5)",
          "null vs shift_side_s\n(want = 0.5)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", choices=["v3", "v4"], default="v4")
    a = ap.parse_args()
    DATA_ = DATA_V4 if a.suite == "v4" else DATA
    names = list(DATA_)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for j, ax in enumerate(axes):
        for i, n in enumerate(names):
            p, lo, hi = DATA_[n][j]
            sig = j < 2 and lo > 0.5
            ax.errorbar(p, len(names) - 1 - i, xerr=[[p - lo], [hi - p]], fmt="o", color="#c0392b" if sig else "#4b6a88",
                        capsize=3, ms=6)
        ax.axvline(0.5, color="grey", ls="--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_title(TITLES[j], fontsize=10)
        ax.set_xlabel("AUROC [95% CI]")
    axes[0].set_yticks(range(len(names)), names[::-1])
    fig.suptitle(f"Contact, violation suite {a.suite}: same displacement, only ground contact differs "
                 "(red = CI above 0.5). 15 source images; sink is ambiguous, not a contact violation", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170)


if __name__ == "__main__":
    main()
