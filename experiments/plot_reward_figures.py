"""Figures for the reward-baseline-vs-our-critics comparison on the controlled-violation suite. CPU only.

    python experiments/plot_reward_figures.py --scoreset runs/rw_violations --e1 runs/e1v2 \
        --scores runs/rw_vio_vie runs/rw_vio_vie2 runs/rw_vio_er --out docs/img

Writes reward_fig1_auroc_heatmap.png (AUROC per model x violation, * = 95% cluster-bootstrap CI above 0.5),
reward_fig2_editreward_pairs.png (null vs violation score of EditReward, one point per source image),
reward_fig3_examples.png (the actual images with their scores) and reward_fig4_score_resolution.png (ties).
Our critics come from the E1 run (raw_scores.json); a CI is drawn only for critics that have a score for all 15
images in every needed variant (others show the point estimate from auroc.csv, no marker).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.report import load_manifest, load_scores, violation_table  # noqa: E402

KINDS = ["float", "sink", "scale_up", "scale_down", "no_shadow", "color_cast", "halo"]
GROUP = {"float": "contact", "sink": "contact", "scale_up": "scale", "scale_down": "scale", "no_shadow": "shadow",
         "color_cast": "light", "halo": "edge"}


def _p(x):
    p = Path(x)
    return p if p.is_absolute() else ROOT / p


def ours_scores(e1: Path, labels_csv: Path):
    """{critic: {id: score}} for critics that have all 15 images in every variant, plus {critic: {kind: auroc}} for the rest."""
    raw = json.loads((e1 / "raw_scores.json").read_text())
    order = [Path(r["filename"]).stem for r in csv.DictReader(open(labels_csv, newline="", encoding="utf-8"))]
    full, points = {}, {}
    for crit, d in raw.items():
        if all(len(d.get(k, [])) == len(order) for k in KINDS + ["null", "foreign_clean"] if k in d) and \
                all(k in d for k in KINDS + ["null", "foreign_clean"] if k != "halo" or crit == "halo" or True) and \
                crit != "halo":
            full[crit] = {f"v/{s}/{k}": v for k, vals in d.items() for s, v in zip(order, vals)}
    with open(e1 / "auroc.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            points[r["critic"]] = {k: (float(v) if v not in ("", None) else None) for k, v in r.items() if k != "critic"}
    return full, points


def table_for(manifest, scores):
    rows = violation_table(manifest, scores, n_boot=2000)
    return {(r["model"], r["kind"]): r for r in rows}


def fig_heatmap(rows, out):
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(rows) + 2.2))
    M = np.full((len(rows), len(KINDS)), np.nan)
    for i, (_, cells) in enumerate(rows):
        for j, k in enumerate(KINDS):
            if cells.get(k) is not None:
                M[i, j] = cells[k][0]
    im = ax.imshow(M, cmap="RdBu", vmin=0.0, vmax=1.0, aspect="auto")
    for i, (_, cells) in enumerate(rows):
        for j, k in enumerate(KINDS):
            c = cells.get(k)
            if c is None:
                ax.text(j, i, "-", ha="center", va="center", color="#666")
                continue
            a, lo = c
            sig = lo is not None and lo > 0.5
            ax.text(j, i, f"{a:.2f}" + ("*" if sig else ""), ha="center", va="center", fontsize=9,
                    fontweight="bold" if sig else "normal", color="white" if abs(a - 0.5) > 0.32 else "black")
    ax.set_xticks(range(len(KINDS)), [f"{k}\n({GROUP[k]})" for k in KINDS], fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.xaxis.tick_top()
    for y in (9.5, 9.5 + 1):
        pass
    ax.set_title("Can a score tell a clean paste from a controlled violation?  AUROC (0.5 = blind, 1 = always separates)\n"
                 "* = 95% cluster-bootstrap CI over 15 source images lies above 0.5; <0.5 = violation scored HIGHER than clean",
                 fontsize=9.5, pad=40)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_pairs(manifest, er, out, name="EditReward"):
    ids = {}
    for r in manifest:
        if r["group"] == "violation":
            ids.setdefault(r["sample_id"], {})[r["kind"]] = r["id"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    stats = {}
    lo = min(er.values()) - 0.2
    hi = max(er.values()) + 0.2
    for ax, kind in zip(axes.ravel(), KINDS):
        ref = "foreign_clean" if kind == "halo" else "null"
        xs = [er[d[ref]] for d in ids.values() if ref in d and kind in d]
        ys = [er[d[kind]] for d in ids.values() if ref in d and kind in d]
        below = sum(y < x for x, y in zip(xs, ys))
        ax.plot([lo, hi], [lo, hi], color="#888", lw=1, ls="--")
        ax.scatter(xs, ys, s=34, c=["#c0392b" if y < x else "#2c7fb8" for x, y in zip(xs, ys)], edgecolor="white", lw=.6)
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
        ax.set_title(f"{kind}: {below}/{len(xs)} below the diagonal", fontsize=10)
        ax.set_xlabel(f"{'foreign_clean' if kind == 'halo' else 'clean paste (null)'} score", fontsize=8.5)
        ax.set_ylabel(f"{kind} score", fontsize=8.5)
        stats[kind] = (below, len(xs), float(np.mean(np.array(xs) - np.array(ys))))
    ax = axes.ravel()[-1]
    ax.axis("off")
    ax.text(0, .75, f"{name} score per source image\n\nred = violation scored LOWER than the\nclean paste (detected)\nblue = scored higher or equal (missed)\n\n"
                    "same source image, same cut-out /\ninpaint / paste pipeline; only one\nthing differs per panel", fontsize=9, va="top")
    fig.suptitle(f"{name}: clean paste vs violation, 15 paired source images", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return stats


def fig_examples(sdir, er, soft, samples, out):
    cols = ["real", "null", "float", "sink", "scale_up", "scale_down", "no_shadow", "color_cast", "foreign_clean", "halo"]
    fig, axes = plt.subplots(len(samples), len(cols), figsize=(2.0 * len(cols), 2.55 * len(samples)))
    for i, s in enumerate(samples):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(Image.open(sdir / "images" / f"{s}_{c}.jpg").convert("RGB"))
            ax.set_xticks([]), ax.set_yticks([])
            t = c if i == 0 else ""
            if c == "real":
                lab = "source photo"
            else:
                _id = f"v/{s}/{c}"
                lab = f"ER {er[_id]:+.2f}\nVIE {soft[_id]:.2f}"
            ax.set_title((t + "\n" if t else "") + lab if i == 0 else lab, fontsize=7.8)
        axes[i, 0].set_ylabel(s, fontsize=7.5)
    fig.suptitle("The images themselves: source photo, clean paste (null) and the seven controlled violations "
                 "(foreign_clean is the reference for halo).  ER = EditReward score, VIE = soft VIEScore-style overall (Qwen2.5-VL)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_smear(sdir, samples, out, kinds=("float", "sink", "scale_down")):
    """Crops around the region where the violations differ from the clean paste: moving / shrinking the subject uncovers the
    inpainted fill that the clean paste hides, so these violations also add a visible smear, not only the physical error."""
    cols = ["null", *kinds]
    fig, axes = plt.subplots(len(samples), len(cols), figsize=(3.3 * len(cols), 4.2 * len(samples)))
    for i, s in enumerate(samples):
        ims = {c: np.asarray(Image.open(sdir / "images" / f"{s}_{c}.jpg").convert("RGB")).astype(int) for c in cols}
        diff = np.zeros(ims["null"].shape[:2], bool)
        for k in kinds:
            diff |= np.abs(ims[k] - ims["null"]).sum(-1) > 60
        ys, xs = np.where(diff)
        h, w = diff.shape
        y0, y1 = max(0, ys.min() - 25), min(h, ys.max() + 25)
        x0, x1 = max(0, xs.min() - 25), min(w, xs.max() + 25)
        for j, c in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(ims[c][y0:y1, x0:x1].astype(np.uint8))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(("clean paste (null)" if c == "null" else c) + (f"\n{s}" if j == 0 else ""), fontsize=9)
    fig.suptitle("Same crop in every column. The clean paste covers the cut-out hole exactly; the violations move or shrink\n"
                 "the subject and so also expose the Telea-inpainted fill (blurry smear) at the old position", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_resolution(series, out):
    fig, axes = plt.subplots(1, len(series), figsize=(3.6 * len(series), 3.2))
    for ax, (name, vals) in zip(axes, series):
        vals = np.asarray(vals, float)
        ax.hist(vals, bins=30, color="#4b6a88")
        ax.set_title(f"{name}\n{len(np.unique(np.round(vals, 6)))} distinct values / {len(vals)}", fontsize=9)
        ax.set_xlabel("score", fontsize=8.5)
    axes[0].set_ylabel("items", fontsize=8.5)
    fig.suptitle("Score resolution on the 135 violation-suite items: a judge that outputs integers ties constantly", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scoreset", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--scores", nargs="+", required=True, help="run directories holding scores_*.jsonl")
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args()
    sdir, out = _p(args.scoreset), _p(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(sdir)
    files = sorted({f for d in args.scores for f in glob.glob(str(_p(d) / "scores_*.jsonl"))})
    S = load_scores(files)
    ours_full, ours_pts = ours_scores(_p(args.e1), ROOT / "data" / "labels.csv")

    chosen = {}
    for nm in ["support", "vlm_support", "vlm_shadow", "vlm_surface", "vlm_integration", "vlm_lighting", "light_harmony",
               "silhouette", "appearance"]:
        if nm in ours_full:
            chosen[f"ours: {nm}" + ("  (keep-type)" if nm in ("appearance", "silhouette") else "")] = ours_full[nm]
    base = {"EditReward": S["editreward"],
            "EditReward 2nd head (dim1)": S["editreward:dim1"],
            "VIE-soft Qwen: overall": S["viescoresoft_vlm:overall"],
            "VIE-soft Qwen: sc (sees source)": S["viescoresoft_vlm:sc"],
            "VIE-soft Qwen: pq (no source)": S["viescoresoft_vlm:pq"],
            "VIE-soft InternVL3: overall": S["viescoresoft_vlm2:overall"],
            "VIE-soft InternVL3: sc (sees source)": S["viescoresoft_vlm2:sc"],
            "VIE-soft InternVL3: pq (no source)": S["viescoresoft_vlm2:pq"],
            "VIE-int Qwen: overall": S["viescore_vlm:overall"],
            "VIE-int InternVL3: overall": S["viescore_vlm2:overall"]}
    T = table_for(manifest, {**chosen, **base})

    rows = []
    for nm in chosen:
        rows.append((nm, {k: ((T[(nm, k)]["auroc"], T[(nm, k)]["auroc_lo"]) if (nm, k) in T else None) for k in KINDS}))
    for nm in ["support", "vlm_integration", "vlm_support", "vlm_shadow", "vlm_surface", "vlm_lighting", "light_harmony"]:
        pass
    for crit in ("support", "halo"):  # critics with missing scores / halo-only: point estimates from E1 auroc.csv, no CI
        if crit in ours_pts and f"ours: {crit}" not in chosen and f"ours: {crit}  (keep-type)" not in chosen:
            rows.append((f"ours: {crit}  (no CI)", {k: ((ours_pts[crit][k], None) if ours_pts[crit].get(k) is not None else None) for k in KINDS}))
    for nm in base:
        rows.append((nm, {k: ((T[(nm, k)]["auroc"], T[(nm, k)]["auroc_lo"]) if (nm, k) in T else None) for k in KINDS}))
    fig_heatmap(rows, out / "reward_fig1_auroc_heatmap.png")
    stats = fig_pairs(manifest, S["editreward"], out / "reward_fig2_editreward_pairs.png")
    samples = []
    for pref in ("adult_sit_river_01", "dog_"):
        c = sorted({r["sample_id"] for r in manifest if r["sample_id"].startswith(pref)})
        if c:
            samples.append(c[0])
    fig_examples(sdir, S["editreward"], S["viescoresoft_vlm:overall"], samples, out / "reward_fig3_examples.png")
    fig_smear(sdir, samples, out / "reward_fig5_inpaint_smear.png")
    fig_resolution([("VIE-int Qwen overall", list(S["viescore_vlm:overall"].values())),
                    ("VIE-int InternVL3 overall", list(S["viescore_vlm2:overall"].values())),
                    ("VIE-soft Qwen overall", list(S["viescoresoft_vlm:overall"].values())),
                    ("EditReward", list(S["editreward"].values()))], out / "reward_fig4_score_resolution.png")

    summary = {"editreward_pairs_below_diag": {k: v for k, v in stats.items()}, "samples_shown": samples,
               "heatmap": {nm: {k: (None if c is None else {"auroc": c[0], "ci_lo": c[1]}) for k, c in cells.items()} for nm, cells in rows}}
    (out / "reward_figures_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["editreward_pairs_below_diag"], indent=1))
    print("samples shown:", samples)
    for nm, cells in rows:
        print(nm.ljust(40), " ".join(("  -  " if cells.get(k) is None else f"{cells[k][0]:.2f}{'*' if (cells[k][1] or 0) > .5 else ' '}") .rjust(6) for k in KINDS))


if __name__ == "__main__":
    main()
