"""Why does the `support` critic call a sunk subject (6% lower) as good as -- or better than -- a clean paste?

For every sample and every variant of the v3 violation suite (null, float, sink, shift_side, shift_side_s, scale_*, ...) this logs the
quantities behind validation of contact and scores three candidate contact detectors. Needs GPU (depth + SAM2 perception), no VLM.

  features (depth is the relative inverse depth in [0, 1] of src/models, normalised per image):
    sub_d    mean depth of the subject's lowest pixels (the 85th-percentile-lowest columns, last 3% of the height)
    gnd_d    mean depth of the ground strip right below them
    gap      gnd_d - sub_d  (SIGNED; the critic uses |gap| and subtracts the same quantity of the real photo, clipped at 0)
    plane    depth of the ground plane at the feet row: a line depth~row fitted on the lower half of the image outside the subject
    res      sub_d - plane (does the subject's depth agree with the ground plane where its feet are?)
  detectors (all 'higher = more plausible', all compare with the real photo of the same sample, none is tuned):
    support_v0     the existing SupportCritic score: exp(-max(0, |gap_a| - |gap_b|) / 0.08)
    contact_dgap   exp(-| |gap_a| - |gap_b| | / 0.08)            no clipping at 0: a change in EITHER direction counts
    contact_dsgap  exp(-|gap_a - gap_b| / 0.08)                  the signed gap itself
    contact_dres   exp(-|res_a - res_b| / 0.08)                  ground-plane residual
  AUROC does not depend on the 0.08 scale (monotone transform of the same |change|), so nothing is tuned here.

    python experiments/support_diagnostics.py --name contactdiag      # then: experiments/violation_compare.py --critics runs/contactdiag/scores_by_sample.json
Writes contact_features.csv, scores_by_sample.json (same format as critic_auroc.py), contact_profiles.json.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import base_parser, build_evaluation_stack, new_run_dir, setup, write_json

SIGMA = 0.08


def lowest_columns(mask: np.ndarray):
    box_cols = np.nonzero(mask.any(0))[0]
    bottoms = np.array([np.nonzero(mask[:, c])[0].max() for c in box_cols])
    sel = bottoms >= np.percentile(bottoms, 85)
    return box_cols[sel], bottoms[sel]


def contact_features(depth: np.ndarray, mask: np.ndarray) -> dict | None:
    """Signed support gap and ground-plane residual at the feet; None if no ground is visible below the subject."""
    h, w = mask.shape
    rows = np.nonzero(mask.any(1))[0]
    if len(rows) < 10 or mask[-3:].any():
        return None
    sh = rows.max() - rows.min() + 1
    band = max(2, int(0.03 * sh))
    cols, bottoms = lowest_columns(mask)
    sub, gnd = [], []
    for c, b in zip(cols, bottoms):
        sub.append(depth[max(0, b - band):b + 1, c].mean())
        below = depth[min(h - 1, b + 1):min(h, b + 1 + band), c]
        if below.size:
            gnd.append(below.mean())
    if not gnd:
        return None
    # ground plane: depth ~ a * row + b on the lower half of the image, outside a margin around the subject
    yy, xx = np.mgrid[0:h, 0:w]
    x1, x2 = np.nonzero(mask.any(0))[0][[0, -1]]
    margin = int(0.15 * (x2 - x1 + 1))
    ground = (yy >= h // 2) & ~((xx >= x1 - margin) & (xx <= x2 + margin))
    if ground.sum() < 200:
        return None
    a, b0 = np.polyfit(yy[ground].astype(float), depth[ground].astype(float), 1)
    feet_row = float(np.mean(bottoms))
    sub_d, gnd_d = float(np.mean(sub)), float(np.mean(gnd))
    return {"sub_d": sub_d, "gnd_d": gnd_d, "gap": gnd_d - sub_d, "plane": float(a * feet_row + b0),
            "res": sub_d - float(a * feet_row + b0), "feet_row": feet_row, "slope": float(a)}


def detectors(fa: dict, fb: dict) -> dict[str, float]:
    e = lambda x: float(np.exp(-x / SIGMA))
    return {
        "support_v0": e(max(0.0, abs(fa["gap"]) - abs(fb["gap"]))),
        "contact_dgap": e(abs(abs(fa["gap"]) - abs(fb["gap"]))),
        "contact_dsgap": e(abs(fa["gap"] - fb["gap"])),
        "contact_dres": e(abs(fa["res"] - fb["res"])),
    }


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--name", default="contactdiag")
    ap.add_argument("--samples", nargs="*")
    args = ap.parse_args()
    cfg = setup(args)
    cfg["critics"]["enabled"] = ["support"]  # nothing here needs the VLM critics

    from src.data import build_tasks, load_samples, prepare_image
    from validation.violations import make_violations

    samples = load_samples(cfg["paths"]["labels"])
    if args.samples:
        samples = [s for s in samples if s.sample_id in args.samples]
    run = new_run_dir(cfg, args.name)
    _, perceiver, _ = build_evaluation_stack(cfg)

    rows = []
    by_sample: dict = defaultdict(lambda: defaultdict(dict))
    profiles: dict = {}
    for k, s in enumerate(samples):
        img = prepare_image(Path(cfg["paths"]["images"]) / s.filename, cfg["paths"]["cache"], cfg["image"]["long_side"])
        spec = build_tasks([s], tuple(cfg["targets"]), cfg["include_swap"])[0][1]
        before = perceiver.perceive_cached(img, spec, s.sample_id)
        if before.subject_count == 0 or before.depth is None:
            continue
        fb = contact_features(before.depth, before.subject_mask)
        if fb is None:
            print(f"[{k + 1}/{len(samples)}] {s.sample_id}: no ground below the real subject, skipped")
            continue
        for v in [("real", img, None)] + [(v.kind, v.image, v.params) for v in make_violations(img, before.subject_mask)]:
            kind, vimg = v[0], v[1]
            after = before if kind == "real" else perceiver.perceive(vimg, spec)
            fa = contact_features(after.depth, after.subject_mask)
            if fa is None:
                continue
            row = {"sample": s.sample_id, "kind": kind, **{f"{n}_a": x for n, x in fa.items()},
                   **{f"{n}_b": x for n, x in fb.items()}}
            for name, sc in detectors(fa, fb).items():
                row[name] = sc
                by_sample[name][kind][s.sample_id] = sc
            rows.append(row)
            if s.sample_id == samples[0].sample_id:  # one example: depth along the subject's centre column, for the figure
                cx = int(np.nonzero(after.subject_mask.any(0))[0].mean())
                profiles[kind] = {"depth_col": [round(float(x), 4) for x in after.depth[:, cx]],
                                  "mask_rows": [int(r) for r in np.nonzero(after.subject_mask[:, cx])[0][[0, -1]]]}
        print(f"[{k + 1}/{len(samples)}] {s.sample_id}: ok")

    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(run / "contact_features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    write_json({"scores": by_sample, "smear_visible": {}}, run / "scores_by_sample.json")
    write_json(profiles, run / "contact_profiles.json")

    # quick summary: mean signed gap / residual change per kind (the sign explains the abs() reversal)
    print("\nmean change vs the real photo (a - b)   kind: d(gap)  d(|gap|)  d(res)   n")
    for kind in sorted({r["kind"] for r in rows}):
        rr = [r for r in rows if r["kind"] == kind]
        print(f"{kind:13s} {np.mean([r['gap_a'] - r['gap_b'] for r in rr]):+.3f} "
              f"{np.mean([abs(r['gap_a']) - abs(r['gap_b']) for r in rr]):+.3f} "
              f"{np.mean([r['res_a'] - r['res_b'] for r in rr]):+.3f}  {len(rr)}")
    print(f"\n-> {run}")


if __name__ == "__main__":
    main()
