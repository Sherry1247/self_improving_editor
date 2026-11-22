"""Item 1: is the paper's light normalisation L (Eq. 5) a valid way to separate 'lighting changed' from 'subject changed'?

Claim under test (Equilibrium Is All You Need, Sec. 3): identity / appearance are compared on L(x) (per-channel standardisation inside
the subject mask), so a change of illumination is not counted as a change of the system, while a change of texture, pattern, colour
of the material or shape is. No editor is involved: every variant is made from the SAME real photo by a known intervention
(validation/interventions.py), so the ground truth is built in, as in the violation suite.

For each of the 15 photos: null (zero-strength relight, same pipeline), nuisance (blur + shift + JPEG), LIGHT variants (gain, side light,
colour cast, combined relight; levels 1-3) and STRUCT variants (blur, texture shuffle, hue rotation, mirror; levels 1-3). Every variant
is perceived again (Grounding DINO + SAM2 + DINOv2 + depth + SigLIP; no VLM) and scored against the original photo by

  repo measures    silhouette IoU, appearance v3 (high-pass ZNCC x chroma), appearance v2 (Reinhard Lab distance), texture (DINOv2 patch cosine),
                   identity (raw DINOv2 CLS cosine of the masked crop, as in src/critics/gate.py)
  paper measures   ksil = |M n M'| / |M| (Eq. 7), kapp_lab / kapp_rgb = cos(phi(L x), phi(L x')) with L = per-channel standardisation
                   in Lab / RGB (Eq. 5 + 7); the same DINOv2 crop encoder as the identity gate, fed the standardised subject

CRITERIA, fixed before the first run (nothing here is tuned; thresholds are not fitted on this data):
  P1  does not penalise lighting:   AUROC(null > LIGHT variants of level 1-2) <= 0.65     (0.5 = lighting is invisible to the measure)
  P2  does penalise structure:      AUROC(null > STRUCT variants of level 2-3) >= 0.85
  P3  separates the two:            P(score of a LIGHT level 1-2 variant > score of a STRUCT level 2-3 variant of the same photo) >= 0.85
  P0  nuisance sanity:              AUROC(null > nuisance) <= 0.65   (an editor's re-encoding must not look like a change)
A measure PASSES when P1, P2 and P3 all hold with the point estimate; it is reported as 'clearly' when the 95% cluster-bootstrap CI
(over the 15 photos) also respects the threshold. The paper's claim is supported if kapp_lab or kapp_rgb passes and the raw identity
cosine (no L) does not, i.e. L is what makes the difference. Dev = photos *_01, *_02; test = *_03 (reported separately, once).

    python experiments/lighting_intervention.py --name lint1
Writes variants.csv (every variant x measure), summary.csv / summary.json (criteria), by_kind.csv (mean score drop per kind and level),
scores_by_sample.json, and examples/<photo>_<variant>.jpg for the first two photos.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import base_parser, new_run_dir, setup, write_json

MEASURES = ["silhouette", "ksil_paper", "appearance_v3", "appearance_v2", "texture", "identity_raw", "kapp_lab", "kapp_rgb"]
THRESH = {"P0": 0.65, "P1": 0.65, "P2": 0.85, "P3": 0.85}


def cos(a, b) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def ksil_paper(regions, after_mask) -> float | None:
    b = regions.aligned_before_subject
    return float((b & after_mask).sum() / b.sum()) if b.any() else None


def split_of(sample_id: str) -> str:
    return "test" if sample_id.endswith("_03") else "dev"


def summarize(rows: list[dict], measure: str, samples: list[str]) -> dict:
    """Criteria for one measure over the given photos. rows: variants.csv rows (dicts)."""
    from src.baselines.stats import auroc, cluster_bootstrap

    by = defaultdict(lambda: defaultdict(list))   # sample -> group -> [scores]
    for r in rows:
        v = r.get(measure)
        if r["sample"] not in samples or v is None:
            continue
        f, lv = r["family"], int(r["level"])
        by[r["sample"]][f].append(v)
        if f == "light" and lv <= 2:
            by[r["sample"]]["light12"].append(v)
        if f == "struct" and lv >= 2:
            by[r["sample"]]["struct23"].append(v)
    out = {}
    clusters = {"P0": {}, "P1": {}, "P2": {}, "P3": {}}
    for s, g in by.items():
        if g["null"]:
            if g["nuisance"]:
                clusters["P0"][s] = (g["null"], g["nuisance"])
            if g["light12"]:
                clusters["P1"][s] = (g["null"], g["light12"])
            if g["struct23"]:
                clusters["P2"][s] = (g["null"], g["struct23"])
        if g["light12"] and g["struct23"]:
            clusters["P3"][s] = (g["light12"], g["struct23"])
    for p, cl in clusters.items():
        pt, lo, hi = cluster_bootstrap(cl, auroc, n_boot=2000, seed=0)
        out[p] = {"auroc": pt, "lo": lo, "hi": hi, "n_photos": len(cl)}
    ok = lambda p, cmp: out[p]["auroc"] is not None and cmp(out[p]["auroc"], THRESH[p])  # noqa: E731
    out["pass_P1"], out["pass_P2"], out["pass_P3"] = ok("P1", lambda a, t: a <= t), ok("P2", lambda a, t: a >= t), ok("P3", lambda a, t: a >= t)
    out["pass_P0"] = ok("P0", lambda a, t: a <= t)
    out["PASS"] = bool(out["pass_P1"] and out["pass_P2"] and out["pass_P3"])
    out["PASS_clearly"] = bool(out["PASS"] and out["P1"]["hi"] is not None and out["P1"]["hi"] <= THRESH["P1"]
                               and out["P2"]["lo"] is not None and out["P2"]["lo"] >= THRESH["P2"]
                               and out["P3"]["lo"] is not None and out["P3"]["lo"] >= THRESH["P3"])
    return out


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--name", default="lighting_intervention")
    ap.add_argument("--samples", nargs="*")
    ap.add_argument("--examples", type=int, default=2, help="save every variant image for the first N photos")
    args = ap.parse_args()
    cfg = setup(args)
    cfg["perception"]["use_vlm"] = False            # no VLM here: only the Keep-type measures are tested

    from src.critics import CriticContext, build_critics
    from src.critics.keep import AppearanceCritic
    from src.data import build_tasks, load_samples, prepare_image, save_image
    from src.models import ModelRegistry
    from src.perception import Perceiver
    from src.regions import partition
    from validation.interventions import build_variants, l_image
    from validation.violations import clean_plate

    samples = load_samples(cfg["paths"]["labels"])
    if args.samples:
        samples = [s for s in samples if s.sample_id in args.samples]
    run = new_run_dir(cfg, args.name)
    cfg["critics"]["enabled"] = ["identity", "silhouette", "appearance", "texture"]
    critics = {c.name: c for c in build_critics(cfg)}
    app_v2 = AppearanceCritic(**{**cfg["critics"].get("appearance", {}), "variant": "v2"})
    perceiver = Perceiver(ModelRegistry(cfg), cfg)

    rows: list[dict] = []
    for k, s in enumerate(samples):
        img = prepare_image(Path(cfg["paths"]["images"]) / s.filename, cfg["paths"]["cache"], cfg["image"]["long_side"])
        spec = build_tasks([s], tuple(cfg["targets"]), cfg["include_swap"])[0][1]
        before = perceiver.perceive_cached(img, spec, s.sample_id)
        if before.subject_count == 0:
            print(f"[{k + 1}/{len(samples)}] {s.sample_id}: no subject, skipped")
            continue
        regions_before = partition(before.subject_mask, before.subject_mask)
        plate = clean_plate(img, before.subject_mask)
        emb_before = {sp: perceiver.dino(l_image(img, before.subject_mask, sp), before.subject_mask)[0] for sp in ("lab", "rgb")}
        for v in build_variants(img, before.subject_mask, plate, bg_mask=None):
            after = perceiver.perceive(v.image, spec)
            regions = partition(after.subject_mask, before.subject_mask)
            ctx = CriticContext(before, after, regions, spec, regions_before)
            res = {n: c.evaluate(ctx) for n, c in critics.items()}
            row = {"sample": s.sample_id, "split": split_of(s.sample_id), "family": v.family, "kind": v.kind, "name": v.name,
                   "level": v.level, "subject_found": int(after.subject_count > 0 and after.subject_mask.any())}
            if row["subject_found"]:
                row["silhouette"] = res["silhouette"].score
                row["ksil_paper"] = ksil_paper(regions, after.subject_mask)
                row["appearance_v3"] = res["appearance"].score          # None when the alignment is unreliable
                row["appearance_v2"] = app_v2.evaluate(ctx).score
                row["texture"] = res["texture"].score
                row["identity_raw"] = res["identity"].evidence.get("cosine")
                row["texture_cos"] = res["texture"].evidence.get("patch_cosine")
                row["app_zncc"] = res["appearance"].evidence.get("zncc")
                row["app_chroma"] = res["appearance"].evidence.get("chroma")
                for sp in ("lab", "rgb"):
                    emb_a = perceiver.dino(l_image(v.image, after.subject_mask, sp), after.subject_mask)[0]
                    row[f"kapp_{sp}"] = cos(emb_before[sp], emb_a)
            rows.append(row)
            if k < args.examples:
                save_image(v.image, run / "examples" / f"{s.sample_id}_{v.name}.jpg")
        print(f"[{k + 1}/{len(samples)}] {s.sample_id}: {sum(r['sample'] == s.sample_id for r in rows)} variants scored")

    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(run / "variants.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    sample_ids = sorted({r["sample"] for r in rows})
    groups = {"all": sample_ids, "dev": [x for x in sample_ids if split_of(x) == "dev"], "test": [x for x in sample_ids if split_of(x) == "test"]}
    summary = {g: {m: summarize(rows, m, ids) for m in MEASURES} for g, ids in groups.items()}
    write_json(summary, run / "summary.json")
    with open(run / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "measure", "P0_nuisance", "P1_light", "P1_lo", "P1_hi", "P2_struct", "P2_lo", "P2_hi", "P3_sep", "P3_lo", "P3_hi", "PASS", "PASS_clearly"])
        fmt = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
        for g in groups:
            for m in MEASURES:
                r = summary[g][m]
                w.writerow([g, m, fmt(r["P0"]["auroc"]), fmt(r["P1"]["auroc"]), fmt(r["P1"]["lo"]), fmt(r["P1"]["hi"]),
                            fmt(r["P2"]["auroc"]), fmt(r["P2"]["lo"]), fmt(r["P2"]["hi"]),
                            fmt(r["P3"]["auroc"]), fmt(r["P3"]["lo"]), fmt(r["P3"]["hi"]), r["PASS"], r["PASS_clearly"]])

    # dose-response: mean (null - variant) per kind and level, per measure; and per-photo scores for later figures
    drop = defaultdict(lambda: defaultdict(list))
    null_by = {(r["sample"]): r for r in rows if r["family"] == "null"}
    for r in rows:
        if r["family"] in ("null",) or r["sample"] not in null_by:
            continue
        for m in MEASURES:
            if r.get(m) is not None and null_by[r["sample"]].get(m) is not None:
                drop[(r["family"], r["kind"], r["level"])][m].append(null_by[r["sample"]][m] - r[m])
    with open(run / "by_kind.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "kind", "level", "n", *[f"drop_{m}" for m in MEASURES]])
        for key in sorted(drop):
            n = max(len(v) for v in drop[key].values())
            w.writerow([*key, n, *[("" if not drop[key][m] else f"{np.mean(drop[key][m]):.3f}") for m in MEASURES]])
    by_sample: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        for m in MEASURES:
            if r.get(m) is not None:
                by_sample[m][r["name"]][r["sample"]] = r[m]
    write_json({"scores": by_sample}, run / "scores_by_sample.json")

    print("\ncriteria (all photos). AUROC: 0.5 = measure does not see the intervention, 1.0 = always scores it below the null")
    print("measure".ljust(15) + "P0 nuis".rjust(9) + "P1 light".rjust(10) + "P2 struct".rjust(11) + "P3 sep".rjust(9) + "   PASS")
    for m in MEASURES:
        r = summary["all"][m]
        f = lambda p: "  -  " if r[p]["auroc"] is None else f"{r[p]['auroc']:.2f}"  # noqa: E731
        print(m.ljust(15) + f("P0").rjust(9) + f("P1").rjust(10) + f("P2").rjust(11) + f("P3").rjust(9)
              + ("   PASS (clearly)" if r["PASS_clearly"] else "   PASS" if r["PASS"] else "   no"))
    print(f"\n-> {run}")


if __name__ == "__main__":
    main()
