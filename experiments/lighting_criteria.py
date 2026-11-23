"""Magnitude criteria for the lighting-intervention variants (re-analysis of runs/<name>/variants.csv; no GPU, no models).

Why this replaces the rank-vs-null criteria P0-P3 of experiments/lighting_intervention.py: the null is the pixel-identical photo, so every
variant (even the harmless re-encoding) ranks below it and AUROC against the null is 1.00 for everything. The question that matters is
whether a change is larger than what an editor's re-encoding does to an unchanged subject. For one photo, a variant is BEYOND NOISE for a
measure when its score is below the score of the same photo's nuisance variant (blur 0.8 + sub-pixel shift + JPEG 80). This equals
"drop vs null > nuisance drop", is unchanged by any monotone rescaling of the measure (so no ramp or threshold is tuned here), and a tie is
not counted as beyond noise.

CRITERIA (written before the lint2 run; read the honesty note below):
  M1  lighting stays inside the noise:  share of LIGHT level 1-2 variants beyond noise <= 0.10                (measure is ELIGIBLE)
  M2  structure leaves the noise:       share of a STRUCT kind (blur, shuffle, hue, mirror) at level >= 2 beyond noise >= 0.80
                                        (the measure COVERS that kind)
  U1  the eligible measures together:   share of LIGHT level 1-2 variants flagged by ANY eligible measure <= 0.20
  U2  ... and every STRUCT kind is covered:  share of that kind's level >= 2 variants flagged by ANY eligible measure >= 0.80
Level 3 lighting is reported, not gated. Shares are over (photo, variant) pairs; 95% CI = cluster bootstrap over photos (2000 resamples).
Dev = photos *_01, *_02; test = *_03, reported separately.

HONESTY NOTE: the thresholds 0.10 / 0.20 / 0.80 were chosen after the first run (lint1) had been summarised, on the same 15 photos and the same
variants, so a pass on those variants is NOT a held-out result. What is new in lint2 are the extra measures (chroma_std, chroma_cov, appearance_v3c,
and raw texture_cos / app_zncc), which were designed from the lint1 findings and are scored here for the first time.

    python experiments/lighting_criteria.py --run runs/lint2 [--run runs/lint1]
Writes criteria.csv and criteria.json into the (first) run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ALL = ["silhouette", "ksil_paper", "appearance_v3", "appearance_v2", "texture", "texture_cos", "identity_raw", "kapp_lab", "kapp_rgb",
       "app_zncc", "app_chroma", "chroma_std", "chroma_cov", "appearance_v3c"]
KINDS = ["blur", "shuffle", "hue", "mirror"]
T_M1, T_U1, T_M2 = 0.10, 0.20, 0.80


def split_of(s: str) -> str:
    return "test" if s.endswith("_03") else "dev"


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    for r in rows:
        for m in ALL + ["level"]:
            if m in r:
                r[m] = None if r[m] in ("", None) else float(r[m])
    return rows


def flags(rows: list[dict], measure: str, sel) -> dict[str, dict[str, bool]]:
    """photo -> {variant name: beyond noise?} for the variants chosen by ``sel``; photos without a nuisance score are skipped."""
    nuis = {r["sample"]: r.get(measure) for r in rows if r["family"] == "nuisance"}
    out = defaultdict(dict)
    for r in rows:
        if not sel(r):
            continue
        n, v = nuis.get(r["sample"]), r.get(measure)
        if n is None or v is None:
            continue
        out[r["sample"]][r["name"]] = bool(v < n)
    return out


def share(fl: dict[str, dict[str, bool]], photos: list[str] | None = None) -> float | None:
    xs = [x for p, v in fl.items() if photos is None or p in photos for x in v.values()]
    return float(np.mean(xs)) if xs else None


def boot(fl: dict[str, dict[str, bool]], n: int = 2000, seed: int = 0):
    ids = list(fl)
    pt = share(fl)
    if pt is None or len(ids) < 2:
        return pt, None, None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = [ids[j] for j in rng.integers(0, len(ids), len(ids))]
        xs = [x for p in pick for x in fl[p].values()]
        if xs:
            vals.append(np.mean(xs))
    return pt, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


LIGHT12 = lambda r: r["family"] == "light" and r["level"] <= 2          # noqa: E731
LIGHT3 = lambda r: r["family"] == "light" and r["level"] == 3           # noqa: E731
STRUCT = lambda k: (lambda r: r["family"] == "struct" and r["kind"] == k and r["level"] >= 2)  # noqa: E731


def analyse(rows: list[dict], photos: list[str]) -> dict:
    rows = [r for r in rows if r["sample"] in photos]
    measures = [m for m in ALL if any(r.get(m) is not None for r in rows)]
    res: dict = {"measures": {}}
    fl_l12: dict[str, dict] = {}
    fl_k: dict[str, dict] = {}
    for m in measures:
        fl_l12[m] = flags(rows, m, LIGHT12)
        fl_k[m] = {k: flags(rows, m, STRUCT(k)) for k in KINDS}
        e = {"light12": boot(fl_l12[m]), "light3": boot(flags(rows, m, LIGHT3)), **{k: boot(fl_k[m][k]) for k in KINDS}}
        e["eligible"] = bool(e["light12"][0] is not None and e["light12"][0] <= T_M1)
        e["covers"] = [k for k in KINDS if e[k][0] is not None and e[k][0] >= T_M2]
        res["measures"][m] = e
    elig = [m for m in measures if res["measures"][m]["eligible"]]

    def union(sel_flags):                                   # a variant is flagged if ANY eligible measure flags it
        out = defaultdict(dict)
        for m in elig:
            for p, d in sel_flags(m).items():
                for name, x in d.items():
                    out[p][name] = out[p].get(name, False) or x
        return out

    res["eligible"] = elig
    if elig:
        res["U1"] = boot(union(lambda m: fl_l12[m]))
        res["U2"] = {k: boot(union(lambda m, k=k: fl_k[m][k])) for k in KINDS}
        res["U1_pass"] = bool(res["U1"][0] is not None and res["U1"][0] <= T_U1)
        res["U2_pass"] = bool(all(res["U2"][k][0] is not None and res["U2"][k][0] >= T_M2 for k in KINDS))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="run directory with variants.csv (first one receives the outputs)")
    args = ap.parse_args()
    out = {}
    for run in args.run:
        rows = load(Path(run) / "variants.csv")
        ids = sorted({r["sample"] for r in rows})
        out[run] = {g: analyse(rows, sel) for g, sel in
                    {"all": ids, "dev": [i for i in ids if split_of(i) == "dev"], "test": [i for i in ids if split_of(i) == "test"]}.items()}
    d = Path(args.run[0])
    (d / "criteria.json").write_text(json.dumps(out, indent=1, default=lambda x: None))
    f = lambda t: "  -  " if t is None or t[0] is None else f"{t[0]:.2f}"  # noqa: E731
    with open(d / "criteria.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "group", "measure", "light12", "light12_lo", "light12_hi", "light3", *KINDS, "eligible", "covers"])
        for run, groups in out.items():
            for g, res in groups.items():
                for m, e in res["measures"].items():
                    w.writerow([run, g, m, *(("" if x is None else f"{x:.3f}") for x in e["light12"]), f(e["light3"]).strip(),
                                *[f(e[k]).strip() for k in KINDS], e["eligible"], " ".join(e["covers"])])
    for run, groups in out.items():
        res = groups["all"]
        print(f"\n== {run}  (all photos). share of variants BEYOND re-encoding noise; lighting should be low, structure high ==")
        print("measure".ljust(16) + "light1-2".rjust(9) + "light3".rjust(8) + "".join(k.rjust(9) for k in KINDS) + "  eligible  covers")
        for m, e in res["measures"].items():
            print(m.ljust(16) + f(e["light12"]).rjust(9) + f(e["light3"]).rjust(8) + "".join(f(e[k]).rjust(9) for k in KINDS)
                  + ("   yes     " if e["eligible"] else "   no      ") + ",".join(e["covers"]))
        if res.get("eligible"):
            print(f"eligible: {', '.join(res['eligible'])}")
            print(f"U1 union false-flag on light1-2: {f(res['U1'])}  ({'pass' if res['U1_pass'] else 'FAIL'}, <= {T_U1})")
            print("U2 union coverage: " + "  ".join(f"{k} {f(res['U2'][k])}" for k in KINDS) + f"  ({'pass' if res['U2_pass'] else 'FAIL'}, >= {T_M2} each)")
        else:
            print("no eligible measure")
    print(f"\n-> {d}/criteria.csv")


if __name__ == "__main__":
    main()
