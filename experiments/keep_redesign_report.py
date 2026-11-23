"""Keep redesign, offline part (CPU, torch-free): compare Keep designs on the SAME raw critic results from experiments/keep_redesign.py.

Designs (all aggregate with the production gated geometric mean; the critic set differs only in the Keep branch and the identity gate):
  old        configs/chtc.yaml as it is: identity gate, appearance v3 (= structure x chroma) in Keep, Keep = mean
  split      identity gate kept, appearance replaced by app_structure + app_chroma
  no_gate    identity gate removed, appearance v3 kept
  pair       identity gate removed, appearance replaced by app_structure + app_chroma, Keep = mean     <- the candidate (primary)
  pair_soft  as pair, Keep = mean of its 2 lowest critics (a union-like rule: one failed Keep factor is not averaged away)
Each of app_structure / app_chroma has weight 1.0, so Keep has 4 critics (silhouette, app_structure, app_chroma, texture) instead of 3.

CRITERIA, fixed in the docstring before the first run (nothing is fitted on the results):
  K1  AUROC kept (suite v4, overall score, null vs violation, 15 photos): for every detect kind (float, sink, buried, scale_up, scale_down,
      no_shadow, color_cast, halo) with AUROC(old) >= 0.70, AUROC(new) - AUROC(old) >= -0.05; and no benign control (shift_side, shift_side_s)
      rises by more than 0.05 above old (a Keep that starts calling harmless moves a violation).
  K2  no stronger preference for the pasted copy (real comp_v2 and qwen outputs; relit1 = relight strength 0.15 / side 0.10 is better by
      construction): mean Keep(relit1) - Keep(pasted) >= -0.02, and the share of tasks where overall(relit1) > overall(pasted) (ties count
      half) is not lower than for old.
  K3  strong colour cast acceptable (synthetic ladder of lint2, Keep branch only): mean Keep drop of cast level 3 <= 0.5 x mean Keep drop of
      blur + shuffle level >= 2.
A design PASSES when K1, K2 and K3 all hold on the point estimates; 95% cluster-bootstrap intervals (over source photos) are reported, not gated.
Thresholds are chosen before looking at the new results but after seeing lint1/lint2/rl1 on the same photos, so a pass is a design check, not a held-out result.

    python experiments/keep_redesign_report.py --suite runs/kr_suite/raw.jsonl --copypaste runs/kr_copy/raw.jsonl --ladder docs/results/lint2/variants.csv --out runs/kr_report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DETECT = ["float", "sink", "buried", "scale_up", "scale_down", "no_shadow", "color_cast", "halo"]
BENIGN = ["shift_side", "shift_side_s"]
NEW_KEEP = ["app_structure", "app_chroma"]


def designs(base_enabled: list[str]) -> dict[str, dict]:
    base = [c for c in base_enabled if c not in NEW_KEEP]
    swap = lambda lst: [x for c in lst for x in (NEW_KEEP if c == "appearance" else [c])]  # noqa: E731
    no_id = lambda lst: [c for c in lst if c != "identity"]  # noqa: E731
    return {"old": {"enabled": base, "keep": "mean"},
            "split": {"enabled": swap(base), "keep": "mean"},
            "no_gate": {"enabled": no_id(base), "keep": "mean"},
            "pair": {"enabled": no_id(swap(base)), "keep": "mean"},
            "pair_soft": {"enabled": no_id(swap(base)), "keep": "softmin2"}}


def aggregate(critics: dict, design: dict, cfg: dict):
    """critics: {name: {"s", "b", "c"}} -> (overall, keep, gate_passed) under ``design``."""
    from src.scoring import build_aggregator
    from src.types import CriticResult

    c2 = json.loads(json.dumps(cfg))
    c2.setdefault("aggregation", {}).setdefault("branch_reduce", {})["keep"] = design["keep"]
    agg = build_aggregator(c2)
    res = {n: CriticResult(n, critics[n]["b"], critics[n]["s"], [], {}, bool(critics[n]["c"])) for n in design["enabled"] if n in critics}
    ev = agg.aggregate(res)
    return ev.overall, ev.branch_scores.get("keep"), ev.gate_passed


def read(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def auroc(pos, neg):
    from src.baselines.stats import auroc as a
    return a(pos, neg)


def kind_ref(kind: str) -> str:
    return "foreign_clean" if kind == "halo" else "null"


def table(rows: list[dict], cfg: dict, D: dict) -> dict:
    """{design: {sample: {kind: (overall, keep, gate)}}}"""
    out = {d: defaultdict(dict) for d in D}
    for r in rows:
        for d, spec in D.items():
            out[d][r["sample"]][r["kind"]] = aggregate(r["critics"], spec, cfg)
    return out


def suite_auroc(T: dict, kind: str, idx: int = 0, n_boot: int = 2000, seed: int = 0):
    from src.baselines.stats import cluster_bootstrap
    ref = kind_ref(kind)
    cl = {s: ([k[ref][idx]], [k[kind][idx]]) for s, k in T.items() if ref in k and kind in k}
    return cluster_bootstrap(cl, auroc, n_boot=n_boot, seed=seed) if cl else (None, None, None)


def paired_diff(TA: dict, TB: dict, kind: str, n_boot: int = 2000, seed: int = 0):
    ref = kind_ref(kind)
    ids = [s for s in TA if ref in TA[s] and kind in TA[s] and s in TB and ref in TB[s] and kind in TB[s]]
    if not ids:
        return None, None, None

    def stat(sel):
        a = auroc([TA[s][ref][0] for s in sel], [TA[s][kind][0] for s in sel])
        b = auroc([TB[s][ref][0] for s in sel], [TB[s][kind][0] for s in sel])
        return None if a is None or b is None else a - b

    pt = stat(ids)
    rng = np.random.default_rng(seed)
    vals = [v for v in (stat([ids[j] for j in rng.integers(0, len(ids), len(ids))]) for _ in range(n_boot)) if v is not None]
    return pt, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def k1(T: dict, new: str, old: str = "old") -> dict:
    res = {"detect": {}, "benign": {}, "fail": []}
    for kind in DETECT + BENIGN:
        a_old, a_new = suite_auroc(T[old], kind)[0], suite_auroc(T[new], kind)[0]
        d = paired_diff(T[new], T[old], kind)
        e = {"old": a_old, "new": a_new, "diff": d}
        if kind in DETECT:
            res["detect"][kind] = e
            if a_old is not None and a_old >= 0.70 and a_new is not None and a_new - a_old < -0.05:
                res["fail"].append(kind)
        else:
            res["benign"][kind] = e
            if a_old is not None and a_new is not None and a_new - a_old > 0.05:
                res["fail"].append(kind)
    res["pass"] = not res["fail"]
    return res


def k2(rows: list[dict], cfg: dict, D: dict, run: str) -> dict:
    by = defaultdict(dict)
    for r in rows:
        if r["run"] == run:
            by[r["task"]][r["kind"]] = r["critics"]
    out = {}
    for d, spec in D.items():
        dk, wins = [], []
        for task, v in by.items():
            if "pasted" not in v or "relit1" not in v:
                continue
            op, kp, _ = aggregate(v["pasted"], spec, cfg)
            orl, krl, _ = aggregate(v["relit1"], spec, cfg)
            if kp is not None and krl is not None:
                dk.append(krl - kp)
            wins.append(1.0 if orl > op + 1e-9 else (0.5 if abs(orl - op) <= 1e-9 else 0.0))
        out[d] = {"n": len(wins), "d_keep_relit1": float(np.mean(dk)) if dk else None, "win_relit1": float(np.mean(wins)) if wins else None}
    old_w = out["old"]["win_relit1"]
    for d, e in out.items():
        e["pass"] = bool(e["d_keep_relit1"] is not None and e["d_keep_relit1"] >= -0.02 and e["win_relit1"] is not None and old_w is not None
                         and e["win_relit1"] >= old_w - 1e-9)
    return out


def ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def keep_scores_ladder(row: dict) -> dict[str, float | None]:
    """Keep-branch score of one ladder variant under each design (critic scores reconstructed from variants.csv)."""
    f = lambda k: None if row.get(k) in (None, "") else float(row[k])  # noqa: E731
    sil, tex, app, zncc, chroma = f("silhouette"), f("texture"), f("appearance_v3"), f("app_zncc"), f("app_chroma")
    st = None if zncc is None else ramp(zncc, 0.2, 0.8)
    mean = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None  # noqa: E731
    soft = lambda xs: (lambda v: float(np.mean(sorted(v)[:2])) if v else None)([x for x in xs if x is not None])  # noqa: E731
    return {"old": mean([sil, app, tex]), "pair": mean([sil, st, chroma, tex]), "pair_soft": soft([sil, st, chroma, tex])}


def k3(path: str) -> dict:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    ks = [(r, keep_scores_ladder(r)) for r in rows]
    null = {r["sample"]: k for r, k in ks if r["family"] == "null"}
    nuis = {r["sample"]: k for r, k in ks if r["family"] == "nuisance"}
    out = {}
    for d in ("old", "pair", "pair_soft"):
        def drops(sel):
            return [null[r["sample"]][d] - k[d] for r, k in ks if sel(r) and r["sample"] in null and k[d] is not None and null[r["sample"]][d] is not None]

        def beyond(sel):
            xs = [k[d] < nuis[r["sample"]][d] for r, k in ks if sel(r) and r["sample"] in nuis and k[d] is not None and nuis[r["sample"]][d] is not None]
            return float(np.mean(xs)) if xs else None
        cast3 = lambda r: r["kind"] == "cast" and int(float(r["level"])) == 3  # noqa: E731
        cast = lambda r: r["kind"] == "cast"  # noqa: E731
        struct = lambda r: r["kind"] in ("blur", "shuffle") and int(float(r["level"])) >= 2  # noqa: E731
        light12 = lambda r: r["family"] == "light" and int(float(r["level"])) <= 2  # noqa: E731
        c3, st = float(np.mean(drops(cast3))), float(np.mean(drops(struct)))
        out[d] = {"drop_cast3": c3, "drop_struct": st, "ratio": c3 / st if st > 0 else None, "beyond_cast": beyond(cast), "beyond_cast3": beyond(cast3),
                  "beyond_light12": beyond(light12), "beyond_struct": beyond(struct), "pass": bool(st > 0 and c3 <= 0.5 * st)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", help="raw.jsonl of keep_redesign.py --part suite")
    ap.add_argument("--copypaste", help="raw.jsonl of keep_redesign.py --part copypaste")
    ap.add_argument("--ladder", help="lint2 variants.csv (synthetic ladder)")
    ap.add_argument("--out", default="runs/keep_redesign_report")
    ap.add_argument("--config", action="append", default=["configs/chtc.yaml"])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    from src.config import load_config
    cfg = load_config(args.config, [])
    base_enabled = [c for c in cfg["critics"]["enabled"]]
    D = designs(base_enabled)
    res: dict = {"designs": D}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f2 = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    f3 = lambda t: "-" if t is None or t[0] is None else f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"  # noqa: E731

    if args.suite:
        rows = read(args.suite)
        T = table(rows, cfg, D)
        res["gate_fired_old"] = sum(1 for r in rows if r["critics"].get("identity", {}).get("c"))
        print(f"\n== K1: violation suite v4, overall score, AUROC (null > violation); {len({r['sample'] for r in rows})} photos ==")
        print("kind".ljust(14) + "".join(d.rjust(11) for d in D))
        for kind in DETECT + BENIGN:
            print(kind.ljust(14) + "".join(f2(suite_auroc(T[d], kind, n_boot=200)[0]).rjust(11) for d in D))
        print("Keep-branch only:")
        for kind in DETECT + BENIGN:
            print(kind.ljust(14) + "".join(f2(suite_auroc(T[d], kind, idx=1, n_boot=200)[0]).rjust(11) for d in D))
        res["K1"] = {}
        for new in ("split", "no_gate", "pair", "pair_soft"):
            r1 = k1(T, new)
            res["K1"][new] = r1
            print(f"\nK1 {new} vs old: " + ("PASS" if r1["pass"] else f"FAIL {r1['fail']}"))
            for kind, e in {**r1["detect"], **r1["benign"]}.items():
                print(f"  {kind.ljust(13)} old {f2(e['old'])} new {f2(e['new'])}  diff {f3(e['diff'])}")
        # per-critic AUROC on the raw critic scores
        crit_names = sorted({n for r in rows for n in r["critics"]})
        print("\nper-critic AUROC (null > violation), raw critic score:")
        print("critic".ljust(16) + "".join(k[:9].rjust(10) for k in DETECT + BENIGN))
        per = {}
        for n in crit_names:
            if not any(n.startswith(p) for p in ("identity", "silhouette", "appearance", "app_", "texture")):
                continue
            line = []
            for kind in DETECT + BENIGN:
                ref = kind_ref(kind)
                by = defaultdict(dict)
                for r in rows:
                    s = r["critics"].get(n, {}).get("s")
                    if s is not None:
                        by[r["sample"]][r["kind"]] = s
                pos = [d[ref] for d in by.values() if ref in d and kind in d]
                neg = [d[kind] for d in by.values() if ref in d and kind in d]
                line.append(auroc(pos, neg))
            per[n] = line
            print(n.ljust(16) + "".join(f2(x).rjust(10) for x in line))
        res["per_critic_auroc"] = per

    if args.copypaste:
        rows = read(args.copypaste)
        res["K2"] = {}
        for run in sorted({r["run"] for r in rows}):
            r2 = k2(rows, cfg, D, run)
            res["K2"][run] = r2
            print(f"\n== K2: {run}, relit (strength 0.15, side 0.10) vs pasted copy, n={r2['old']['n']} tasks ==")
            for d, e in r2.items():
                dk = "-" if e["d_keep_relit1"] is None else "%+.3f" % e["d_keep_relit1"]
                print(f"  {d.ljust(10)} mean dKeep {dk}  relit>pasted overall {f2(e['win_relit1'])}  " + ("PASS" if e["pass"] else "fail"))

    if args.ladder:
        res["K3"] = k3(args.ladder)
        print("\n== K3: synthetic ladder, Keep branch: drop for strong cast (level 3) vs blur/shuffle (level >= 2) ==")
        for d, e in res["K3"].items():
            print(f"  {d.ljust(10)} cast3 drop {e['drop_cast3']:.3f}  struct drop {e['drop_struct']:.3f}  ratio {f2(e['ratio'])}  "
                  f"beyond Keep-noise: cast {f2(e['beyond_cast'])} cast3 {f2(e['beyond_cast3'])} light1-2 {f2(e['beyond_light12'])} struct {f2(e['beyond_struct'])}  "
                  + ("PASS" if e["pass"] else "FAIL"))
    (out / "report.json").write_text(json.dumps(res, indent=1, default=lambda x: None))
    print(f"\n-> {out}/report.json")


if __name__ == "__main__":
    main()
