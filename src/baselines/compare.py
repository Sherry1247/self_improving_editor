"""Violation-suite comparison with matched controls, uniform bootstrap CIs and (exploratory) score combinations.

Everything is a table ``T[model][kind][sample_id] = score`` (higher = more plausible). Sources:
  * our critics: ``runs/<e1>/scores_by_sample.json`` written by experiments/critic_auroc.py,
  * reward baselines: a scoring-set manifest (experiments/build_scoring_set.py) + ``scores_*.jsonl`` files.
Pairs (positive kind, negative kind):
  detect   (null, k) for every violation k           ('foreign_clean', 'halo' for halo)
  benign   (null, shift_side), (null, shift_side_s)  -- ~0.5 for a critic that is not just a displacement detector
  contrast (shift_side, float), (shift_side_s, sink) -- the matched test: same displacement, only contact differs
Combination = mean of the two models' z-scores, each standardised over ALL items both models scored (no labels used, no
fitted weights). It is exploratory: the choice of pair is not corrected for multiplicity.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.baselines.report import load_manifest
from src.baselines.stats import auroc, cluster_bootstrap

VIOLATIONS = ["float", "sink", "scale_up", "scale_down", "no_shadow", "color_cast", "halo"]
BENIGN = ["shift_side", "shift_side_s"]
CONTRASTS = [("shift_side", "float"), ("shift_side_s", "sink")]
NON_NEGATIVE = {"real", "foreign_clean"}


def pairs() -> list[tuple[str, str, str]]:
    out = [("detect", "foreign_clean" if k == "halo" else "null", k) for k in VIOLATIONS]
    out += [("benign", "null", k) for k in BENIGN]
    out += [("contrast", p, n) for p, n in CONTRASTS]
    return out


def table_from_critics(by_sample: dict) -> dict[str, dict[str, dict[str, float]]]:
    """critic_auroc.py's scores_by_sample.json['scores'] -> T with model names prefixed 'ours:'."""
    return {"ours:" + c: {k: dict(v) for k, v in kinds.items()} for c, kinds in by_sample.items()}


def table_from_rewards(manifest: list[dict], scores: dict[str, dict[str, float]]) -> dict:
    T: dict = {}
    for col, sc in scores.items():
        d: dict = defaultdict(dict)
        for r in manifest:
            if r["group"] == "violation" and r["id"] in sc:
                d[r["kind"]][r["sample_id"]] = sc[r["id"]]
        T[col] = dict(d)
    return T


def _z(values: dict[tuple, float]) -> dict[tuple, float]:
    v = np.array(list(values.values()), float)
    sd = v.std()
    return {k: (x - v.mean()) / sd if sd > 0 else 0.0 for k, x in values.items()}


def combine(a: dict, b: dict) -> dict:
    """z-score mean of two models over the (kind, sample) cells both scored."""
    cells = {(k, s) for k, d in a.items() for s in d} & {(k, s) for k, d in b.items() for s in d}
    za = _z({c: a[c[0]][c[1]] for c in cells})
    zb = _z({c: b[c[0]][c[1]] for c in cells})
    out: dict = defaultdict(dict)
    for (k, s) in cells:
        out[k][s] = (za[(k, s)] + zb[(k, s)]) / 2
    return dict(out)


def add_combinations(T: dict, ref: str, others: list[str] | None = None) -> dict:
    """T plus 'combo:<ref>+<other>' for every other model (default: all our critics)."""
    out = dict(T)
    for m in others if others is not None else [m for m in T if m.startswith("ours:")]:
        if m in T and ref in T and m != ref:
            out[f"combo:{ref}+{m}"] = combine(T[ref], T[m])
    return out


def compare_table(T: dict, n_boot: int = 2000) -> list[dict]:
    rows = []
    for model, kinds in T.items():
        for role, pk, nk in pairs():
            cl = {s: ([kinds[pk][s]], [kinds[nk][s]]) for s in kinds.get(pk, {}) if s in kinds.get(nk, {})}
            if not cl:
                continue
            a, lo, hi = cluster_bootstrap(cl, auroc, n_boot)
            rows.append({"model": model, "role": role, "pos": pk, "neg": nk, "n_images": len(cl), "auroc": a,
                         "lo": lo, "hi": hi})
    return rows


def smear_summary(smear: dict) -> dict[str, float]:
    """kind -> mean share of the inpainted region left uncovered (critic_auroc.py's scores_by_sample.json['smear_visible'])."""
    return {k: float(np.mean(list(v.values()))) for k, v in smear.items() if v}


def load_all(critics_json: dict | None, scoreset: str | None, score_files: list[str]) -> tuple[dict, dict]:
    T: dict = {}
    smear: dict = {}
    if critics_json:
        T.update(table_from_critics(critics_json["scores"]))
        smear = critics_json.get("smear_visible", {})
    if scoreset and score_files:
        from src.baselines.report import load_scores

        T.update(table_from_rewards(load_manifest(scoreset), load_scores(score_files)))
    return T, smear
