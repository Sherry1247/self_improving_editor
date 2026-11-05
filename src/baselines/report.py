"""Join a scoring-set manifest with reward-model score files and tabulate the comparisons.

manifest.jsonl rows (written by experiments/build_scoring_set.py):
    id, group ("violation" | "run"), sample_id, kind (violation kind or run name), task, src, edit, instruction,
    and for runs the project's own scores: ours_overall / ours_keep / ours_follow / ours_world.
scores_<model>.jsonl rows: id plus either "score" (one number) or several metric fields (e.g. viescore: overall, sc, pq,
naturalness, ...). Every (model, metric) pair becomes one column called "<model>:<metric>" (or just "<model>").
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from src.baselines.stats import auroc, cluster_bootstrap, paired_accuracy

SKIP_KINDS = {"real", "null", "foreign_clean"}


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_manifest(set_dir) -> list[dict]:
    return load_jsonl(Path(set_dir) / "manifest.jsonl")


def load_scores(paths) -> dict[str, dict[str, float]]:
    """{column: {item id: score}}; None / non-numeric values are dropped."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for p in paths:
        name = Path(p).stem.removeprefix("scores_")
        for r in load_jsonl(p):
            metrics = {k: v for k, v in r.items() if k != "id" and not k.startswith("raw")}
            for k, v in metrics.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                out[name if k == "score" else f"{name}:{k}"][r["id"]] = float(v)
    return dict(out)


def violation_table(manifest: list[dict], scores: dict[str, dict[str, float]], n_boot: int = 2000) -> list[dict]:
    """Per (column, violation kind): does the score separate the clean paste ('null'; 'foreign_clean' for halo) from the
    violation? AUROC over all pairs and paired accuracy within a source image, both with a cluster bootstrap over
    source images."""
    ids = defaultdict(dict)  # sample -> kind -> id
    for r in manifest:
        if r["group"] == "violation":
            ids[r["sample_id"]][r["kind"]] = r["id"]
    kinds = sorted({k for d in ids.values() for k in d} - SKIP_KINDS)
    rows = []
    for col, sc in scores.items():
        for kind in kinds:
            pos_kind = "foreign_clean" if kind == "halo" else "null"
            clusters = {}
            for s, d in ids.items():
                if pos_kind in d and kind in d and d[pos_kind] in sc and d[kind] in sc:
                    clusters[s] = ([sc[d[pos_kind]]], [sc[d[kind]]])
            if not clusters:
                continue
            a, alo, ahi = cluster_bootstrap(clusters, auroc, n_boot)
            p, plo, phi = cluster_bootstrap(clusters, paired_accuracy, n_boot)
            rows.append({"model": col, "kind": kind, "n_images": len(clusters), "auroc": a, "auroc_lo": alo,
                         "auroc_hi": ahi, "paired_acc": p, "paired_lo": plo, "paired_hi": phi})
    return rows


def run_table(manifest: list[dict], scores: dict[str, dict[str, float]]) -> list[dict]:
    """Per column: mean score per run, pairwise wins on common tasks, and Spearman with the project's own overall score."""
    from scipy.stats import spearmanr

    runs = defaultdict(dict)  # run -> task -> row
    for r in manifest:
        if r["group"] == "run":
            runs[r["kind"]][r["task"]] = r
    rows = []
    for col, sc in scores.items():
        means = {run: st.mean(sc[r["id"]] for r in d.values() if r["id"] in sc) for run, d in runs.items()
                 if any(r["id"] in sc for r in d.values())}
        row = {"model": col, **{f"mean_{k}": v for k, v in means.items()}}
        names = sorted(means)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = [t for t in runs[a] if t in runs[b] and runs[a][t]["id"] in sc and runs[b][t]["id"] in sc]
                row[f"{a}>{b}"] = sum(sc[runs[a][t]["id"]] > sc[runs[b][t]["id"]] for t in common)
                row[f"{b}>{a}"] = sum(sc[runs[a][t]["id"]] < sc[runs[b][t]["id"]] for t in common)
        pairs = [(sc[r["id"]], r["ours_overall"]) for d in runs.values() for r in d.values()
                 if r["id"] in sc and r.get("ours_overall") is not None]
        if len(pairs) > 2:
            row["spearman_ours"] = float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs])[0])
        for run, d in runs.items():
            pr = [(sc[r["id"]], r["ours_overall"]) for r in d.values() if r["id"] in sc and r.get("ours_overall") is not None]
            if len(pr) > 2:
                row[f"spearman_ours_in_{run}"] = float(spearmanr([p[0] for p in pr], [p[1] for p in pr])[0])
        rows.append(row)
    return rows


def our_critics_rows(auroc_csv) -> list[dict]:
    """Our critics' E1 AUROCs (runs/e1*/auroc.csv, the cross-sample AUROC) in the same long format, without CIs."""
    import csv

    rows = []
    with open(auroc_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k != "critic" and v not in ("", None):
                    rows.append({"model": "ours:" + r["critic"], "kind": k, "auroc": float(v)})
    return rows
