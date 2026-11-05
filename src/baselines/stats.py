"""Small statistics helpers: AUROC / paired accuracy with a cluster bootstrap over source images."""

from __future__ import annotations

import numpy as np


def auroc(pos, neg) -> float | None:
    """P(pos > neg) over all pairs, ties count half. None if either side is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def paired_accuracy(pos, neg) -> float | None:
    """Mean of [pos_i > neg_i] (ties half) for aligned pairs."""
    if len(pos) == 0 or len(pos) != len(neg):
        return None
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    return float(((p > n) + 0.5 * (p == n)).mean())


def cluster_bootstrap(clusters: dict, stat, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """clusters: {cluster_id: (pos_list, neg_list)}. Resample clusters with replacement and recompute
    ``stat(pos_all, neg_all)``. Returns (point, lo, hi); lo/hi are None with fewer than 2 clusters."""
    ids = list(clusters)

    def run(sel):
        pos = [x for i in sel for x in clusters[i][0]]
        neg = [x for i in sel for x in clusters[i][1]]
        return stat(pos, neg)

    point = run(ids)
    if len(ids) < 2 or point is None:
        return point, None, None
    rng = np.random.default_rng(seed)
    vals = [v for v in (run([ids[j] for j in rng.integers(0, len(ids), len(ids))]) for _ in range(n_boot))
            if v is not None]
    if not vals:
        return point, None, None
    return point, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
