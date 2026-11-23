"""experiments/keep_redesign_report: designs, offline aggregation, K1 / K2 / K3 on synthetic critic results (torch-free)."""
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from keep_redesign_report import aggregate, designs, k1, k2, k3, keep_scores_ladder, table  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config([str(ROOT / "configs" / "chtc.yaml")], [])


def crits(identity=0.9, ident_cat=False, app_structure=1.0, app_chroma=1.0, sil=1.0, tex=1.0, world=1.0, appearance=None):
    c = lambda s, b, cat=False: {"s": s, "b": b, "c": cat}  # noqa: E731
    return {"subject_count": c(1.0, "gate"), "identity": c(identity, "gate", ident_cat), "bg_changed": c(1.0, "follow"),
            "silhouette": c(sil, "keep"), "appearance": c(app_structure * app_chroma if appearance is None else appearance, "keep"), "app_structure": c(app_structure, "keep"),
            "app_chroma": c(app_chroma, "keep"), "texture": c(tex, "keep"), "bg_semantic": c(1.0, "follow"), "old_bg_residue": c(1.0, "follow"),
            "support": c(world, "world"), "light_harmony": c(world, "world"), "halo": c(world, "world")}


def test_designs_swap_the_appearance_critic_and_drop_the_identity_gate():
    D = designs(CFG["critics"]["enabled"] + ["app_structure", "app_chroma"])
    assert "appearance" in D["old"]["enabled"] and "identity" in D["old"]["enabled"] and "app_structure" not in D["old"]["enabled"]
    assert "appearance" not in D["pair"]["enabled"] and "identity" not in D["pair"]["enabled"]
    assert {"app_structure", "app_chroma"} <= set(D["pair"]["enabled"]) and D["pair_soft"]["keep"] == "softmin2"
    assert "identity" in D["split"]["enabled"] and "identity" not in D["no_gate"]["enabled"] and "appearance" in D["no_gate"]["enabled"]


def test_identity_gate_zeroes_old_but_not_pair():
    D = designs(CFG["critics"]["enabled"] + ["app_structure", "app_chroma"])
    c = crits(identity=0.2, ident_cat=True)
    assert aggregate(c, D["old"], CFG)[0] == 0.0 and aggregate(c, D["old"], CFG)[2] is False
    assert aggregate(c, D["pair"], CFG)[0] > 0.5 and aggregate(c, D["pair"], CFG)[2] is True


def test_keep_branch_mean_vs_softmin():
    D = designs(CFG["critics"]["enabled"] + ["app_structure", "app_chroma"])
    c = crits(app_chroma=0.2)                                  # one failed factor
    keep_mean, keep_soft = aggregate(c, D["pair"], CFG)[1], aggregate(c, D["pair_soft"], CFG)[1]
    assert keep_mean == pytest.approx((1 + 1 + 0.2 + 1) / 4) and keep_soft == pytest.approx((0.2 + 1) / 2)
    assert aggregate(c, D["old"], CFG)[1] == pytest.approx((1 + 0.2 + 1) / 3)       # old: appearance = structure * chroma = 0.2


def test_k1_detects_a_lost_auroc_and_a_benign_rise():
    D = designs(CFG["critics"]["enabled"] + ["app_structure", "app_chroma"])
    rows = []
    for i in range(6):
        s = f"p_{i:02d}"
        for kind in ("null", "float", "shift_side"):
            # a Keep failure on the benign control that only app_structure sees (the old appearance critic stays at 1); 'float' is seen by the world critics only
            rows.append({"sample": s, "kind": kind, "critics": crits(world=0.4 if kind == "float" else 1.0,
                                                                     app_structure=0.3 if kind == "shift_side" else 1.0, appearance=1.0)})
    T = table(rows, CFG, D)
    r_pair, r_old = k1(T, "pair"), k1(T, "old")
    assert "float" not in r_pair["fail"] and r_pair["detect"]["float"]["new"] == 1.0
    assert "shift_side" in r_pair["fail"] and not r_pair["pass"]         # the new Keep calls a benign control a violation, the old one did not
    assert r_old["pass"] or r_old["fail"] == []


def test_k2_counts_relit_wins():
    D = {"old": designs(CFG["critics"]["enabled"] + ["app_structure", "app_chroma"])["old"], "pair": designs(CFG["critics"]["enabled"] + ["app_structure", "app_chroma"])["pair"]}
    rows = []
    for i in range(4):
        t = f"t{i}"
        rows.append({"run": "comp_v2", "task": t, "sample": "p", "kind": "pasted", "critics": crits(world=0.5)})
        rows.append({"run": "comp_v2", "task": t, "sample": "p", "kind": "relit1", "critics": crits(world=0.8, app_chroma=0.95)})
    r = k2(rows, CFG, D, "comp_v2")
    assert r["old"]["win_relit1"] == 1.0 and r["pair"]["win_relit1"] == 1.0
    assert r["pair"]["d_keep_relit1"] == pytest.approx(-0.05 / 4) and r["pair"]["pass"]


def test_k3_ladder(tmp_path):
    def row(sample, family, kind, level, **v):
        return {"sample": sample, "family": family, "kind": kind, "level": level, "silhouette": 1.0, "texture": 1.0, "appearance_v3": 1.0,
                "app_zncc": 1.0, "app_chroma": 1.0, **{k: str(x) for k, x in v.items()}}
    rows = [row("p", "null", "null", 0), row("p", "nuisance", "degrade", 0, app_zncc=0.95, app_chroma=0.95, appearance_v3=0.9),
            row("p", "light", "cast", 3, app_chroma=0.85, appearance_v3=0.85),
            row("p", "struct", "blur", 2, app_zncc=0.4, appearance_v3=0.4), row("p", "struct", "shuffle", 3, app_zncc=0.3, appearance_v3=0.3)]
    path = tmp_path / "v.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = k3(str(path))
    assert out["pair"]["pass"] and out["pair"]["drop_cast3"] < out["pair"]["drop_struct"]
    assert keep_scores_ladder(rows[0])["old"] == 1.0
