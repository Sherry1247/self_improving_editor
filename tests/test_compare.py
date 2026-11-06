"""Violation comparison with matched controls (no torch needed)."""


# violation comparison with matched controls --------------------------------------------------------------------
def _mk_T(n=6):
    import numpy as np

    rng = np.random.default_rng(0)
    T = {"ours:a": {}, "editreward": {}}
    for m in T:
        for kind, base in (("null", 1.0), ("float", 0.2), ("sink", 0.3)):
            T[m][kind] = {f"s{i}": base + 0.05 * rng.standard_normal() for i in range(n)}
        T[m]["shift_side"] = dict(T[m]["null"])    # identical to the null -> AUROC exactly 0.5 (all ties across pairs of equals)
        T[m]["shift_side_s"] = dict(T[m]["null"])
    return T


def test_compare_table_contrast_and_benign():
    from src.baselines.compare import compare_table

    rows = {(r["model"], r["role"], r["neg"]): r for r in compare_table(_mk_T(), n_boot=50)}
    assert rows[("ours:a", "contrast", "float")]["auroc"] > 0.95            # shift_side scores above float
    assert abs(rows[("ours:a", "benign", "shift_side")]["auroc"] - 0.5) < 0.2   # null ~ shift_side
    assert rows[("ours:a", "detect", "float")]["n_images"] == 6


def test_combine_is_symmetric_unfitted_and_keeps_order():
    from src.baselines.compare import add_combinations, combine

    T = _mk_T()
    c1, c2 = combine(T["ours:a"], T["editreward"]), combine(T["editreward"], T["ours:a"])
    assert all(abs(c1[k][s] - c2[k][s]) < 1e-12 for k in c1 for s in c1[k])
    mean = lambda d: sum(d.values()) / len(d)
    assert mean(c1["null"]) > mean(c1["float"])
    assert "combo:editreward+ours:a" in add_combinations(T, "editreward")


def test_filter_split_dev_test():
    from src.baselines.compare import filter_split

    T = {"m": {"null": {"a_01": 1.0, "a_02": 2.0, "a_03": 3.0}}}
    assert filter_split(T, "test")["m"]["null"] == {"a_03": 3.0}
    assert filter_split(T, "dev")["m"]["null"] == {"a_01": 1.0, "a_02": 2.0}
    assert filter_split(T, "all") is T


def test_paired_difference_detects_a_real_gap_and_not_a_tie():
    from src.baselines.compare import paired_difference

    T = _mk_T(8)
    T["good"] = {k: dict(v) for k, v in T["ours:a"].items()}
    T["bad"] = {k: dict(v) for k, v in T["ours:a"].items()}
    for s in T["bad"]["float"]:                       # 'bad' cannot tell the float from the shift
        T["bad"]["float"][s] = T["bad"]["shift_side"][s]
    d, lo, hi, n = paired_difference(T, "good", "bad", "shift_side", "float", n_boot=100)
    assert n == 8 and d > 0.3 and lo > 0
    d0, lo0, hi0, _ = paired_difference(T, "good", "good", "shift_side", "float", n_boot=100)
    assert d0 == 0 and lo0 == 0 and hi0 == 0
