"""experiments/lighting_criteria: share of variants beyond the re-encoding noise, eligibility, coverage, union (synthetic rows)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from lighting_criteria import analyse, flags, share  # noqa: E402


def rows_for(photos, scores):
    """scores: {measure: {(family, kind, level): value}}; nuisance value under key ('nuisance','degrade',0)."""
    out = []
    for p in photos:
        keys = set().union(*[set(d) for d in scores.values()])
        for (fam, kind, lv) in keys:
            r = {"sample": p, "family": fam, "kind": kind, "level": float(lv), "name": f"{kind}{lv}"}
            for m, d in scores.items():
                r[m] = d.get((fam, kind, lv))
            out.append(r)
    return out


def test_beyond_noise_and_union():
    keys = [("null", "null", 0), ("nuisance", "degrade", 0), ("light", "gain", 1), ("light", "gain", 2), ("light", "gain", 3),
            ("struct", "blur", 2), ("struct", "blur", 3), ("struct", "shuffle", 2), ("struct", "hue", 2), ("struct", "mirror", 3)]
    # A sees blur and shuffle only; B sees hue and mirror only; C flags everything (lighting too); D never beats noise
    A = dict(zip(keys, [1.0, .90, .95, .93, .9, .5, .4, .5, .95, .95]))
    B = dict(zip(keys, [1.0, .90, .95, .93, .9, .95, .95, .95, .5, .5]))
    C = dict(zip(keys, [1.0, .90, .5, .5, .5, .5, .5, .5, .5, .5]))
    D = dict(zip(keys, [1.0, .90, .95, .95, .95, .95, .95, .95, .95, .95]))
    photos = [f"p_{i:02d}" for i in range(6)]
    res = analyse(rows_for(photos, {"silhouette": A, "texture": B, "identity_raw": C, "kapp_lab": D}), photos)
    m = res["measures"]
    assert m["silhouette"]["eligible"] and m["texture"]["eligible"] and m["kapp_lab"]["eligible"] and not m["identity_raw"]["eligible"]
    assert set(m["silhouette"]["covers"]) == {"blur", "shuffle"} and set(m["texture"]["covers"]) == {"hue", "mirror"} and m["kapp_lab"]["covers"] == []
    assert res["eligible"] == ["silhouette", "texture", "kapp_lab"] and res["U1_pass"] and res["U2_pass"]      # A and B together cover all four kinds


def test_tie_is_not_beyond_noise_and_missing_values_are_skipped():
    rows = [{"sample": "p", "family": "nuisance", "kind": "degrade", "level": 0.0, "name": "degrade", "m": 1.0},
            {"sample": "p", "family": "light", "kind": "gain", "level": 1.0, "name": "g1", "m": 1.0},
            {"sample": "p", "family": "light", "kind": "gain", "level": 2.0, "name": "g2", "m": None}]
    fl = flags(rows, "m", lambda r: r["family"] == "light")
    assert fl == {"p": {"g1": False}} and share(fl) == 0.0
