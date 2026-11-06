"""support_diagnostics: signed gap / plane residual / detectors on a synthetic ground-plane depth (no torch needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from support_diagnostics import contact_features, detectors  # noqa: E402


def scene(feet_row: int, subject_depth: float | None = None, h=200, w=200):
    depth = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))  # nearer = lower in the image
    mask = np.zeros((h, w), bool)
    mask[feet_row - 60:feet_row, 80:120] = True
    if subject_depth is not None:
        depth = depth.copy()
        depth[mask] = subject_depth
    return depth, mask


def test_features_and_none_cases():
    d, m = scene(150)
    f = contact_features(d, m)
    assert f is not None and abs(f["res"]) < 0.02 and f["gap"] > 0   # standing on the plane: residual ~0, ground below is nearer
    d2, m2 = scene(150)
    m2[-3:, 90:100] = True                                           # touches the bottom edge -> contact not visible
    assert contact_features(d2, m2) is None


def test_displaced_subject_with_fixed_object_depth_has_signed_residual():
    d0, m0 = scene(150)
    sd = float(d0[145, 100])                                          # the subject's own depth stays what it was
    fb = contact_features(d0, m0)
    up = contact_features(*scene(135, sd))                             # lifted: its feet row is higher (farther), depth is not
    down = contact_features(*scene(160, sd))                           # sunk: feet row lower (nearer)
    assert up["res"] > fb["res"] + 0.03 and down["res"] < fb["res"] - 0.03   # opposite signs
    det_up, det_down = detectors(up, fb), detectors(down, fb)
    assert det_up["contact_dres"] < 0.8 and det_down["contact_dres"] < 0.8   # both directions are caught
    assert detectors(fb, fb)["contact_dres"] == 1.0 and detectors(fb, fb)["support_v0"] == 1.0


def test_old_clip_ignores_a_gap_that_shrinks():
    fb = {"gap": 0.10, "res": 0.0}
    shrunk = {"gap": 0.02, "res": 0.0}
    d = detectors(shrunk, fb)
    assert d["support_v0"] == 1.0                    # |gap| went down -> excess 0 -> 'perfect', although the contact changed
    assert d["contact_dgap"] < 0.9 and d["contact_dsgap"] < 0.9   # the unclipped versions see it
