"""Violation suite v3: one shared plate (same smear everywhere), and the shift_side benign controls."""
import numpy as np

from fakes import make_scene
from src.utils.geometry import fill_holes, mask_box
from validation.violations import make_violations


def _suite():
    img, m = make_scene("river")
    return img, fill_holes(m), {v.kind: v for v in make_violations(img, m, make_scene("snow")[0])}


def test_new_kinds_present_and_shapes():
    img, m, vs = _suite()
    assert {"null", "float", "sink", "scale_up", "scale_down", "no_shadow", "color_cast", "shift_side", "shift_side_s",
            "halo", "foreign_clean"} <= set(vs)
    for v in vs.values():
        assert v.image.shape == img.shape and v.mask.shape == m.shape


def test_shift_side_moves_horizontally_only_and_stays_in_frame():
    img, m, vs = _suite()
    b0, b1 = mask_box(m), mask_box(vs["shift_side"].mask)
    assert b1[3] == b0[3] and b1[1] == b0[1]          # same feet row, same top row: still on the ground
    assert b1[0] != b0[0]                                 # but moved sideways
    assert 0 <= b1[0] and b1[2] <= img.shape[1]
    sh = b0[3] - b0[1]
    assert abs(abs(b1[0] - b0[0]) - int(0.10 * sh)) <= 1  # same displacement as float


def test_shared_plate_far_background_identical_across_variants():
    img, m, vs = _suite()
    far = np.zeros_like(m)
    far[:, :8] = True
    ref = vs["null"].image[far].astype(int)
    for k in ("float", "sink", "scale_down", "shift_side", "no_shadow"):
        assert np.abs(vs[k].image[far].astype(int) - ref).max() == 0, k


def test_smear_visible_recorded_and_ordered():
    _, _, vs = _suite()
    for k in ("null", "float", "sink", "scale_down", "shift_side", "shift_side_s", "scale_up"):
        assert 0.0 <= vs[k].params["smear_visible"] <= 1.0, k
    # shrinking uncovers part of the hole the null covers; growing covers more of the plate than the null does
    assert vs["scale_down"].params["smear_visible"] > vs["null"].params["smear_visible"]
    assert vs["scale_up"].params["smear_visible"] < vs["null"].params["smear_visible"]


def test_null_unchanged_by_union_in_subject_region():
    img, m, vs = _suite()
    import cv2
    inside = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0  # the 1-2 px edge is alpha-blended with the plate
    assert np.abs(vs["null"].image[inside].astype(int) - img[inside].astype(int)).max() <= 3
