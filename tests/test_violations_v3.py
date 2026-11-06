"""Violation suite v4: protected ground strip, shift_side controls, buried, shadow lift without inpainting, smear_ctrl."""
import cv2
import numpy as np

from fakes import make_scene
from src.utils.geometry import dilate, fill_holes, mask_box
from validation.violations import contact_band, make_violations


def _suite():
    img, m = make_scene("river")
    m = fill_holes(m)
    return img, m, {v.kind: v for v in make_violations(img, m, make_scene("snow")[0])}


def _L(img):
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)[..., 0]


def test_kinds_present_and_shapes():
    img, m, vs = _suite()
    assert {"null", "float", "sink", "scale_up", "scale_down", "no_shadow", "color_cast", "shift_side", "shift_side_s",
            "buried", "smear_ctrl", "halo", "foreign_clean"} <= set(vs)
    for v in vs.values():
        assert v.image.shape == img.shape and v.mask.shape == m.shape


def test_shift_side_moves_horizontally_only_and_stays_in_frame():
    img, m, vs = _suite()
    b0, b1 = mask_box(m), mask_box(vs["shift_side"].mask)
    assert b1[3] == b0[3] and b1[1] == b0[1] and b1[0] != b0[0]
    assert 0 <= b1[0] and b1[2] <= img.shape[1]
    sh = b0[3] - b0[1]
    assert abs(abs(b1[0] - b0[0]) - int(0.10 * sh)) <= 1  # same displacement as float


def test_far_background_identical_across_variants():
    img, m, vs = _suite()
    far = np.zeros_like(m)
    far[:, :8] = True
    ref = vs["null"].image[far].astype(int)
    for k in ("float", "sink", "scale_down", "shift_side", "shift_side_s", "no_shadow", "buried", "smear_ctrl"):
        assert np.abs(vs[k].image[far].astype(int) - ref).max() == 0, k


def test_ground_below_the_feet_is_real_in_every_variant_but_the_shifts():
    img, m, vs = _suite()
    strip = contact_band(m, 0.15)
    for k in ("null", "float", "sink", "scale_up", "scale_down", "buried", "smear_ctrl", "color_cast"):
        free = strip & ~dilate(vs[k].mask | m, 3)
        assert free.sum() > 100
        assert np.abs(vs[k].image[free].astype(int) - img[free].astype(int)).max() == 0, k


def test_shift_side_carries_the_real_strip_with_the_subject():
    img, m, vs = _suite()
    v = vs["shift_side"]
    b0, b1 = mask_box(m), mask_box(v.mask)
    dx = int(round(b1[0] - b0[0]))
    strip = contact_band(m, 0.15)
    sel = strip & np.roll(strip, dx, axis=1) & ~dilate(v.mask | m, 3)  # pixels whose source is also strip
    ys, xs = np.nonzero(sel)
    assert len(ys) > 50
    moved = v.image[ys, xs].astype(int)
    src = img[ys, xs - dx].astype(int)
    assert np.abs(moved - src).max() <= 4  # alpha-blended edges of the carried strip


def test_smear_visible_recorded_and_matched():
    _, _, vs = _suite()
    s = {k: v.params.get("smear_visible") for k, v in vs.items()}
    assert all(0 <= s[k] for k in ("null", "float", "sink", "scale_up", "scale_down", "buried", "smear_ctrl", "no_shadow"))
    assert abs(s["float"] - s["null"]) < 0.08                      # moving up uncovers about what the null hides anyway
    assert s["scale_down"] > s["null"] + 0.2 and s["scale_up"] < s["null"] - 0.2
    assert abs(s["smear_ctrl"] - s["scale_down"]) < 0.15 * s["scale_down"]   # the matched control for scale_down


def test_buried_hides_the_feet_below_the_original_ground_line():
    img, m, vs = _suite()
    b0, bb = mask_box(m), mask_box(vs["buried"].mask)
    sh = b0[3] - b0[1]
    assert bb[3] <= b0[3] + 0.02 * sh + 1                         # nothing below the (wavy) ground line
    assert bb[1] > b0[1] + 0.05 * sh                              # but the head is lower: the subject was moved down
    below = contact_band(m, 0.15) & ~dilate(vs["buried"].mask | m, 3)
    assert np.abs(vs["buried"].image[below].astype(int) - img[below].astype(int)).max() == 0


def test_no_shadow_lifts_the_dark_strip_without_inpainting():
    img, m, vs = _suite()
    strip = contact_band(m, 0.15)
    outside = ~dilate(strip, 6) & ~dilate(m, 3)
    assert np.abs(vs["no_shadow"].image[outside].astype(int) - vs["null"].image[outside].astype(int)).max() == 0
    Ln, Lu = _L(vs["null"].image)[strip], _L(vs["no_shadow"].image)[strip]
    assert Lu.mean() >= Ln.mean() + 0.5 and (Lu >= Ln - 0.5).all()  # only lighter, never darker


def test_null_unchanged_inside_the_subject():
    img, m, vs = _suite()
    inside = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0  # the 1-2 px edge is alpha-blended with the plate
    assert np.abs(vs["null"].image[inside].astype(int) - img[inside].astype(int)).max() <= 3
