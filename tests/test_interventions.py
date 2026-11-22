"""validation/interventions + experiments/lighting_intervention.summarize: torch-free checks on synthetic images."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from lighting_intervention import summarize  # noqa: E402
from validation.interventions import (build_variants, hue_rotate, l_image, light_cast, light_gain, light_side,  # noqa: E402
                                      mirror_subject, standardize, texture_blur, texture_shuffle)


def photo(h=160, w=200, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (90, 120, 80), np.uint8)
    mask = np.zeros((h, w), bool)
    mask[30:140, 60:150] = True
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5)
    tex = (tex / tex.std() * 18).astype(np.float32)
    subj = np.clip(np.array([170, 120, 90], np.float32)[None, None] + tex[..., None], 0, 255)
    img[mask] = subj[mask].astype(np.uint8)
    return img, mask


def test_everything_outside_the_mask_is_untouched_except_mirror():
    img, mask = photo()
    far = ~cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    for out in (light_gain(img, mask, 0.7), light_side(img, mask, 0.4), light_cast(img, mask, 8, 12), texture_blur(img, mask, 3),
                texture_shuffle(img, mask, 10, 6), hue_rotate(img, mask, 60)):
        assert out.shape == img.shape and (out[far] == img[far]).all()
        assert (out[mask] != img[mask]).any()


def test_variant_suite_shape():
    img, mask = photo()
    plate = img.copy()
    plate[mask] = (90, 120, 80)
    vs = build_variants(img, mask, plate)
    fams = {v.family for v in vs}
    assert fams == {"null", "nuisance", "light", "struct"}
    assert len({v.name for v in vs}) == len(vs)
    assert {v.level for v in vs if v.family in ("light", "struct")} == {1, 2, 3}
    null = next(v for v in vs if v.family == "null")
    assert np.abs(null.image.astype(int) - img.astype(int)).max() <= 2     # the null is the photo up to the Lab round trip


def test_standardisation_removes_gain_and_cast_but_not_texture_changes():
    img, mask = photo()
    ref = l_image(img, mask, "lab")[mask].astype(float)
    d = lambda x: float(np.abs(l_image(x, mask, "lab")[mask].astype(float) - ref).mean())  # noqa: E731
    light = max(d(light_gain(img, mask, 0.7)), d(light_gain(img, mask, 1.3)), d(light_cast(img, mask, 8, 12)))
    struct = min(d(texture_blur(img, mask, 3)), d(texture_shuffle(img, mask, 10, 6)), d(hue_rotate(img, mask, 60)))
    # global standardisation removes a uniform gain / cast (residual = 8-bit + gamut-clip noise) but NOT a left-right gradient (side light),
    # which is exactly what experiments/lighting_intervention.py measures on real photos; so 'side' is not part of this check
    assert light < 8.0 and struct > 1.5 * light, (light, struct)
    assert d(light_side(img, mask, 0.3)) > 3 * light
    z = standardize(img, mask, "rgb")
    assert abs(z[mask].mean()) < 1e-3 and abs(z[mask].std(0).mean() - 1) < 1e-2 and not z[~mask].any()


def test_mirror_flips_inside_bbox():
    img, mask = photo()
    img[40:60, 60:80] = (255, 0, 0)          # a red patch on the subject's left edge
    plate = np.full_like(img, (90, 120, 80))
    out = mirror_subject(img, mask, plate)
    assert out[50, 140, 0] > 200 and out[50, 70, 0] < 200


def test_summarize_separates_a_good_measure_from_a_blind_one():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(8):
        s = f"p_{i:02d}"
        base = {"sample": s}
        rows.append({**base, "family": "null", "level": 0, "good": 0.994 + 0.002 * rng.random(), "blind": 1.0 - 0.1 * rng.random()})
        rows.append({**base, "family": "nuisance", "level": 0, "good": 0.994 + 0.002 * rng.random(), "blind": 0.95 - 0.1 * rng.random()})
        for lv in (1, 2, 3):
            rows.append({**base, "family": "light", "level": lv, "good": 0.994 + 0.002 * rng.random(), "blind": 0.95 - 0.1 * rng.random()})
            rows.append({**base, "family": "struct", "level": lv, "good": 0.9 - 0.1 * lv - 0.01 * rng.random(), "blind": 0.95 - 0.1 * rng.random()})
    ids = sorted({r["sample"] for r in rows})
    good, blind = summarize(rows, "good", ids), summarize(rows, "blind", ids)
    assert good["PASS"] and good["PASS_clearly"] and good["P1"]["auroc"] <= 0.65 and good["P3"]["auroc"] == 1.0
    assert not blind["PASS"]
