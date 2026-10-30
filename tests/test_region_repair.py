import numpy as np

from src.refinement.region_repair import blend_region, contact_region, footprint_region, misplaced_region
from src.regions import partition


def _subject(h=200, w=200):
    m = np.zeros((h, w), bool)
    m[60:140, 80:120] = True
    m[130:140, 70:130] = True  # wider feet
    return m


def test_footprint_follows_feet_not_box():
    s = _subject()
    reg = partition(s, s)
    rect = contact_region(s, reg.contact, "rect")
    fp = contact_region(s, reg.contact, "footprint")
    assert fp.any() and not (fp & s).any()
    assert fp.sum() < rect.sum()  # tighter than the rectangle
    ys, xs = np.where(fp)
    assert ys.min() >= 128 and xs.min() >= 60 and xs.max() <= 140  # hugs the feet columns


def test_cast_shears_with_light():
    s = _subject()
    reg = partition(s, s)
    right = contact_region(s, reg.contact, "cast", 0.8)
    left = contact_region(s, reg.contact, "cast", -0.8)
    assert np.where(right)[1].mean() > np.where(left)[1].mean()


def test_misplaced_has_same_area_and_avoids_subject():
    s = _subject()
    reg = partition(s, s)
    c = contact_region(s, reg.contact, "footprint")
    m = misplaced_region(c, s, c, seed=1)
    assert m is not None and m.sum() == c.sum() and not (m & s).any() and not (m & c).any()


def test_blend_freezes_subject_and_outside():
    s = _subject()
    img = np.full((200, 200, 3), 50, np.uint8)
    gen = np.full((200, 200, 3), 250, np.uint8)
    reg = partition(s, s)
    c = contact_region(s, reg.contact, "footprint")
    out = blend_region(img, gen, c, s)
    assert (out[s] == 50).all()
    far = np.zeros_like(s); far[:20, :20] = True
    assert (out[far] == 50).all()
    assert (out[c & ~s].mean() > 100)
