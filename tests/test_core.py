import numpy as np
import pytest

from src.config import apply_overrides, load_config
from src.refinement.prompts import MAX_CLAUSES, add_clause, build_prompt
from src.regions import partition
from src.spec import build_spec, targets_for
from src.utils.geometry import box_iou, mask_iou, nms, similarity_align
from fakes import subject_mask


def test_config_defaults_and_overrides():
    cfg = load_config()
    assert cfg["loop"]["editor"] == "ip2p"
    cfg2 = apply_overrides(cfg, ["loop.max_edits=3", "critics.identity.lo=0.5"])
    assert cfg2["loop"]["max_edits"] == 3 and cfg2["critics"]["identity"]["lo"] == 0.5
    assert cfg["loop"]["max_edits"] != 3  # original untouched


def test_spec_and_targets():
    s = build_spec("dog_sit_river_01", "dog", "sit", "river", "snow", submerged=True)
    assert s.subject == "dog" and s.pose == "sitting" and s.task_id == "dog_sit_river_01__to_snow"
    assert "reflection_removed" in s.covariants
    assert not s.old_bg_may_reappear
    assert build_spec("x", "dog", "sit", "river", "beach").old_bg_may_reappear  # beach has water
    assert targets_for("river") == ["snow", "beach", "city", "forest", "indoor", "mountain"]
    with pytest.raises(ValueError):
        build_spec("x", "cat", "sit", "river", "snow")


def test_prompt_clause_cap():
    s = build_spec("x", "adult_person", "stand", "mountain", "city")
    c = ()
    for k in ["preserve", "target", "contact", "light", "edges"]:
        c = add_clause(c, k)
    assert len(c) == MAX_CLAUSES and c[-1] == "edges"
    p = build_prompt(s, "instruction", c)
    assert p.startswith("replace the background") and "clean natural edges" in p
    assert build_prompt(s, "scene").startswith("a photo of a person standing")


def test_geometry():
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert nms([(0, 0, 10, 10), (1, 1, 10, 10), (20, 20, 30, 30)], [0.9, 0.8, 0.7], 0.5) == [0, 2]
    a = subject_mask(cx=0.4, bottom=0.8, sw=0.2, shh=0.3)
    b = subject_mask(cx=0.6, bottom=0.9, sw=0.3, shh=0.45)  # shifted + scaled
    assert mask_iou(a, b) < 0.5
    warped, tf = similarity_align(a, b)
    assert mask_iou(warped, b) > 0.9 and tf["scale"] == pytest.approx(1.5, rel=0.05)


def test_partition_regions():
    m = subject_mask()
    r = partition(m, m)
    assert not (r.subject & r.background).any()
    assert not (r.contact & r.subject).any()
    ys = np.nonzero(r.contact)[0]
    assert ys.mean() > np.nonzero(m)[0].mean()  # contact band lies at the feet
    assert r.boundary_outer.any() and not (r.boundary_outer & r.subject).any()
    assert r.boundary_inner.any() and (r.boundary_inner <= r.subject).all()
    assert mask_iou(r.aligned_before_subject, m) > 0.99
