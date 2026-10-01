"""Critic behaviour on controlled synthetic edits: each test changes ONE thing and checks the right critic reacts."""

import numpy as np
import pytest

from src.config import load_config
from src.critics import CriticContext, build_critics
from src.pipelines import Evaluator
from src.regions import partition
from src.scoring import GatedGeometricAggregator, WeightedSumAggregator
from src.spec import build_spec
from src.types import CriticResult, IssueType
from fakes import FakePerceiver, make_scene, subject_mask

CFG = load_config()
CFG["critics"]["enabled"] = CFG["critics"]["enabled"] + ["contact_shadow"]  # ablation critic, tested here
SPEC = build_spec("dog_sit_river_01", "dog", "sit", "river", "snow")


@pytest.fixture(scope="module")
def stack():
    per = FakePerceiver()
    ev = Evaluator(per, build_critics(CFG), GatedGeometricAggregator(CFG))
    before_img, _ = make_scene("river")
    return per, ev, per.perceive(before_img, SPEC)


def run(stack, after_img):
    per, ev, before = stack
    res, _, _ = ev.evaluate(before, after_img, SPEC)
    return res


def test_good_swap_passes(stack):
    after, _ = make_scene("snow")
    r = run(stack, after)
    assert r.gate_passed, r.to_dict()
    assert r.critics["silhouette"].score > 0.95
    assert r.critics["bg_semantic"].evidence["top"] == "snow"
    assert r.overall > 0.6


def test_unchanged_image_is_gated(stack):
    """The classic reward hack: return the input. Preservation is perfect, but the gate must zero it."""
    r = run(stack, stack[2].image.copy())
    assert r.critics["bg_changed"].is_catastrophic
    assert IssueType.BG_UNCHANGED in r.issues
    assert r.overall == 0.0
    # the legacy flat weighted sum is fooled by it
    assert r.branch_scores["weighted_sum_baseline"] > 0.6


def test_extra_subject_is_gated(stack):
    after, m = make_scene("snow")
    second = subject_mask(cx=0.15, bottom=0.95, sw=0.15, shh=0.2)
    after[second] = np.array([200, 40, 40], np.uint8)
    r = run(stack, after)
    assert r.critics["subject_count"].is_catastrophic and IssueType.EXTRA_SUBJECT in r.issues


def test_lighting_change_is_not_drift(stack):
    """A global colour/brightness shift on the subject is allowed (it should adapt to the new scene)."""
    after, _ = make_scene("snow", subject_rgb=(222, 58, 52))
    r = run(stack, after)
    app = r.critics["appearance"]
    assert app.evidence["raw_dist"] > app.evidence["normalized_dist"] * 3
    assert app.score > 0.8


def test_texture_loss_is_drift(stack):
    after, _ = make_scene("snow", stripes=False)
    good, _ = make_scene("snow")
    assert run(stack, after).critics["appearance"].score < run(stack, good).critics["appearance"].score


def test_silhouette_drift(stack):
    m = subject_mask(sw=0.42, shh=0.25)  # very different shape
    after, _ = make_scene("snow", mask=m)
    r = run(stack, after)
    assert r.critics["silhouette"].score < 0.75 and IssueType.SUBJECT_DRIFT in r.issues


def test_missing_shadow(stack):
    with_s, _ = make_scene("snow", shadow=0.3)
    without, _ = make_scene("snow", shadow=0.0)
    s_with = run(stack, with_s).critics["contact_shadow"]
    s_without = run(stack, without).critics["contact_shadow"]
    assert s_with.score > s_without.score
    assert IssueType.MISSING_SHADOW in s_without.issues


def test_support_floating():
    per = FakePerceiver()
    before_img, m = make_scene("river")
    before = per.perceive(before_img, SPEC)
    after_img, _ = make_scene("snow")
    after = per.perceive(after_img, SPEC)
    # make the subject 'float': its depth no longer matches the ground right below it
    after.depth = after.depth.copy()
    after.depth[after.subject_mask] = 0.3
    regions = partition(after.subject_mask, before.subject_mask)
    from src.critics.world import SupportCritic

    res = SupportCritic(**CFG["critics"]["support"]).evaluate(
        CriticContext(before, after, regions, SPEC, partition(before.subject_mask, before.subject_mask)))
    assert res.score < 0.5 and IssueType.FLOATING in res.issues


def test_halo(stack):
    clean, m = make_scene("snow")
    halo = clean.copy()
    from src.utils.geometry import dilate

    ring = dilate(m, 6) & ~m
    halo[ring] = stack[2].image[ring]  # old river pixels left around the subject
    assert run(stack, halo).critics["halo"].score < run(stack, clean).critics["halo"].score
    assert IssueType.HALO in run(stack, halo).issues


def test_aggregators():
    cr = {
        "g": CriticResult("g", "gate", 1.0),
        "k": CriticResult("k", "keep", 0.9),
        "f": CriticResult("f", "follow", 0.1),
        "w": CriticResult("w", "world", None),
    }
    geo = GatedGeometricAggregator(CFG).aggregate(cr)
    ws = WeightedSumAggregator(CFG).aggregate(cr)
    assert geo.overall == pytest.approx(np.sqrt(0.9 * 0.1), rel=1e-3)  # world n/a -> skipped
    assert ws.overall > geo.overall  # one strong branch cannot buy back a failed one under geo
    cr["g"] = CriticResult("g", "gate", 0.0, is_catastrophic=True)
    assert GatedGeometricAggregator(CFG).aggregate(cr).overall == 0.0
