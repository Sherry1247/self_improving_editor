"""Critics v3: lighting is covariant (must follow the new scene), structure is invariant.

Constructed pairs with a known right answer, no humans involved:
  pasted : subject keeps its ORIGINAL lighting in the new scene (copy-paste look)
  relit  : same subject, same texture, but re-lit by the new scene (gain gradient = side light, desaturated tint)
A covariance-aware critic must (1) not call the re-lit subject 'drifted' and (2) prefer it over the pasted one.
"""

import numpy as np
import pytest

from src.config import load_config
from src.critics import CriticContext
from src.critics.keep import AppearanceCritic
from src.critics.world import LightHarmonyCritic
from src.regions import partition
from src.spec import build_spec
from validation.relight import relight_subject
from fakes import FakePerceiver, make_scene, subject_mask

CFG = load_config()
SPEC = build_spec("dog_sit_river_01", "dog", "sit", "river", "snow")


class MaskPerceiver(FakePerceiver):
    """Always returns the known subject mask, so relit colours cannot confuse the colour-key detector."""

    def find_subjects(self, img):
        return [subject_mask()]


def relight(img, mask, bg_rgb):
    """Fixed-colour variant of validation.relight for the flat synthetic scenes."""
    return relight_subject(img, mask, strength=0.35, side=0.18)


def context(after_img):
    per = MaskPerceiver()
    before_img, _ = make_scene("river")
    before, after = per.perceive(before_img, SPEC), per.perceive(after_img, SPEC)
    return CriticContext(before, after, partition(after.subject_mask, before.subject_mask), SPEC,
                         partition(before.subject_mask, before.subject_mask))


def crit(cls, **kw):
    params = dict(CFG["critics"][cls.name])
    params.update(kw)
    return cls(**params)


@pytest.fixture(scope="module")
def pair():
    pasted, m = make_scene("snow")
    relit = relight(pasted, m, (232, 232, 242))
    return context(pasted), context(relit)


def test_v3_appearance_ignores_relighting_v2_does_not(pair):
    _, ctx_relit = pair
    v3 = crit(AppearanceCritic, variant="v3").evaluate(ctx_relit)
    v2 = crit(AppearanceCritic, variant="v2").evaluate(ctx_relit)
    assert v3.score > 0.8, v3.evidence
    assert v3.score > v2.score + 0.15, (v3.score, v2.score)


def test_v3_appearance_still_catches_texture_loss():
    good, _ = make_scene("snow")
    flat, _ = make_scene("snow", stripes=False)
    c = crit(AppearanceCritic, variant="v3")
    assert c.evaluate(context(flat)).score < 0.4 < 0.8 < c.evaluate(context(good)).score


def test_v3_light_harmony_prefers_relit_over_pasted(pair):
    ctx_pasted, ctx_relit = pair
    v3 = crit(LightHarmonyCritic, variant="v3")
    p, r = v3.evaluate(ctx_pasted), v3.evaluate(ctx_relit)
    assert r.score > p.score + 0.2, (p.score, r.score, p.evidence, r.evidence)
    assert p.evidence["follow"] < 0.05  # the subject did not follow the scene at all
    assert r.evidence["follow"] > 0.25, r.evidence


def test_v3_light_harmony_penalises_random_cast_more_than_relight(pair):
    pasted, m = make_scene("snow")
    cast = pasted.copy()
    cast[m] = np.clip(cast[m].astype(np.float32) * np.array([0.7, 1.0, 1.35]) + np.array([0, 25, 0]), 0, 255)
    cast = cast.astype(np.uint8)
    v3 = crit(LightHarmonyCritic, variant="v3")
    assert v3.evaluate(context(cast)).score < v3.evaluate(pair[1]).score


def test_light_harmony_skips_cochange_when_background_light_is_unchanged():
    same, m = make_scene("river")
    res = crit(LightHarmonyCritic, variant="v3").evaluate(context(same))
    assert res.evidence["cochange"] is None and res.score == pytest.approx(res.evidence["mismatch_score"])
