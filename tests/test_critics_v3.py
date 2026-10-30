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
from src.critics.keep import AppearanceCritic, refine_zncc, structure_zncc
from src.critics.follow import UnchangedBackgroundCritic
from src.critics.world import LightHarmonyCritic
from src.regions import partition
from src.spec import build_spec
from validation.relight import relight_subject
from fakes import FakePerceiver, make_scene, subject_mask

CFG = load_config()
CFG["critics"]["unchanged_bg"] = {"tau": 7.0, "min_area_frac": 0.004, "sigma": 0.08}
SPEC = build_spec("dog_sit_river_01", "dog", "sit", "river", "snow")


class MaskPerceiver(FakePerceiver):
    """Always returns the known subject mask, so relit colours cannot confuse the colour-key detector."""

    def find_subjects(self, img):
        return [subject_mask()]


def relight(img, mask, bg_rgb):
    """Fixed-colour variant of validation.relight for the flat synthetic scenes."""
    return relight_subject(img, mask, strength=0.6, side=0.8)  # strong side light: local shading a global fit cannot undo


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
    assert v3.score > v2.score + 0.08, (v3.score, v2.score)


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


def test_null_relight_is_pipeline_noise_only():
    pasted, m = make_scene("snow")
    null = relight_subject(pasted, m, strength=0.0, side=0.0)
    assert np.abs(null.astype(int) - pasted.astype(int)).max() <= 3
    a, b = crit(AppearanceCritic).evaluate(context(pasted)), crit(AppearanceCritic).evaluate(context(null))
    assert abs(a.score - b.score) < 0.05


def test_relight_keeps_texture_and_changes_light():
    pasted, m = make_scene("snow")
    relit = relight_subject(pasted, m, strength=0.3, side=0.15)
    assert np.abs(relit.astype(int) - pasted.astype(int))[m].mean() > 3  # it really changed the subject
    assert (relit[~m] == pasted[~m]).mean() > 0.95  # and left the scene alone


def test_refine_zncc_recovers_a_misregistered_texture():
    rng = np.random.default_rng(0)
    base = cv2_blur(rng.normal(0, 1, (120, 120)).astype(np.float32))
    region = np.zeros((120, 120), bool)
    region[30:90, 30:90] = True
    shifted = np.roll(base, 4, axis=1)
    raw = structure_zncc(base, shifted, region, 4.0)
    refined, (dx, dy) = refine_zncc(base, shifted, region, 4.0, 8.0, raw)
    assert raw < 0.7 and refined > 0.9 and abs(abs(dx) - 4) < 1.5, (raw, refined, dx, dy)


def cv2_blur(a):
    import cv2

    return cv2.GaussianBlur(a, (0, 0), 1.5)


def test_appearance_flags_unreliable_alignment_instead_of_scoring_zero():
    m = subject_mask(sw=0.5, shh=0.12)  # a very different silhouette (aligned IoU ~0.3)
    after, _ = make_scene("snow", mask=m)
    per = FakePerceiver()
    before_img, _ = make_scene("river")
    before, aft = per.perceive(before_img, SPEC), per.perceive(after, SPEC)
    ctx = CriticContext(before, aft, partition(aft.subject_mask, before.subject_mask), SPEC,
                        partition(before.subject_mask, before.subject_mask))
    res = crit(AppearanceCritic, variant="v3").evaluate(ctx)
    assert res.score is None and res.evidence["unreliable_alignment"] is True
    scored = crit(AppearanceCritic, variant="v3", unreliable_policy="score").evaluate(ctx)
    assert scored.score is not None


def test_cochange_testability_is_reported(pair):
    pasted, relit = pair
    lh = crit(LightHarmonyCritic, variant="v3")
    assert lh.evaluate(pasted).evidence["cochange_tested"] is True
    same, _ = make_scene("river")
    assert lh.evaluate(context(same)).evidence["cochange_tested"] is False  # scene lighting unchanged: untested


def test_unchanged_bg_detects_kept_old_scene_chunks():
    clean, m = make_scene("snow")
    before, _ = make_scene("river")
    kept = clean.copy()
    kept[:60, :50] = before[:60, :50]  # a chunk of the old scene stays pixel-identical (railing / bench)
    c = crit(UnchangedBackgroundCritic)
    good, bad = c.evaluate(context(clean)), c.evaluate(context(kept))
    assert bad.evidence["blob_frac"] > good.evidence["blob_frac"] + 0.05
    assert bad.score < good.score
