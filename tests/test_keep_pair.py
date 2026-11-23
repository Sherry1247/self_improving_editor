"""app_structure + app_chroma: the two factors of the v3 appearance score as separate Keep critics (identity gate replaced, docs/results/2026-11-23-*)."""
import numpy as np
import pytest

from src.config import load_config
from src.critics import CRITICS, CriticContext, build_critics
from src.critics.keep import AppearanceChromaCritic, AppearanceCritic, AppearanceStructureCritic
from src.regions import partition
from validation.relight import relight_subject
from fakes import FakePerceiver, make_scene, subject_mask
from test_critics_v3 import SPEC, context, crit

CFG = load_config()


def pair(pasted, relit_strength=0.3, side=0.15):
    return relight_subject(pasted, subject_mask(), strength=relit_strength, side=side)


def scores(ctx):
    return (crit(AppearanceCritic, variant="v3").evaluate(ctx), crit(AppearanceStructureCritic).evaluate(ctx),
            crit(AppearanceChromaCritic).evaluate(ctx))


def test_the_two_factors_multiply_to_the_v3_score():
    pasted, _ = make_scene("snow")
    flat, _ = make_scene("snow", stripes=False)
    for img in (pasted, pair(pasted), flat):
        app, st, ch = scores(context(img))
        assert app.score == pytest.approx(st.score * ch.score, abs=1e-9), (app.score, st.score, ch.score)


def test_structure_ignores_relighting_and_sees_lost_texture():
    pasted, _ = make_scene("snow")
    flat, _ = make_scene("snow", stripes=False)
    _, st_relit, ch_relit = scores(context(pair(pasted)))
    _, st_flat, ch_flat = scores(context(flat))
    assert st_relit.score > 0.8 and ch_relit.score > 0.8
    assert st_flat.score < 0.4 < st_relit.score


def test_a_mean_colour_shift_is_not_counted_by_the_chroma_factor():
    pasted, m = make_scene("snow")
    cast = pasted.copy()
    cast[m] = np.clip(cast[m].astype(np.float32) + np.array([6, 0, -10]), 0, 255)
    _, st, ch = scores(context(cast.astype(np.uint8)))
    assert ch.score > 0.9 and st.score > 0.8, (st.score, ch.score)


def test_unreliable_alignment_is_not_applicable_for_both_factors():
    m = subject_mask(sw=0.5, shh=0.12)
    after, _ = make_scene("snow", mask=m)
    per = FakePerceiver()
    before_img, _ = make_scene("river")
    before, aft = per.perceive(before_img, SPEC), per.perceive(after, SPEC)
    ctx = CriticContext(before, aft, partition(aft.subject_mask, before.subject_mask), SPEC, partition(before.subject_mask, before.subject_mask))
    app, st, ch = scores(ctx)
    assert app.score is None and st.score is None and ch.score is None


def test_registered_in_the_keep_branch_and_parameters_match_appearance():
    assert CRITICS["app_structure"].branch == "keep" and CRITICS["app_chroma"].branch == "keep"
    shared = {k: v for k, v in CFG["critics"]["appearance"].items() if k not in ("variant", "issue_below")}
    for name in ("app_structure", "app_chroma"):
        assert all(CFG["critics"][name].get(k) == v for k, v in shared.items()), name   # same alignment / ramp parameters as v3
    cfg = load_config()
    cfg["critics"]["enabled"] = ["silhouette", "app_structure", "app_chroma", "texture"]
    assert [c.name for c in build_critics(cfg)] == ["silhouette", "app_structure", "app_chroma", "texture"]
