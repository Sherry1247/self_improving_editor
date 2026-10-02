"""Tests for the fixes made after the first CHTC runs (docs/results/2026-10-01-first-chtc-runs.md)."""

import numpy as np
import pytest

from src.config import apply_overrides, load_config
from src.critics import CriticContext, build_critics
from src.critics.keep import AppearanceCritic, SilhouetteCritic
from src.critics.vlm import VLM_CRITICS
from src.pipelines import Evaluator, RefinementLoop
from src.refinement import ActionRouter
from src.regions import partition
from src.scoring import GatedGeometricAggregator
from src.spec import build_spec
from src.types import CriticResult, IssueType
from src.utils.geometry import choose_alignment, fill_holes, similarity_align
from fakes import FakeEditor, FakePerceiver, make_scene, subject_mask

CFG = load_config()
SPEC = build_spec("dog_sit_river_01", "dog", "sit", "river", "snow", submerged=True)


# 1. alignment: a pixel-exact paste whose mask GREW (revealed legs) must align with identity ---------------
def test_identity_alignment_when_mask_grows():
    before = subject_mask(bottom=0.75, shh=0.35)
    after = before | subject_mask(bottom=0.85, sw=0.12, shh=0.25)  # legs revealed below
    warped_m, tf_m = similarity_align(before, after)
    assert tf_m["scale"] > 1.02 or abs(tf_m["dy"]) > 2  # the moment method drifts
    warped, tf = choose_alignment(before, after)
    assert tf == {"dx": 0.0, "dy": 0.0, "scale": 1.0}
    assert (warped == before).all()


def test_moment_alignment_when_subject_moved():
    before = subject_mask(cx=0.4)
    after = subject_mask(cx=0.6)
    _, tf = choose_alignment(before, after)
    assert tf["dx"] > 10


def test_fill_holes():
    m = subject_mask()
    holed = m.copy()
    ys, xs = np.nonzero(m)
    holed[ys[len(ys) // 2], xs[len(xs) // 2]] = False
    assert fill_holes(holed).sum() == m.sum()


# 2. revealed legs are not "drift" -------------------------------------------------------------------------
def test_revealed_parts_not_penalised():
    per = FakePerceiver()
    img_b, m_b = make_scene("river", mask=subject_mask(bottom=0.75, shh=0.35))
    before = per.perceive(img_b, SPEC)
    # pretend the lower part of the image was water (old background) in the before photo
    before.old_bg_mask = np.zeros_like(m_b)
    before.old_bg_mask[int(0.6 * m_b.shape[0]):] = True
    before.old_bg_mask &= ~m_b
    grown = m_b | subject_mask(bottom=0.85, sw=0.12, shh=0.25)
    img_a, _ = make_scene("snow", mask=grown)
    after = per.perceive(img_a, SPEC)
    ctx = CriticContext(before, after, partition(after.subject_mask, before.subject_mask), SPEC)
    sil = SilhouetteCritic(**CFG["critics"]["silhouette"]).evaluate(ctx)
    assert sil.evidence["revealed_px"] > 0
    assert sil.score > sil.evidence["raw_iou"] and sil.score > 0.95
    app = AppearanceCritic(**CFG["critics"]["appearance"]).evaluate(ctx)
    assert app.score > 0.9  # identical subject pixels, identity alignment -> no drift
    # growth SIDEWAYS (not into the old water) is still penalised
    side = m_b | subject_mask(cx=0.65, bottom=0.55, sw=0.3, shh=0.2)  # attached arm-like growth, above centre
    img_s, _ = make_scene("snow", mask=side)
    after_s = per.perceive(img_s, SPEC)
    ctx_s = CriticContext(before, after_s, partition(after_s.subject_mask, before.subject_mask), SPEC)
    assert SilhouetteCritic().evaluate(ctx_s).score < sil.score


# 3. bg_changed is graded in the follow branch and still gates ------------------------------------------
def test_bg_changed_graded_and_gating():
    per = FakePerceiver()
    ev = Evaluator(per, build_critics(CFG), GatedGeometricAggregator(CFG))
    before = per.perceive(make_scene("river")[0], SPEC)
    good, _, _ = ev.evaluate(before, make_scene("snow")[0], SPEC)
    assert good.critics["bg_changed"].branch == "follow" and good.critics["bg_changed"].score > 0
    same, _, _ = ev.evaluate(before, before.image.copy(), SPEC)
    assert same.gate_passed is False and same.overall == 0.0


# 4. VLM critics are relative to the real photo ---------------------------------------------------------
def _ctx_with_vlm(vb, va):
    per = FakePerceiver()
    b = per.perceive(make_scene("river")[0], SPEC)
    a = per.perceive(make_scene("snow")[0], SPEC)
    b.vlm, a.vlm = vb, va
    return CriticContext(b, a, partition(a.subject_mask, b.subject_mask), SPEC)


def test_vlm_critics_relative_and_absolute():
    crit = {c.name: c(**CFG["critics"].get(c.name, {})) for c in VLM_CRITICS}
    ctx = _ctx_with_vlm({"shadow": 0.2, "support": 0.9}, {"shadow": 0.45, "support": 0.3, "background": 0.8})
    sh = crit["vlm_shadow"].evaluate(ctx)  # real photo itself shows little shadow -> requirement relaxed
    assert sh.score == pytest.approx(0.9) and not sh.issues
    sp = crit["vlm_support"].evaluate(ctx)
    assert sp.score == pytest.approx(0.3 / 0.9) and IssueType.FLOATING in sp.issues
    bg = crit["vlm_background"].evaluate(ctx)
    assert bg.branch == "follow" and bg.score == pytest.approx(0.8)
    assert crit["vlm_lighting"].evaluate(ctx).score is None  # not asked -> not applicable


def test_vlm_question_regions_and_scorer():
    import torch

    from src.perception.vlm import VLMScorer, build_questions, question_images, region_box

    qs = build_questions(SPEC)
    assert {q.key for q in qs} == {"support", "shadow", "surface", "integration", "lighting", "background"}
    img, m = make_scene("snow")
    box = region_box(m, "contact")
    ys = np.nonzero(m)[0]
    assert box[1] < ys.max() < box[3]  # contact box straddles the feet
    assert len(question_images(img, m, "contact")) == 2 and len(question_images(img, m, "full")) == 1

    class Tok:
        def encode(self, t, add_special_tokens=False):
            return [{"Yes": 1, "yes": 2, " Yes": 1, "No": 3, "no": 4, " No": 3}[t]]

    class Proc:
        tokenizer = Tok()

        def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
            assert sum(c["type"] == "image" for c in msgs[0]["content"]) == 2
            return "prompt"

        def __call__(self, text, images, return_tensors):
            return {"input_ids": torch.zeros(1, 3, dtype=torch.long), "pixel_values": torch.zeros(1, 3)}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **kw):
            logits = torch.full((1, 3, 6), -10.0)
            logits[0, -1, 1] = 2.0  # "Yes"
            logits[0, -1, 3] = 0.0  # "No"
            return type("O", (), {"logits": logits})()

    bundle = type("B", (), {"processor": Proc(), "model": Model(), "dtype": torch.float32})()
    p = VLMScorer(bundle).p_yes(question_images(img, m, "contact"), "q?")
    assert p == pytest.approx(torch.sigmoid(torch.logsumexp(torch.tensor([2.0, -10.0]), 0)
                                            - torch.logsumexp(torch.tensor([0.0, -10.0]), 0)).item(), rel=1e-4)


# 5. full-budget loops ignore the threshold ----------------------------------------------------------------
def test_full_budget():
    cfg = apply_overrides(load_config(), ["loop.max_edits=6", "loop.n_candidates=2", "loop.threshold=0.1",
                                          "loop.full_budget=true", "loop.patience=99"])
    ed = FakeEditor(start_igs=1.5)
    ev = Evaluator(FakePerceiver(), build_critics(cfg), GatedGeometricAggregator(cfg))
    res = RefinementLoop(ed, ev, ActionRouter(ed), cfg).run(FakePerceiver().perceive(make_scene("river")[0], SPEC),
                                                          SPEC)
    assert res.edits_used == 6 and res.stopped_reason == "budget"


# 6. violation suite has a same-pipeline positive -----------------------------------------------------------
def test_violation_null_positive():
    from validation.violations import make_violations

    img, m = make_scene("river")
    vs = {v.kind: v for v in make_violations(img, m, make_scene("snow")[0])}
    assert "null" in vs
    assert (vs["null"].mask == fill_holes(m)).all()
    # null and float differ only around the subject; far background identical
    far = np.zeros_like(m)
    far[:, :10] = True
    assert np.abs(vs["null"].image[far].astype(int) - vs["float"].image[far].astype(int)).max() == 0


def test_any_catastrophic_critic_gates():
    cr = {"k": CriticResult("k", "keep", 0.9), "f": CriticResult("f", "follow", 0.0, is_catastrophic=True)}
    assert GatedGeometricAggregator(CFG).aggregate(cr).overall == 0.0


def test_semantic_gate_on_restyle():
    per = FakePerceiver()
    ev = Evaluator(per, build_critics(CFG), GatedGeometricAggregator(CFG))
    before = per.perceive(make_scene("river")[0], SPEC)
    after = per.perceive(make_scene("snow")[0], SPEC)
    after.bg_probs = {k: (0.9 if k == "river" else 0.1 / 6) for k in after.bg_probs}  # still looks like the river
    res, _, _ = ev.evaluate(before, after.image, SPEC, after=after)
    assert res.critics["bg_semantic"].is_catastrophic and res.overall == 0.0
    forest = build_spec("dog_sit_river_01", "dog", "sit", "river", "forest")  # forest may contain a river
    res2, _, _ = ev.evaluate(before, after.image, forest, after=after)
    assert not res2.critics["bg_semantic"].is_catastrophic


def test_world_softmin():
    cr = {"a": CriticResult("a", "world", 1.0), "b": CriticResult("b", "world", 1.0),
          "c": CriticResult("c", "world", 0.2), "d": CriticResult("d", "world", 0.4)}
    assert GatedGeometricAggregator(CFG).branch_scores(cr)["world"] == pytest.approx(0.3)
