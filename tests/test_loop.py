"""End-to-end closed loop on CPU with the fake editor / perceiver."""

import numpy as np

from src.config import apply_overrides, load_config
from src.critics import build_critics
from src.pipelines import Evaluator, RefinementLoop
from src.refinement import ActionMemory, ActionRouter
from src.scoring import GatedGeometricAggregator
from src.spec import build_spec
from fakes import FakeEditor, FakePerceiver, make_scene

SPEC = build_spec("dog_sit_river_01", "dog", "sit", "river", "snow")


def make_loop(cfg, editor, memory=None):
    ev = Evaluator(FakePerceiver(), build_critics(cfg), GatedGeometricAggregator(cfg))
    return RefinementLoop(editor, ev, ActionRouter(editor, memory), cfg)


def test_loop_escapes_unchanged_and_respects_budget(tmp_path):
    cfg = apply_overrides(load_config(), ["loop.max_edits=8", "loop.n_candidates=2", "loop.threshold=0.99",
                                          "loop.patience=3"])
    editor = FakeEditor(start_igs=2.25)  # starts in the "returns the input" regime
    loop = make_loop(cfg, editor, ActionMemory(tmp_path / "mem.json"))
    before = FakePerceiver().perceive(make_scene("river")[0], SPEC)
    res = loop.run(before, SPEC)

    assert editor.calls == res.edits_used <= 8
    first = res.rounds[0]
    assert first["round_best"] == 0.0  # gated: background unchanged
    assert res.best.score > 0.5 and res.best.evaluation.gate_passed
    assert res.best.state.params["image_guidance_scale"] < 2.25  # router turned the preserve knob down
    assert res.rounds[1]["action"] == "edit_more" and res.rounds[1]["issue"] == "bg_unchanged"
    assert (tmp_path / "mem.json").exists()


def test_best_is_kept_when_a_round_regresses():
    cfg = apply_overrides(load_config(), ["loop.max_edits=6", "loop.n_candidates=1", "loop.threshold=0.99",
                                          "loop.patience=5"])
    editor = FakeEditor(start_igs=1.5)  # clean swap from the start; pushing "edit_more" loses texture
    loop = make_loop(cfg, editor)
    before = FakePerceiver().perceive(make_scene("river")[0], SPEC)
    res = loop.run(before, SPEC)
    scores = [c.score for c in res.candidates]
    assert res.best.score == max(scores)
    # first result has no contact shadow -> router adds the 'contact' clause -> shadow appears
    assert res.rounds[1]["issue"] == "missing_shadow" and "contact" in res.rounds[1]["clauses"]
    assert res.rounds[1]["round_best"] > res.rounds[0]["round_best"]
    # later rounds that do not improve are reverted, never replacing the best
    for r in res.rounds[2:]:
        assert r["accepted"] == (r["round_best"] > r["best_before"] + 1e-3)


def test_memory_ranking(tmp_path):
    from src.refinement.router import ISSUE_ACTIONS
    from src.types import IssueType

    mem = ActionMemory(tmp_path / "m.json")
    acts = ISSUE_ACTIONS[IssueType.BG_WRONG]  # [clause_target, edit_more]
    assert [a.name for a in mem.rank(IssueType.BG_WRONG, "ip2p", acts)][0] == "clause_target"
    mem.update(IssueType.BG_WRONG, "ip2p", "clause_target", -0.2)
    mem.update(IssueType.BG_WRONG, "ip2p", "edit_more", 0.3)
    assert mem.rank(IssueType.BG_WRONG, "ip2p", acts)[0].name == "edit_more"
    assert ActionMemory(tmp_path / "m.json").stats == mem.stats  # persisted


def test_registry_swap_policy():
    import torch

    from src.models.registry import ModelBundle, ModelRegistry

    loads = []

    def loader(mcfg, dtype, device):
        loads.append(mcfg["id"])
        return ModelBundle("lin", mcfg["id"], torch.nn.Linear(2, 2))

    cfg = {"device": "cpu", "memory_policy": "swap", "models": {"a": {"id": "A"}, "b": {"id": "B"}}}
    reg = ModelRegistry(cfg, loaders={"a": loader, "b": loader})
    with reg.use("a") as m:
        assert m.model(torch.ones(1, 2)).shape == (1, 2)
    with reg.use("a"):
        pass
    with reg.use("b"):
        pass
    assert loads == ["A", "B"]  # loaded once each
    assert sorted(reg.loaded()) == ["a", "b"]


def test_violation_suite_shapes():
    from validation.violations import auroc, make_violations

    img, m = make_scene("river")
    foreign = make_scene("snow")[0]
    vs = make_violations(img, m, foreign)
    kinds = {v.kind for v in vs}
    assert {"float", "sink", "scale_up", "scale_down", "no_shadow", "color_cast", "halo", "foreign_clean"} <= kinds
    for v in vs:
        assert v.image.shape == img.shape and v.mask.shape == m.shape
    fl = next(v for v in vs if v.kind == "float")
    assert np.nonzero(fl.mask)[0].mean() < np.nonzero(m)[0].mean()  # moved up
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0 and auroc([0.5], [0.5]) == 0.5
