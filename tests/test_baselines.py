import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.baselines import rewards, variance
from src.baselines.report import load_scores, run_table, violation_table
from src.baselines.stats import auroc, cluster_bootstrap, paired_accuracy
from src.perception.vlm import build_questions
from src.spec import build_spec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


def test_parse_scores():
    assert rewards.parse_scores("7, 9") == [7.0, 9.0]
    assert rewards.parse_scores("Score: 6.5 and 8") == [6.5, 8.0]
    assert rewards.parse_scores("12, 3, 4") == [3.0, 4.0]  # 12 is out of range
    assert rewards.parse_scores("7") is None and rewards.parse_scores("no idea") is None


def test_viescore_combination():
    replies = iter(["7, 9", "6, 8"])
    r = rewards.viescore(lambda imgs, prompt: next(replies), "src", "edit", "do it")
    assert r["success"] == 0.7 and r["over_edit"] == 0.9 and r["sc"] == 0.7 and r["pq"] == 0.6
    assert r["overall"] == pytest.approx(math.sqrt(0.7 * 0.6))


def test_viescore_unparsable_reply_gives_none_not_zero():
    r = rewards.viescore(lambda imgs, prompt: "I cannot rate this", "s", "e", "x")
    assert r["overall"] is None and r["sc"] is None and r["pq"] is None


def test_viescore_soft_routes_prompts_and_combines_like_viescore():
    seen = []

    def expect(images, prompt):
        seen.append((len(images), prompt))
        return {"follow": 0.9, "unchanged": 0.8, "natural": 0.6, "artifacts": 0.7}[
            "follow" if "follow the instruction" in prompt else "unchanged" if "left unchanged" in prompt
            else "natural" if "natural photograph" in prompt else "artifacts"]

    r = rewards.viescore_soft(expect, "s", "e", "do it")
    assert (r["success"], r["over_edit"], r["naturalness"], r["artifacts"]) == (0.9, 0.8, 0.6, 0.7)
    assert r["sc"] == 0.8 and r["pq"] == 0.6 and r["overall"] == pytest.approx(math.sqrt(0.8 * 0.6))
    assert [n for n, _ in seen] == [2, 2, 1, 1] and "do it" in seen[0][1]  # edit pair vs single-image prompts


def test_paraphrases_cover_every_vlm_question():
    spec = build_spec("dog_sit_river_01", "dog", "sit", "river", "snow", False)
    para = variance.paraphrases(spec)
    assert set(para) == {q.key for q in build_questions(spec)}
    assert all(len(v) == 3 and len(set(v)) == 3 for v in para.values())
    assert variance.view_params("pad_small")["pad"] < variance.view_params("orig")["pad"] < variance.view_params("pad_large")["pad"]


def test_auroc_and_paired_accuracy():
    assert auroc([1, 1], [0, 0]) == 1.0 and auroc([0, 0], [1, 1]) == 0.0 and auroc([1], [1]) == 0.5
    assert auroc([], [1]) is None
    assert paired_accuracy([1, 0, 1], [0, 1, 1]) == pytest.approx((1 + 0 + 0.5) / 3)
    assert paired_accuracy([1], [1, 2]) is None


def test_cluster_bootstrap_is_tight_for_a_perfect_separator_and_wide_for_noise():
    perfect = {f"s{i}": ([1.0], [0.0]) for i in range(8)}
    assert cluster_bootstrap(perfect, auroc, 200) == (1.0, 1.0, 1.0)
    rng = np.random.default_rng(1)
    noise = {f"s{i}": ([float(rng.random())], [float(rng.random())]) for i in range(8)}
    p, lo, hi = cluster_bootstrap(noise, paired_accuracy, 500)
    assert lo < p < hi + 1e-9 and hi - lo > 0.2
    assert cluster_bootstrap({"a": ([1.0], [0.0])}, auroc)[1:] == (None, None)  # one cluster -> no CI


def _rows(run, task, image, question, vals):
    return [{"run": run, "task": task, "image": image, "question": question, "view": v, "p_yes": p}
            for v, p in zip(variance.VIEWS, vals)]


def test_variance_summary_separates_repeat_floor_from_view_spread():
    # orig, orig_rep, para1, para2, pad_small, pad_large, side512
    rows = _rows("a", "t1", "best", "support", [0.80, 0.80, 0.70, 0.90, 0.80, 0.60, 0.80])
    s = variance.summarise(rows)
    assert s["repeat_floor_max"] == 0.0
    q = s["questions"]["support"]
    assert q["mean_range"] == pytest.approx(0.30) and q["max_range"] == pytest.approx(0.30)
    v = s["views"]["support"]
    assert v["pad_large"] == pytest.approx(0.20) and v["pad_small"] == 0.0 and v["para2"] == pytest.approx(0.10)


def test_ranking_stability_counts_flips():
    rows = (_rows("a", "t1", "best", "q", [0.9] * 7) + _rows("b", "t1", "best", "q", [0.5] * 7)   # stable a>b
            + _rows("a", "t2", "best", "q", [0.6, 0.6, 0.6, 0.6, 0.4, 0.6, 0.6])                 # flips under pad_small
            + _rows("b", "t2", "best", "q", [0.5] * 7))
    d = variance.ranking_stability(rows, "a", "b")["q"]
    assert d["tasks"] == 2 and d["sign_stable_tasks"] == 1
    assert d["views"]["orig"]["a_wins"] == 2 and d["views"]["pad_small"]["a_wins"] == 1


def test_load_scores_flattens_metrics(tmp_path):
    p1, p2 = tmp_path / "scores_viescore_vlm.jsonl", tmp_path / "scores_editreward.jsonl"
    p1.write_text(json.dumps({"id": "x", "overall": 0.5, "sc": None, "raw_sc": "7, 8"}) + "\n")
    p2.write_text(json.dumps({"id": "x", "score": 1.5, "raw_dims": [1.5]}) + "\n")
    sc = load_scores([p1, p2])
    assert sc == {"viescore_vlm:overall": {"x": 0.5}, "editreward": {"x": 1.5}}


def test_violation_table_detects_a_sensitive_model_and_a_blind_one():
    manifest, good, blind = [], {}, {}
    for i in range(6):
        s = f"s{i}"
        for kind in ("null", "float", "foreign_clean", "halo"):
            _id = f"v/{s}/{kind}"
            manifest.append({"id": _id, "group": "violation", "sample_id": s, "kind": kind})
            good[_id] = 0.2 if kind in ("float", "halo") else 0.9
            blind[_id] = 0.5
    rows = violation_table(manifest, {"good": good, "blind": blind}, n_boot=50)
    by = {(r["model"], r["kind"]): r for r in rows}
    assert by[("good", "float")]["auroc"] == 1.0 and by[("good", "halo")]["auroc"] == 1.0
    assert by[("blind", "float")]["auroc"] == 0.5 and by[("blind", "halo")]["paired_acc"] == 0.5
    assert {r["kind"] for r in rows} == {"float", "halo"}  # null / foreign_clean are references, not violations


def test_run_table_wins_and_correlation():
    manifest, sc = [], {}
    for i in range(6):
        for run, bonus in (("comp", 0.2), ("qwen", 0.0)):
            _id = f"r/{run}/t{i}"
            manifest.append({"id": _id, "group": "run", "kind": run, "task": f"t{i}", "ours_overall": i / 10 + bonus})
            sc[_id] = i / 10 + bonus  # perfectly rank-aligned with ours
    row = run_table(manifest, {"m": sc})[0]
    assert row["comp>qwen"] == 6 and row["qwen>comp"] == 0
    assert row["spearman_ours"] == pytest.approx(1.0)


def test_build_scoring_set_parses_kinds_and_filters_split(tmp_path):
    import build_scoring_set as bss

    ex = tmp_path / "e1" / "examples"
    ex.mkdir(parents=True)
    for s in ("adult_sit_river_01", "dog_sit_river_03"):
        for k in ("real", "null", "no_shadow", "halo", "foreign_clean"):
            Image.new("RGB", (8, 8), (10, 20, 30)).save(ex / f"{s}_{k}.jpg")
    out = tmp_path / "set"
    (out / "images").mkdir(parents=True)
    items: list = []
    bss.add_violations(tmp_path / "e1", out, items, "dev", 0)
    assert {i["sample_id"] for i in items} == {"adult_sit_river_01"}          # _03 is test
    assert {i["kind"] for i in items} == {"null", "no_shadow", "halo", "foreign_clean"}  # 'real' only as src
    assert all((out / i["src"]).exists() and (out / i["edit"]).exists() for i in items)


def _fake_run(root, name, task, good_before=True):
    d = root / name / task
    d.mkdir(parents=True)
    Image.new("RGB", (16, 16), (1, 2, 3)).save(d / "best.png")
    if good_before:
        Image.new("RGB", (16, 16), (4, 5, 6)).save(d / "before.png")
    else:
        (d / "before.png").write_bytes(b"\x89PNG\r\n\x1a\n-truncated")
    (d / "report.json").write_text(json.dumps({
        "spec": {"sample_id": "dog_sit_river_01", "subject": "dog", "target_bg": {"name": "forest"}},
        "best_evaluation": {"overall": 0.5, "branch_scores": {"keep": 1, "follow": 1, "world": 0.5}}}))


def test_build_scoring_set_replaces_a_truncated_before_png_from_a_sibling_run(tmp_path, capsys):
    import build_scoring_set as bss

    _fake_run(tmp_path, "comp", "t1")
    _fake_run(tmp_path, "qwen", "t1", good_before=False)
    out = tmp_path / "set"
    (out / "images").mkdir(parents=True)
    items: list = []
    runs = (tmp_path / "comp", tmp_path / "qwen")
    for r in runs:
        bss.add_runs(r, out, items, "all", 0, runs)
    assert {i["kind"] for i in items} == {"comp", "qwen"}          # qwen task was kept
    assert "using the one from comp" in capsys.readouterr().out
    q = next(i for i in items if i["kind"] == "qwen")
    assert np.asarray(Image.open(out / q["src"]))[0, 0].tolist() != [0, 0, 0]  # readable sibling image, not garbage
    assert q["ours_overall"] == 0.5 and "forest" in q["instruction"]


@pytest.mark.parametrize("script", ["build_scoring_set", "score_viescore", "score_editreward", "reward_report",
                                    "vlm_variance"])
def test_experiment_cli_builds_its_parser(script, monkeypatch, capsys):
    """Guards against option clashes with the shared base parser (--set/--config are taken) -- found the hard way."""
    import importlib

    mod = importlib.import_module(script)
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 0 and "usage" in capsys.readouterr().out
