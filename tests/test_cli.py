"""CLI plumbing: run experiments/run_loop.py end-to-end on a synthetic dataset with fake models."""

import csv
import json
import sys
from pathlib import Path

from PIL import Image

from fakes import FakeEditor, FakePerceiver, make_scene

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))


def test_run_loop_cli(tmp_path, monkeypatch):
    import common
    import run_loop

    from src.critics import build_critics
    from src.pipelines import Evaluator
    from src.scoring import build_aggregator

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.fromarray(make_scene("river")[0]).save(img_dir / "dog_sit_river_01.png")
    labels = tmp_path / "labels.csv"
    labels.write_text("filename,object,action,background,submerged\ndog_sit_river_01.png,dog,sit,river,1\n")

    class CachedFake(FakePerceiver):
        def perceive_cached(self, image, spec, key):
            return self.perceive(image, spec)

    def fake_stack(cfg):
        per = CachedFake()
        return None, per, Evaluator(per, build_critics(cfg), build_aggregator(cfg))

    monkeypatch.setattr(run_loop, "build_evaluation_stack", fake_stack)
    monkeypatch.setattr(common, "build_evaluation_stack", fake_stack)
    import src.editors

    monkeypatch.setattr(src.editors, "build_editor", lambda name, cfg, reg: FakeEditor(2.25))
    monkeypatch.setattr(sys, "argv", [
        "run_loop.py", "--name", "t", "--targets", "snow", "beach",
        "--set", f"paths.labels={labels}", "--set", f"paths.images={img_dir}",
        "--set", f"paths.cache={tmp_path / 'cache'}", "--set", f"paths.runs={tmp_path / 'runs'}",
        "--set", "image.long_side=192", "--set", "loop.max_edits=4", "--save-candidates"])
    run_loop.main()

    run = tmp_path / "runs" / "t"
    rows = list(csv.DictReader(open(run / "summary.csv")))
    assert [r["target"] for r in rows] == ["snow", "beach"]
    assert rows[0]["submerged"] == "1"
    rep = json.loads((run / "dog_sit_river_01__to_snow" / "report.json").read_text())
    assert rep["best"]["score"] == max(c["score"] for c in rep["candidates"])
    assert (run / "dog_sit_river_01__to_snow" / "panel_best.jpg").exists()
    assert any((run / "dog_sit_river_01__to_snow" / "candidates").iterdir())
