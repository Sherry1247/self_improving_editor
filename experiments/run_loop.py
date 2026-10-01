"""Closed-loop background replacement over the dataset.

    python experiments/run_loop.py --samples dog_sit_river_01 --targets snow          # smoke test
    python experiments/run_loop.py --name ip2p_full                                    # everything
    python experiments/run_loop.py --name comp --set loop.editor=compositing
    python experiments/run_loop.py --name oneshot --set loop.max_edits=1 --set loop.n_candidates=1

Outputs runs/<name>/<task_id>/{before.png, best.png, panel_best.jpg, candidates/*.jpg, report.json}
and runs/<name>/summary.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

from common import base_parser, build_evaluation_stack, new_run_dir, setup, write_json


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--name")
    ap.add_argument("--samples", nargs="*", help="sample ids (filename stems)")
    ap.add_argument("--targets", nargs="*", help="override target backgrounds")
    ap.add_argument("--limit", type=int, help="max number of tasks")
    ap.add_argument("--task-index", type=int, help="run only the i-th task (CHTC array jobs)")
    ap.add_argument("--save-candidates", action="store_true")
    args = ap.parse_args()
    cfg = setup(args)

    from src.data import build_tasks, load_samples, prepare_image, save_image
    from src.editors import build_editor
    from src.refinement import ActionMemory, ActionRouter
    from src.pipelines import RefinementLoop
    from src.utils.visualization import make_panel

    targets = tuple(args.targets or cfg["targets"])
    tasks = build_tasks(load_samples(cfg["paths"]["labels"]), targets, cfg["include_swap"] and not args.targets,
                        set(args.samples) if args.samples else None)
    if args.task_index is not None:
        tasks = [tasks[args.task_index]]
    if args.limit:
        tasks = tasks[: args.limit]
    run = new_run_dir(cfg, args.name)
    write_json(cfg, run / "config.json")

    registry, perceiver, evaluator = build_evaluation_stack(cfg)
    editor = build_editor(cfg["loop"]["editor"], cfg, registry)
    memory = ActionMemory(run / "memory.json") if cfg["loop"].get("memory", True) else None
    loop = RefinementLoop(editor, evaluator, ActionRouter(editor, memory), cfg)

    summary_path = run / "summary.csv"
    fields = ["task_id", "sample", "target", "submerged", "editor", "best_score", "gate_passed", "keep", "follow",
              "world", "weighted_sum_baseline", "first_score", "edits_used", "rounds", "stopped", "issues"]
    new_file = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as fsum:
        writer = csv.DictWriter(fsum, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for k, (sample, spec) in enumerate(tasks):
            tdir = run / spec.task_id
            if (tdir / "report.json").exists():
                print(f"[{k + 1}/{len(tasks)}] {spec.task_id}: done, skipping")
                continue
            print(f"[{k + 1}/{len(tasks)}] {spec.task_id}")
            img = prepare_image(Path(cfg["paths"]["images"]) / sample.filename, cfg["paths"]["cache"],
                                cfg["image"]["long_side"])
            before = perceiver.perceive_cached(img, spec, sample.sample_id)
            if before.subject_count == 0:
                print("   ! no subject detected in the before image — skipped (check prepare_data output)")
                continue

            def on_candidate(c, tdir=tdir):
                if args.save_candidates:
                    save_image(c.image, tdir / "candidates" / f"r{c.round}_{c.index}_{c.score:.3f}.jpg")

            res = loop.run(before, spec, on_candidate)
            best = res.best
            save_image(img, tdir / "before.png")
            save_image(best.image, tdir / "best.png")
            save_image(make_panel(img, best.image, best.regions, best.evaluation,
                                  f"{spec.task_id}  ({res.stopped_reason})"), tdir / "panel_best.jpg")
            write_json({"spec": spec.to_dict(), "editor": editor.name, "stopped_reason": res.stopped_reason,
                        "edits_used": res.edits_used, "rounds": res.rounds,
                        "candidates": [c.summary() for c in res.candidates],
                        "best": best.summary(), "best_evaluation": best.evaluation.to_dict()}, tdir / "report.json")
            bs = best.evaluation.branch_scores
            writer.writerow({"task_id": spec.task_id, "sample": sample.sample_id, "target": spec.target_bg.key,
                             "submerged": int(spec.submerged), "editor": editor.name,
                             "best_score": round(best.score, 4), "gate_passed": best.evaluation.gate_passed,
                             "keep": _r(bs.get("keep")), "follow": _r(bs.get("follow")), "world": _r(bs.get("world")),
                             "weighted_sum_baseline": _r(bs.get("weighted_sum_baseline")),
                             "first_score": round(res.candidates[0].score, 4) if res.candidates else None,
                             "edits_used": res.edits_used, "rounds": len(res.rounds), "stopped": res.stopped_reason,
                             "issues": " ".join(i.value for i in best.evaluation.issues)})
            fsum.flush()
            print(f"   best={best.score:.3f} gate={'PASS' if best.evaluation.gate_passed else 'FAIL'} "
                  f"edits={res.edits_used} stop={res.stopped_reason}")
    print(f"\nDone -> {run}")


def _r(x):
    return None if x is None else round(float(x), 4)


if __name__ == "__main__":
    main()
