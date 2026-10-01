"""Score one (before, after) pair with the full critic stack — no editing.

    python experiments/evaluate_pair.py --before a.jpg --after b.jpg \
        --object dog --action sit --background river --target snow --out runs/pair
"""

from __future__ import annotations

from pathlib import Path

from common import base_parser, build_evaluation_stack, setup, write_json


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--object", required=True, help="adult_person | dog")
    ap.add_argument("--action", required=True, help="sit | stand")
    ap.add_argument("--background", required=True, help="source background key")
    ap.add_argument("--target", required=True, help="target background key")
    ap.add_argument("--out", default="runs/pair")
    args = ap.parse_args()
    cfg = setup(args)

    from src.data import load_image, save_image
    from src.data.images import match_size, resize_long_side
    from src.spec import build_spec
    from src.utils.visualization import make_panel, score_lines

    after = resize_long_side(load_image(args.after), cfg["image"]["long_side"])
    before = match_size(load_image(args.before), after.shape[:2])
    spec = build_spec(Path(args.before).stem, args.object, args.action, args.background, args.target)
    _, perceiver, evaluator = build_evaluation_stack(cfg)
    ev, _, regions = evaluator.evaluate(perceiver.perceive(before, spec), after, spec)
    out = Path(args.out)
    write_json(ev.to_dict(), out / "evaluation.json")
    save_image(make_panel(before, after, regions, ev, spec.task_id), out / "panel.jpg")
    print("\n".join(score_lines(ev)))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
