"""Downscale originals (aspect preserved, EXIF fixed) and cache the before-image perception.

    python experiments/prepare_data.py                    # all samples
    python experiments/prepare_data.py --no-perception    # only resize (no GPU needed)

Writes a contact sheet of subject masks to runs/prepare/ so you can eyeball the segmentation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from common import base_parser, build_evaluation_stack, new_run_dir, setup, write_json


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--no-perception", action="store_true")
    ap.add_argument("--samples", nargs="*")
    args = ap.parse_args()
    cfg = setup(args)

    from src.data import build_tasks, load_samples, prepare_image, save_image
    from src.regions import partition
    from src.utils.visualization import overlay_regions

    samples = load_samples(cfg["paths"]["labels"])
    if args.samples:
        samples = [s for s in samples if s.sample_id in args.samples]
    out = new_run_dir(cfg, "prepare")
    perceiver = None if args.no_perception else build_evaluation_stack(cfg)[1]
    rows = []
    for s in samples:
        img = prepare_image(Path(cfg["paths"]["images"]) / s.filename, cfg["paths"]["cache"], cfg["image"]["long_side"])
        row = {"sample": s.sample_id, "shape": list(img.shape[:2])}
        if perceiver:
            spec = build_tasks([s], tuple(cfg["targets"]), cfg["include_swap"])[0][1]
            per = perceiver.perceive_cached(img, spec, s.sample_id)
            reg = partition(per.subject_mask, per.subject_mask)
            save_image(overlay_regions(img, reg), out / f"{s.sample_id}_regions.jpg")
            top = max(per.bg_probs, key=per.bg_probs.get) if per.bg_probs else None
            row.update(subjects=per.subject_count, subject_area=float(per.subject_mask.mean()),
                       old_bg_area=float(per.old_bg_mask.mean()), bg_top=top,
                       bg_p_source=per.bg_probs.get(s.background) if per.bg_probs else None,
                       scores=[round(d.score, 3) for d in per.subject_detections])
            flag = "" if per.subject_count == 1 else "   <-- check"
            print(f"{s.sample_id:28s} subjects={per.subject_count} area={row['subject_area']:.3f} "
                  f"bg_top={top}{flag}")
        rows.append(row)
    write_json(rows, out / "prepare_summary.json")
    print(f"\nSummary -> {out / 'prepare_summary.json'}")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
