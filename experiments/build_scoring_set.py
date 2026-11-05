"""Build a scoring set (manifest.jsonl + images/) for the reward-model baselines. No GPU, no models.

    # the controlled-violation suite, from a finished E1 run (needs runs/<e1>/examples/*.jpg)
    python experiments/build_scoring_set.py --name rw_violations --violations runs/e1v2
    # finished editor runs (before.png / best.png / report.json per task), optionally only dev or test sources
    python experiments/build_scoring_set.py --name rw_runs --runs runs/comp_v2 runs/qwen runs/ip2p_v2 --split dev

Violation items: src = the untouched photo (<sample>_real.jpg), edit = <sample>_<kind>.jpg. 'null' (clean paste) is the
positive for every violation, 'foreign_clean' for halo -- the same protocol as experiments/critic_auroc.py.
Run items carry the project's own scores (ours_*) so the report can correlate the baselines with them.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from common import ROOT, base_parser, new_run_dir, setup

KINDS = ["color_cast", "float", "foreign_clean", "halo", "no_shadow", "null", "real", "scale_down", "scale_up", "sink"]
VIOLATION_INSTRUCTION = ("Place the subject into this scene so that the result looks like a natural, physically "
                         "plausible photograph.")


def split_of(sample_id: str) -> str:
    return "test" if sample_id.endswith("_03") else "dev"


def _p(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def add_violations(src: Path, out: Path, items: list[dict], split: str, limit: int) -> None:
    ex = src / "examples"
    by_sample: dict[str, dict[str, Path]] = {}
    for f in sorted(ex.glob("*.jpg")):
        for k in KINDS:
            if f.name.endswith(f"_{k}.jpg"):
                by_sample.setdefault(f.name[: -len(f"_{k}.jpg")], {})[k] = f
                break
    for n, (sample, d) in enumerate(sorted(by_sample.items())):
        if (limit and n >= limit) or (split != "all" and split_of(sample) != split) or "real" not in d:
            continue
        for kind, f in d.items():
            if kind == "real":
                continue
            for role, path in (("src", d["real"]), ("edit", f)):
                shutil.copy(path, out / "images" / f"{sample}_{path.stem.removeprefix(sample + '_')}.jpg")
            items.append({"id": f"v/{sample}/{kind}", "group": "violation", "sample_id": sample, "kind": kind,
                          "task": f"{sample}/{kind}", "src": f"images/{sample}_real.jpg",
                          "edit": f"images/{sample}_{kind}.jpg", "instruction": VIOLATION_INSTRUCTION})


def _rgb(path: Path):
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:  # missing, or truncated by an interrupted copy / extraction
        return None


def add_runs(src: Path, out: Path, items: list[dict], split: str, limit: int, siblings: tuple = ()) -> None:
    """``siblings``: the other run directories. before.png is the same prepared source image in every run, so an unreadable
    one is replaced by the same task's before.png from a sibling run (with a warning); an unreadable best.png skips the task."""
    for n, rep_path in enumerate(sorted(src.glob("*/report.json"))):
        d = rep_path.parent
        if limit and n >= limit:
            break
        rep = json.loads(rep_path.read_text())
        sp = rep["spec"]
        if split != "all" and split_of(sp["sample_id"]) != split:
            continue
        best, before = _rgb(d / "best.png"), _rgb(d / "before.png")
        if before is None:
            for other in siblings:
                before = _rgb(other / d.name / "before.png") if other != src else None
                if before is not None:
                    print(f"WARNING {src.name}/{d.name}: before.png unreadable, using the one from {other.name}")
                    break
        if best is None or before is None:
            print(f"WARNING {src.name}/{d.name}: skipped (unreadable best.png or before.png)")
            continue
        stem = f"{src.name}__{d.name}"
        for name, im in (("before", before), ("best", best)):  # JPEG q95: ~5x smaller to ship, same for every editor
            im.save(out / "images" / f"{stem}__{name}.jpg", quality=95)
        ev = rep["best_evaluation"]
        bs = ev["branch_scores"]
        items.append({"id": f"r/{src.name}/{d.name}", "group": "run", "sample_id": sp["sample_id"], "kind": src.name,
                      "task": d.name, "src": f"images/{stem}__before.jpg", "edit": f"images/{stem}__best.jpg",
                      "instruction": f"Replace the background with a {sp['target_bg']['name']}, keeping the "
                                     f"{sp['subject']} unchanged.",
                      "ours_overall": ev["overall"], "ours_keep": bs.get("keep"), "ours_follow": bs.get("follow"),
                      "ours_world": bs.get("world")})


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--violations", help="finished E1 run directory with examples/")
    ap.add_argument("--runs", nargs="*", default=[], help="finished editor run directories")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all",
                    help="dev = source images _01/_02, test = _03 (thresholds are tuned on dev only)")
    ap.add_argument("--limit", type=int, default=0, help="at most N samples / tasks (smoke tests)")
    args = ap.parse_args()
    cfg = setup(args)
    out = new_run_dir(cfg, args.name)
    (out / "images").mkdir(exist_ok=True)
    items: list[dict] = []
    if args.violations:
        add_violations(_p(args.violations), out, items, args.split, args.limit)
    run_dirs = tuple(_p(r) for r in args.runs)
    for r in run_dirs:
        add_runs(r, out, items, args.split, args.limit, run_dirs)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    n_v = sum(i["group"] == "violation" for i in items)
    print(f"{len(items)} items ({n_v} violation, {len(items) - n_v} run) -> {out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
