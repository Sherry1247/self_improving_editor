"""E2 step 1 — build the blind human-annotation set from finished runs.

    python experiments/e2_build.py                       # comp_v2 vs qwen + 30 ip2p_v2, into annotation/e2/
    python experiments/e2_build.py --weak ip2p_v2 --n-weak 30 --repeats 20 --seed 0

Writes
    annotation/e2/manifest.json   items shown to annotators (no editor names, no scores)
    annotation/e2/key.json        item -> (run, task); DO NOT open this before labelling is finished
    annotation/e2/images/*.jpg    anonymised, resized images (zip this folder + manifest for a teammate)

Items
    single : one edited image (+ the source photo) -> 6 yes/no questions + 1-5 overall
    pair   : the comp and qwen result for the same task, sides randomised -> 3 preference questions
    repeat : ~10 % of the single items shown a second time (intra-rater consistency)
    pilot  : a fixed subset (20 single + 10 pair) for the calibration round
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_reports(runs_dir: Path, run: str) -> dict[str, dict]:
    return {f.parent.name: json.loads(f.read_text()) for f in sorted((runs_dir / run).glob("*/report.json"))}


def export(src: Path, dst: Path, size: int) -> None:
    if dst.exists():
        return
    im = Image.open(src).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    im.save(dst, quality=92)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--pair", nargs=2, default=["comp_v2", "qwen"], help="the two strong editors to compare")
    ap.add_argument("--weak", default="ip2p_v2", help="weak editor for low-score single items ('' to skip)")
    ap.add_argument("--n-weak", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--n-pilot-single", type=int, default=20)
    ap.add_argument("--n-pilot-pair", type=int, default=10)
    ap.add_argument("--size", type=int, default=1024, help="longest side of exported images")
    ap.add_argument("--out", default=str(ROOT / "annotation" / "e2"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    runs_dir, out = Path(args.runs_dir), Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    if (out / "key.json").exists():
        raise SystemExit(f"{out/'key.json'} exists - delete {out} first if you really want to rebuild "
                         "(labels refer to the item ids in the current manifest)")

    R = {r: load_reports(runs_dir, r) for r in args.pair}
    tasks = sorted(set(R[args.pair[0]]) & set(R[args.pair[1]]))
    if not tasks:
        raise SystemExit("no common tasks - check --runs-dir / --pair")

    used_ids: set[str] = set()

    def new_id(prefix: str) -> str:
        while True:
            i = f"{prefix}{rng.getrandbits(24):06x}"
            if i not in used_ids:
                used_ids.add(i)
                return i

    def info(report: dict) -> dict:
        s = report["spec"]
        return {"subject": s["subject"], "pose": s.get("pose", ""), "target_key": s["target_bg"]["key"],
                "target": s["target_bg"]["name"], "source": s["source_bg"]["name"],
                "support": s["target_bg"].get("support", "")}

    image_of: dict[tuple[str, str], str] = {}

    def after_image(run: str, task: str) -> str:
        if (run, task) not in image_of:
            name = f"x{rng.getrandbits(32):08x}.jpg"
            export(runs_dir / run / task / "best.png", img_dir / name, args.size)
            image_of[(run, task)] = f"images/{name}"
        return image_of[(run, task)]

    def before_image(run: str, task: str) -> str:
        sample = R_all[run][task]["spec"]["sample_id"]
        name = f"src_{sample}.jpg"
        export(runs_dir / run / task / "before.png", img_dir / name, args.size)
        return f"images/{name}"

    R_all = dict(R)
    singles, pairs, key = [], [], {}

    # strong editors: every task, both editors
    for run in args.pair:
        for t in tasks:
            i = new_id("s")
            singles.append({"id": i, "kind": "single", "before": before_image(run, t),
                            "after": after_image(run, t), **info(R[run][t])})
            key[i] = {"run": run, "task": t}

    # weak editor: stratified by critic score (gated / low / high) so the human scale is covered
    if args.weak:
        W = load_reports(runs_dir, args.weak)
        R_all[args.weak] = W
        ws = [t for t in tasks if t in W]
        score = {t: W[t]["best_evaluation"]["overall"] for t in ws}
        strata = [[t for t in ws if score[t] == 0],
                  [t for t in ws if 0 < score[t] < 0.6],
                  [t for t in ws if score[t] >= 0.6]]
        per = args.n_weak // 3
        chosen = []
        for s in strata:
            rng.shuffle(s)
            chosen += s[:per]
        rest = [t for t in ws if t not in chosen]
        rng.shuffle(rest)
        chosen += rest[: args.n_weak - len(chosen)]
        for t in chosen:
            i = new_id("s")
            singles.append({"id": i, "kind": "single", "before": before_image(args.weak, t),
                            "after": after_image(args.weak, t), **info(W[t])})
            key[i] = {"run": args.weak, "task": t}

    # pairs: comp vs qwen, side randomised
    a_run, b_run = args.pair
    for t in tasks:
        i = new_id("p")
        sides = [a_run, b_run]
        rng.shuffle(sides)
        pairs.append({"id": i, "kind": "pair", "before": before_image(a_run, t),
                      "A": after_image(sides[0], t), "B": after_image(sides[1], t), **info(R[a_run][t])})
        key[i] = {"task": t, "A": sides[0], "B": sides[1]}

    # repeats for intra-rater consistency
    repeats = []
    for orig in rng.sample(singles, min(args.repeats, len(singles))):
        i = new_id("s")
        repeats.append({**orig, "id": i})
        key[i] = {**key[orig["id"]], "repeat_of": orig["id"]}

    # pilot subset: drawn from strong + weak singles and pairs
    pilot = [s["id"] for s in rng.sample(singles, args.n_pilot_single)] + \
            [p["id"] for p in rng.sample(pairs, args.n_pilot_pair)]

    manifest = {"version": 1, "seed": args.seed, "pilot": pilot,
                "single": singles + repeats, "pair": pairs}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (out / "key.json").write_text(json.dumps({"pair_runs": args.pair, "weak_run": args.weak, "items": key},
                                             indent=1))
    n_img = len(list(img_dir.glob("*.jpg")))
    print(f"{len(singles)} single (+{len(repeats)} repeats), {len(pairs)} pairs, pilot {len(pilot)}, "
          f"{n_img} images -> {out}")


if __name__ == "__main__":
    main()
