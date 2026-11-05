"""VIEScore-style VLM judge over a scoring set (GPU, ~one chat VLM; no editing).

    python experiments/score_viescore.py --scoreset runs/rw_violations --name rw_violations_vie --vlm-kind vlm
    python experiments/score_viescore.py --scoreset runs/rw_violations --name rw_violations_vie2 --vlm-kind vlm2

Writes runs/<name>/scores_viescore_<vlm-kind>.jsonl (id, success, over_edit, naturalness, artifacts, sc, pq, overall, raw
replies). Resumable: ids already in the file are skipped. Use vlm2 to see whether the baseline depends on the judge.
"""

from __future__ import annotations

import json
from pathlib import Path

from common import ROOT, base_parser, new_run_dir, setup


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--scoreset", required=True, help="scoring set directory (manifest.jsonl + images/)")
    ap.add_argument("--name", default="viescore")
    ap.add_argument("--vlm-kind", default="vlm", choices=["vlm", "vlm2"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = setup(args)

    from PIL import Image

    from src.baselines.rewards import HFChatJudge, viescore
    from src.baselines.report import load_manifest
    from src.models import ModelRegistry

    sdir = Path(args.scoreset) if Path(args.scoreset).is_absolute() else ROOT / args.scoreset
    items = load_manifest(sdir)[: args.limit or None]
    out = new_run_dir(cfg, args.name) / f"scores_viescore_{args.vlm_kind}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    registry = ModelRegistry(cfg)
    with registry.use(args.vlm_kind) as bundle, open(out, "a", encoding="utf-8") as f:
        judge = HFChatJudge(bundle)
        for k, it in enumerate(items):
            if it["id"] in done:
                continue
            src = Image.open(sdir / it["src"]).convert("RGB")
            edit = Image.open(sdir / it["edit"]).convert("RGB")
            res = viescore(judge, src, edit, it["instruction"])
            f.write(json.dumps({"id": it["id"], **res}) + "\n")
            f.flush()
            if k % 20 == 0:
                print(f"[{k + 1}/{len(items)}] {it['id']} overall={res['overall']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
