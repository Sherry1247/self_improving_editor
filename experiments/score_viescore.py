"""VIEScore-style VLM judge over a scoring set (GPU, ~one chat VLM; no editing).

    python experiments/score_viescore.py --scoreset runs/rw_violations --name rw_violations_vie --vlm-kind vlm
    python experiments/score_viescore.py --scoreset runs/rw_violations --name rw_violations_vie2 --vlm-kind vlm2

Writes runs/<name>/scores_viescore_<vlm-kind>.jsonl (id, success, over_edit, naturalness, artifacts, sc, pq, overall, raw
replies); with --soft also scores_viescoresoft_<vlm-kind>.jsonl (the same fields from digit-token expectations, no ties). Resumable: ids already in the file are skipped. Use vlm2 to see whether the baseline depends on the judge.
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
    ap.add_argument("--soft", action="store_true",
                    help="also write scores_viescoresoft_<kind>.jsonl (expected digit instead of a generated integer)")
    args = ap.parse_args()
    cfg = setup(args)

    from PIL import Image

    from src.baselines.rewards import HFChatJudge, viescore, viescore_soft
    from src.baselines.report import load_manifest
    from src.models import ModelRegistry

    sdir = Path(args.scoreset) if Path(args.scoreset).is_absolute() else ROOT / args.scoreset
    items = load_manifest(sdir)[: args.limit or None]
    rd = new_run_dir(cfg, args.name)
    out, out_soft = rd / f"scores_viescore_{args.vlm_kind}.jsonl", rd / f"scores_viescoresoft_{args.vlm_kind}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    done_soft = ({json.loads(l)["id"] for l in out_soft.read_text().splitlines() if l.strip()}
                 if args.soft and out_soft.exists() else set())
    registry = ModelRegistry(cfg)
    with registry.use(args.vlm_kind) as bundle, open(out, "a", encoding="utf-8") as f, \
            open(out_soft, "a", encoding="utf-8") as fs:
        judge = HFChatJudge(bundle)
        for k, it in enumerate(items):
            need_text, need_soft = it["id"] not in done, args.soft and it["id"] not in done_soft
            if not (need_text or need_soft):
                continue
            src = Image.open(sdir / it["src"]).convert("RGB")
            edit = Image.open(sdir / it["edit"]).convert("RGB")
            res = viescore(judge, src, edit, it["instruction"]) if need_text else {"overall": None}
            if need_text:
                f.write(json.dumps({"id": it["id"], **res}) + "\n")
                f.flush()
            if need_soft:
                soft = viescore_soft(judge.expect, src, edit, it["instruction"])
                fs.write(json.dumps({"id": it["id"], **soft}) + "\n")
                fs.flush()
                res["soft_overall"] = soft["overall"]
            if k % 20 == 0:
                print(f"[{k + 1}/{len(items)}] {it['id']} overall={res['overall']} soft={res.get('soft_overall')}")
    print(f"-> {out}" + (f" and {out_soft}" if args.soft else ""))


if __name__ == "__main__":
    main()
