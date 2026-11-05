"""EditReward (TIGER-Lab) over a scoring set. Runs in its OWN environment: EditReward pins transformers==4.57.0 and
trl==0.8.6, this repo needs transformers>=5.5. chtc/run.sh MODE=editreward builds that environment.

    python experiments/score_editreward.py --scoreset rw_violations_in --out scores_editreward.jsonl \
        --config /path/to/EditReward/config/EditReward-Qwen2.5-7B-VL.yaml --checkpoint /path/to/EditReward-Qwen2.5-VL-7B

Usage follows the official README (EditRewardInferencer, reward_dim="overall_detail", rm_head_type="ranknet_multi_head");
the score is reward[0] of each returned tensor. The README does not state the score range, so only RANKINGS / AUROC
are meaningful, never absolute values. This script is standalone (no imports from src/) on purpose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scoreset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True, help="EditReward repo config yaml")
    ap.add_argument("--checkpoint", required=True, help="local dir of TIGER-Lab/EditReward-Qwen2.5-VL-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from EditReward import EditRewardInferencer

    sdir = Path(args.scoreset)
    items = [json.loads(l) for l in (sdir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    items = items[: args.limit or None]
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    inf = EditRewardInferencer(config_path=args.config, checkpoint_path=args.checkpoint, device=args.device,
                               reward_dim="overall_detail", rm_head_type="ranknet_multi_head")
    with open(out, "a", encoding="utf-8") as f:
        for k, it in enumerate(items):
            if it["id"] in done:
                continue
            try:
                with torch.no_grad():
                    rewards = inf.reward(prompts=[it["instruction"]], image_src=[str(sdir / it["src"])],
                                         image_paths=[str(sdir / it["edit"])])
                score = float(rewards[0][0].item())
                dims = [float(x) for x in rewards[0].flatten().tolist()]
            except Exception as e:  # keep going; a missing score is reported, not silently zero
                score, dims = None, [f"{type(e).__name__}: {e}"]
            # `score` = reward[0] as in the README; the second head is kept as `dim1` (its meaning is not documented)
            dim1 = dims[1] if len(dims) > 1 and isinstance(dims[1], float) else None
            f.write(json.dumps({"id": it["id"], "score": score, "dim1": dim1, "raw_dims": dims}) + "\n")
            f.flush()
            if k % 20 == 0:
                print(f"[{k + 1}/{len(items)}] {it['id']} score={score}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
