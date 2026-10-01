#!/bin/bash
# Unpack results/result_<RUN>_*.tar.gz into runs/<RUN>/ and merge the per-job summary.csv files.
#   bash chtc/collect.sh ip2p
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="$1"
mkdir -p runs
for f in results/result_${RUN}_*.tar.gz; do tar xzf "$f"; done   # each holds runs/<RUN>/...
# every job appended to its own copy of summary.csv; the last extracted one wins, so rebuild it
python3 - "$RUN" <<'EOF'
import csv, json, sys
from pathlib import Path
run = Path("runs") / sys.argv[1]
rows = []
for rep in sorted(run.glob("*/report.json")):
    r = json.loads(rep.read_text())
    ev = r["best_evaluation"]; bs = ev["branch_scores"]; spec = r["spec"]
    rows.append({"task_id": spec["task_id"], "sample": spec["sample_id"], "target": spec["target_bg"]["key"],
                 "submerged": int(spec["submerged"]), "editor": r["editor"], "best_score": r["best"]["score"],
                 "gate_passed": ev["gate_passed"], "keep": bs.get("keep"), "follow": bs.get("follow"),
                 "world": bs.get("world"), "weighted_sum_baseline": bs.get("weighted_sum_baseline"),
                 "first_score": r["candidates"][0]["score"], "edits_used": r["edits_used"],
                 "rounds": len(r["rounds"]), "stopped": r["stopped_reason"], "issues": " ".join(ev["issues"])})
if rows:
    with open(run / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} tasks -> {run/'summary.csv'}")
EOF
