#!/bin/bash
# Unpack results/result_<RUN>_*.tar.gz into runs/<RUN>/, rebuild summary.csv from the report.json files,
# and print the run analysis (experiments/analyze_runs.py).
#   bash chtc/collect.sh ip2p_v2
#   bash chtc/collect.sh comp_v2 qwen        # several runs -> also prints the pairwise comparison
# Every job writes its own runs/<RUN>/summary.csv; extracting them one after another leaves only the last
# one, so summary.csv is always REBUILT from report.json (the source of truth) after unpacking.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$#" -ge 1 ] || { echo "usage: bash chtc/collect.sh <RUN> [<RUN> ...]"; exit 2; }
mkdir -p runs
for RUN in "$@"; do
  shopt -s nullglob
  files=(results/result_${RUN}_*.tar.gz)
  shopt -u nullglob
  [ "${#files[@]}" -gt 0 ] || { echo "no results/result_${RUN}_*.tar.gz found"; exit 1; }
  for f in "${files[@]}"; do tar xzf "$f"; done   # each holds runs/<RUN>/...
  echo "unpacked ${#files[@]} job result(s) for $RUN"
done
PYTHON="${PYTHON:-python3}"
"$PYTHON" experiments/analyze_runs.py "$@" --csv | tee "runs/analysis_$(IFS=_; echo "$*").txt"
for RUN in "$@"; do echo "$RUN: $(($(wc -l < runs/$RUN/summary.csv) - 1)) tasks in summary.csv"; done
