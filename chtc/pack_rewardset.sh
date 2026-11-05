#!/bin/bash
# Bundle ONE scoring set (manifest.jsonl + images/, built by experiments/build_scoring_set.py) for MODE=viescore / editreward.
#   bash chtc/pack_rewardset.sh rw_violations      -> chtc/rewardset_in.tar.gz   (contains rw_violations/...)
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$#" -eq 1 ] || { echo "usage: bash chtc/pack_rewardset.sh <scoring set name under runs/>"; exit 2; }
SET="$1"
[ -f "runs/$SET/manifest.jsonl" ] || { echo "runs/$SET/manifest.jsonl not found"; exit 1; }
tar czf chtc/rewardset_in.tar.gz -C runs "$SET/manifest.jsonl" "$SET/images"
echo "scoring set '$SET': $(wc -l < runs/$SET/manifest.jsonl) items, $(du -h chtc/rewardset_in.tar.gz | cut -f1)"
