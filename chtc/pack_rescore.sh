#!/bin/bash
# Bundle before.png / best.png / report.json of finished runs for MODE=rescore / MODE=covtest.
#   bash chtc/pack_rescore.sh ip2p comp      -> chtc/rescore_in.tar.gz
# Extracts only those three files per task into a temp dir inside the repo (home and /tmp share a small quota on CHTC; the repo dir does not),
# and always cleans the temp dir, even on failure.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$PWD/chtc/rescore_in.tar.gz"
TMP=$(mktemp -d -p "${PACK_TMP:-$PWD/chtc}")
trap 'rm -rf "$TMP"' EXIT
for RUN in "$@"; do
  for f in results/result_${RUN}_*.tar.gz; do
    tar xzf "$f" -C "$TMP" --wildcards '*/report.json' '*/best.png' '*/before.png'
  done
done
(cd "$TMP" && find runs -name report.json -o -name best.png -o -name before.png | tar czf "$OUT" -T -)
echo "rescore input: $(du -h "$OUT" | cut -f1)"
