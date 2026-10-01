#!/bin/bash
# Bundle before.png / best.png / report.json of finished runs for MODE=rescore.
#   bash chtc/pack_rescore.sh ip2p comp      -> chtc/rescore_in.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
for RUN in "$@"; do
  for f in results/result_${RUN}_*.tar.gz; do tar xzf "$f" -C "$TMP"; done
done
cd "$TMP"
find runs -name report.json -o -name best.png -o -name before.png | tar czf "$OLDPWD/chtc/rescore_in.tar.gz" -T -
cd "$OLDPWD"; rm -rf "$TMP"
echo "rescore input: $(du -h chtc/rescore_in.tar.gz | cut -f1)"
