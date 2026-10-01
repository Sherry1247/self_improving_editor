#!/bin/bash
# Run on the CHTC submit server from the repo root. Bundles code + data into chtc/payload.tar.gz
# (HTCondor flattens transferred directories, so one tarball keeps the repo layout intact).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p chtc/logs results
tar czf chtc/payload.tar.gz src configs experiments validation requirements.txt data/labels.csv data/images/original
echo "payload: $(du -h chtc/payload.tar.gz | cut -f1)"
