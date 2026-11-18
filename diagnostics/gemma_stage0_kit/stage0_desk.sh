#!/usr/bin/env bash
# GEMMA-SPLIT Stage 0 on the analysis host: drafts from the stated facts, then what the frozen rules still require.
#   GS=<frozen checkout>/diagnostics/deployment_gemma_split bash stage0_desk.sh [edge hostname]
# Writes only into GS0_OUT (default: gemma_stage0_evidence beside the repository). Never writes into GS/capture.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$HERE/../../.." && pwd)"
GS="${GS:?set GS=<frozen checkout>/diagnostics/deployment_gemma_split}"
OUT="${GS0_OUT:-$BASE/gemma_stage0_evidence}"
python3 -I -B "$HERE/gs0_kit.py" drafts --gs-dir "$GS" --out "$OUT/drafts" ${1:+--edge-host "$1"} || exit $?
python3 -I -B "$HERE/gs0_kit.py" status --gs-dir "$GS" --drafts "$OUT/drafts" --work "$OUT/status"
rc=$?
echo "[stage0] exit $rc (0 = the real capture would CONTINUE; 3 = facts are still owed, listed above)"
exit $rc
