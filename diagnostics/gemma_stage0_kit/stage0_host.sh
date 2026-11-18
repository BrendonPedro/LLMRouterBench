#!/usr/bin/env bash
# GEMMA-SPLIT Stage 0 evidence, on a lab host. READ-ONLY: loads no model, starts and stops nothing, sends no
# generation request, never reads another user's process environment, writes only into its own output directory.
#   bash stage0_host.sh edge      # on the DGX Spark
#   bash stage0_host.sh server    # on asus-ESC4000-E11
# GS = the experiment's frozen checkout of diagnostics/deployment_gemma_split (tag gemma-split-prereg-v1). If unset, the
# checkout beside this repository that already holds capture/<role>/inventory.json is used.
# Optional: GS0_SOURCE_ROOT (Brian's code directory, default the path the PI stated), GS0_OUT, GS0_NO_API=1 (edge: skip
# the single GET of the API's /openapi.json).
set -uo pipefail
ROLE="${1:?usage: stage0_host.sh edge|server}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$HERE/../../.." && pwd)"
if [ -z "${GS:-}" ]; then
  for c in "$BASE"/LLMRouterBench*/diagnostics/deployment_gemma_split "$HOME"/research/LLMRouterBench*/diagnostics/deployment_gemma_split; do
    [ -f "$c/capture/$ROLE/inventory.json" ] && GS="$c" && break
  done
fi
if [ -z "${GS:-}" ] || [ ! -f "$GS/gs_bench.py" ]; then
  echo "STOP: set GS=<frozen checkout>/diagnostics/deployment_gemma_split (the one that holds capture/$ROLE/)" >&2; exit 3
fi
OUT="${GS0_OUT:-$BASE/gemma_stage0_evidence}"
echo "[stage0] role $ROLE on $(hostname); frozen checkout $GS at $(git -C "$GS" describe --tags --always 2>/dev/null); output $OUT"
python3 -I -B "$HERE/gs0_kit.py" collect --role "$ROLE" --gs-dir "$GS" --out "$OUT" \
  ${GS0_SOURCE_ROOT:+--source-root "$GS0_SOURCE_ROOT"} ${GS0_NO_API:+--no-api} || exit $?
echo "[stage0] frozen capture files of this host (written only by the frozen gs_bench.py, RUNBOOK Stage 0):"
for f in inventory.json checkpoint.json deployment.json; do
  if [ -f "$GS/capture/$ROLE/$f" ]; then echo "  present  capture/$ROLE/$f"; else echo "  MISSING  capture/$ROLE/$f   -> cd $GS && \$PY -I -B gs_bench.py ${f%.json} --role $ROLE ... (RUNBOOK.md, Stage 0)"; fi
done
echo "[stage0] copy $OUT/evidence_$ROLE.{json,md} to the analysis host; fill capture/split_capture.json from them."
