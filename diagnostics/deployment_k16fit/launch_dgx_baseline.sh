#!/usr/bin/env bash
# K16-HWFIT, DGX Spark only: the matched UNSPLIT Qwen3-32B latency baseline for a completed split run.
# CPU/static validation -> preflight (no server, no tunnel) -> memory gate (about 72 GiB free needed) -> load the whole
# model on the GB10 -> the split run's own latency requests (same prompts, warm-ups, repetitions, 256 forced tokens,
# same clocks) -> comparison with the split run. The Asus server is not used. Kills nothing; installs nothing.
#   K16_SPLIT_RUN=<split run directory>   default: the newest *_edge run under K16_OUT whose gates passed
# Optional: K16_ALLOW_SHARED=1 (run beside another process if memory suffices; rows are flagged), K16_PY,
# K16_MODEL_DIR, K16_OUT.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_launch_common.sh"
k16_run baseline k16_baseline.py "$@"
