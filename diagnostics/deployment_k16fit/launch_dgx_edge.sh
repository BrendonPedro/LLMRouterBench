#!/usr/bin/env bash
# K16-HWFIT, DGX Spark: run this SECOND, after the ASUS command printed READY. CPU/static validation -> preflight
# -> unsplit reference -> in-host split check -> two-host correctness -> generation -> latency -> optional static
# mode -> summary. Stops at the first failed gate. Kills nothing; installs nothing.
# Environment: K16_SERVER=host:port of the ASUS server (default asus-ESC4000-E11:29616). Optional: K16_PY,
# K16_MODEL_DIR, K16_OUT (~/k16fit_runs), K16_ALLOW_SHARED=1, K16_NO_STATIC=1, K16_KEEP_SERVER=1.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_launch_common.sh"
k16_run edge k16_edge.py "$@"
