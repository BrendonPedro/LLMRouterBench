#!/usr/bin/env bash
# K16-HWFIT, ASUS main server: run this FIRST. CPU/static validation -> preflight and memory gate -> load layers
# 16-47 on one free RTX A6000 -> serve the DGX edge -> exit when the edge is done. Kills nothing; installs nothing.
# Optional environment: K16_PY (python), K16_MODEL_DIR (checkpoint), K16_GPU (index or UUID; default: a free A6000),
# K16_PORT (29616), K16_OUT (~/k16fit_runs), K16_ALLOW_SHARED=1, K16_NO_STATIC=1, K16_SERVER_WAIT_S (14400).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_launch_common.sh"
k16_run server k16_server.py "$@"
