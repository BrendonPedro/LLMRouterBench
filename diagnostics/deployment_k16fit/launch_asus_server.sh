#!/usr/bin/env bash
# K16-HWFIT, ASUS main server: run this FIRST. CPU/static validation -> preflight and memory gate -> load layers
# 16-47 on one free RTX A6000 -> listen on 127.0.0.1 ONLY -> serve the DGX edge (which arrives through its SSH
# tunnel) -> exit when the edge is done. Kills nothing; installs nothing; opens no port on the lab network.
# Lab paths: repository /mnt/mydisk/brendon/LLMRouterBench (this worktree beside it); checkpoint
# /mnt/mydisk/brian/splitllm-cloud/models/Qwen3-32B/<revision> (found automatically, read-only, hash-verified).
# Optional environment: K16_PY (python), K16_MODEL_DIR (another checkpoint path), K16_GPU (index or UUID; default:
# a free A6000), K16_PORT (29616), K16_OUT (default: k16fit_runs beside the repository), K16_ALLOW_SHARED=1,
# K16_NO_STATIC=1, K16_SERVER_WAIT_S (14400).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_launch_common.sh"
k16_run server k16_server.py "$@"
