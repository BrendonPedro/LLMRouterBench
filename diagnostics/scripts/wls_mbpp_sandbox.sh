#!/usr/bin/env bash
# Run a script from diagnostics/scripts/wls_sandbox/ inside the verified MBPP sandbox.
#
# This is the Bubblewrap 0.9.0 configuration verified on 2026-11-03
# (WINNER_LABEL_STABILITY_MISSION.md §7a), with paths taken from the repo instead of a
# temporary session directory. MBPP evaluation executes model-generated code: only run it this way.
#
#   - new user/PID/net/IPC/UTS/cgroup namespaces, nested user namespaces disabled
#   - uid/gid 65534, all capabilities dropped, die-with-parent, new session, cleared env
#   - fresh root remounted read-only; visible read-only: /usr, the repo .venv,
#     evaluation/, generators/, common/, the verified MBPP test.json, the input dir,
#     and diagnostics/scripts/wls_sandbox/
#   - writable: private 1 GB tmpfs /tmp and bwrap's private /dev
#   - outer limits (user systemd scope): MemoryMax 6G, no swap, TasksMax 512; 6 h wall clock
#
# Usage: wls_mbpp_sandbox.sh <input_dir> <script.py> [args...]
#   WLS_LFS_DIR  verified LFS copy (default ~/.cache/LLMRouterBench/wls_lfs_data)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LFS="${WLS_LFS_DIR:-$HOME/.cache/LLMRouterBench/wls_lfs_data}"
RUN_DIR="$REPO/diagnostics/scripts/wls_sandbox"

[[ $# -ge 2 ]] || { echo "usage: $0 <input_dir> <script.py> [args...]" >&2; exit 2; }
IN_DIR="$(cd "$1" && pwd)"; SCRIPT="$2"; shift 2
[[ -f "$RUN_DIR/$SCRIPT" ]] || { echo "no such sandbox script: $RUN_DIR/$SCRIPT" >&2; exit 2; }
[[ -f "$LFS/data/MBPP/test.json" ]] || { echo "missing $LFS/data/MBPP/test.json; run wls_fetch_lfs.py" >&2; exit 2; }
command -v bwrap >/dev/null || { echo "bwrap not installed (apt package: bubblewrap)" >&2; exit 2; }

exec systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -p TasksMax=512 \
  timeout 21600 \
  bwrap \
    --unshare-user --unshare-pid --unshare-net --unshare-ipc --unshare-uts --unshare-cgroup \
    --disable-userns --uid 65534 --gid 65534 --cap-drop ALL \
    --die-with-parent --new-session --hostname sbx \
    --clearenv --setenv PATH /usr/bin --setenv HOME /tmp --setenv LANG C.UTF-8 \
    --setenv PYTHONDONTWRITEBYTECODE 1 --setenv HF_HOME /tmp/hf --setenv HF_DATASETS_OFFLINE 1 --setenv HF_HUB_OFFLINE 1 \
    --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/sbin /sbin \
    --ro-bind /etc/ld.so.cache /etc/ld.so.cache \
    --proc /proc --dev /dev --size 1073741824 --tmpfs /tmp \
    --ro-bind "$REPO/.venv" /sbx/venv \
    --ro-bind "$REPO/evaluation" /sbx/code/evaluation \
    --ro-bind "$REPO/generators" /sbx/code/generators \
    --ro-bind "$REPO/common" /sbx/code/common \
    --ro-bind "$LFS/data/MBPP/test.json" /sbx/data/MBPP/test.json \
    --ro-bind "$IN_DIR" /sbx/in \
    --ro-bind "$RUN_DIR" /sbx/run \
    --remount-ro / \
    --chdir /tmp \
    /sbx/venv/bin/python -I "/sbx/run/$SCRIPT" "$@"
