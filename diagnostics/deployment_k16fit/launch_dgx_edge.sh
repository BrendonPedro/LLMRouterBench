#!/usr/bin/env bash
# K16-HWFIT, DGX Spark: run this SECOND, after the ASUS command printed READY. Opens an SSH tunnel to the server's
# loopback port, then: CPU/static validation -> preflight -> unsplit reference -> in-host split check -> two-host
# correctness -> generation -> latency -> optional static mode -> summary. Stops at the first failed gate. Closes
# its own tunnel at exit. Kills nothing else; installs nothing.
#
#   K16_SSH=<ssh target of the ASUS server>   required: whatever `ssh <target>` reaches it from here WITHOUT a prompt
#                                             (key or agent; BatchMode is on so an unattended run cannot hang)
# Optional: K16_PORT (server port, 29616), K16_LOCAL_PORT (local end, default the same), K16_SSH_OPTS (extra ssh
# options, e.g. "-J jumphost"), K16_PY, K16_MODEL_DIR, K16_OUT, K16_ALLOW_SHARED=1, K16_NO_STATIC=1, K16_KEEP_SERVER=1.
# If you keep your own tunnel open instead, set K16_SERVER=127.0.0.1:<local port> and leave K16_SSH unset.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_launch_common.sh"

PORT="${K16_PORT:-29616}"
if [ -n "${K16_SERVER:-}" ] && [ -z "${K16_SSH:-}" ]; then
  echo "[launch] using K16_SERVER=$K16_SERVER as given; no tunnel is opened here (the server listens on its loopback only)"
  export K16_TRANSPORT="${K16_TRANSPORT:-as given by K16_SERVER (tunnel managed by the user)}"
else
  if [ -z "${K16_SSH:-}" ]; then
    echo "[launch] STOP: set K16_SSH to the ssh target of the ASUS server, for example K16_SSH=brendon@asus-ESC4000-E11" >&2; exit 3
  fi
  LPORT="${K16_LOCAL_PORT:-$PORT}"
  if (exec 3<>"/dev/tcp/127.0.0.1/$LPORT") 2>/dev/null; then
    echo "[launch] STOP: local port $LPORT is already in use on this machine; set K16_LOCAL_PORT to a free one" >&2; exit 3
  fi
  mkdir -p "$K16_OUT"
  TLOG="$K16_OUT/$(date -u +%Y%m%dT%H%M%SZ)_$(hostname)_edge_tunnel.log"
  # shellcheck disable=SC2086
  ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
      -o Compression=no ${K16_SSH_OPTS:-} -L "127.0.0.1:$LPORT:127.0.0.1:$PORT" "$K16_SSH" >"$TLOG" 2>&1 &
  TUNNEL_PID=$!
  trap 'kill "$TUNNEL_PID" 2>/dev/null' EXIT          # only the tunnel this script started
  up=0
  for _ in $(seq 1 60); do
    kill -0 "$TUNNEL_PID" 2>/dev/null || break
    if (exec 3<>"/dev/tcp/127.0.0.1/$LPORT") 2>/dev/null; then up=1; break; fi
    sleep 0.5
  done
  if [ "$up" != 1 ]; then
    echo "[launch] STOP: the SSH tunnel to $K16_SSH did not come up: $(tail -3 "$TLOG" 2>/dev/null)" >&2
    echo "[launch] check that 'ssh -o BatchMode=yes $K16_SSH true' works from this machine; nothing was started." >&2; exit 3
  fi
  echo "[launch] SSH tunnel up: 127.0.0.1:$LPORT -> $K16_SSH 127.0.0.1:$PORT (pid $TUNNEL_PID, log $TLOG)"
  export K16_SERVER="127.0.0.1:$LPORT" K16_TRANSPORT="ssh tunnel (ssh -L) from the edge to the server's loopback"
fi
k16_run edge k16_edge.py "$@"
exit $?
