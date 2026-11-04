#!/usr/bin/env bash
# Serve ONE winner-label-stability model with vLLM (reconstructed evaluation setup).
#
# Reconstruction choices (WINNER_LABEL_STABILITY_MISSION.md, Generation-Setting Policy):
#   - official checkpoints at the exact revisions from the 2026-11-03 serving audit
#   - bf16, no quantization, --max-model-len 32768 (the observed total-context cap)
#   - vLLM 0.8.4, the version the LLMRouterBench paper reports for its ~7B pool (App. B.1)
#   - --generation-config auto (the 0.8.4 default; the paper left non-temperature/top_p decoding
#     parameters at their defaults): fields the request omits come from the model's
#     generation_config.json. Qwen3-8B: top_k=20 applies (its temperature 0.6 / top_p 0.95 are
#     overridden by the request). DeepSeek has no generation_config.json, so nothing is added.
#   - prefix caching disabled: a deliberate reconstruction difference, NOT an original setting
#     (0.8.4's V1 engine enables it by default). It reuses computation only, but keeping it off
#     rules out any confusion with cached-response reuse across repetitions.
#   - no reasoning parser (stored outputs carry inline <think>)
#   - served under the exact original model name
#   - no sampling flags and no --seed (engine default) are added; sampling comes only from the
#     request (temperature 0.2, top_p 1.0) plus the model defaults above
#   - the server log is written to server_<name>.log; the pre-flight client checks its startup
#     lines (V1 engine, version, default sampling params) before sending anything
# This is not a claim about the unknown original server.
#
# Usage:
#   wls_serve_vllm.sh deepseek|qwen [--print-cmd]
# Env:
#   WLS_PORT (8000)  WLS_HOST (127.0.0.1)
#   WLS_RUN_DIR      where server_<name>.json/.log are written (default: the pre-flight dir)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_VERSION="0.8.4"
HOST="${WLS_HOST:-127.0.0.1}"
PORT="${WLS_PORT:-8000}"
RUN_DIR="${WLS_RUN_DIR:-$REPO/diagnostics/winner_label_stability/preflight}"

case "${1:-}" in
  deepseek)
    HF_REPO="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    REVISION="6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
    SERVED="DeepSeek-R1-0528-Qwen3-8B" ;;
  qwen)
    HF_REPO="Qwen/Qwen3-8B"
    REVISION="b968826d9c46dd6066d109eabc6255188de91218"
    SERVED="Qwen3-8B" ;;
  *) echo "usage: $0 deepseek|qwen [--print-cmd]" >&2; exit 2 ;;
esac

CMD=(vllm serve "$HF_REPO"
  --revision "$REVISION"
  --tokenizer-revision "$REVISION"
  --served-model-name "$SERVED"
  --dtype bfloat16
  --max-model-len 32768
  --generation-config auto
  --no-enable-prefix-caching
  --host "$HOST"
  --port "$PORT")

if [[ "${2:-}" == "--print-cmd" ]]; then
  printf '%s\n' "${CMD[@]}"
  exit 0
fi

# One model at a time: refuse if any vLLM server is already running or the port is taken.
if pgrep -f "vllm serve" >/dev/null; then
  echo "another 'vllm serve' process is running; stop it first (one model at a time)" >&2; exit 1
fi
if (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; then
  echo "port $HOST:$PORT is already in use" >&2; exit 1
fi

GOT="$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)"
if [[ "$GOT" != "$VLLM_VERSION" ]]; then
  echo "vllm $VLLM_VERSION required, found '${GOT:-none}' (pip install vllm==$VLLM_VERSION)" >&2; exit 1
fi

mkdir -p "$RUN_DIR"
MANIFEST="$RUN_DIR/server_${SERVED}.json"
LOG="$RUN_DIR/server_${SERVED}.log"
python - "$MANIFEST" "$LOG" "$HF_REPO" "$REVISION" "$SERVED" "$REPO" "${CMD[@]}" <<'PY'
import datetime, json, platform, subprocess, sys
out, log, repo, rev, served, repo_dir, *cmd = sys.argv[1:]
def sh(*a):
    try: return subprocess.run(a, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e: return f"unavailable: {type(e).__name__}"
import torch, vllm
m = {"launched_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "served_model_name": served, "hf_repo": repo, "revision": rev,
     "engine": "vllm", "vllm_version": vllm.__version__, "torch_version": torch.__version__,
     "cuda_version": torch.version.cuda, "dtype": "bfloat16", "max_model_len": 32768,
     "generation_config": "auto", "prefix_caching": False, "reasoning_parser": None,
     "server_seed": "engine default (not set)", "quantization": None,
     "gpus": sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
     "host": platform.node(), "python": platform.python_version(),
     "repo_commit": sh("git", "-C", repo_dir, "rev-parse", "HEAD"),
     "server_log": log, "command": cmd}
open(out, "w").write(json.dumps(m, indent=2) + "\n")
print(f"wrote {out}", file=sys.stderr)
PY

# Fresh log per launch (overwritten), so the client never reads a previous server's lines.
exec "${CMD[@]}" > >(tee "$LOG") 2>&1
