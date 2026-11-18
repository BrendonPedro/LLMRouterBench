# K16-HWFIT launcher helpers (sourced). Finds a Python that already has the required packages; installs nothing.
K16_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K16_REQ='import os, sys, torch, transformers, safetensors, numpy
ok = transformers.__version__ == "4.51.3" and (torch.cuda.is_available() or os.environ.get("K16_DEV") == "1")   # K16_DEV: CPU tests only
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} transformers {transformers.__version__}")
sys.exit(0 if ok else 1)'

k16_pick_python() {
  local c out
  if [ -n "${K16_PY:-}" ]; then                 # an explicit choice is used or refused, never replaced
    if out=$("$K16_PY" -I -c "$K16_REQ" 2>&1); then echo "$K16_PY"; return 0; fi
    echo "K16_PY=$K16_PY is not usable: $out" >&2; return 1
  fi
  for c in "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}" \
           "$HOME"/research/LLMRouterBench*/.venv*/bin/python "$HOME"/.cache/LLMRouterBench/venv*/bin/python \
           "$HOME"/venv*/bin/python "$HOME"/.venv*/bin/python "$HOME"/*/venv*/bin/python "$HOME"/*/.venv*/bin/python \
           "$HOME"/miniconda3/envs/*/bin/python "$HOME"/anaconda3/envs/*/bin/python "$HOME"/miniforge3/envs/*/bin/python \
           "$(command -v python3 2>/dev/null)" "$(command -v python 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] || continue
    if out=$("$c" -I -c "$K16_REQ" 2>&1); then echo "$c"; return 0; fi
    echo "  not usable: $c ($(echo "$out" | tail -1))" >&2
  done
  echo "BLOCKER: no Python here has torch with CUDA and transformers 4.51.3. Set K16_PY=/path/to/python (the E1-OPT venv has both)." >&2
  return 1
}

k16_run() {   # k16_run <role> <driver.py> [args...]: static/CPU validation first, then the driver; everything is logged
  local role="$1" driver="$2"; shift 2
  export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  local out="${K16_OUT:-$HOME/k16fit_runs}"; mkdir -p "$out"
  local log="$out/$(date -u +%Y%m%dT%H%M%SZ)_$(hostname)_${role}_console.log"
  {
    echo "[launch] K16-HWFIT $role on $(hostname) at $(date -u +%FT%TZ); console log: $log"
    echo "[launch] code: $(git -C "$K16_HERE" rev-parse HEAD 2>/dev/null) $(git -C "$K16_HERE" status --porcelain -- . 2>/dev/null | head -3)"
    PY=$(k16_pick_python) || exit 3
    echo "[launch] python: $PY ($("$PY" -I -c "$K16_REQ" 2>&1 | tail -1))"
    echo "[launch] ===== stage 0: CPU/static validation (tiny random model on the CPU; no GPU memory is used) ====="
    if ! CUDA_VISIBLE_DEVICES="" K16_TEST_STRICT=0 "$PY" -I "$K16_HERE/tests/test_k16.py" 2>&1 | grep -v "Sliding Window"; then
      echo "[launch] STOP: CPU/static validation failed on this host; nothing else was started."; exit 3
    fi
    "$PY" -I "$K16_HERE/$driver" all "$@"
    rc=$?
    echo "[launch] $role finished with exit code $rc (0 = complete, 3 = preflight or memory STOP, 4 = gate FAIL, 5 = no edge connected)"
    exit $rc
  } 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}
