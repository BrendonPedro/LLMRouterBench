# S1 on RunPod: exact commands

The design is pre-registered in `PREREG.md`. Everything below runs on the pod. Nothing here has been run on a GPU yet.
Times and GPU-hours are **planning estimates, not measurements**.

## 0. Pod

| Item | Value |
|---|---|
| GPU | **1× H200 SXM 141 GB** (simplest). Fallback: **2× H100/A100 80 GB SXM** with `S1_TP=2`. A single 80 GB GPU cannot hold Qwen3-32B bf16 plus one 32k-token KV sequence, and `preflight` refuses it. Do not pick Blackwell (B200/RTX 50xx): the vLLM 0.8.5.post1 wheels are CUDA 12.4. |
| Image | Any RunPod PyTorch/CUDA template whose driver supports CUDA ≥ 12.4, e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. The image's own Python and torch are not used. |
| Storage | **Network volume ≥ 250 GB at `/workspace`**, plus a 50 GB container disk. |
| Volume contents | Weights 117 GB, LiveCodeBench 4.5 GB, bench tarball 1.3 GB, two venvs ≈ 15 GB, outputs < 1 GB. Everything that must survive a restart (weights, venvs, data, results) lives on the volume. |
| Estimate | ≈ 5–8 GPU-h on 1× H200 for smoke + parity + the 4,000-generation run (planning estimate). |

**Before creating the pod**, push the branch that contains `diagnostics/deployment_s1/` (the pod clones it):

```bash
# on your machine
git -C ~/research/LLMRouterBench push -u origin deployment-s1-quality-gate
```

## 1. Environment setup (once per volume)

```bash
apt-get update && apt-get install -y git tmux
pip install -U uv

cd /workspace
# The repo's data/ files are Git-LFS stubs whose LFS budget is exhausted; S1 does not need them.
GIT_LFS_SKIP_SMUDGE=1 git clone --branch deployment-s1-quality-gate https://github.com/BrendonPedro/LLMRouterBench.git
git -C /workspace/LLMRouterBench log --oneline -1

# vLLM venv: dependencies resolved as of 2025-05-20, so later releases cannot break 0.8.5.post1.
uv venv /workspace/venv-vllm --python 3.12
uv pip install --python /workspace/venv-vllm/bin/python --exclude-newer 2025-05-20 \
    "vllm==0.8.5.post1" "transformers==4.51.3" hf_transfer

# Scoring/driver venv: the evaluator dependencies exactly as verified locally (Python 3.12).
uv venv /workspace/venv-eval --python 3.12
uv pip install --python /workspace/venv-eval/bin/python \
    datasets==5.1.0 numpy==2.5.4 pandas==2.2.3 pyarrow==26.0.0 sympy==1.14.0 mpmath==1.3.0 \
    pylatexenc==2.11 loguru==0.7.3 openai==2.54.0 httpx==0.28.1 tiktoken==0.14.0 tenacity==9.2.1 \
    PyYAML==6.0.3 tqdm==4.70.1 PyMySQL==1.2.3 python-dotenv==1.2.4 dill==0.4.1 multiprocess==0.70.19 \
    fsspec==2026.7.0 huggingface_hub==1.33.0

cat > /workspace/s1_env.sh <<'EOF'
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export S1_DATA_DIR=/workspace/s1_data
export S1_RUNS_DIR=/workspace/s1_runs
export S1_VLLM_BIN=/workspace/venv-vllm/bin/vllm
export S1_TP=1                      # set to 2 on 2x80 GB
export PY=/workspace/venv-eval/bin/python
export S1=/workspace/LLMRouterBench/diagnostics/deployment_s1
mkdir -p "$HF_HOME" "$S1_DATA_DIR" "$S1_RUNS_DIR"
cd /workspace/LLMRouterBench
EOF
source /workspace/s1_env.sh
```

Every later shell starts with `source /workspace/s1_env.sh`.

## 2. Disk / GPU preflight

```bash
source /workspace/s1_env.sh
nvidia-smi
df -h /workspace
free -g
$PY $S1/run_s1.py preflight          # writes $S1_RUNS_DIR/preflight.json; exit 0 = PASS
```

`preflight` checks:
- GPU count and memory, including the Qwen3-32B KV headroom, and that no GPU memory is already in use;
- free disk for the missing weights;
- RAM;
- the vLLM version at `$S1_VLLM_BIN`;
- that the evaluators import;
- that Hugging Face is reachable;
- that no stale `vllm serve` is running.

Download the pinned weights (≈ 117 GB) and the data, then run the offline checks:

```bash
source /workspace/s1_env.sh
for spec in \
  Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137 \
  Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218 \
  Qwen/Qwen3-8B-AWQ@4da05a8edb55c6046cce958586c33b61da07bb79 \
  Qwen/Qwen3-14B-AWQ@31c69efc29464b6bb0aee1398b5a7b50a99340c3 \
  Qwen/Qwen3-32B-AWQ@0499c3ac83fdef8810b907a23894ba91e95eddd8; do
  /workspace/venv-vllm/bin/huggingface-cli download "${spec%@*}" --revision "${spec#*@}" || exit 1
done

$PY $S1/run_s1.py fetch-data      # LiveCodeBench release_v6 (4.5 GB) + cached Qwen3-8B outputs, sha256-verified
$PY $S1/run_s1.py check --hash-lcb --execute-lcb
```

`check` must end with `0 FAIL`. `--execute-lcb` re-runs the cached Qwen3-8B LiveCodeBench programs. It prints an
INFO agreement line, which is reported and not gated.

## 3. Smoke test (≈ 10 min)

```bash
source /workspace/s1_env.sh
$PY $S1/run_s1.py smoke 2>&1 | tee -a $S1_RUNS_DIR/smoke_driver.log
cat $S1_RUNS_DIR/smoke/smoke.json
```

The smoke test covers Qwen3-8B-AWQ on 2 queries per benchmark, in both modes, with seed 0 and `max_tokens` 1024.
It checks:
- server launch, the identity checks and shutdown;
- that every request succeeds;
- that thinking outputs open `<think>` and non-thinking outputs never do (so `enable_thinking` reaches the template);
- that a second pass sends nothing (resume);
- that all 16 outputs are scored, including LiveCodeBench execution.

It also writes a same-seed probe (`seed_probe.json`, informational). The verdict must be `PASS`, because the full run
is gated on it.

## 4. Harness-parity test (≈ 30–60 min)

```bash
source /workspace/s1_env.sh
$PY $S1/run_s1.py parity 2>&1 | tee -a $S1_RUNS_DIR/parity_driver.log
cat $S1_RUNS_DIR/parity/parity.md
```

Parity runs Qwen3-8B bf16 at the cached run's settings (T 0.2, top_p 1.0, top_k 20, thinking, 32k total cap) on all
200 queries, then applies the PREREG §5 rule.
- **PASS:** go on to the full run.
- **FAIL:** fix the harness and re-run `parity`. Completed generations are kept, so delete
  `$S1_RUNS_DIR/parity` only if the harness changed. Otherwise record the gap as a limitation (§5).

## 5. Full unattended run

```bash
source /workspace/s1_env.sh
tmux new -d -s s1 "source /workspace/s1_env.sh && \
  $PY $S1/run_s1.py run 2>&1 | tee -a $S1_RUNS_DIR/s1_driver.log; \
  $PY $S1/run_s1.py score --run s1 2>&1 | tee -a $S1_RUNS_DIR/s1_driver.log; \
  $PY $S1/summarize_s1.py 2>&1 | tee -a $S1_RUNS_DIR/s1_driver.log"
```

If parity did not pass and you decide to proceed with a stated limitation, add the following after `run`. The text
is recorded in `invocations.jsonl`.

```
--accept-parity-limitation "<residual gap and reason>"
```

**What the run does.**
- It serves the five models one at a time, in the order Qwen3-32B → 8B-AWQ → 14B-AWQ → 32B-AWQ → 8B.
- Each model gets 800 requests (2 modes × 2 seeds × 200) at client concurrency 64.
- Each completion is appended to `$S1_RUNS_DIR/s1/generations.jsonl` and fsynced.
- Crashed requests are retried twice per invocation, with at most 5 attempts per key.

**Logs.**
- Each server launch writes `servers/<model>-<UTC>.{json,log}`, holding the command, versions, `pip freeze` and
  `nvidia-smi`.

**Monitoring.**

```bash
source /workspace/s1_env.sh
$PY $S1/run_s1.py status
tail -f $S1_RUNS_DIR/s1_driver.log
tmux attach -t s1        # detach: Ctrl-b d
```

## 6. Resume after interruption (pod restart, crash, preemption)

Completed generations are never re-requested; a torn last line is moved aside to
`generations.jsonl.torn-*`.

```bash
source /workspace/s1_env.sh       # after a pod re-create: apt-get install -y tmux first
tmux kill-session -t s1 2>/dev/null || true              # a stopped driver also stops its server
pkill -f "vllm serve" || true; sleep 15; nvidia-smi    # a SIGKILLed driver can leave one; GPU memory must be free
$PY $S1/run_s1.py status
# then re-issue exactly the §5 tmux command
```

To redo only some models, add `--models Qwen3-14B-AWQ,Qwen3-8B` to `run`. Scoring resumes the same way:
`$PY $S1/run_s1.py score --run s1`.

## 7. Summarization

```bash
source /workspace/s1_env.sh
$PY $S1/run_s1.py score --run s1                    # idempotent; scores only new generations
$PY $S1/summarize_s1.py                             # -> $S1_RUNS_DIR/s1/summary/{S1_SUMMARY.md,s1_summary.json,q3_scores.jsonl}
cat $S1_RUNS_DIR/s1/summary/S1_SUMMARY.md
# with S6's measured fits:  --fit-json /workspace/s6_fit.json   ({"12": [...], "16": [...], "24": [...]})
# interim look while running: --allow-incomplete   (labelled PROVISIONAL, not a G1 decision)
```

| Decision | Meaning (PREREG §6) |
|---|---|
| `CONTINUE` | G1 passes; the 16/24 GB regimes stay |
| `DROP_16_24` | only the 12 GB class and the 6 core cells remain |
| `STOP_NO_HEADROOM` | R fails by the S1 bound; stop before split-runtime work |

## 8. Copying / backing up results

The volume persists across pod stops. Still, keep off-pod copies.

```bash
# periodic snapshot on the volume while the run is going (24 rotating hourly files)
tmux new -d -s backup 'while true; do tar czf /workspace/s1_backup_$(date +%H).tgz -C /workspace s1_runs; sleep 1800; done'

# final bundle
source /workspace/s1_env.sh
B=/workspace/s1_results_$(date -u +%Y%m%dT%H%MZ).tgz
tar czf $B -C /workspace s1_runs && sha256sum $B | tee $B.sha256

# to your machine, option A (RunPod CLI, preinstalled on pods): prints a one-time code
runpodctl send $B
#   locally:  runpodctl receive <code>

# option B (pod "SSH over exposed TCP"):
#   locally:  scp -P <PORT> root@<POD_IP>:/workspace/s1_results_*.tgz ~/research/LLMRouterBench/
```

To re-check the summary locally, on CPU:

```bash
mkdir -p /tmp/s1 && tar xzf s1_results_*.tgz -C /tmp/s1
.venv/bin/python diagnostics/deployment_s1/summarize_s1.py --runs-dir /tmp/s1/s1_runs --out-dir /tmp/s1/summary
```

**Stop the pod** when the bundle is copied. The network volume keeps weights and results until you delete it.
