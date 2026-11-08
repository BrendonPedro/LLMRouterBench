# E0 runbook: hardware, exact commands, runtime and cost

The protocol and the pass/fail rules are in `PREREG.md`. This file is operational only. **Nothing here has run
on a GPU, and no hardware has been rented.** Runtimes and costs are **planning estimates, not measurements.**
Prices are assumed on-demand list prices and must be re-checked at booking.

## 0. Files

| File | Role |
|---|---|
| `e0_spec.py` | every frozen constant (models, classes, budgets, L_F, cells, thresholds, seeds) |
| `split_runtime.py` | U-shaped split runtime (edge runner, segment server, echo stub, wire format, app shaper) |
| `e0_memory.py` | analytic memory model (prediction only) |
| `e0_instrument.py` | memory cap, NVML sampler, allocator statistics, environment capture |
| `e0_f.py` | F: capped fit attempts, U-max search, EDGE competitor fit (+ S5 profile) |
| `e0_q.py` | Q: continuations, teacher-forced reference/compare, Q2 free-run, Q decision |
| `e0_latency.py` | E0-L: workload, plan, SPLIT/REMOTE/EDGE runners, segment profile, load generator, netem, linkcheck |
| `e0_r.py` | R: prior, upper-bound gates, profile fitting + validation, R cells, final decision |
| `e0_selftest.py` | local and pod self-test |
| `frozen/` | `workload.json`, `q2_keys.json`, `memory_model.md`, R priors |
| `requirements-e0.txt` | pinned HF/runtime environment (identical on both hosts) |

## 1. Hardware plan

### 1.1 What runs where

| Stage | Machine | Needs | Local WSL? | NYCU? | RunPod? |
|---|---|---|---|---|---|
| 0: preparation, self-test, prior | this WSL box (Ryzen 5 8500G, 9 GB RAM, no CUDA) | CPU | **done** | — | — |
| 1: edge single-machine (F, EDGE fit + profiles, edge-segment profiles) | edge GPU | 1× 24 GB NVIDIA, exclusive | no (no CUDA) | **yes**, if a ≥ 24 GB Ampere/Ada GPU with exclusive access exists (e.g. one RTX 3090 of the manuscript workstation) | otherwise 1× RTX 3090 / 4090 / A5000 pod |
| G-F, G-UB1 | any | CPU | **yes** | yes | — |
| 2: server single-machine (Q references, in-host floor, server-segment profiles, REMOTE-HF) | server GPU | ≥ 80 GB; H200 preferred | no | a DGX **A100/H100 80 GB** can run all of Stage 2 (see note 1) | 1× H200 SXM pod |
| G-UB2 | any | CPU | **yes** | yes | — |
| 3: joint (G-link, Q1(a), Q2, hardware-floor edge half, E0-L cells) | edge **and** server at once, same datacenter | the Stage 1 edge GPU model + **1× H200 141 GB** (co-resident contention, A10) + native RTT ≤ 5 ms | no | only if NYCU has an H200-class server on the same LAN as the edge | **yes** (default): H200 pod + 24 GB pod in one datacenter |
| fit, R, decide | any | CPU | **yes** | yes | — |

**Notes**
1. On an A100/H100 **80 GB** server:
   - The in-host floor fits on one GPU (about 76 GB) or uses 2 GPUs (record which).
   - The co-resident contention block does **not** fit (60 GB segment + 61 GB vLLM). On 80 GB servers the four
     load cells and the κ > 0 held-out cells are not run, and R is evaluated in the good state only (PREREG A10).
   - A V100 DGX (no bf16) or a DGX Spark / GB10 (273 GB/s) is not a server for E0 (HARDWARE-PLAN §2.2).
2. A **WAN pairing** (a NYCU edge and a RunPod server abroad) can run Q1(a)/Q2: correctness does not depend on
   RTT. It fails G-link for latency, so E0-L then needs a same-datacenter pairing.
3. The **same edge GPU model** must be used in Stages 1 and 3 (A13), because the R latency for EDGE comes from
   Stage 1.

### 1.2 Edge machine (exact)

- 1× NVIDIA GPU with **24 GiB** (24,576 MiB), compute capability ≥ 8.6 (bf16).
  - Selection order: **RTX 3090** (the manuscript edge) → RTX 4090 → RTX A5000.
  - Not Blackwell: the vLLM 0.8.5.post1 / torch 2.6.0 wheels are CUDA 12.4.
- Driver supporting CUDA 12.4 (≥ 550).
- No other process on the GPU, no display attached, MIG off.
- Host: ≥ 8 vCPU, ≥ 48 GB RAM, ≥ 200 GB disk (volume).
  - Disk: 32B edge shards 1–4 and 14–17 (≈ 31 GB), the three AWQ models (35.4 GB), Qwen3-8B (16.4 GB), two
    venvs (≈ 15 GB) and logs.
- Python 3.12, `uv`.
- Optional: root / CAP_NET_ADMIN for `tc netem` (otherwise the app shaper, PREREG §6.3) and `nvidia-smi -lgc` for
  clock locking.
- Reachable from the server: TCP 29500 (segment), 29501 (echo), 8000 (vLLM), 8090 (file transfer),
  8099 (load status).

### 1.3 Server machine (exact)

- **1× NVIDIA H200 SXM 141 GB** (S1 recorded 143,771 MiB) with a CUDA 12.4-capable driver.
- Exclusive for Stage 3 (≈ 5 h).
- Host: ≥ 16 vCPU, ≥ 128 GB RAM, ≥ 250 GB disk (volume).
  - Disk: Qwen3-32B 65.5 GB, Qwen3-8B 16.4 GB, venvs, Q reference files ≈ 1 GB.
- Stage 3 memory plan on one H200:
  - vLLM Qwen3-32B with `--gpu-memory-utilization 0.50` (70.2 GiB: 61 GiB weights + ≈ 9 GiB KV, enough for
    load 32 × 512 tokens);
  - plus the HF segment server with layers [1, 63) (56.3 GiB weights + ≤ 4.3 GiB KV + transients);
  - ≈ 133 GiB of 140.4 GiB in total. Start vLLM first and record `nvidia-smi` with both resident.

## 2. Common setup (each pod / machine)

```bash
apt-get update && apt-get install -y git tmux iproute2 curl
pip install -U uv
cd /workspace
GIT_LFS_SKIP_SMUDGE=1 git clone --branch deployment-e0-split-prep https://github.com/BrendonPedro/LLMRouterBench.git
git -C /workspace/LLMRouterBench fetch --tags
git -C /workspace/LLMRouterBench checkout e0-prereg-v1     # the frozen pre-registration (PREREG §10)
git -C /workspace/LLMRouterBench describe --tags --exact-match   # must print e0-prereg-v1

uv venv /workspace/venv-e0 --python 3.12
uv pip install --python /workspace/venv-e0/bin/python "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install --python /workspace/venv-e0/bin/python -r /workspace/LLMRouterBench/diagnostics/deployment_e0/requirements-e0.txt
uv venv /workspace/venv-vllm --python 3.12                  # identical to S1 (deployment_s1/RUNPOD.md)
uv pip install --python /workspace/venv-vllm/bin/python --exclude-newer 2025-05-20 \
    "vllm==0.8.5.post1" "transformers==4.51.3" hf_transfer

cat > /workspace/e0_env.sh <<'EOF'
export E0=/workspace/LLMRouterBench/diagnostics/deployment_e0
export PY=/workspace/venv-e0/bin/python
export VLLM=/workspace/venv-vllm/bin/vllm
export MODELS=/workspace/models RUNS=/workspace/e0_runs HF_HOME=/workspace/hf
export Q32=9216db5781bf21249d130ec9da846c4624c16137 Q8=b968826d9c46dd6066d109eabc6255188de91218
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # deterministic cuBLAS (E0 §7); set on both hosts
mkdir -p $RUNS/env $RUNS/stage1 $RUNS/stage2 $RUNS/stage3/lat $RUNS/q $RUNS/gates $RUNS/R
kv() { $PY -c "import json;v=json.load(open('$RUNS/stage1/f_results.json'))['u_max']['$1'];print(v[0] if v else '')"; }
EOF
source /workspace/e0_env.sh
$PY -c "import sys;sys.path.insert(0,'$E0');import torch,e0_instrument as I;I.write_json('$RUNS/env/'+__import__('socket').gethostname()+'.json',I.environment(torch))"
```

Every later shell starts with `source /workspace/e0_env.sh`.

## 3. Stage 1: edge single-machine (≈ 2.5 h)

**What Stage 1 is (PREREG §9.4, §9.5):**
- It is only a CUDA, memory and upper-bound feasibility gate.
- Its first `split-fit` is the **first CUDA execution** of this harness; nothing has been validated on CUDA
  before it.
- A CONTINUE here is **not** evidence that SPLIT meets any latency or deadline requirement.
- A non-memory exception in any `split-fit` makes Stage 1 **INVALID**, not an F failure (PREREG §5.6). Fix it
  without changing anything memory-relevant, commit, and re-run Stage 1 from scratch into a fresh `$RUNS/stage1`.

```bash
source /workspace/e0_env.sh
HF=/workspace/venv-e0/bin/huggingface-cli
$HF download Qwen/Qwen3-32B --revision $Q32 --local-dir $MODELS/Qwen3-32B \
    --include "model-0000[1-4]-of-00017.safetensors" "model-0001[4-7]-of-00017.safetensors" "*.json" "*.txt"
$HF download Qwen/Qwen3-8B --revision $Q8 --local-dir $MODELS/Qwen3-8B
$PY $E0/e0_selftest.py --only spec s1 template runtime f latency r --tokenizer-dir $MODELS/Qwen3-32B \
    --report $RUNS/selftest_edge.json                     # must be 0 FAIL

# F: U-max search per class (fresh process per attempt) + U-min at 12 GB
for M in 12 16 24; do $PY $E0/e0_f.py split-umax --model-dir $MODELS/Qwen3-32B --class $M --out-dir $RUNS/stage1; done
$PY $E0/e0_f.py split-fit --model-dir $MODELS/Qwen3-32B --class 12 --k 1 --out $RUNS/stage1/split_fit_c12_k1.json

# EDGE competitors: measured fit for all 4 models x 3 classes; S5 latency profile when the fit passes
for M in 12 16 24; do for m in Qwen3-8B-AWQ Qwen3-14B-AWQ Qwen3-8B Qwen3-32B-AWQ; do
  $PY $E0/e0_f.py edge-fit --model $m --class $M --vllm-bin $VLLM --out-dir $RUNS/stage1 --profile-out $RUNS/stage1
done; done
$PY $E0/e0_f.py summarize --dir $RUNS/stage1               # -> f_results.json, edge_fit.json

# edge-segment compute profiles under each class cap, against a loopback echo stub
$PY $E0/split_runtime.py serve --echo --host 127.0.0.1 --port 29501 > $RUNS/stage1/echo.log 2>&1 &
for MK in "12 1" "12 $(kv 12)" "16 $(kv 16)" "24 $(kv 24)"; do set -- $MK; [ -n "$2" ] || continue
  $PY $E0/e0_latency.py run-split --model-dir $MODELS/Qwen3-32B --server 127.0.0.1:29501 --k $2 --j $2 \
      --cap-class $1 --with-depth 1 --out-dir $RUNS/stage1
done
kill %1
```

**Gates G-F and G-UB1.** These can run on the pod or locally after copying `$RUNS/stage1`.

```bash
$PY -c "import json;r=json.load(open('$RUNS/stage1/f_results.json'));print('valid',r['valid'],r['harness_defects']);print('u_max',r['u_max']);print('eligible',r['edge_eligible']);print('blocked',r['no_split_pass_classes']);print('F_pass_static',r['F_pass_static'])"
$PY $E0/e0_r.py gate-ub1 --stage1-dir $RUNS/stage1 --server-gpu H200 --out $RUNS/gates/gate_ub1.json
```

Read the outcome in this order:

| Condition | Outcome |
|---|---|
| `valid` false (a `split-fit` raised a non-memory exception) | **INVALID**: fix the harness defect, re-run all of Stage 1. Not an F result |
| `F_pass_static` false (no class has a U-max; (1,1) does not fit 24 GB) | **STOP**: F fails, SPLIT removed |
| `gate_ub1.json` decision `INCOMPLETE` | measure the missing edge-segment profile, re-run the gate |
| `gate_ub1.json` decision `STOP` (no bound cell reaches Δ ≥ 3.0 pt) | **STOP**: R fails by bound, SPLIT removed |
| `gate_ub1.json` decision `CONTINUE` | proceed to Stage 2. This is **not** latency evidence |

- On any STOP, write `DECISION.md` from the gate file and release the hardware.
- `no_split_pass_classes` lists classes blocked by PREREG §8.4 (b), where a desk-eligible EDGE competitor failed
  its fit for a non-memory reason. If every class with a U-max is blocked, the gate STOPs.

## 4. Stage 2: server single-machine (≈ 2.5 h on 1× H200)

```bash
source /workspace/e0_env.sh                 # copy $RUNS/stage1/f_results.json here first (for kv)
HF=/workspace/venv-e0/bin/huggingface-cli
$HF download Qwen/Qwen3-32B --revision $Q32 --local-dir $MODELS/Qwen3-32B
$HF download Qwen/Qwen3-8B --revision $Q8 --local-dir $MODELS/Qwen3-8B
$PY $E0/e0_selftest.py --only spec s1 template runtime --tokenizer-dir $MODELS/Qwen3-32B --report $RUNS/selftest_server.json

# Q references (single-host HF)
$PY $E0/e0_q.py cont-gen --model-dir $MODELS/Qwen3-32B --out $RUNS/q/cont.jsonl --batch 32
$PY $E0/e0_q.py ref --model-dir $MODELS/Qwen3-32B --cont $RUNS/q/cont.jsonl --out-dir $RUNS/q/ref32
$PY $E0/e0_q.py ref --model-dir $MODELS/Qwen3-8B --cont $RUNS/q/cont.jsonl --out-dir $RUNS/q/ref8_server
$PY $E0/e0_q.py q2-ref --model-dir $MODELS/Qwen3-32B --out $RUNS/q/q2_ref.jsonl

# segment server holding every middle layer a legal config can need
$PY $E0/split_runtime.py serve --model-dir $MODELS/Qwen3-32B --layers 1-63 --host 127.0.0.1 --port 29500 \
    --device cuda:0 > $RUNS/stage2/segment.log 2>&1 &
sleep 120
# in-host floor (c): the edge half on the same GPU, cut like each Q configuration
for K in 1 $(kv 16); do
  $PY $E0/e0_q.py cmp --config split --model-dir $MODELS/Qwen3-32B --cont $RUNS/q/cont.jsonl --ref-dir $RUNS/q/ref32 \
      --out-dir $RUNS/q/inhost_k$K --server 127.0.0.1:29500 --k $K --j $K
done
# server-segment compute profiles (synthetic activations)
for K in 1 $(kv 12) $(kv 16) $(kv 24); do
  $PY $E0/e0_latency.py profile-segment --server 127.0.0.1:29500 --k $K --j $K --model-dir $MODELS/Qwen3-32B \
      --out-dir $RUNS/stage2
done
kill %1
# informational engine-gap cell (A8)
$PY $E0/e0_latency.py run-hf-local --model-dir $MODELS/Qwen3-32B --out-dir $RUNS/stage2
```

**Gate G-UB2** (locally, with `stage1/` and `stage2/` copied):

```bash
$PY $E0/e0_r.py gate-ub2 --stage1-dir $RUNS/stage1 --stage2-dir $RUNS/stage2 --out $RUNS/gates/gate_ub2.json
```

- Decision `STOP`: **STOP**. Do not book Stage 3.
- `INCOMPLETE`: measure the missing server-segment profile, then re-run the gate.
- `CONTINUE`: book Stage 3. It is not latency evidence (PREREG §9.4).

## 5. Stage 3: joint (≈ 5 h, both pods in one datacenter)

**Server:**

```bash
source /workspace/e0_env.sh
$VLLM serve $MODELS/Qwen3-32B --served-model-name Qwen3-32B --tokenizer $MODELS/Qwen3-32B --dtype bfloat16 \
    --max-model-len 32768 --gpu-memory-utilization 0.50 --seed 0 --host 0.0.0.0 --port 8000 > $RUNS/stage3/vllm.log 2>&1 &
sleep 240
$PY $E0/split_runtime.py serve --model-dir $MODELS/Qwen3-32B --layers 1-63 --host 0.0.0.0 --port 29500 \
    --device cuda:0 > $RUNS/stage3/segment.log 2>&1 &
$PY $E0/split_runtime.py serve --echo --host 0.0.0.0 --port 29501 > $RUNS/stage3/echo.log 2>&1 &
nvidia-smi > $RUNS/stage3/nvidia_smi_resident.txt          # both processes resident, < 141 GB
tar czf $RUNS/q_refs.tgz -C $RUNS q/cont.jsonl q/ref32 q/ref8_server q/q2_ref.jsonl q/inhost_k*
python3 -m http.server 8090 --directory $RUNS > /dev/null 2>&1 &
```

**Edge:** link check (G-link), then Q.

```bash
source /workspace/e0_env.sh
export SRV=<server address>   IF=<edge interface, e.g. eth0>
$PY $E0/e0_latency.py linkcheck --server $SRV:29501 --out $RUNS/stage3/linkcheck.json
# G-link: joint_allowed = (rt_ms_p50 <= 5.0) and (bulk_mbps_median >= 900). False => BLOCKED: stop here and
# re-provision the pairing in one datacenter (not an E0 outcome).
curl -fsS http://$SRV:8090/q_refs.tgz | tar xz -C $RUNS
K16=$(kv 16)
$PY $E0/e0_q.py cmp --config hf --model-dir $MODELS/Qwen3-8B --cont $RUNS/q/cont.jsonl --ref-dir $RUNS/q/ref8_server \
    --out-dir $RUNS/q/hw8_edge                             # hardware floor, edge half
for K in 1 $K16; do
  $PY $E0/e0_q.py cmp --config split --model-dir $MODELS/Qwen3-32B --cont $RUNS/q/cont.jsonl --ref-dir $RUNS/q/ref32 \
      --out-dir $RUNS/q/split_k$K --server $SRV:29500 --k $K --j $K
  $PY $E0/e0_q.py q2-run --model-dir $MODELS/Qwen3-32B --ref $RUNS/q/q2_ref.jsonl --out $RUNS/q/q2_split_k$K.jsonl \
      --server $SRV:29500 --k $K --j $K
done
$PY $E0/e0_q.py decide --pipeline-dirs $RUNS/q/inhost_k1 $RUNS/q/inhost_k$K16 --hw-dir $RUNS/q/hw8_edge \
    --split-dirs $RUNS/q/split_k1 $RUNS/q/split_k$K16 --q2-ref $RUNS/q/q2_ref.jsonl \
    --q2-runs $RUNS/q/q2_split_k1.jsonl $RUNS/q/q2_split_k$K16.jsonl --out $RUNS/q/q_decision.json
```

- **G-Q:** `pass` false: apply **one** implementation fix, re-run the four Q commands into fresh directories,
  then `decide ... --fix-attempt 1`.
- If it fails again: **STOP** (SPLIT removed). No latency cell is run before Q passes.

**E0-L cells.** These are the decision cells from `plan`, except `core-04..06`, which Stage 1 measured.

```bash
$PY $E0/e0_latency.py plan > $RUNS/stage3/plan.jsonl
SHAPING=$(sudo -n tc qdisc show dev $IF >/dev/null 2>&1 && echo netem || echo app)   # recorded per request
# for every cell id C in plan order (core, load, rtt, heldout, bw, cut):
#  1. link:   netem: $PY $E0/e0_latency.py netem --dev $IF --rtt-ms <rtt> --bw-mbps <bw> --action apply   (native: --action clear)
#  2. load:   on the SERVER, for load > 0, wait 60 s after starting:
#             $PY $E0/e0_latency.py load-gen --base-url http://127.0.0.1:8000/v1 --concurrency <load> \
#                 --status-file $RUNS/stage3/load_status.json --status-port 8099 &
#  3. run (EDGE):
$PY $E0/e0_latency.py run-cell --label C --out-dir $RUNS/stage3/lat --shaping $SHAPING --dev $IF \
    --model-dir $MODELS/Qwen3-32B --server $SRV:29500 --base-url http://$SRV:8000/v1 --served-model Qwen3-32B \
    --f-results $RUNS/stage1/f_results.json --load-status http://$SRV:8099/status --nvml
#  4. stop the load generator (kill %N on the server); next cell
```

**Co-residency contingency (pre-declared; no criterion changes).** If the segment server cannot allocate next to
vLLM at util 0.50:
1. Run the two U-min cells (`core-01`, `cut-01`) in a session with vLLM stopped and the segment holding [1, 63).
2. Restart with vLLM at 0.50 and the segment holding only [k12, 64 − k12), where k12 = `$(kv 12)`, for every other
   cell.
3. Record the split sessions in `$RUNS/stage3/sessions.txt`.

`run-cell` refuses to start if:
- the netem state does not match the cell;
- a shaped cell has no shaping;
- the load generator's concurrency, uptime (≥ 60 s) or error count do not match;
- the cell's class failed F.

SPLIT cells run under their class cap automatically. A transport error ends the cell (an F failure), and the
request is never retried.

## 6. Analysis (local, CPU)

```bash
$PY $E0/e0_r.py fit --lat-dir $RUNS/stage3/lat --f-results $RUNS/stage1/f_results.json --out $RUNS/R/bundle.json
$PY $E0/e0_r.py r --bundle $RUNS/R/bundle.json --stage1-dir $RUNS/stage1 --out-dir $RUNS/R
$PY $E0/e0_r.py decide --f-results $RUNS/stage1/f_results.json --lat-dir $RUNS/stage1 $RUNS/stage3/lat \
    --q $RUNS/q/q_decision.json --r $RUNS/R/R_cells.json --out-dir $RUNS/R
```

- `R/R_cells.md` lists every cell.
- `R/DECISION.md` gives Q / F / R, the E0 §8 consequence and `memory_class_scope`.
- `E0 = PENDING` has exactly two causes:
  - the §8.5 extension is triggered: run it once with `frozen/extension_400_keys.json`;
  - a 24 GB-only pass awaits §8.4 (a): add Qwen3-30B-A3B, re-run `r`, then `decide --a3b-added`.

  Nothing else changes.

## 7. Runtime and cost (planning estimates; verify prices at booking)

Assumed on-demand prices: H200 SXM ≈ $3.6–4.0/h; RTX 3090/4090 ≈ $0.4–0.7/h; network volumes ≈ $0.07/GB-month.

| Stage | Hardware | Wall-clock (estimate) | Dominant items | GPU cost (estimate) |
|---|---|---|---|---|
| 0 | local WSL | done | self-test 15 min | $0 |
| 1 | 1× 24 GB edge (NYCU if available, else RunPod) | 2–3 h | downloads ≈ 100 GB; 12 vLLM fits + 7 S5 profiles; ≈ 8 split-fit attempts; 4 edge-segment profiles | $0 at NYCU, ≈ $1.5–2 on RunPod |
| 2 | 1× H200 | 2–3 h | downloads 82 GB; 200 greedy continuations ≤ 4,096 (batched); 32B/8B references; 30 × 1,024 Q2 references; in-host floors; 4 segment profiles; REMOTE-HF | ≈ $8–12 |
| 3 | 1× H200 + 1× 24 GB, together | 4.5–6 h | Q1(a) 2 × 200 sequences; Q2 2 × 30 × 1,024 tokens; 19 E0-L cells × 32–38 requests × 256 tokens (12 SPLIT cells dominate, ≈ 2 h) | ≈ $20–28 |
| 4 | local | < 10 min | fit / R / decide | $0 |
| **Total** | | **≈ 9–12 h of GPU wall-clock** | | **≈ $30–45**; ≤ $70 with a 1.5× contingency, plus ≈ $5 of volume storage |

Early stops save the later stages:
- G-F or G-UB1 failing costs ≤ $2.
- G-UB2 failing costs ≤ $14.
- G-Q failing saves about 3 h of the joint window.
