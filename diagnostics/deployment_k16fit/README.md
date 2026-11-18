# deployment_k16fit: K16-HWFIT, a runnable Qwen3-32B split for the NYCU lab pair (2026-11-18)

**Status: exploratory hardware-fit experiment. Not pre-registered. Not a continuation of E0 or of E1-OPT (v1, v2, v3).**
Nothing here has run on a GPU yet. Results from it must be labelled exploratory.

Why it exists: E1-OPT inherited E0's cuts (k = 3 and k = 5), whose server segment needs about 59 GiB on one GPU.
One RTX A6000 has 48 GiB, so the lab's role assignment could not run. This experiment keeps the model, the runtime
and the wire, and moves the cut to k = 16.

| | EDGE: DGX Spark (GB10, unified memory) | SERVER: Asus main server, one RTX A6000 |
|---|---|---|
| holds | embeddings, layers 0-15, layers 48-63, final norm, LM head, sampling, tokenizer | layers 16-47 |
| KV cache | layers 0-15 and 48-63 | layers 16-47 |
| wire | sends the bf16 residual stream after layer 15 | returns the bf16 residual stream after layer 47 |
| network | opens the SSH tunnel | listens on 127.0.0.1 only |

Model: `Qwen/Qwen3-32B` at revision `9216db57`, bf16, verified on each host against
`frozen/checkpoint_manifest.json` (size, safetensors headers, SHA-256 of every file the host loads).

## What is reused and what is new

Reused unchanged, by import: `deployment_e0/split_runtime.py` (weight loading), `deployment_e1opt/e1_segment.py`
(`Segment`, `USplitEdge`), `deployment_e1opt/e1_protocol.py` (wire, HELLO checks), `deployment_e0/e0_q.py`
(comparison metrics). No file of `deployment_e0/` or `deployment_e1opt/` is modified. The runtime already accepts
any cut; only E1-OPT's drivers were tied to the frozen cuts, the freeze record and the inventory-decided roles.

New, in this directory only: `k16_spec.py` (constants, criterion, spec hash), `k16_memory.py`, `k16_checkpoint.py`,
`k16_preflight.py`, `k16_criteria.py`, `k16_runtime.py` (glue), `k16_server.py`, `k16_edge.py`, two launchers,
`tests/test_k16.py`.

## Order of execution and gates

| stage | where | what | stops the run if |
|---|---|---|---|
| 0 | both | CPU/static validation: `tests/test_k16.py` on the host's own Python | any test fails |
| 1 | both | preflight: software, code revision, checkpoint, GPU type, GPU occupancy, free memory incl. loading transients; server: loopback port free; edge: HELLO answered through the tunnel | any blocking check fails |
| 2a | DGX | unsplit Hugging Face model on the GB10: the reference | not finite |
| 2b | DGX | in-host k = 16 split over the real wire on loopback, reference mode | K16-CORRECT-v1 fails |
| 2c | both | two-host split, reference mode | K16-CORRECT-v1 fails |
| 3 | both | two-host generation of three chat prompts, decoded text | a request fails |
| 4 | both | two-host latency (10 prompts x 2 repetitions, 256 forced tokens, 2 warm-ups) | a request fails |
| 5 | both | optional: static mode under the same criterion; timed only if it passes | never (does not change stages 2-4) |

Every GPU stage re-reads the free memory first and refuses to load if it is short. Latency is written only behind
the passed two-host correctness gate of the same mode. A failed gate ends the run and still releases the server.

## Correctness criterion K16-CORRECT-v1 (fixed 2026-11-18, before any k = 16 GPU run)

Reference: the unsplit model on the GB10. Check set: E0's ten frozen workload prompts (5 x 256 and 5 x 2,048
tokens), each with the reference's own 64 greedy tokens. A configuration passes iff all hold:

1. C1: no NaN or Inf.
2. C2: where the reference's top-1 minus top-2 log-probability is at least 0.5 nats, the argmax equals the
   reference's at 99.5 % of positions or more.
3. C3: mean coarse KL(reference || configuration) over all positions is at most 0.05 nats.
4. C4: per prompt, the 64 greedy tokens are identical, or first differ where the reference margin is below 0.5 nats.
   At most one of the ten prompts may fail. None may be degenerate.

E1-OPT's failed check B was an absolute 2^-6 bound on bf16 hidden states, which no change of kernel or device can
meet (`PREREG_E1OPT.md` §17). This rule tests the function instead, and it is not relative to a floor measured
after the fact. Reference mode ("prefix", E0's computation) runs first. Static mode is not assumed correct: it
must pass the same rule against the same unsplit reference before it is timed. Bitwise equality and the largest
absolute difference are recorded as descriptive fields. The test suite includes a negative control (a wrong layer
order on the server), which fails the rule with KL above 6 nats.

## Expected memory (`python -I k16_memory.py`; weights from the checkpoint's actual tensors)

| process | weights GiB | KV GiB | loading transient GiB | expected peak GiB | required free GiB |
|---|---|---|---|---|---|
| ASUS server, layers 16-47 | 29.06 | 1.00 | 0.24 | 33.4 | 35.4 of 48 |
| DGX edge, two-host stages | 31.96 | 1.00 | 2.90 | 38.9 | 42.9 |
| DGX, unsplit reference (stage 2a) | 61.02 | 0.52 | 2.90 | 67.6 | 71.6 of about 119.7 |
| DGX, in-host split check (stage 2b) | 61.02 | 1.00 | 2.90 | 67.5 | 71.5 |

KV is preallocated for 4,096 tokens per owned layer, for the reference-mode cache and the static-mode cache.

Loading never makes a full host copy. Every loader reads one tensor at a time from the safetensors shards straight
onto the device: `split_runtime.Slice` for the split segments, and `k16_runtime.hf_model` for the unsplit reference
(the Hugging Face `Qwen3ForCausalLM` is built on the meta device and each shard's tensors are assigned as its
parameters; a test checks it is bit-identical to `from_pretrained`, in bf16). On the DGX Spark host and device
memory are one pool, so the gate counts a host staging buffer plus a device copy of the largest tensor (2 x 1.45
GiB). On the ASUS only the device copy counts, and the preflight separately requires 8 GiB of free host RAM. Each
stage records what its load actually peaked at (`load_memory`: device peak, device transient, host peak RSS).

## Transport

The wire protocol has no authentication of its own, so the server listens on `127.0.0.1` only and refuses any other
bind address. `launch_dgx_edge.sh` opens `ssh -N -L 127.0.0.1:29616:127.0.0.1:29616 $K16_SSH` from the DGX Spark,
runs the experiment against the local end, and closes that tunnel when it exits. SSH authenticates both ends and
encrypts the residual stream. No port is opened on the lab network. The tunnel runs with `BatchMode=yes`, so it
needs a key or agent and can never hang on a prompt. Latency rows carry `transport`; the network component of a
measurement includes the tunnel.

## Commands

Lab paths: on the ASUS the repository is `/mnt/mydisk/brendon/LLMRouterBench` and the checkpoint is
`/mnt/mydisk/brian/splitllm-cloud/models/Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137` (found automatically,
never written to, verified against the manifest). The worktree is created beside the repository and run
directories go to `k16fit_runs/` beside it.

ASUS first, inside `tmux`:
```
cd /mnt/mydisk/brendon/LLMRouterBench && git fetch origin k16-hwfit-exploratory && REV=$(git rev-parse FETCH_HEAD) && { git worktree add --detach ../LLMRouterBench-k16fit $REV 2>/dev/null || git -C ../LLMRouterBench-k16fit checkout --detach $REV; } && bash ../LLMRouterBench-k16fit/diagnostics/deployment_k16fit/launch_asus_server.sh
```
Wait for `[server] READY`. Then on the DGX Spark, inside `tmux` (`K16_SSH` is whatever `ssh <target>` reaches the
ASUS from the DGX without a prompt; change the first path if the DGX clone lives elsewhere):
```
cd ~/research/LLMRouterBench && git fetch origin k16-hwfit-exploratory && REV=$(git rev-parse FETCH_HEAD) && { git worktree add --detach ../LLMRouterBench-k16fit $REV 2>/dev/null || git -C ../LLMRouterBench-k16fit checkout --detach $REV; } && K16_SSH=asus-ESC4000-E11 bash ../LLMRouterBench-k16fit/diagnostics/deployment_k16fit/launch_dgx_edge.sh
```

Options (environment): `K16_PY` (Python with torch + transformers 4.51.3), `K16_MODEL_DIR`, `K16_GPU` (index or
UUID; default: a free card of the role's type), `K16_ALLOW_SHARED=1` (run beside another process if memory
suffices; timing rows are flagged `gpu_shared`), `K16_NO_STATIC=1`, `K16_KEEP_SERVER=1`, `K16_OUT`, `K16_PORT`,
and on the DGX `K16_LOCAL_PORT`, `K16_SSH_OPTS` (for example `-J jumphost`). With your own tunnel already open, set
`K16_SERVER=127.0.0.1:<local port>` and leave `K16_SSH` unset.
Exit codes: 0 complete, 3 preflight, memory or tunnel STOP, 4 gate FAIL, 5 server saw no edge. The scripts install
nothing and never signal a process they did not start.

## Outputs (`k16fit_runs/<utc>_<host>_<role>/` beside the repository)

`preflight.json`, `gates/*.json`, `reference/`, `correctness/{inhost_prefix,twohost_prefix,twohost_static}.json`,
`generation/chat.jsonl`, `latency/{prefix,static}.jsonl` + `_summary.json`, `summary.json`, `SUMMARY.md`,
`<run>.tgz` + `.sha256`; on the server `server_ready.json`, `server_events.jsonl`, `server_done.json`. Every file
carries the spec hash (constants + criterion + every source file + the checkpoint manifest), which also crosses
in HELLO: the server refuses an edge on another code revision, criterion or manifest.

## Tests

`<python> -I tests/test_k16.py` (CPU, about 2 minutes, 16 tests): partition, unchanged cut and criterion, unmodified
reused runtime, memory from tensors, manifest and checkpoint verification, the direct-to-device reference loader
(bit-identical to `from_pretrained`; refuses a missing tensor or another dtype), GPU selection and memory gate on
recorded inventories, the criterion with boundary cases, split = unsplit bitwise over the wire at k = 16 on a
64-layer model, KV ownership, chunked prefill, wire byte count, HELLO refusals, a rounding-noise positive control,
the wrong-layer-order negative control, two-process end-to-end run, gate order, preflight STOP, and the shipped
launchers (loopback listener, refusal of a public bind, tunnel options, tunnel closed at exit; `ssh` is a local
stand-in there).
