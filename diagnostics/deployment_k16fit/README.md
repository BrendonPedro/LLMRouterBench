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
| 1 | both | preflight: software, code revision, checkpoint, GPU type, GPU occupancy, free memory, port or peer | any blocking check fails |
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

| process | weights GiB | KV GiB | expected peak GiB | required free GiB |
|---|---|---|---|---|
| ASUS server, layers 16-47 | 29.06 | 1.00 | 33.4 | 35.4 of 48 |
| DGX edge, two-host stages | 31.96 | 1.00 | 37.5 | 41.5 |
| DGX, unsplit reference (stage 2a) | 61.02 | 0.52 | 66.2 | 70.2 of about 119.7 |
| DGX, in-host split check (stage 2b) | 61.02 | 1.00 | 66.0 | 70.0 |

KV is preallocated for 4,096 tokens per owned layer, for the reference-mode cache and the static-mode cache.

## Commands

ASUS first, inside `tmux`:
```
cd ~/research/LLMRouterBench && git fetch origin k16-hwfit-exploratory && REV=$(git rev-parse FETCH_HEAD) && { git worktree add --detach ../LLMRouterBench-k16fit $REV 2>/dev/null || git -C ../LLMRouterBench-k16fit checkout --detach $REV; } && bash ../LLMRouterBench-k16fit/diagnostics/deployment_k16fit/launch_asus_server.sh
```
Wait for `[server] READY`. Then on the DGX Spark, inside `tmux`:
```
cd ~/research/LLMRouterBench && git fetch origin k16-hwfit-exploratory && REV=$(git rev-parse FETCH_HEAD) && { git worktree add --detach ../LLMRouterBench-k16fit $REV 2>/dev/null || git -C ../LLMRouterBench-k16fit checkout --detach $REV; } && K16_SERVER=asus-ESC4000-E11:29616 bash ../LLMRouterBench-k16fit/diagnostics/deployment_k16fit/launch_dgx_edge.sh
```
If the DGX cannot resolve the server's name, use the `K16_SERVER=<address>:29616` line the ASUS command prints.

Options (environment): `K16_PY` (Python with torch + transformers 4.51.3), `K16_MODEL_DIR`, `K16_GPU` (index or
UUID; default: a free card of the role's type), `K16_ALLOW_SHARED=1` (run beside another process if memory
suffices; timing rows are flagged `gpu_shared`), `K16_NO_STATIC=1`, `K16_KEEP_SERVER=1`, `K16_OUT`, `K16_PORT`.
Exit codes: 0 complete, 3 preflight or memory STOP, 4 gate FAIL, 5 server saw no edge. The scripts install
nothing and never signal a process they did not start.

## Outputs (`~/k16fit_runs/<utc>_<host>_<role>/`)

`preflight.json`, `gates/*.json`, `reference/`, `correctness/{inhost_prefix,twohost_prefix,twohost_static}.json`,
`generation/chat.jsonl`, `latency/{prefix,static}.jsonl` + `_summary.json`, `summary.json`, `SUMMARY.md`,
`<run>.tgz` + `.sha256`; on the server `server_ready.json`, `server_events.jsonl`, `server_done.json`. Every file
carries the spec hash (constants + criterion + every source file + the checkpoint manifest), which also crosses
in HELLO: the server refuses an edge on another code revision, criterion or manifest.

## Tests

`<python> -I tests/test_k16.py` (CPU, about 90 s): partition, unmodified reused runtime, memory from tensors,
manifest and checkpoint verification, GPU selection and memory gate on recorded inventories, the criterion with
boundary cases, split = unsplit bitwise over the wire at k = 16 on a 64-layer model, KV ownership, wire byte
count, HELLO refusals, the negative control, two-process end-to-end run, gate order, preflight STOP.
