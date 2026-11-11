# E1-OPT: optimised-runtime split follow-up. Pre-registration DRAFT, revision 3 (2026-11-11)

Status: **DRAFT, NOT FROZEN.** No E1-OPT measurement exists. Nothing in this directory has run on a GPU.
To freeze: review this file and the code it names, then tag the commit `e1opt-prereg-v1`. Until that tag exists,
no document may call E1-OPT "pre-registered". The E0 pre-registration (`deployment_e0/PREREG.md`, tag
`e0-prereg-v1`, validated harness commit `81461b6`) is **not changed** by this file and stays the record of the
E0 result (G-UB2 STOP, best bounded advantage +0.5 pt, 0 of 20 cells). E0 is a valid pre-registered result for
the frozen eager runtime. E1-OPT is a new experiment, not a re-run, motivated by a post hoc CPU analysis showing
that runtime performance may be decisive.

**The runtime and all executable experiment code are frozen by repository tag before any GPU executes E1-OPT
code.** The tag comes first; Stage 0 (inventory: facts only, no model, no E1 GPU kernels) runs after it; the
freeze record is then written from the inventory and committed in a commit that touches only `frozen/`.

Revision 3 (implementation pass of 2026-11-11) incorporates five approved decisions: the arithmetic rule
(§7.2), one server GPU (§6.3), fixed cuts (§3), the compatibility fallback (§4.2) and the target pair with roles
decided by inventory (§6). All four executable components the review notes asked for exist and are tested on CPU
(§15). The unresolved items (§14) are facts that only Stage 0 can supply.

Everything not listed here is inherited unchanged from E0 PREREG: the model (Qwen3-32B bf16 at revision
`9216db57`), the U-shaped cut and ownership (§2), the transport accounting (§2.1), the memory classes and budgets
(§3), the S1 scores and the 200-query sample (§1, §8.1), the frozen workload (§6.1), the utility, grid, statistic,
bootstrap, exclusions and completeness rules (§8), the extension rule (§8.5), and the consequences table (§9.3).

## 0. Why a new experiment and not a re-run of E0

E0 asked whether the split is feasible with a *frozen* runtime and answered no by bound (G-UB2). E0 §11 threat 1
pre-stated that the eager Hugging Face runtime penalises SPLIT, that this is never used to overturn a FAIL, and
that the Stage-3 cell separating "this runtime" from "split itself" was the only way to tell them apart. That cell
was never reached. A CPU reanalysis of the archived E0 profiles (`reanalysis/`, E0 gate code unchanged, f = 1
reproduces `gate_ub2.json` bitwise) shows, with the measured server-segment time scaled by f (server time =
0.90 f × measured) and the archived RTX 3090 edge and competitor profiles:

| Target | f | server speedup | raw server step at k = 3 (measured 66.9 ms) |
|---|---|---|---|
| Δ ≥ +3.0 in some cell, as `e0_r.py` computes it | 0.8297 | 1.21× | 55.5 ms |
| Δ ≥ +3.0 in exact arithmetic (Δ = 3.00; see §7.2) | 0.8410 | 1.19× | 56.3 ms |
| full pass: Δ ≥ +3.0 **and** CI lower bound > 0 (the gate) | 0.6888 | 1.45× | 46.1 ms |

The first draft reported a full pass at f = 0.705 (1.42×). That figure used CI low ≥ 0; the gate uses > 0, and the
CI lower bound of the best cell is exactly 0.0 for f in (0.6888, 0.7054]. It is withdrawn.

The measured server step is 5.7× the H200 bandwidth roofline (11.8 ms). The per-layer cost (about 1.2 ms against a
0.2 ms roofline) is roughly constant across the segment, which is consistent with per-kernel launch and Python
dispatch overhead in the eager loop. This is a hypothesis; no CUDA profiler run exists. CUDA-graph replay removes
exactly that overhead and leaves memory traffic unchanged, so E1-OPT tests the hypothesis directly. The roofline is
an optimistic lower bound on the step, not a level an implementation is known to reach.

E0's rule removed *the frozen implementation*. Whether the split *design* opens a quality tier is untested. E1-OPT
tests it with a runtime frozen before any GPU performance observation, under E0's decision rule, on the lab's
target hardware with every decision profile measured there.

## 1. Research question

> With the server segment executed by a runtime whose configuration is frozen before any GPU performance
> measurement (fixed-shape KV cache, CUDA-graph decode step, pre-chosen attention kernel), does a U-shaped layer
> split of Qwen3-32B (edge: embedding, layers [0, k) and [64−k, 64), norm, LM head, sampling; server: layers
> [k, 64−k) on one server GPU) keep an on-time quality advantage of at least 3.0 points over the best edge-feasible
> model at an emulated 12 or 16 GB edge budget under a no-raw-text policy, on the laboratory's two machines in the
> roles the inventory assigns? The primary regime is 12 GB, D = 60 s.

## 2. Hypothesis

H-E1: at 12 GB and/or 16 GB, no-raw-text, some deadline D in {30, 60, 120, 300, 600} s, SPLIT (fixed cut, better of
thinking / non-thinking) reaches Δ ≥ 3.0 pt over the best eligible EDGE competitor with a 95 % paired bootstrap
lower bound > 0. The pre-hardware expectation from the reanalysis is that, *if* H-E1 holds, it holds first at
12 GB, D = 60 s, non-thinking SPLIT against Qwen3-8B-AWQ thinking (the E0 best cell). That expectation was formed
with RTX 3090 edge profiles and is not a prediction for a different edge device.

## 3. Model, cut, competitors (APPROVED: fixed cuts)

- SPLIT model: Qwen3-32B bf16, revision `9216db57` (E0 §1). No other model pool. No quantisation anywhere in SPLIT.
- Cut: **fixed** at E0's U-max(M): (k, j) = (3, 3) at 12 GB and (5, 5) at 16 GB (`e1_spec.FIXED_CUTS`). It is not
  re-derived on the target edge and not re-optimised after any timing. Every driver asserts the cut
  (`e1_spec.validate_cut`): the experiment path takes no cut argument; a developer/test mode (`--dev-mode`) may run
  another cut, and every row it produces is marked `profile_role = screening`, which the gates reject. The E0
  "SPLIT U-min" cells (k = j = 1) are part of E0's frozen cell grid and are run as such; they are not a new cut.
- A class whose fixed cut does not pass the Stage-3 fit on the target edge is **not evaluable** there (reported,
  not re-cut). If neither primary class is evaluable, E1-OPT ends INCOMPLETE for hardware reasons. The 24 GB
  class stays governed by E0 §8.4(b) and is not the primary target.
- Competitors: the EDGE models of E0 §3 that pass the measured fit at each class **on the target edge**, served with
  the pinned vLLM (0.8.5.post1) as E0 Stage 1 if it installs on the target edge; if it does not, the newest vLLM
  release that installs is pinned in the freeze record (§4.3) before any timing and the difference is reported.
  REMOTE in open cells: Qwen3-32B with vLLM on the server GPU, as in E0.
- **Fresh profiles rule.** Every profile that enters the final decision (split edge segment, every eligible edge
  competitor, memory fit, server segment, link, joint cells) is measured on the physical target machines under
  the freeze record and carries `profile_role = target` provenance (§10.1). Archived E0 profiles (rented RTX 3090
  and H200) carry no E1 provenance and are rejected by the gates; rented or dev profiles are `screening`.

## 4. Runtime: the candidate configuration (APPROVED: fallback rule; torch.compile off)

### 4.1 What changes relative to E0, and only this

Implementation: this directory. It imports E0's weight loading and the transformers 4.51.3 Qwen3 modules unchanged
(`e1_segment.py`), and changes:

1. KV cache addressing: fixed-shape per-layer tensors written in place at `cache_position` (`e1_static_cache.py`).
   Mode `prefix` returns prefix views (bitwise equal to E0; used for equivalence only). Mode `static` returns the
   full tensors and is the measured mode.
2. Decode launch: in static mode the single-token step of each segment runs with static input, position and output
   buffers and is captured once as a CUDA graph (`GraphedStep`). Prefill chunks (2,048 tokens) run uncaptured.
3. Decode attention kernel in static mode: `flash_attn_with_kvcache` over the static cache (`e1_attention.py`),
   because the pinned SDPA path copies the GQA cache 8× per step. Compatibility fallback: SDPA with a static causal
   mask (§4.2).
4. Wire: E0's message set and 72-byte header, protocol version 2, host and device buffers allocated once
   (`e1_protocol.py`). Bytes per step are unchanged: n × 10,240 + 72 each way.

Not permitted (unchanged from E0 §0.3, §2): batching, speculation, activation compression or quantisation, any
dtype other than bf16 on the wire or in the weights, any change to the cut, any change to what crosses the link.

**`torch.compile`: off** (`e1_spec.TORCH_COMPILE = False`, checked by the freeze checker). Decided before any
timing: the mechanism under test is launch and dispatch overhead, which CUDA graphs address on their own;
compilation adds a second mechanism whose kernels differ from the stock modules and would widen the equivalence
question; it adds recompilation and version risk on an aarch64 target; and leaving it optional would be a degree
of freedom after the freeze. A compiled variant, if ever wanted, is a separate pre-registration.

### 4.2 Compatibility fallback (deterministic; never a speed comparison)

Attention. `flash_attn_with_kvcache` is used if it imports on that machine for its compute capability and passes
the equivalence gate on that device. If it is unavailable for a documented architecture or software reason (no
wheel, compute capability below 8.0, import error), or it fails the equivalence gate with a reproducible
compatibility failure (wrong result on this device only, unsupported head dimension or stride, CUDA error,
recorded verbatim), the frozen SDPA static-cache fallback is used on that device. FlashAttention and SDPA are never
benchmarked against each other and the faster one is never chosen.

CUDA graphs. Capture is used if it succeeds and passes equivalence. If capture fails for a reproducible
compatibility reason (driver capture error, recorded verbatim), the static-cache eager decode step is used on that
device. Graphed and eager execution are never benchmarked against each other and selected by speed.

The branch is derived from inventory facts by `e1_freeze.branch_for_device` and written into the freeze record
with its reason **before any timing**. A fallback decided at Stage 1b (a capture or kernel failure) produces a NEW
freeze record (§4.4). The equivalence driver records the branch actually in use and fails if it differs from the
record. A device on a fallback is reported as such in every table.

### 4.3 Freeze point and freeze record

Order: tag `e1opt-prereg-v1` (code and protocol) → Stage 0 inventory (facts) → Stage 1a: `e1_bench.py freeze`
writes `frozen/runtime_freeze.json` from the inventory and the role record; it is committed in a commit touching
only `frozen/`. No GPU executes E1 code before that record exists. Every later command takes `--freeze <path>`,
runs `e1_freeze.check` first and refuses on any mismatch (non-zero exit, nothing written). A record is never
regenerated or edited; a new record is a new file and a new commit.

The record (`e1_freeze.build`; machine-readable JSON) contains: experiment name and record version; the
experiment repository commit and the tag; the E1 directory path; SHA-256 of every file that can affect E1 execution
(`e1_*.py`, `tests/*.py`, `reanalysis/*.py`) and of `PREREG_E1OPT.md`; SHA-256 of the E0 files E1 imports or
reads (`e0_spec.py`, `e0_r.py`, `e0_latency.py`, `e0_instrument.py`, `e0_memory.py`, `e0_f.py`, `e0_q.py`,
`split_runtime.py`, `frozen/workload.json`, `frozen/q2_keys.json`) plus the E0 validated commit, the S1 score and
sample hashes and the two E0 archive hashes; protocol and header version; model repo, revision, dtype, wire dtype
and config; the required transformers version and the competitor vLLM pin; the cuts, U-min and primary classes;
the one-server-GPU rule; the static-cache configuration (L_F = 17,920 tokens, prefill chunk 2,048, layout, measured
and reference modes); the buffer strategy; the no-raw-text ownership definition; the seeds (bootstrap 20261107,
n = 10,000, order, Q2, held-out, synthetic activations); the decision rule (margin 3.0, rounding 10⁻⁶ pt, strict
CI, E0 flag reported, gate code path, commit and SHA-256, segment discount 0.9); the role record; and per machine:
role, GPU (index, name, UUID, memory, compute capability), GPU count, software versions (python, torch, torch
CUDA, transformers, flash-attn, vLLM, numpy, pynvml, NVIDIA driver, OS, kernel) and the runtime branch (attention
path with reason, CUDA-graph mode with reason, `torch_compile: false`). `freeze_sha256` is the SHA-256 of the
canonical record; it is exchanged in HELLO and stamped into every row.

The checker (`e1_bench.py check-freeze`, and the same function inside every driver) recomputes every hash, refuses
files not covered by the record, checks that HEAD equals the frozen commit or differs only under `frozen/`, checks
that the tag points at the frozen commit, checks that this machine is in the record with the expected role, and
compares the live python, torch, torch-CUDA, transformers and flash-attn versions with the frozen ones. Exit
status 2 on mismatch. `--host` overrides the machine identity only in `--dev-mode`.

### 4.4 Bug fixes after the freeze

If Stage 1b or any later stage exposes a defect in E1 code: fix it in a new commit; write a new freeze record;
rerun Stage 1b on every device; **discard every timing recorded under the previous freeze**. No prior timing result
is carried forward, compared, or mentioned as a result. There is no "small fix" exception. A fix may not change the
kernel choice, the graph setting or the compile setting except through §4.2.

## 5. Computation-equivalence tests (Stage 1b; `e1_bench.py equivalence`)

CPU (this host; `tests/test_e1_cpu.py` and `tests/test_e1_equivalence.py`): prefix mode bitwise equal to HF
`Qwen3Model.forward` and to E0's runtime (tiny random Qwen3 and Qwen3-0.6B real weights); greedy tokens equal to HF
`generate` with chunked prefill of 5, 16 and 64; static mode final hidden states within 2⁻⁶ (bf16) of prefix mode
with identical argmax and greedy tokens equal to HF; wire byte accounting; refusals of version, hash, cut and
position errors; the driver's output schema contains no timing field.

GPU, after the freeze, in this order, with the freeze checker first (check I):
- `equivalence reference` (server machine; it holds the whole model): HF unsplit final hidden states and greedy
  tokens for the check sequence; optionally the E0-format Q1 top-K references and Q2 references (`e0_q.py`
  procedures) for the joint stage.
- `equivalence local --role server` (in-host, both segments on the server GPU): **A** prefix mode bitwise equal to
  the HF reference and to E0's runtime; **B** static mode with the frozen kernel and graph branch within 2⁻⁶ of
  prefix mode, identical argmax; **C** greedy tokens equal to HF; **D** wire invariants over a loopback echo stub
  (bytes equal the hidden-state formula, HELLO keys within the control set, the server saw only HELLO/RESET/FWD/
  BYE); **E** ownership invariants (edge holds exactly layers [0,k) ∪ [L−j,L), embedding, norm, LM head; server
  holds exactly [k,L−j) and no embedding or head); **F/G** the attention kernel and CUDA-graph mode actually in use
  equal the freeze record; **H** nothing but the bf16 residual stream and headers crossed.
- `equivalence local --role edge --segments-only` (the capped edge cannot hold the model): head and tail segments,
  prefix vs E0's `run_layers` bitwise and static vs prefix within tolerance, plus D, E, F/G, H.
- `equivalence joint` (edge ↔ server over the lab link): Q1 teacher-forced comparison and Q2 greedy runs with the E1
  runtime in E0's file formats; then `equivalence decide` applies E0's Q1 and Q2 rules with E0's thresholds
  (`e0_q.q1_rule`, `e0_q.q2_rule`: D ≤ max(D_pipeline, D_hw) + 0.2 pp; KL ≤ 2 × max floors; ≥ 27 of 30 Q2 prompts
  agree or diverge only at a margin < 0.5 nats; no degenerate split sequence). Floors as in E0 §4.1: the pipeline
  floor is the in-host E1 static split on the server GPU at the same cut; the hardware floor is Qwen3-8B HF on
  the edge GPU versus on the server GPU (E0 `e0_q.py cmp --config hf`, unchanged).
- **No performance peeking.** The equivalence driver never calls a timer for a result, prints no durations and
  stores none; `assert_no_timing_fields` rejects any key that looks like a latency, throughput, duration, event
  time or rate before a file is written. A timing field in any Stage-1b output makes the stage INVALID.
- One implementation fix is allowed (then §4.4 applies), then one re-run. A second failure is STOP (§11).

## 6. Hardware inventory, feasibility and role assignment (Stage 0; APPROVED)

### 6.1 Inventory (`e1_bench.py inventory collect --peer <other>`; both machines; no model; no timing)

Per machine: hostname; `uname -a`; architecture; CPU; RAM; `nvidia-smi` (GPU names, count, memory, driver,
compute capability, UUIDs); CUDA toolkit; Python; PyTorch version, CUDA build, availability and device
enumeration; vLLM importable and version; flash-attn importable, version and whether `flash_attn_with_kvcache`
exists; pynvml; NICs and rated speeds; the route to the peer; whether `tc` exists; `sudo -n` and `capsh` output for
CAP_NET_ADMIN (to be confirmed manually); unified-memory indication. Output: `env/<hostname>.json` (machine-readable)
and `env/<hostname>.txt` (summary). The script runs no benchmark, no model and no CUDA timing.

Also to be asked of the lab, as facts: how the two machines are connected; whether `tc netem` is permitted;
whether the lab assigns roles or fixes a device; whether the 2 × RTX 3090 machine of the October 2026 manuscript
is the Asus server.

### 6.2 Feasibility (`e1_inventory.feasibility`; arithmetic and availability only)

A machine **can be the server** iff one of its physical GPUs (or its unified memory) holds the server segment at the
largest frozen segment (k = 3): weights + KV(L_F) + E1 buffers + 2 GiB margin ≈ 59 GiB (`e1_inventory.server_need_gib`),
with compute capability ≥ 8.0 and a CUDA PyTorch build. A machine **can be the edge** iff one GPU can emulate the
largest primary budget (16 GiB, E0 §3.1 cap method), compute capability ≥ 8.0, and vLLM imports for the
competitors. The pair is **feasible** iff a role assignment exists with one server and one edge and both machines
have a route to each other.

### 6.3 Role assignment (`e1_bench.py inventory roles`; deterministic; before any measurement)

1. An explicit lab assignment (`--lab-server`, `--lab-edge`) is used if it is feasible; if the lab's assignment
   is infeasible, the pair is INCOMPLETE with the reason.
2. If exactly one assignment is feasible, it is used.
3. If both are feasible, the machine whose qualifying server GPU has the **larger single-GPU memory capacity** is
   the server; on a tie, the smaller hostname in byte order is the server. Capacity is a datasheet fact; no
   bandwidth figure and no measured speed enters the decision. Consequence, stated now: a 128 GB unified-memory
   machine would be the server over an 80 GB GPU unless the lab assigns otherwise; the bound in Stage 3 then
   decides, not the rule.
4. If none is feasible, the pair is INCOMPLETE (a hardware result, §11).

**One server GPU (APPROVED).** The server segment runs on exactly one physical GPU: the lowest CUDA-visible index
on the server machine that holds the segment, unless the lab fixes another device for operational reasons, which
is recorded in the role record before any measurement. No tensor parallelism, no pipeline parallelism across
server GPUs, no sharding; multiple GPUs are never benchmarked to choose. If no single GPU holds the segment, the
server role is infeasible on that machine (`e1_inventory.feasibility`). The rule is in `e1_spec.ONE_SERVER_GPU`,
in the freeze record (`one_server_gpu`, `server_parallelism`), and `serve` refuses a GPU whose UUID is not the
inventoried server GPU. Reason: the runtime implementation is the intended major variable relative to E0; a
multi-GPU server would change the execution architecture as well.

The edge GPU is the lowest-index GPU on the edge machine that can emulate the largest primary budget. **Stage 0
uses no performance timing and no archived RTX 3090 or H200 profile.** The first performance decision is G-UB-E1
after Stage 3 with both segments and all competitors measured on the target machines.

## 7. Decision rule, metric, threshold (E0 §8, unchanged) and arithmetic rule (APPROVED)

### 7.1 Rule
- U(a, D) = mean over the 200 S1 queries × 2 seeds of score × 1[T ≤ D]; Δ = max_mode U_SPLIT − max U_competitor in
  points; paired stratified bootstrap, 10,000 resamples, seed 20261107, both maxima re-selected per replicate.
- Pass: some evaluated cell has Δ ≥ 3.0 pt and CI lower bound **strictly** > 0. Bonferroni label reported, not decisive.
- Exclusions, completeness rules and the 400-query extension: E0 §8.3 to §8.5, unchanged.
- The gate code is E0's `e0_r.py` at commit `81461b6`, unchanged; E1 profiles are written in E0's schema.

### 7.2 Arithmetic rule (APPROVED 2026-11-11; `e1_spec.decide_cell`, `e1_gate.py`)
`e0_r.py` compares `delta_pt >= 3.0` and `ci95[0] > 0` on floating-point values. With binary scores and 200 × 2
queries both quantities are multiples of 0.25 pt, but the float difference of two means can land at
2.9999999999999973 for an exact 3.00 (observed in the reanalysis for f in [0.8297, 0.8410]). E0's own decision was
not near either boundary and is unaffected; E0's code is not modified. For every evaluated E1 cell the gates
compute the raw E0-compatible result and save the raw `delta_pt`; `delta_rounded = round(delta_pt, 6)`; the E1
threshold condition is `delta_rounded >= 3.0`; the CI condition stays `ci95_low > 0`, strictly; **E1 pass = both**.
`e0_raw_pass` (E0's own float flag) is saved alongside for transparency. The sidecar `<gate>.e1decision.json`
lists every cell with both flags and every disagreement. The E1 flag decides. This does not change the scientific
threshold.

## 8. Stages and stopping rules (each stage runs only if the previous one did not STOP)

| Stage | Where | Command(s) | Rule |
|---|---|---|---|
| tag | repository | `git tag e1opt-prereg-v1` after review | nothing below runs before the tag |
| 0 inventory | both lab machines, no model | `inventory collect`, `inventory roles` | **INCOMPLETE** iff no feasible assignment. No performance input |
| 1a freeze | analysis host | `freeze --inventory --roles --tag e1opt-prereg-v1 --out frozen/runtime_freeze.json`; commit (frozen/ only); `check-freeze` on each machine | no GPU has executed E1 code before this record exists |
| 1b equivalence | server GPU, edge GPU, then both | `equivalence reference`, `local --role server`, `local --role edge --segments-only`, `joint`, `decide` | **STOP** iff Q fails after one fix (then §4.4); INVALID on harness defects or any timing field |
| 2 server segment | server GPU | `profile-server-segment --class 12`, `--class 16` | descriptive only: server step vs E0's 66.9 / 61.3 ms and vs the device roofline. **No decision.** |
| 3 edge | edge machine, capped | `fit-split --class 12/16`; `fit-competitor` for every E0 §3 model and class; `profile-edge-segment --class 12/16`; `summarize-fit`; `gate-ub` | **G-UB-E1**: E0 `gate-ub2` on {Stage-3 edge and competitor profiles} × {Stage-2 server profiles}, 0.90 discount on both segments, zero link, provenance-checked: **STOP** iff no cell has Δ ≥ 3.0 (rounded, §7.2). A class whose cut or competitor fails the fit is not evaluable |
| 4 joint | edge + server on the lab link | `serve` (server); `linkcheck`; `run-cell <id>` for every SPLIT cell of E0's grid, `run-remote-cell` for REMOTE cells; `gate-r` | **G-link** first (E0 §9: p50 ≤ 5 ms, ≥ 900 Mbps on the native link); then **G-R** (§7): **retain SPLIT** iff a cell passes; **remove** otherwise; PENDING only by E0 §8.4(a) / §8.5 |

Rented hardware may be used for Stages 1b and 2 as a **screen only**, labelled (`--screening`), with its own
freeze record; nothing measured there enters the decision. Stages 3 and 4 run only on the lab machines.

## 9. Measurements

- Latency (Stages 2 to 4 only): per-request `token_ns` on the edge (E0 §6.2), per-step component timers (edge head
  / tail / LM head; server t_recv / t_compute / t_prep; net = round trip − server), each step labelled `prefill`
  or `decode`, CUDA-synchronised; per request the summed link, server-compute and edge-compute contributions.
  Profiles fitted by E0 `fit_e2e` / `fit_component`.
- Network: G-link over the split protocol itself on the lab link (`linkcheck`), plus `tc netem` cells at the E0
  RTT and bandwidth levels if CAP_NET_ADMIN is available; otherwise E0's application-level shaper, recorded as
  such (E0 §6.3).
- Memory: NVML peak and allocator peak under the cap (E0 §5.1), on the target edge, per request and per fit.
- Quality: Q1/Q2 (§5); S1 scores are reused for utility (E0 §8.1), licensed by Q.
- Requests: the E0 frozen workload per profile (36 usable after 2 warm-ups; failed requests recorded, never
  retried, never erased); 200 queries × 2 seeds for utility.
- Identity: every row carries the edge device, the server identity returned in HELLO (host, GPU, branch, freeze
  hash, active layers) and the provenance object.
- Seeds: bootstrap 20261107; synthetic activations torch seed 0; Q2 prompt draw E0 §4.2.

## 10. Invalid-run rules and provenance

### 10.1 Provenance (`e1_provenance.py`)
Every profile row and every summary JSON carries `provenance`: experiment = E1-OPT, stage, role, machine, GPU
(name, UUID, index), freeze hash, `profile_role` (target | screening), model repo@revision, cut, memory class,
runtime branch, decision-rule version, protocol version. The gates (`gate-ub`, `gate-r`) refuse to run, and write
nothing, if any required input row lacks provenance, has a different freeze hash, a different experiment id, is
marked screening, lacks a device identity, names another model revision, carries a different runtime branch than
the freeze record for its machine, or comes from a machine or GPU other than the inventoried one for its role.
`--allow-screening` produces only a file named `*.screening.json`, never a decision file.

### 10.2 INVALID
INVALID (stop, fix, new freeze record, repeat from Stage 1b): harness exception; freeze-record mismatch on any
machine; prefix mode not bitwise equal to E0 on the GPU; a timing field in any Stage-1b output; NVML or allocator
cap not applied; the server refusing HELLO; any failed request in a decision cell (E0 §5.5); a decision profile
taken from an archive, a rented machine or a dev run. INVALID is never an outcome and never becomes PASS.

## 11. What each result means

- **Retain SPLIT** (G-R pass): the paper's deployment conclusion changes to "an optimised split of Qwen3-32B keeps a
  ≥ 3 pt on-time advantage in the stated cells on the lab's hardware", scoped to emulated budgets and the
  no-raw-text policy, with the leakage caveat; the E0 STOP is reported as the frozen-runtime screen.
- **Remove SPLIT** (STOP at G-UB-E1 or G-R): the paper reports that the split failed its gate with a runtime whose
  server step was X % of the measured E0 step and Y × its device roofline (both recorded), which closes E0 threat 1
  for this design on this hardware.
- **INCOMPLETE at Stage 0**: the pair cannot host the server segment or the edge budget; the paper reports the
  memory and software facts. A hardware result, not a split result.
- **Stage-1b STOP**: the optimised runtime is not the same computation on this hardware; an implementation result.
- **INVALID**: no claim; repeat.

## 12. Post-result tuning: prevented how

The thresholds, cells, competitors, bootstrap and gate code are E0's and are not editable here. The runtime
configuration, kernel, graph and compile settings are fixed by §4 before any GPU execution; the only post-freeze
branch is the §4.2 compatibility rule, which uses availability facts, not speed. Equivalence runs without timers.
Every stage verifies the freeze record and stamps its hash. Decision profiles are fresh, target-marked and from
the inventoried machines only. The joint driver runs only the frozen cells; there is no configuration search. The
reanalysis that motivated E1-OPT (§0) was computed from E0's archives with E0's gate code and is reproducible from
the two archive hashes (`tests/test_reanalysis.py`). No E1-OPT number is compared with E0's except as a stated
change of runtime and hardware.

## 13. Variables: E0 versus E1-OPT

| Variable | E0 | E1-OPT | Changed? | Why necessary | Could affect quality? | Could affect latency? | Frozen before measurement? |
|---|---|---|---|---|---|---|---|
| Model, revision, dtype | Qwen3-32B bf16 `9216db57` | same | no | | | | E0 |
| Cut (k, j) per class | U-max from Stage-1 fit: (3,3), (5,5), (9,9) blocked | fixed at E0's values; asserted in code | no (rule: "fixed") | avoids a hidden server-segment change | no | no | yes (§3, approved) |
| Ownership, wire content, bytes per step | E0 §2 | same; protocol version 2, same header and payload; HELLO keys whitelisted | no | | no | no | yes |
| Runtime: KV addressing | growing `DynamicCache` | fixed-shape static cache | **yes** | the variable under test | no (Q-tested; prefix mode bitwise) | yes | yes (§4) |
| Runtime: decode launch | eager per-kernel | CUDA graph per segment step (eager fallback by §4.2) | **yes** | the variable under test | no (same kernels) | yes | yes (§4) |
| Runtime: decode attention kernel | SDPA over prefix views | `flash_attn_with_kvcache`, SDPA static fallback by §4.2 | **yes** | avoids the 8× GQA copy of the static SDPA path | small numeric differences (Q-tested, thresholds E0's) | yes | yes (§4.2, §4.3) |
| `torch.compile` | not used | not used | no | | | | yes (approved) |
| Buffers | per-step allocation | allocated once; pinned host buffers | **yes** | part of the runtime change | no | yes (small) | yes |
| Prefill chunk | 2,048 | 2,048 | no | | | | E0 |
| Transformers / torch versions | 4.51.3 / 2.6.0+cu124 | 4.51.3; torch build per target architecture | torch build may change | aarch64 target has no cu124 x86 wheel | no (bitwise prefix check on each GPU) | yes | yes (§4.3) |
| Competitor engine | vLLM 0.8.5.post1 | same if installable; else newest installable, pinned | possibly | compatibility | no (same models and scores) | yes | yes (§3, §4.3) |
| Edge hardware | rented RTX 3090, capped | lab edge machine, capped | **yes** | advisor specified final hardware | no | yes (both SPLIT and competitors) | roles fixed at Stage 0 |
| Server hardware | rented H200 | lab server machine, one GPU | **yes** | advisor specified final hardware | no | yes | roles fixed at Stage 0 (approved) |
| Link | loopback echo (profiles); never measured end to end | the lab link, measured; `tc netem` cells | **yes** | E0 never reached Stage 3 | no | yes | gate E0 §9 |
| Memory classes and cap method | 12 / 16 / 24 GB, E0 §3.1 | same | no | | | | E0 |
| Workload, queries, seeds | E0 §6.1, 200 × 2 | same | no | | | | E0 |
| Utility, deadlines, Δ, bootstrap, margin | E0 §8 | same | no | | | | E0 |
| Pass predicate | `delta >= 3.0 and ci_lo > 0` (float) | same, delta rounded to 10⁻⁶ pt (§7.2), E0 flag also reported | **arithmetic only** | float boundary defect found in the reanalysis | no | no | yes (approved) |
| Gate code | `e0_r.py` @ 81461b6 | same, unchanged, behind a provenance check | no | | | | E0 |
| Stage order | F → UB1 → UB2 → link → R | tag → inventory → freeze → equivalence → server → edge + bound → link → R | **yes** | fresh profiles; no archived-profile STOP | no | no | yes |

Hidden variables flagged and closed: the cut rule (fixed, asserted), the competitor engine version on a new
architecture (pinned at freeze), the torch build (bitwise prefix check), the attention kernel and graph mode
(compatibility rule only, recorded branch checked against use), the machine identity (`--host` only in dev mode).

## 14. Unresolved items: facts that only Stage 0 can supply

1. The Asus server's GPU model, count and memory; its OS, driver, CUDA and PyTorch; whether vLLM 0.8.5.post1 and
   flash-attn install on each machine (§6.1).
2. The physical link between the two machines, its rated speed and route; whether `tc netem` is permitted.
3. Whether the lab assigns roles or fixes a server device (§6.3 rule 1); otherwise the capacity rule applies.
4. Whether the 2 × RTX 3090 machine of the October 2026 manuscript is the Asus server (if so, it cannot be the
   server under §6.2: 24 GB < 59 GiB).
5. A technical constraint to check at inventory: E0's `e0_f.py edge-fit` (reused unchanged for the competitor fits)
   samples NVML device index 0. `fit-competitor` refuses to run on a multi-GPU edge whose inventoried edge GPU is
   not index 0. If the edge machine has several GPUs, the edge GPU must be index 0 at the system level.

## 15. Executable components and the hash set

| Component | File(s) | CPU tests |
|---|---|---|
| Spec, approved rules, cut validation, decision rows | `e1_spec.py` | `tests/test_e1_rules.py` |
| Runtime (static cache, segments, graphs, attention branch) | `e1_static_cache.py`, `e1_segment.py`, `e1_attention.py`, `e1_memory.py` | `tests/test_e1_cpu.py` |
| Wire (hardened HELLO, freeze hash, fail-closed) | `e1_protocol.py` | `tests/test_e1_cpu.py`, `tests/test_e1_joint.py` |
| Freeze writer / checker | `e1_freeze.py` | `tests/test_e1_freeze.py` |
| Stage-0 inventory, feasibility, roles | `e1_inventory.py` | `tests/test_e1_rules.py` |
| Provenance and gate rejections | `e1_provenance.py`, `e1_gate.py` | `tests/test_e1_rules.py`, `tests/test_e1_profile_gate.py` |
| Equivalence driver (no timing fields) | `e1_equivalence.py` | `tests/test_e1_equivalence.py` |
| Target profilers (fit, competitors, edge segment, server segment, summary) | `e1_profile.py` | `tests/test_e1_profile_gate.py` |
| Joint edge/server cell driver, linkcheck | `e1_joint.py` | `tests/test_e1_joint.py` |
| Command hub | `e1_bench.py` | all of the above (subprocess) |
| Reanalysis (motivation; not an experiment input) | `reanalysis/*.py` | `tests/test_reanalysis.py` |

The freeze hash set is every `e1_*.py`, every `tests/*.py`, every `reanalysis/*.py` and this file; the E0 files E1
imports are hashed separately in the record. `tests/e1_testlib.py` is included (it matches `tests/*.py`). Any new
helper script that could affect a scientific output must be placed in this directory under one of these patterns
or it will fail the "file not covered by the freeze" check at the next run only if it is hashed; a script outside
the patterns cannot be imported by the drivers, which import siblings by name only.
