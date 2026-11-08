# E0 physical split viability: pre-registration

Frozen **2026-11-08**, before any E0 hardware access. **No E0 measurement has been made.** Nothing in this
directory has run on a GPU.

The one question:

> Can Qwen3-32B be physically layer-split across an edge and a server while preserving model quality and
> satisfying the E0 memory and latency feasibility requirements?

Design source (not changed here, only made exact): `~/dev/cloud_slm-router/deployment-pivot/E0-SPLIT-VIABILITY.md`
(frozen 2026-11-07), `HARDWARE-PLAN.md`, `WEEK-EXECUTION-PLAN.md`, `PROBLEM-FORMULATION.md` §3.2–§3.4.
This file pins every choice E0 leaves open and lists each deviation as a pre-data amendment (§10), with the
direction of its bias. **Where this file and E0 differ, this file governs the physical run.** Every constant is
in `e0_spec.py`. Commands, hardware stages and costs are in `RUNBOOK.md`. Local results are in
`LOCAL_VALIDATION.md`.

## 0. Discipline (binding)

1. **Split is not assumed useful.** E0 decides whether SPLIT enters F3's action set at all.
2. **No privacy claim.** "Raw text stays on the edge" is an *admissibility* condition: k ≥ 1 and j ≥ 1, so
   token ids, text and logits never leave the device. Activations at shallow cuts are invertible (Cunningham:
   about 35 % token recovery at 8 layers). The word "privacy-preserving" is never used for SPLIT.
3. **Failure removes SPLIT.** It is not rescued by activation compression, quantised activations, extra
   split depths, new memory classes, other models or engine changes after data are seen. One
   implementation fix is allowed for Q (§4.5), and only for Q.
4. **No sweeps, no post-hoc criteria.** All thresholds, cells, seeds, samples and procedures are fixed below.
   The only pre-declared extension is E0 §4.4's 200 → 400 query extension (§8.5).
5. **Implementation and interpretation are separate.** Harness checks (self-test, parity, link check)
   validate the measurement and decide nothing about SPLIT. Scientific verdicts come only from §9.
6. **S1 is not re-run.** Q3 quality is the committed S1 result (§1).

## 1. Verified inputs (checked 2026-11-08, `e0_selftest.py` groups `spec`, `s1` and `template`)

| Item | Check | Outcome |
|---|---|---|
| S1 verdict | `results/s1_2026-11-08/s1_summary.json`, recomputed from per-query data | **CONTINUE**: rule16 false, rule12 false; 4,000 / 4,000 scored, 0 missing, 0 duplicate keys |
| S1 per-query file | `q3_scores.jsonl` sha256 `e00736dc…93ce` (commit `d849ed6`) | pinned in `e0_spec.S1_Q3_SHA256` |
| S1 gaps | recomputed independently from `q3_scores.jsonl` | identical to `S1_SUMMARY.md` (e.g. mmlupro thinking at 16 GB +5.8; gpqa non-thinking at 16 GB +7.5) |
| S1 serving | `servers/*.json` | 1× H200 (143,771 MiB), vLLM 0.8.5.post1, torch 2.6.0+cu124, transformers 4.51.3 |
| S1 scoring anomaly | traceback in `s1_driver.log` | frame `<string>` line 48 in `main`, started by `threading.Thread`: a **model-generated LiveCodeBench program** (run through `exec` in `testing_util.py:193`) raised on a test input. `run_s1.score_run` catches per-item exceptions into `eval_error`, and 0/4,000 records carry one. **No score is affected.** |
| Harness parity | `s1_runs/parity/parity.md` | PASS on all 4 benchmarks |
| Chat template | pinned tokenizer (`tokenizer.json` sha256 `aeb13307…dae4`) applied to the 200 prompts × 2 modes | reproduces **400 / 400** S1 `prompt_tokens` exactly |
| Qwen3-32B config | `config.json` @ `9216db57` | 64 layers, hidden 5120, 64 query heads, 8 KV heads × 128, intermediate 25,600, vocab 151,936, untied embeddings, bf16 |
| Parameter count | exact count from the config vs `model.safetensors.index.json` total_size | 32,762,123,264 params = 65,524,246,528 bytes: identical |

**Negative findings carried forward (stated now, before any E0 data):**
- At **16 GB**, S1's gap is ≥ 3.0 pt in 5 of 8 benchmark × mode cells (point estimates; the S1 rule).
  **Every 16 GB CI includes 0.** At **12 GB**, 5 of 8 CIs exclude 0, and a sixth (LiveCodeBench non-thinking)
  touches 0.
- At **24 GB**, Qwen3-32B-AWQ beats Qwen3-32B bf16 on LiveMathBench thinking (−6.2 pt, CI [−12.5, −1.2]).
- The informational prior (§12) finds the R regime, if it exists, only at **12 GB**.

## 2. Split architecture (frozen)

```
EDGE (raw text, ids, logits never leave)                 SERVER
tokens ─► embed_tokens ─► layers [0, k) ──h_k (bf16)──►  layers [k, 64−j)
                                                          │
next token ◄─ argmax ◄─ lm_head ◄─ norm ◄─ layers [64−j, 64) ◄──h_{64−j} (bf16)──┘
```

| Component | Placement | Notes |
|---|---|---|
| Tokeniser, chat template, sampling (argmax) | edge | the server never sees ids or text |
| `embed_tokens` (777.9 M params, 1.556 GB) | edge | |
| Layers `[0, k)` | edge | own KV cache, preallocated for L_F |
| Layers `[k, 64−j)` | server | own KV cache, preallocated for L_F |
| Layers `[64−j, 64)` | edge | own KV cache |
| Final RMSNorm + `lm_head` (777.9 M params, 1.556 GB; untied) | edge | the edge therefore holds 3.11 GB that does not shrink with k, j |
| Rotary embedding, causal mask | both | recomputed from positions; each side checks the other's position |

- **Activation movement.** Per forward step the edge sends the residual stream after layer k−1
  (`[n, 5120]`, raw bf16, `n·10,240` bytes plus a 64-byte header). It receives the residual stream after layer
  63−j (same size). That is 10 KB each way per decoded token, and n × 10 KB each way per prefill chunk.
  TCP with `TCP_NODELAY`, pinned host buffers, non-blocking copies. There is one round trip per forward step,
  and no pipelining. Socket byte counters are recorded and cross-checked against `n·5120·2 + header`.
- **dtype.** The wire is bf16 only. The runtime raises on any other dtype (E0 §7 forbids an fp16 cast).
- **Prefill.** Chunks of 2,048 tokens. Each chunk crosses edge → server → edge before the next one starts.
- **KV cache.** Each side holds K/V for its own layers only (4,096 B per token per layer, bf16),
  preallocated for L_F (§3.1). The server rejects a forward step whose start position differs from its cache
  length. This catches KV and position bugs.
- **Identity.** HELLO carries the model id + revision, a config hash, torch / transformers / CUDA versions and
  (k, j). The server refuses any mismatch.
- **Legal splits.** k ≥ 1, j ≥ 1, symmetric k = j (E0 §3.2), and edge peak ≤ the class budget.
  Frozen configurations: **U-min = (1, 1)**; **U-max(M)** = the largest symmetric (k, k) that passes F at class M
  (§5.3).
- **Runtime.** A two-process Hugging Face runtime (`split_runtime.py`). It reuses the transformers 4.51.3
  Qwen3 modules unchanged (decoder layers, rotary embedding, `_update_causal_mask`), with SDPA attention.
  The single optimisation pass E0 §3.3 allows (pinned buffers + non-blocking copies) is built in. No other
  optimisation is permitted before R: no CUDA graphs, batching, speculation, compression or compilation.

## 3. Edge memory classes

### 3.1 Budget and context

| Class | Budget (per process, CUDA context included) | Emulated on |
|---|---|---|
| 12 GB | 12 × 1024 − 512 = **11,776 MiB** | one 24 GB NVIDIA GPU, capped |
| 16 GB | **15,872 MiB** | same GPU |
| 24 GB | **24,064 MiB** | same GPU |

- 512 MiB per class is unusable on a real device (driver, display, ECC).
- **L_F = 17,920 tokens** of KV for every action, SPLIT and EDGE alike. That is S1's longest prompt
  (1,030 tokens) + 16,384 new tokens = 17,414, rounded up to a multiple of 512.

### 3.2 Predicted feasibility (analytic, `e0_memory.py`, `frozen/memory_model.md`)

The edge total is:
- weights: embedding + LM head + 2k layers × 975.2 MB;
- KV: 2k × 70 MiB;
- the largest transient: the GQA-expanded K/V that transformers 4.51.3's SDPA path materialises per layer
  (2 × 64 heads × context × 128 × 2 B, about 560 MiB at L_F), plus MLP intermediates of the largest prefill
  chunk;
- an assumed 400 MiB CUDA context and 256 MiB of allocator slack.

| (k, j) | edge layers | predicted MiB | 12 GB | 16 GB | 24 GB | server layers / MiB |
|---|---|---|---|---|---|---|
| (1, 1) = U-min | 2 | 6,377 | fits | fits | fits | 62 / 63,391 |
| (3, 3) | 6 | 10,377 | **U-max(12)** | fits | fits | 58 / 59,391 |
| (4, 4) | 8 | 12,377 | no | fits | fits | 56 |
| (5, 5) | 10 | 14,377 | no | **U-max(16)** | fits | 54 / 55,391 |
| (6, 6) | 12 | 16,377 | no | no | fits | 52 |
| (9, 9) | 18 | 22,377 | no | no | **U-max(24)** | 46 / 47,391 |
| (10, 10) | 20 | 24,377 | no | no | no (by 313 MiB) | 44 |

- The prediction equals the E0 desk check: (3,3), (5,5), (9,9).
- (10,10) misses the 24 GB budget by only 313 MiB, so the measured search (§5.3) must try it.
- Asymmetric (4,3) would fit 12 GB, but E0 freezes symmetric splits.

**EDGE competitors** (vLLM; weights + KV at L_F + an assumed 1.5 GiB of engine overhead):

| Model | Predicted MiB | Predicted class fit |
|---|---|---|
| Qwen3-8B-AWQ | 9,873 | 12, 16, 24 |
| Qwen3-14B-AWQ | 13,873 | 16, 24 |
| Qwen3-8B bf16 | 19,696 | 24 |
| Qwen3-32B-AWQ | 22,182 with fp8 KV | 24 only (E0 §1) |

**Measured fit replaces the prediction** (§5.4). The measured eligibility JSON is in `summarize_s1.py --fit-json`
format.

## 4. Q: quality equivalence (gate before any latency claim)

### 4.1 Q1: teacher-forced agreement

- **Sample:** the 200 S1 queries (`sample_200.jsonl`, sha `2c4df493…6955`), thinking template.
- **Continuations:** greedy (argmax), ≤ 4,096 new tokens, stopping at `<|im_end|>` / `<|endoftext|>`. They are
  generated once on the server by single-host HF Qwen3-32B (batched, left-padded) and stored as token ids.
  They are used **only** as fixed teacher-forcing text, so their numerics do not enter any metric (A5).
- **Positions:** every position that predicts a continuation token: len(continuation) positions per query,
  pooled over all queries.

Configurations, all HF bf16, SDPA, TF32 off, `use_deterministic_algorithms(True, warn_only)`:

| Label | What | Host(s) |
|---|---|---|
| (b) reference | single-host HF Qwen3-32B: `model.model(ids)` + the shared chunked LM head | server |
| (a) SPLIT | U-min and U-max(16), edge ↔ server over the real link | edge + server |
| (c) in-host floor | the split runtime with both processes on **one** server GPU over loopback, cut at the same layers as each (a) | server (A4) |
| hardware floor | Qwen3-8B single-host HF on the edge vs on the server (same continuations) | edge, server |

**Metrics** (per configuration x against its reference r):
- **D** = 100 × the share of positions where argmax_x ≠ argmax_r (full-vocabulary argmax on both sides).
- **KL** = the mean over positions of the coarse-grained KL(r ‖ x) on the partition {each of r's top-32 tokens,
  "rest"}. It is computed from log-softmax in fp32 of the bf16 logits.
  - It is a lower bound on the full-vocabulary KL and is computed identically for SPLIT and both floors (A6).
  - A full-vocabulary dump would be about 270 GB.

### 4.2 Q2: free-running greedy

- **Prompts:** 30 prompts (`frozen/q2_keys.json`: 9 MMLU-Pro, 9 GPQA, 6 LiveMathBench, 6 LiveCodeBench; a seeded
  permutation per benchmark), thinking template.
- **Decoding:** greedy, up to 1,024 new tokens, stopping at EOS, batch 1.
- **Reference:** single-host HF `generate(do_sample=False)` on the server, storing each step's top-1/top-2
  log-probability margin.
- **SPLIT:** the split runtime with the same prompts, for U-min and U-max(16).
- **First divergence:** the first index where the tokens differ. A sequence that ends (EOS) where the other
  continues diverges at that index.
- **Degenerate:** a token id ≥ 151,669 (the tokeniser length), any NaN margin, or a last 200 tokens that are
  periodic with a period ≤ 50.

### 4.3 Q pass rule (E0 §8, unchanged thresholds)

For **both** U-min and U-max(16):
- **Q1:** D_split ≤ max(D_pipeline(cfg), D_hw) + 0.2 pp **and** KL_split ≤ 2 × max(KL_pipeline(cfg), KL_hw).
  - If both floors are exactly 0, the KL threshold is 1e-6.
  - D_pipeline comes from the in-host floor cut at the same layers as that configuration.
- **Q2:** ≥ 27 / 30 prompts either do not diverge or first diverge where the reference margin is < 0.5 nats,
  **and** no SPLIT sequence is degenerate unless the reference for the same prompt also is.

**Q = Q1 ∧ Q2 for both configurations.**

### 4.4 Seeds and determinism

- Greedy decoding everywhere in Q, so there is no sampling seed.
- Q3 quality is S1's (seeds 0 and 1; vendor sampling; per mode).
- Bit-reproducibility across GPU architectures is not claimed. The floors measure exactly that noise.

### 4.5 If Q fails

- One implementation fix is allowed, then Q1 + Q2 are re-run once (`decide --fix-attempt 1`).
- If Q still fails, **SPLIT is removed.**
- A harness fix found by the self-test before Stage 3 does not consume this fix.

## 5. F: memory feasibility

### 5.1 Instrumentation (`e0_instrument.py`, `e0_f.py`)

| Quantity | How |
|---|---|
| Idle baseline | the median NVML device `used` over 10 samples **before** the process initialises CUDA. The GPU must be exclusive; container PID namespaces make per-process NVML unreliable |
| Non-torch overhead | after CUDA + cuBLAS initialisation: NVML used − baseline − `torch.cuda.memory_reserved()` |
| Cap | `torch.cuda.set_per_process_memory_fraction((budget − non-torch) / total)`, so an over-budget allocation raises `OutOfMemoryError` instead of succeeding |
| Static weights | exact: the bytes of the parameters materialised on the device (`Slice.param_bytes`); everything else stays on `meta` |
| KV cache | exact: the preallocated bytes (`SliceKVCache.nbytes`) |
| Activation buffers | allocated peak in each phase − weights − KV (prefill and decode separately) |
| Peak | (i) the **device view**: max NVML used − baseline, sampled every 20 ms; (ii) the **allocator view**: non-torch overhead + `max_memory_reserved` |
| Allocator pressure | `num_alloc_retries` and `num_ooms` from `torch.cuda.memory_stats` |
| OOM behaviour | the exception is caught, and the phase (cap / load / prefill / decode), the allocator statistics and the tail of `memory_summary` are recorded. **The attempt fails.** There is no retry with other settings |

### 5.2 One F attempt (`e0_f.py split-fit`)

1. Fresh process.
2. Cap at the class budget.
3. Load the edge slice U(k, k) and preallocate KV for L_F.
4. Run the depth prompt (17,664 tokens, 9 chunks of ≤ 2,048) and then 256 greedy decode tokens. The final
   context is exactly L_F. The middle is an in-process echo segment.
5. **PASS** iff there is no exception **and** device view ≤ budget **and** allocator view ≤ budget.

### 5.3 U-max(M) search (frozen; `umax_search`)

1. Start at the predicted k (3, 5 and 9).
2. If it passes, step up by 1 while passing.
3. If it fails, step down until one passes.
4. U-max(M) = the largest passing k. Each attempt is a fresh process. There is no other tuning.
5. If (1,1) fails at class M, the class has no SPLIT configuration.

### 5.4 EDGE competitor fit (`e0_f.py edge-fit`), all 4 models × 3 classes

1. Launch vLLM 0.8.5.post1 with `--gpu-memory-utilization = budget/total`, `--max-model-len 17920`, the S1
   dtype and revision, and `--kv-cache-dtype fp8` only for (Qwen3-32B-AWQ, 24).
2. Send one depth request (17,664 + 256 forced tokens).
3. **PASS** iff the server starts, the request completes and the NVML device peak ≤ budget.
4. If the peak exceeds the budget, one retry is made with the utilisation reduced by the measured excess + 0.01.
5. The S5 latency profile runs on the same server when the fit passes.

### 5.5 F rule (E0 §8)

- **Per class:** U-max(M) exists, **and** zero OOM or transport errors over every E0-L request that uses that
  class.
- A class that fails F is dropped from R. That alone is not an E0 failure.
- **F fails iff no class passes.**

## 6. E0-L: latency

### 6.1 Fixed workload (`frozen/workload.json`, ids sha256 `615f0392…3d47`)

- 5 prompts with n_in = 256, 5 with n_in = 2,048, and (core block and single-machine profiles only) 2 with
  n_in = 17,664. The 17,664-token prompts are **depth probes** (A3).
- The token sequences are concatenated thinking-template S1 prompts, truncated to exactly n_in.
- Each request generates exactly 256 tokens: greedy, EOS ignored (`min_tokens = max_tokens = 256`,
  `ignore_eos` for vLLM). Batch 1.
- Each cell runs 2 warm-ups, then every (prompt, rep) for 3 reps in a cell-seeded random order. Block order is
  fixed (core, load, rtt, heldout, bw, cut); the cell order within a block is seeded (`e0_latency.py plan`).

### 6.2 Per request (recorded, `e0_latency.py`)

- Token timestamps (ns from the start of the request), giving TTFT, TPOT p50/p95 and total latency.
- SPLIT, per forward step:
  - edge compute (head layers, tail layers + norm, LM head + argmax; CUDA-synchronised);
  - edge serialisation (d2h + h2d);
  - application shaping;
  - the edge-observed round trip;
  - server t_recv / t_compute / t_prep (returned in the reply header);
  - network = round trip − server time;
  - bytes each way;
  - the server cache length.
- NVML edge power at 100 ms (J/request, integrated), the SM clock range, the device-memory peak and allocator
  statistics.
- Server power is reported, not optimised.
- REMOTE / EDGE: the vLLM streaming `/v1/completions` with prompt token ids. Per-chunk token counts come from
  `continuous_usage_stats`. `usage.prompt_tokens` must equal n_in.

### 6.3 Link

- **Native link.** It is first characterised over the split protocol itself (`linkcheck`: 1,000 per-token
  round trips with 10 KB each way, plus 20 bulk round trips of 2,048-token chunks).
  - HARDWARE-PLAN go/no-go #3: RTT p50 < 1 ms and ≥ 1 Gbps.
  - **If the native RTT p50 is > 5 ms, Stage 3 is not run on that pairing.** The 5 ms cells would be physically
    unrealisable. The pods are re-provisioned in one datacenter instead.
  - Between 1 and 5 ms, "native" is the measured value, recorded as such.
- **RTT levels {5, 20, 50} ms and 100 Mbps:**
  - with `tc netem` + `ifb` on the edge egress and ingress (delay RTT/2 each way) where CAP_NET_ADMIN exists;
    the qdisc state is verified before and logged before and after every cell;
  - **otherwise** with the application shaper (A11). It delays each SPLIT round trip by RTT + both payloads'
    serialisation time at B, inside the decode loop, so GPU idle effects are real. REMOTE requests get RTT
    added to every timestamp (computed, labelled `app_adjust_ns`).
  - The shaping mode is recorded per request.

### 6.4 Server state κ

- A load generator keeps exactly {8, 32} chat requests in flight against the server's vLLM Qwen3-32B:
  thinking, max_tokens 512, vendor sampling, S1 prompts. It serves `GET /status`, and `run-cell` refuses to start
  unless concurrency matches, uptime ≥ 60 s and there are 0 errors.
- The SPLIT server segment is **co-resident on the same H200** as the vLLM REMOTE instance (A10), so SPLIT's
  server layers feel the same contention REMOTE does.
- The realised slowdown is measured per path, not assumed equal.

### 6.5 Cells (E0 §6, 22 decision cells; `e0_spec.cells()`)

| Block | Paths | Factors | n |
|---|---|---|---|
| core | SPLIT U-min, SPLIT U-max(16), REMOTE, EDGE-best(12/16/24) | native, 1 Gbps, κ = 0, + depth probes | 6 |
| rtt | SPLIT U-max(16), REMOTE | RTT 5 / 20 / 50 ms | 6 |
| cut | SPLIT U-min | RTT 20 ms | 1 |
| bw | SPLIT U-max(16) | 100 Mbps | 1 |
| load | SPLIT U-max(16), REMOTE | κ = 8 / 32 | 4 |
| heldout | drawn once with seed `deployment-e0:heldout:20261107` | U-max(12) 50 ms / 100 Mbps / κ0; U-max(24) 5 ms / κ32; REMOTE 50 ms / κ8; U-max(24) 20 ms / κ8 | 4 |
| info (not a decision cell) | REMOTE-HF: unsplit HF on the server | engine gap only (A8) | 1 |

- The EDGE core cells are measured as single-machine S5 profiles for **every** eligible (model, class) pair (A16).
- E0 §6's server-local REMOTE profile (S4) is subsumed by the REMOTE cells measured from the edge client.

## 7. Profiles and the latency model

### 7.1 Per-cell profile (`fit_e2e`)

- T(n_in, N) = TTFT(n_in) + Σ_{i=1..N−1} (a + b·(n_in + i)).
- TTFT is piecewise-linear through the measured n_in means.
- a and b are an OLS fit over every decode step's (context, step time).
- Cells without depth probes borrow b from the core cell of the same path (A15).

### 7.2 Single-machine bound profiles

- Edge-segment compute: Stage 1, the split runtime against the echo stub at each class cap.
- Server-segment compute: Stage 2, a synthetic-activation client.

### 7.3 Additive SPLIT model (E0 §6; `AdditiveModel`), fitted on the non-held-out SPLIT cells' component timers

| Component | Model |
|---|---|
| Edge per step | e0 + e1·L_e + e2·L_e·ctx (L_e = k + j) |
| Server per step | r(κ)·(s0 + s1·L_s + s2·L_s·ctx), with r(0) = 1 and r(κ) = measured / predicted on the U-max(16) load cells |
| Serialisation | a constant |
| Network | l0 + λ·RTT + bytes·8/B |
| Prefill | the same structure per chunk, linear in L·n |

**Validation (frozen ±10 %):** for each held-out cell, the predicted TPOT p50 must be within ±10 % of the measured
one.
- If all 4 pass, R uses the model for every (M, state, RTT).
- Otherwise R is evaluated **only at measured cells**: (16, good, native) = core-02; (16, good, 20) = rtt-02;
  (16, contended32, native) = load-02. All 12 and 24 GB cells are then not evaluable.

## 8. R: a useful regime

### 8.1 Utility

- U(a, D) = the mean over the 200 S1 queries and both seeds of score(q, s) × 1[T_a(q, s) ≤ D].
- T uses that action's own per-query S1 prompt length and completion tokens (model and mode). SPLIT uses
  Qwen3-32B's S1 records, which Q licenses.
- The queries are pooled with the sample's natural weights (60/60/40/40).

### 8.2 Grid

- M ∈ {12, 16, 24} × privacy ∈ {no-raw-text, open} × state ∈ {good, contended32} × RTT ∈ {native, 20 ms} ×
  D ∈ {30, 60, 120, 300, 600} s: **120 cells**.
- SPLIT uses U-max(M) and the better of its two modes.
- Competitors are taken over (model, mode) pairs:
  - **no-raw-text:** every EDGE model that passed the measured fit at M;
  - **open:** those plus REMOTE Qwen3-32B.
- The remote-down state is excluded by construction (SPLIT needs the server).

### 8.3 Statistic and pass rule (E0 §8, unchanged)

- Δ = max_mode U_SPLIT − max U_competitor, in points.
- The 95 % CI comes from a paired bootstrap with 10,000 resamples (seed 20261107), stratified by benchmark,
  with **both maxima re-selected inside every replicate**.
- **R passes iff at least one evaluated cell has Δ ≥ 3.0 pt and CI lower bound > 0.**
- Every cell is reported.
- A Bonferroni lower bound across the evaluated cells is reported as a label, `pass_robust_to_multiplicity`.
  **It is not a decision input.**

**Exclusions:**
- cells where every action has U < 0.05;
- no-raw-text cells with **no** EDGE competitor that fits (reported as a capacity statement, never a regime);
- classes that failed F.

### 8.4 Pre-declared competitor completeness rules (E0 §3.1, §1)

- **(a) 24 GB passes.** If any 24 GB cell passes, the 24 GB pass is held **pending**. Qwen3-30B-A3B (an official
  4-bit checkpoint, pinned at that time) must first be added as an EDGE competitor (S1-style quality on the same
  200 queries and seeds, edge fit + profile), and the cells re-evaluated.
- **(b) 32B-AWQ fails to fit.** If Qwen3-32B-AWQ fails the 24 GB fit for an *engine* reason (for example fp8 KV
  unsupported by the pinned engine on that GPU) rather than for memory, no 24 GB cell may pass. Both rules can only
  remove a SPLIT pass.

### 8.5 Sample extension

- E0 §4.4 pre-declares one extension. If R's best cell has Δ ≥ 3.0 pt but its CI straddles 0, the sample is
  extended once to 400, using S1's permutation prefix and the same seeds and settings, and R is re-evaluated.
- This needs new S1-style generations for the extra 200 queries: a separate, pre-declared server job. It is not
  run otherwise.

## 9. E0 PASS / FAIL and falsification gates (frozen before any result)

**E0 PASS = Q ∧ F ∧ R.**

Gates in execution order. Each STOP writes `DECISION.md` and ends hardware spending.

| Gate | When | Rule | If it fails |
|---|---|---|---|
| G1 (done) | S1 | E0 §8 early stop | not triggered: **CONTINUE** |
| G-F | after Stage 1 | F: some class has a U-max (§5.5) | **STOP: F fails, SPLIT removed** |
| G-UB1 | after Stage 1 | R must be possible when SPLIT time is lower-bounded by the measured edge-segment compute + the server's **datasheet** roofline (100 % of peak HBM bandwidth and FLOPs) + zero link (§9.2) | **STOP: R fails by bound** |
| G-UB2 | after Stage 2 | the same with the **measured** server-segment compute in place of the roofline | **STOP: R fails by bound** |
| G-link | start of Stage 3 | native RTT p50 ≤ 5 ms over the split protocol | do not run Stage 3 on this pairing (blocked, not failed) |
| G-Q | Stage 3, before any latency cell | §4.3, one fix | **STOP: Q fails, SPLIT removed** |
| G-R | after Stage 3 | §8.3 (+ §8.4) | **SPLIT removed for lack of a regime** |

### 9.1 Consequences (E0 §8 table, unchanged)

| Outcome | Consequence |
|---|---|
| PASS, no-raw-text cells only | SPLIT enters F3's action set as the no-raw-text quality tier. The claim is "a larger model under a raw-text-stays-local constraint". Measured leakage must be reported; never "privacy-preserving". If all passing cells are 12 GB: "restricted to memory-starved edges" |
| PASS including open cells | unexpected. Re-check against the component floor and the netem logs before believing it |
| Q or F fails | SPLIT removed. F3 = EDGE/REMOTE |
| Q ∧ F, R fails | SPLIT removed for lack of a regime. Report "weight-quantised edge models dominate layer split" as a negative result |

### 9.2 Why the upper-bound gates can never cause a false STOP

- U_SPLIT is monotone non-increasing in T. A per-query lower bound on T is therefore a per-query, per-replicate
  upper bound on U_SPLIT, and an upper bound on Δ and on its CI's lower bound. `e0_selftest` checks this
  monotonicity.
- In no-raw-text cells the competitor (EDGE) is unaffected by the server and the link, so:
  - contention can only slow SPLIT's server segment;
  - RTT > 0 only adds time;
  - the open class only adds REMOTE to the competitors.
- So the bound computed at (no-raw-text, good, RTT 0) dominates every cell. **If the bound cannot pass, the real
  R cannot pass.**
- The edge-segment compute is a measurement of the same code, on the same GPU, under the same cap.
- The server bound is physics: weights + KV read at the datasheet peak bandwidth.

## 10. Amendments to E0-SPLIT-VIABILITY.md (all made 2026-11-08, before any E0 data)

| # | E0 said | Now | Why | Bias direction |
|---|---|---|---|---|
| A1 | "16k context" | L_F = 17,920 for every action | S1 outputs reach 16,384 + 1,030 prompt tokens | stricter for all actions; at 24 GB slightly against both SPLIT and 32B-AWQ |
| A2 | memory capped by `set_per_process_memory_fraction` | budget = M GiB − 512 MiB, CUDA context included; device and allocator views must both hold | the fraction alone ignores the CUDA context, so a pass would be optimistic | none intended; the same rule for EDGE |
| A3 | 256 / 2,048-token prompts | + 2 depth prompts (17,664) in core cells and Stage 1/2 profiles | thinking traces reach 16k context; the decode slope must be measured there, not extrapolated | none (all paths) |
| A4 | in-host floor on 2 GPUs | two processes on one server GPU over loopback (2-GPU is allowed on 80 GB servers) | the H200 holds both halves; this removes inter-GPU heterogeneity | a lower floor, so a **stricter** Q for SPLIT |
| A5 | "REMOTE greedy" continuations | HF batched greedy on the server | they are only teacher-forcing text; vLLM token-id output adds API risk | none on Q metrics |
| A6 | "mean KL" | coarse-grained KL over top-32 + rest | full-vocabulary storage is about 270 GB | same estimator for floors and SPLIT; tail-only differences are caught by the exact D and by Q2 |
| A7 | "degenerates into repetition" | the operational definition in §4.2 | needed to make Q2 decidable | none |
| A8 | — | informational REMOTE-HF cell | separates the engine gap from the split cost in interpretation | not a criterion |
| A9 | E0 §8 early stop only | + G-F, G-UB1, G-UB2, G-link | cheap falsification before expensive stages | can only stop when R or F is impossible; cannot create a pass |
| A10 | load generator on "the server GPU(s)" | 1× H200: SPLIT segment co-resident with vLLM (util 0.50); load max_tokens 512 | on 80 GB GPUs the two cannot share a GPU, so SPLIT would not feel the load | none; on 2×80 GB the contention block is not run and R is evaluated in the good state only |
| A11 | `tc netem` | netem, otherwise the application shaper (recorded) | rented containers usually lack CAP_NET_ADMIN | the app shaper omits TCP slow-start on prefill transfers, slightly favouring SPLIT at RTT > native; reported |
| A12 | "on-time utility for each cell" | pooled 200 queries + both seeds; stratified bootstrap with re-selection; no-competitor exclusion; Bonferroni label | E0 left these implicit | re-selection penalises the more-tried side; neutral overall |
| A13 | edge = "a ≥ 24 GB NVIDIA GPU" | RunPod order RTX 3090 → RTX 4090 → RTX A5000; the same GPU model in Stages 1 and 3; clocks locked only if permitted | reproducibility | a faster edge favours EDGE more than SPLIT (E0 §5) |
| A14 | 30B-A3B "must be added" | §8.4 (a) and (b) | make the rule operational | can only remove SPLIT passes |
| A15 | — | slope borrowed from the core cell when a cell has no depth probes | the decode slope is only identified with long contexts | none |
| A16 | EDGE core cells in the joint window | EDGE profiles measured single-machine in Stage 1 for every eligible pair | no link is involved; saves joint time | none |

## 11. Known threats (stated before data)

1. **HF eager vs vLLM.** SPLIT runs in an eager HF runtime while EDGE and REMOTE run in vLLM. This penalises
   SPLIT's latency.
   - R is judged on measured values (E0 §3.3).
   - The component floor and the REMOTE-HF cell let a FAIL be read as "this runtime" vs "split itself".
   - It is never used to overturn a FAIL.
2. **Emulated memory classes.** A capped 24 GB card has 3090/4090 bandwidth. Real 12–16 GB devices are slower,
   which would slow EDGE more than SPLIT. Any SPLIT pass is therefore conservative (E0 §5).
3. **S1 quality is vLLM's.** Q licenses Qwen3-32B-bf16 quality for the split path only to the extent HF split ≈ HF
   unsplit, and HF ≈ vLLM. The latter is not tested here. It is at numerical-noise level and the same for every
   Qwen3-32B action.
4. **Selection.** Both sides take a maximum over modes and models on the same queries. The bootstrap re-selects.
5. **fp8 KV on the edge engine** may be unsupported on Ampere with vLLM 0.8.5.post1 V1 (unverified). Rule §8.4 (b)
   applies.
6. **Rented networking.** Pod-to-pod RTT and CAP_NET_ADMIN are unverified until the G-link check runs.

## 12. Pre-hardware prior (informational; decides nothing): `frozen/R_prior_s1_roofline_overhead{0,10,20}ms.md`

Inputs:
- the S1 per-query (score, tokens) for all 5 models × 2 modes;
- spec-sheet roofline profiles (an RTX 3090 edge capped to each class, an H200 server, 0.7 bandwidth / 0.5
  compute efficiency for every action);
- the predicted U-max and the desk-check eligibility;
- the good state only;
- an extra SPLIT per-token overhead of 0, 10 or 20 ms.

| Extra SPLIT overhead / token | Passing cells / 60 | Where | Best cell |
|---|---|---|---|
| 0 ms (SPLIT-favouring) | 5 | 12 GB no-raw-text only: D = 30, 60, 600 (native); 60, 600 (20 ms) | +13.0 pt, D = 60 s, CI [+7.3, +17.5] |
| 10 ms | 2 | 12 GB no-raw-text, native: D = 60 (non-thinking SPLIT vs thinking 8B-AWQ), D = 600 (thinking) | +11.2 pt, D = 60 s |
| 20 ms | 2 | same two cells; no longer robust to multiplicity | +8.5 pt, D = 60 s |

- 16 GB, 24 GB and every `open` cell fail at every overhead.
- **Reading:** if E0 can pass at all, it is expected at 12 GB, no-raw-text, with non-thinking SPLIT at D ≈ 60 s
  or thinking SPLIT at D ≈ 600 s, against Qwen3-8B-AWQ.
- It is a prior, not a result. G-UB1/G-UB2 and Stage 3 decide.
