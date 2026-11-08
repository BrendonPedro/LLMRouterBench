# E0 physical split viability: pre-registration

Frozen **2026-11-08**, before any E0 hardware access. **No E0 measurement has been made.** Nothing in this
directory has run on a GPU. A pre-hardware audit on the same day (still before any E0 measurement) corrected
the implementation and wording listed in §10.1. The frozen version is git tag `e0-prereg-v1`.

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
   token ids, text and logits never leave the device. Activations and control metadata do cross (§2.1).
   Activations at shallow cuts are invertible (Cunningham: about 35 % token recovery at 8 layers). The word
   "privacy-preserving" is never used for SPLIT.
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
  (`[n, 5120]`, raw bf16, `n·10,240` bytes plus a 72-byte header). It receives the residual stream after layer
  63−j (same size, same header). That is 10,312 B each way per decoded token, and n·10,240 + 72 B each way per
  prefill chunk. TCP with `TCP_NODELAY`, pinned host buffers, non-blocking copies. There is one round trip per
  forward step, and no pipelining. Socket byte counters are recorded per request (`bytes_out`, `bytes_in`)
  and cross-checked against `n·5120·2 + 72`. The full accounting, including control metadata, is in §2.1.
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

### 2.1 Transport accounting (exact; `split_runtime.py`)

One TCP connection per cell or profile run. The protocol is strictly synchronous: one message, one reply.
Every message is a 72-byte header (`struct "<4sHHQIIIIQQQQQ"`) plus a payload. The header fields are:
- magic `E0SP`, protocol version, message type;
- request counter, start position, n tokens, hidden size, flags (use-cache);
- payload length;
- three server timings in ns (zero edge → server);
- the server's cache length after the step.

| Message | Direction | When | Payload | Content |
|---|---|---|---|---|
| HELLO | edge → server | once per connection | JSON, about 0.3 KB: model id + revision, config hash, torch / transformers / CUDA / protocol versions, k, j, n_layers, hidden, max_len, wire dtype | control |
| HELLO_OK | server → edge | once per connection | JSON, about 0.35–0.5 KB: active layer range, KV bytes, server identity (model id, software, hostname, device, GPU name, loaded layers, config hash) | control |
| RESET / RESET_OK | edge → server / reply | once per request (each generation or teacher-forced pass) | none | control |
| FWD | edge → server | every forward step | n·10,240 B: the bf16 residual stream after layer k−1 | **model content** |
| FWD_OUT | server → edge | reply to every FWD | n·10,240 B: the bf16 residual stream after layer 63−j | **model content** |
| ERROR | server → edge | a rejected step, then disconnect | UTF-8 error text | control |
| BYE | edge → server | end of connection | none | control |

- **Model-content tensors:** exactly two, h_k (edge → server) and h_{64−j} (server → edge).
- **Never on the link:** raw text, token ids, logits or probabilities, sampled tokens, the embedding table,
  KV entries, the attention mask, position ids. Both sides build the causal mask and rotary positions from
  `start` and `n`.
- **Control metadata does cross.** It is the header on every message plus the HELLO identity. "Only the
  hidden state crosses" is therefore wrong: what is true is that only one model-content tensor crosses per
  direction.

**Bytes per request.** For a prompt of P tokens, N generated tokens and c = ⌈P / 2,048⌉ prefill chunks, each
direction carries:
- RESET: 72 B;
- prefill: 10,240·P + 72·c;
- decode: (N − 1) × 10,312. The first token comes from the prefill pass, and no step is sent after the last
  token;
- **total: 10,240·(P + N − 1) + 72·(c + N)**;
- plus, once per connection, HELLO / HELLO_OK (about 0.4 KB + 72 B) and BYE (72 B, edge → server).

| P (N = 256) | each direction |
|---|---|
| 256 | 5,251,144 B |
| 1,030 (S1 max prompt) | 13,176,904 B |
| 2,048 | 23,601,224 B |
| 17,664 (depth probe) | 183,509,640 B |

- **Q1 teacher forcing:** one cache-less FWD over the whole sequence after RESET, so 10,240·n + 72 B each way.
- **`linkcheck`** sends zero tensors to an echo stub only.

**What the server can observe.** Stated so that no reader infers a privacy property:
- the activations (invertible at shallow cuts, §0.2);
- from the metadata: the exact prompt length, the prefill chunking, the number of generated tokens and the
  end of each request;
- per-step timing (hence edge-side stalls) and the request count;
- the edge's software versions and (k, j).

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

### 3.3 What a capped 24 GB GPU shows, and what it does not

**How the classes are run.** Every attempt is a fresh process on one exclusive 24 GB GPU (§5.1):
1. Take the NVML idle baseline before CUDA initialisation.
2. Initialise CUDA and cuBLAS, and measure the non-torch overhead (NVML used − baseline − torch reserved).
3. Cap the caching allocator with `set_per_process_memory_fraction((budget − non-torch) / total)`.
4. Run, with NVML sampled every 20 ms.

**What the cap controls.** It controls only the PyTorch caching allocator's reserved bytes in that process. An
over-budget allocation raises OutOfMemoryError inside the run, as it would on a smaller device.

**What the cap does not control:**
- memory outside the caching allocator: the CUDA context and driver memory, lazily loaded kernel modules,
  library handles, other processes. Its growth after the cap is set is not prevented; only the device view
  detects it, after the fact, as a FAIL;
- the smaller device itself: bandwidth, SM count, L2, clocks, power and thermal limits, PCIe, another
  architecture's context size and kernels, a display sharing the device, or unified-memory platforms.

**The gate uses the device view.**
- An attempt passes only if both views are within budget:
  - the NVML device view: device-level `used` − idle baseline, max over the run. It includes the CUDA context
    and every non-PyTorch allocation of every process on the GPU;
  - the allocator view: the non-torch overhead + `max_memory_reserved`.
- `torch.cuda` statistics alone never pass an attempt.
- Device-level rather than per-process NVML is used because container PID namespaces hide processes. With
  the GPU exclusive, it can over-count but never under-count.
- The vLLM EDGE fits are judged on the NVML device view only; vLLM's own accounting is not trusted.
- Residual blind spot: a non-allocator transient shorter than 20 ms. Allocator transients are exact
  (`max_memory_reserved`, and the cache does not release memory during a run).

**What a pass is evidence for (conservative):**
- **24 GB.** The class is the physical device (24,064 of 24,576 MiB). A 24 GB pass is a physical result for
  that GPU model, driver and stack.
- **12 and 16 GB.** A pass is evidence only that the run *fits a 12 (16) GiB device-memory budget, CUDA context
  included, on this 24 GB GPU model, driver and software stack*.
  - It is **not sufficient** evidence that SPLIT fits, or meets any latency, on a physical 12 or 16 GB device.
  - Any such claim requires a separate physical validation on a true-capacity device: F re-run, plus SPLIT and
    best-competitor latency for the passing R cells. That validation is outside E0 and not prepared.
  - DECISION labels such results `memory_class_scope: EMULATED`.
- **The emulation is not conservative for SPLIT's latency** (this corrects §11.2). Per decoded token, SPLIT's
  edge segment reads more bytes than the class's EDGE competitor:
  - 7.41 GB at U(3,3) vs 6.1 GB for Qwen3-8B-AWQ at 12 GB;
  - 11.31 GB vs 10.0 GB for Qwen3-14B-AWQ at 16 GB.
  A slower real device would therefore lengthen SPLIT's per-token edge time more than the competitor's. EDGE's
  longer thinking traces push the other way, so the net direction is unknown.

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

### 5.5 F rule (E0 §8, read literally)

- **Per class (memory):** U-max(M) exists (§5.2–§5.3). A class without one is dropped from R. That alone is
  not an E0 failure.
- **Errors (global):** E0 §8 F requires "zero transport errors or OOMs over every E0-L request". This is applied
  literally and conservatively to every E0-L request on every path (SPLIT, REMOTE, EDGE profiles, edge-segment
  profiles) in every latency directory given to `decide`.
  - It covers OOM, transport errors and any request that did not complete exactly (256 tokens, prompt tokens
    = n_in).
  - Failed rows are appended and never erased: a resume re-runs the request, but the failure still counts.
  - One failed request means F fails and SPLIT is removed.
- **F fails iff no class has a U-max, or any E0-L request failed.**
- (The pre-audit text read the error condition per class. The code always applied it globally; §10.1 C8.)

### 5.6 Memory failure vs harness defect (Stage 1 is the first CUDA execution, §9.5)

- **Memory failures** are evidence for F:
  - an exception whose text matches `e0_f.MEMORY_ERROR_PATTERNS` (CUDA OOM, cuBLAS allocation failure, vLLM
    KV-capacity errors);
  - a device or allocator peak above the budget.
- **Any other exception in a split-fit attempt is a harness defect.** It invalidates Stage 1 rather than
  failing F (`f_results.json: valid = false`; G-F is not evaluated).
  - The defect is fixed, committed and recorded in DECISION.
  - Then the whole Stage 1 is re-run from scratch.
  - A fix may not change anything memory-relevant: the weights loaded, KV size, L_F, chunk size, attention
    implementation, dtypes, budget or cap procedure.
- **EDGE competitor fits.** A desk-eligible competitor (§3.2) that fails its fit for a non-memory reason
  ("engine") makes the competitor set incomplete. That class can then never produce a SPLIT pass (§8.4 (b)).
  An unrecognised error text is classified "engine", so a misclassification can only block SPLIT.

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
  - **If the native RTT p50 is > 5 ms, or the median bulk goodput is < 900 Mbps, Stage 3 is not run on that
    pairing.** The 5 ms cells, or the "native, 1 Gbps" cells (run unshaped), would be physically
    unrealisable. 900 Mbps is the TCP goodput of a nominal 1 Gbps link. The pods are re-provisioned in one
    datacenter instead.
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
- **(b) A competitor fails for an engine reason.** If Qwen3-32B-AWQ fails the 24 GB fit for an *engine* reason
  (for example fp8 KV unsupported by the pinned engine on that GPU) rather than for memory, no 24 GB cell may
  pass.
  - Generalised (§10.1 C11): the same holds at class M for any desk-eligible EDGE model that fails its fit for a
    non-memory reason (§5.6).
  - The code enforces it: `f_results.no_split_pass_classes`, and those cells are blocked in `r` and excluded
    from the bounds.
- Both rules can only remove a SPLIT pass. (a) is enforced by `decide`: a 24 GB-only pass is PENDING until the
  re-evaluation, which is run with `--a3b-added`.

### 8.5 Sample extension (E0 §4.4)

- **Trigger** (mechanical, `e0_r.extension_triggered`). All of the following:
  - Q and F pass;
  - R fails on the 200 queries;
  - R's best evaluated cell (largest Δ) has **Δ ≥ 3.0 pt with CI lower bound ≤ 0**.

  DECISION then reads `E0 = PENDING`, not FAIL. Q or F failures are never pending.
- **Sample (pinned now):** the first 2n of the same S1 permutation per benchmark (`run_s1.draw_sample(scale=2)`).
  - 120 / 120 / 80 / 80 queries, containing the 200.
  - `frozen/extension_400_keys.json`: sample sha256 `b6ae701a…b908`, keys sha256 `005ca514…5b9f`
    (`e0_spec.EXT400_*`).
  - The self-test re-draws it.
- **Data:** new S1-style generations for the 200 added queries, for all 10 (model, mode) actions × seeds 0 and 1,
  with S1's frozen settings. That is 8,000 generations, about the cost of S1 itself.
  - It is a separate server job. It is not prepared and runs only on trigger.
  - It is the one pre-declared exception to "S1 is not re-run": the 200 S1 queries are not re-generated.
- **Re-evaluation:** every R cell is re-evaluated, with no change to:
  - the measured profiles (no new latency data);
  - the thresholds (Δ ≥ 3.0, CI lower bound > 0);
  - the bootstrap seed and procedure, the competitors and the exclusions.

  It happens once, with no third look. Its result is final.
- **Why it cannot become a post-hoc rescue:**
  - the trigger, the sample and the rule were fixed before any E0 data: the trigger in E0 §4.4 (2026-11-07),
    the sample hash here;
  - the added queries are fixed by a seeded permutation, not chosen;
  - they add data to both arms equally (paired), and can lower Δ as easily as raise it;
  - nothing else may change, and the trigger cannot fire on the 400-query result (`sample_n = 400`);
  - an ad hoc extension (other queries, more queries, a second extension, a changed cell) is forbidden by §0.4.
- **Its statistical cost, stated now:** the extension is a second look at the same hypothesis.
  - The two-stage false-pass rate exceeds the single-look nominal 2.5 % one-sided level, by at most about
    double.
  - Together with E0 §8's "any of up to 120 cells" rule, the R decision is not multiplicity-controlled.
  - The Bonferroni label (§8.3) must be quoted with every pass.

### 8.6 What S1 already implies for 16 and 24 GB (computed 2026-11-08 from S1 per-query data; decides nothing)

R compares on-time utility pooled over the 200 queries, with both modes on each side and deadlines in force. S1's
per-benchmark × mode CIs are therefore not R's CIs, and S1 does not by itself exclude a 16 GB pass.

| Class | Every action instantaneous (no deadline effect) | SPLIT instantaneous, EDGE at the §12 roofline timing |
|---|---|---|
| 12 | Δ = +11.75 pt, CI [+7.25, +16.50] | passes at every D |
| 16 | Δ = **+3.25** pt, CI [**−0.50**, +7.00] | D = 30 / 60 / 120: +21.25 / +20.75 / +18.25, CI > 0; D = 300 / 600: +3.25, CI low −0.50 |
| 24 | Δ = **+2.50** pt, CI [−1.25, +5.25] (vs 32B-AWQ thinking) | D = 30 / 60 / 120: +18 to +21, CI > 0; D = 300: +3.25, CI low −0.50; D = 600: +2.50 |

- **16 GB.** A 16 GB pass is mathematically possible under G-R.
  - It cannot come from quality alone at 200 queries. Where every action finishes in time, the best possible
    Δ is +3.25 with CI low −0.50.
  - It needs either SPLIT U-max(16) to beat the edge models on speed at short deadlines (the §12 prior says
    it does not), or the extension.
  - That ceiling cell sits exactly in the §8.5 trigger zone. If it were also R's best cell and the extra 200
    queries reproduced the gap, the CI half-width would shrink by about 1/√2 (≈ 3.75 → 2.65 pt). That could
    make the CI exclude 0, so the extension can change the 16 GB outcome.
- **24 GB.** Where every competitor finishes in time, the ceiling is +2.50 < 3.0. No such cell can pass or
  trigger the extension.
  - A 24 GB pass needs SPLIT U-max(24) to be faster on-time than every 24 GB edge model.
  - Any 24 GB pass is then held pending §8.4 (a) (Qwen3-30B-A3B) and voided by §8.4 (b).

## 9. E0 PASS / FAIL and falsification gates (frozen before any result)

**E0 PASS = Q ∧ F ∧ R.** Each STOP writes `DECISION.md` and ends hardware spending. BLOCKED, INCOMPLETE and
INVALID are not outcomes: they stop the stage until the named condition is met, and they never turn into PASS.

| Gate | When | Outcomes |
|---|---|---|
| G1 (done) | S1 | E0 §8 early stop: not triggered, **CONTINUE** |
| G-F | after Stage 1 | STOP (F fails, SPLIT removed) / proceed / INVALID (harness defect, §5.6) |
| G-UB1 | after Stage 1 | STOP (R fails by bound, SPLIT removed) / CONTINUE / INCOMPLETE |
| G-UB2 | after Stage 2 | the same |
| G-link | start of Stage 3 | BLOCKED (re-provision the pairing; not an E0 outcome) / proceed |
| G-Q | Stage 3, before any latency cell | STOP (Q fails after the one fix, SPLIT removed) / proceed |
| G-R | after Stage 3 | E0 PASS / FAIL (SPLIT removed for lack of a regime) / PENDING (§8.4 a, §8.5) |

### 9.1 Exact conditions

Notation:
- B_M = 1024·M − 512 MiB.
- Δ, CI and "evaluated cell" are as in §8.1–§8.3: 200 queries × 2 seeds, a stratified paired bootstrap with
  10,000 resamples, seed 20261107, re-selection of both maxima, and the §8.3 exclusions.
- U-max(M) = (k_M, k_M).

**G-F** (`e0_f.py summarize` → `f_results.json`)
- attempt(k, M) passes iff all of:
  - a fresh process completes cap → load U(k,k) → 17,664-token prefill (9 chunks) → 256 decode tokens, final
    context = 17,920;
  - it raises no exception;
  - max_t (NVML_used(t) − baseline) ≤ B_M;
  - non-torch + max_memory_reserved ≤ B_M.
- U-max(M) = the §5.3 search from the predicted k, or ∅.
- If any attempt has a non-memory exception, the outcome is **INVALID** (§5.6).
- Otherwise **STOP iff U-max(12) = U-max(16) = U-max(24) = ∅**. The 24 GB search reaches k = 1, so this
  means (1,1) does not fit even 24 GB.

**G-UB1** (`e0_r.py gate-ub1`)
- For each class M with a U-max and not blocked by §8.4 (b), bound SPLIT's time from below:
  T̲_M(n, N) = 0.90 · E_k(n, N) + S̲_k(n, N), with zero link.
  - E_k is the Stage 1 edge-segment compute profile for U(k_M, k_M), under cap M, against the echo stub: the
    head + tail + LM-head CUDA-synchronised timers, fitted as in §7.1.
  - S̲_k is the H200 datasheet roofline for layers [k, 64−k):
    - decode step ≥ (weight bytes + KV bytes · ctx) / 4.8 TB/s;
    - prefill ≥ max(2 · params · n / 989 TFLOP/s, weight bytes / 4.8 TB/s);
    - no first-token step.
- Evaluate the 30 cells M ∈ {12, 16, 24} × {no-raw-text, open} × good × RTT 0 × D ∈ {30, 60, 120, 300, 600}.
  - The EDGE competitors are the measured Stage 1 profiles and eligibility, exactly as R uses them.
  - REMOTE in open cells is bounded below by U = 0.
- Δ̄ = max Δ_bound over the evaluated cells.
- Outcomes:
  - **INCOMPLETE** if a class with a U-max lacks its edge-segment profile;
  - else **STOP iff Δ̄ < 3.0 pt** (or no cell is evaluable);
  - else **CONTINUE**.

**G-UB2** (`e0_r.py gate-ub2`)
- Identical to G-UB1, with S̲_k replaced by 0.90 × the Stage 2 measured server-segment compute profile
  (`t_compute` only, synthetic activations, H200, κ = 0).

**G-link** (`e0_latency.py linkcheck`, server-side echo stub, over the split protocol)
- Measure:
  - the p50 round trip of 1,000 n = 1 steps (10,312 B each way);
  - the median bulk goodput of 20 n = 2,048 round trips.
- **Proceed iff p50 ≤ 5.0 ms and goodput ≥ 900 Mbps.** Otherwise the pairing is BLOCKED.

**G-Q** (`e0_q.py decide`), for each cfg ∈ {U-min, U-max(16)}:
- Q1(cfg): both of
  - D_cfg ≤ max(D_pipe(cfg), D_hw) + 0.2 pp;
  - KL_cfg ≤ 2 · max(KL_pipe(cfg), KL_hw), or KL_cfg ≤ 1e-6 if both floors are 0.
- Q2(cfg): all of
  - exactly 30 prompts are run;
  - #{prompts with no divergence, or reference margin at the first divergence < 0.5 nats} ≥ 27;
  - no prompt has a degenerate SPLIT sequence with a non-degenerate reference.
- Q = ∧_cfg (Q1 ∧ Q2).
- If Q fails: one implementation fix, then Q1 + Q2 are re-run once. **STOP** if Q fails again.

**G-R** (`e0_r.py r`, then `decide`)
- R passes iff some evaluated cell of the 120 has **Δ ≥ 3.0 pt and CI lower bound > 0**.
  - SPLIT = U-max(M), profiled by the validated additive model, or by the measured cells only (§7.3).
  - Cells are blocked by §8.4 (b).
- §8.4 (a): a 24 GB-only pass is PENDING.
- §8.5: the extension trigger means PENDING.
- **E0 = PASS iff Q ∧ F ∧ R**, where F includes zero failed E0-L requests (§5.5).

### 9.2 Why G-UB1 and G-UB2 can STOP without discarding a configuration that could still pass R

1. **The same configurations and competitors.** R evaluates SPLIT only at U-max(M), the same (k, k) as the
   bound.
   - The EDGE competitors and their eligibility are the same Stage 1 files.
   - §8.4 can only add competitors or block classes. A blocked class can never pass, so excluding it from the
     bound is safe.
2. **A per-(n, N) lower bound on SPLIT's time.** Real SPLIT time ≥ edge compute + server compute + link + serialisation.
   - The edge term is the same code, on the same GPU model, under the same cap. The 0.90 factor absorbs timer
     noise, fit noise and unit-to-unit variation, since the Stage 3 card may be another physical unit (A13).
   - The UB1 server term is physics: no kernel exceeds peak HBM bandwidth or peak dense FLOPs, and attention
     FLOPs and the first-token step are left out.
   - The UB2 server term is the measured compute of the same server code, × 0.90.
3. **Monotonicity.** U_SPLIT is non-increasing in T, per query and per bootstrap replicate. So Δ_real ≤ Δ_bound
   and CI-low_real ≤ CI-low_bound for the same cell.
   - In no-raw-text cells, contention and RTT > 0 only add SPLIT time, and the competitors are unchanged.
   - In open cells, the real REMOTE has U ≥ 0.
   - So the 30 good-state, RTT-0 bound cells dominate all 120 R cells. That includes open cells at a class
     whose no-raw-text cells are excluded for lack of an EDGE competitor.
4. **Δ̄ < 3.0 is a certain R failure under the full preregistered rule.**
   - Every real cell has Δ < 3.0.
   - No cell can pass, since a pass needs Δ ≥ 3.0.
   - The §8.5 extension cannot fire, since it needs the best cell's Δ ≥ 3.0.
   - The pre-audit rule (STOP unless some bound cell passes Δ ≥ 3 ∧ CI low > 0) could STOP while a real cell
     would trigger the extension. Corrected (§10.1 C1).
5. **Missing data never STOPs.** A missing segment profile gives INCOMPLETE.
6. **Stated residuals:**
   - the bound holds for the fitted profiles R uses (means), not for every single request;
   - a validated additive model may undershoot measured SPLIT times by up to its ±10 % tolerance. The 0.90
     edge factor covers that for the edge segment, and the datasheet server bound is far below any real
     server time;
   - the Stage 3 edge GPU must be the Stage 1 GPU model (A13).

### 9.3 Consequences (E0 §8 table; the last two rows added by the audit)

| Outcome | Consequence |
|---|---|
| PASS, no-raw-text cells only | SPLIT enters F3's action set as the no-raw-text quality tier. The claim is "a larger model under a raw-text-stays-local constraint". Measured leakage must be reported; never "privacy-preserving". If all passing cells are 12 GB: "restricted to memory-starved edges" |
| PASS including open cells | unexpected. Re-check against the component floor and the netem logs before believing it |
| Q or F fails | SPLIT removed. F3 = EDGE/REMOTE |
| Q ∧ F, R fails | SPLIT removed for lack of a regime. Report "weight-quantised edge models dominate layer split" as a negative result |
| Any PASS at 12 or 16 GB | scope: "under an emulated 12 (16) GiB budget on a capped 24 GB GPU" (§3.3); no physical-device claim without a separate validation |
| PENDING | §8.5 extension triggered, or 24 GB-only pass awaiting §8.4 (a): run exactly that pre-declared step, nothing else |

### 9.4 Stage 1 is a feasibility gate only

- **What Stage 1 measures:**
  - memory feasibility (F, U-max per class);
  - the EDGE competitors' fit and single-machine profiles;
  - the edge segment's compute against a loopback echo stub.
- **CONTINUE after G-F and G-UB1 means only this:** a lower bound on SPLIT time does not yet rule R out. That
  bound ignores the link, server contention, server-side inefficiency, serialisation and the HF/eager runtime
  overhead.
- **It is not evidence that SPLIT meets any latency, TTFT or deadline requirement, and must not be presented as
  such.**
- No latency or deadline statement about SPLIT is made before the Stage 3 measured cells and G-R.
- The gate files say so in their decision string.

### 9.5 CUDA: Stage 1 is the first physical validation

- Nothing in E0 has run on CUDA.
- **What the local checks validate.** The local checks show bitwise equality with HF on CPU, including real
  Qwen3-0.6B weights over TCP. They validate the protocol, the KV/position bookkeeping and the numerics on
  one CPU.
- **What they do not validate:**
  - CUDA kernels and SDPA on a GPU;
  - the allocator cap and its OOM behaviour;
  - pinned-memory non-blocking copies;
  - CUDA-synchronised timers;
  - NVML on a real device;
  - vLLM launches and their real API fields;
  - cross-GPU numerics;
  - any execution of Qwen3-32B.
- The pod self-test (`--only spec s1 template runtime f latency r`) also runs on CPU.
- **Stage 1's first `split-fit` is the first CUDA execution of this harness.** Defects found there are handled
  by §5.6 (INVALID, fix, re-run Stage 1), never read as an F result.

## 10. Amendments to E0-SPLIT-VIABILITY.md (all made 2026-11-08, before any E0 data)

**Type:**
- **A** = an implementation or operational clarification (it makes an E0 instruction executable without
  changing what is tested);
- **B** = a change to a scientific criterion or protocol.

**Easier?** says whether the amendment can make SPLIT easier to pass than E0 as written.

| # | Type | E0 said | Now | Why | Bias direction | Easier? |
|---|---|---|---|---|---|---|
| A1 | B | "16k context" | L_F = 17,920 for every action | S1 outputs reach 16,384 + 1,030 prompt tokens | stricter for all actions; at 24 GB slightly against both SPLIT and 32B-AWQ | no |
| A2 | B | memory capped by `set_per_process_memory_fraction` | budget = M GiB − 512 MiB, CUDA context included; device and allocator views must both hold | the fraction alone ignores the CUDA context, so a pass would be optimistic | stricter; the same rule for EDGE | no |
| A3 | B | 256 / 2,048-token prompts | + 2 depth prompts (17,664) in core cells and Stage 1/2 profiles | thinking traces reach 16k context; the decode slope must be measured there, not extrapolated | none (all paths) | no |
| A4 | B | in-host floor on 2 GPUs | two processes on one server GPU over loopback (2-GPU is allowed on 80 GB servers) | the H200 holds both halves; this removes inter-GPU heterogeneity | a lower or equal floor, so a **stricter** Q for SPLIT | no |
| A5 | A | "REMOTE greedy" continuations | HF batched greedy on the server | they are only teacher-forcing text; vLLM token-id output adds API risk | none on Q metrics | no |
| A6 | B | "mean KL" | coarse-grained KL over top-32 + rest | full-vocabulary storage is about 270 GB | the same estimator for the floors and SPLIT. It under-reads tail-only differences, so a SPLIT whose divergence sits beyond the top-32 more than the floors' does would be judged more leniently. Mitigated by the exact full-vocabulary D and by Q2 | **possibly (Q1-KL only)** |
| A7 | A | "degenerates into repetition" | the operational definition in §4.2 | needed to make Q2 decidable | none | no |
| A8 | A | — | informational REMOTE-HF cell | separates the engine gap from the split cost in interpretation | not a criterion | no |
| A9 | B | E0 §8 early stop only | + G-F, G-UB1, G-UB2, G-link | cheap falsification before expensive stages | can only stop when R or F is impossible (with §10.1 C1–C4; §9.2); cannot create a pass | no |
| A10 | B | load generator on "the server GPU(s)" | 1× H200: SPLIT segment co-resident with vLLM (util 0.50); load max_tokens 512 | on 80 GB GPUs the two cannot share a GPU, so SPLIT would not feel the load | stricter (SPLIT feels the load). On 2×80 GB the contention block is not run and R is evaluated in the good state only, so fewer cells | no |
| A11 | A | `tc netem` | netem, otherwise the application shaper (recorded) | rented containers usually lack CAP_NET_ADMIN | the app shaper omits TCP slow-start on prefill transfers, slightly favouring SPLIT at RTT > native; reported | **yes, slightly (app-shaper mode only)** |
| A12 | A (+ one B element) | "on-time utility for each cell" | pooled 200 queries + both seeds; stratified bootstrap with re-selection; no-competitor exclusion (B); Bonferroni label | E0 left these implicit | re-selection penalises the side with more options (the competitors: up to 8 vs 2), so slightly against SPLIT; the exclusion removes cells SPLIT would win trivially | no |
| A13 | A | edge = "a ≥ 24 GB NVIDIA GPU" | RunPod order RTX 3090 → RTX 4090 → RTX A5000; the same GPU model in Stages 1 and 3; clocks locked only if permitted | reproducibility | direction unknown (corrected; §3.3): per token, SPLIT's edge segment reads more bytes than the class competitor | unknown |
| A14 | B | 30B-A3B "must be added" | §8.4 (a) and (b) | make the rule operational | can only remove SPLIT passes | no |
| A15 | A | — | slope borrowed from the core cell when a cell has no depth probes | the decode slope is only identified with long contexts | load cells borrow the uncontended slope, which can under-state contended times for SPLIT and REMOTE alike. In no-raw-text cells a contended SPLIT profile is still dominated by the good-state one | **possibly (contended open cells)** |
| A16 | A | EDGE core cells in the joint window | EDGE profiles measured single-machine in Stage 1 for every eligible pair | no link is involved; saves joint time | none | no |

Not an amendment but inherited from E0 §8, and liberal: R passes on **any one** of up to 120 cells without a
multiplicity correction. The Bonferroni label is reported next to every pass (§8.3, §8.5).

**Timing.** A1–A16 and C1–C12 below were all written on 2026-11-08, before any E0 measurement. The E0
directory contains no GPU result at all: only `results_local/` (CPU self-test). The frozen state is tag
`e0-prereg-v1`. Every E0 GPU result must record a `git rev-parse HEAD` equal to or descending from it, and
`environment()` records it per host.

### 10.1 Pre-hardware audit corrections (2026-11-08, before any E0 measurement)

| # | Type | Before | After | Effect on SPLIT |
|---|---|---|---|---|
| C1 | A (bug) | G-UB STOPped unless a bound cell had Δ ≥ 3 ∧ CI low > 0, which could STOP while a real cell would trigger the §8.5 extension, violating §9.2 | STOP iff max Δ_bound < 3.0 pt | STOPs less often; cannot create a pass |
| C2 | A (bug) | a class with a U-max but no segment profile was silently left out of the bound | INCOMPLETE, never STOP | the same |
| C3 | A (bug) | the bound covered no-raw-text cells only, so a class with no fitting EDGE competitor had no bound for its open cells | open cells bounded with REMOTE at U = 0 | the same |
| C4 | B (gate constant) | measured segment compute used as an exact lower bound | × 0.90 allowance for noise and unit variation (`UB_SEGMENT_DISCOUNT`) | the same; gate only, R never uses it |
| C5 | A (bug) | `decide` printed FAIL where §8.5 triggers, and left §8.4 (a) to the reader | `E0 = PENDING` for the §8.5 trigger and for a 24 GB-only pass; the extension cannot fire twice | none on the final rule |
| C6 | A (bug) | after an over-budget first edge-fit attempt, the resume kept that attempt's S5 profile | the profile is kept only from the passing attempt; the others become `.rejected` | removes contamination of EDGE profiles; direction unknown, small |
| C7 | B (blocking condition) | G-link checked RTT only, although the "native, 1 Gbps" cells run unshaped | also median bulk goodput ≥ 900 Mbps, else BLOCKED | prevents an unrealised, SPLIT-penalising cell; it can move SPLIT's measured latency *towards* its definition |
| C8 | A (wording) | §5.5 read the error condition per class; the code applied E0's literal, global rule | the text now states the global rule | none (the code was unchanged); stricter than the old text |
| C9 | A (wording) | "64-byte header"; "only activations cross"; §11.2 and A13 claimed the emulation was conservative | §2.1 transport accounting, §3.3 memory-class validity, §9.4 Stage 1 scope, §9.5 CUDA; §11.2 and A13 corrected | claims only |
| C10 | B (new rule for an unspecified case) | a non-memory crash in Stage 1 would have been read as an F failure | §5.6: harness defect, Stage 1 INVALID, fix without memory-relevant changes, re-run all of Stage 1 | can keep SPLIT alive after a *bug*; cannot create a pass, since all memory measurements are re-taken |
| C11 | B | §8.4 (b) covered 32B-AWQ at 24 GB only, and was not enforced in code | any desk-eligible EDGE model failing its fit for a non-memory reason blocks that class; enforced in `summarize`, the gates and `r`; unrecognised errors count as non-memory | stricter |
| C12 | A | the 400-query extension sample was defined by a procedure | the sample is pinned by hash now (`frozen/extension_400_keys.json`) | removes flexibility |

## 11. Known threats (stated before data)

1. **HF eager vs vLLM.** SPLIT runs in an eager HF runtime while EDGE and REMOTE run in vLLM. This penalises
   SPLIT's latency.
   - R is judged on measured values (E0 §3.3).
   - The component floor and the REMOTE-HF cell let a FAIL be read as "this runtime" vs "split itself".
   - It is never used to overturn a FAIL.
2. **Emulated memory classes** (corrected by the audit; §3.3). A capped 24 GB card has 3090/4090 bandwidth and
   compute, while real 12–16 GB devices are usually slower. The earlier claim that this "would slow EDGE more
   than SPLIT, so any SPLIT pass is conservative" is **withdrawn**.
   - Per decoded token, SPLIT's edge segment reads more bytes than the class's EDGE competitor: 7.41 vs 6.1 GB
     at 12 GB, and 11.31 vs 10.0 GB at 16 GB.
   - EDGE's longer thinking traces push the other way, so the direction is unknown.
   - A 12 or 16 GB result is a statement about an emulated budget only. A physical-device claim needs a
     separate validation.
3. **S1 quality is vLLM's.** Q licenses Qwen3-32B-bf16 quality for the split path only to the extent HF split ≈ HF
   unsplit, and HF ≈ vLLM. The latter is not tested here. It is at numerical-noise level and the same for every
   Qwen3-32B action.
4. **Selection.** Both sides take a maximum over modes and models on the same queries. The bootstrap re-selects.
5. **fp8 KV on the edge engine** may be unsupported on Ampere with vLLM 0.8.5.post1 V1 (unverified). Rule §8.4 (b)
   applies.
6. **Rented networking.** Pod-to-pod RTT and CAP_NET_ADMIN are unverified until the G-link check runs.
7. **No CUDA validation yet** (§9.5). Stage 1 is the first CUDA execution; §5.6 governs harness defects.
8. **Second look and multiplicity** (§8.5). The extension and the any-of-120-cells rule make R
   liberal; the Bonferroni label is reported with every pass.

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
