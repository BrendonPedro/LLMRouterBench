# E0 local validation (CPU only; 2026-11-08, re-run after the pre-hardware audit (PREREG §10.1) and on 2026-11-09 after harness fixes H1 and H2)

**Host:** the WSL box (Ryzen 5 8500G, 9 GB RAM, **no CUDA**).

**Environment:** `~/.cache/LLMRouterBench/venv-e0` with torch 2.6.0+cpu, transformers 4.51.3, accelerate 1.6.0,
numpy 2.2.6 (`requirements-e0.txt`).

**What this is:** implementation validation. **It decides nothing about SPLIT.**

**Command:**

```bash
~/.cache/LLMRouterBench/venv-e0/bin/python diagnostics/deployment_e0/e0_selftest.py \
    --tokenizer-dir ~/.cache/LLMRouterBench/e0_hf/qwen3-32b-tok \
    --real-model-dir ~/.cache/LLMRouterBench/e0_hf/qwen3-0.6b \
    --report diagnostics/deployment_e0/results_local/selftest_report.json
```

**Inputs:**
- Qwen3-32B tokenizer and index at revision `9216db57`; `tokenizer.json` sha256 `aeb13307…dae4`.
- Qwen3-0.6B at revision `c1899de2`; `model.safetensors` sha256 `f47f7117…874b`.

**Result: 41 PASS, 0 FAIL, 0 SKIP (914 s)** after harness fix H2 (`HARNESS_FIXES.md`). The 2 new checks cover vLLM server cleanup and edge-fit against a stand-in vLLM. After H1: 39 PASS, 0 FAIL, 0 SKIP (413 s); its 2 new checks cover clean process exit and exit-status auditing. The log's one `terminate called without an active exception` line comes from the deliberate positive control in the exit-status check. On the tag `e0-prereg-v1`: 37 PASS, 0 FAIL, 0 SKIP (603 s). Before the audit: 33 PASS, 0 FAIL, 0 SKIP (455 s) on commit `3600904`. The 4 new checks cover the audit corrections (PREREG §10.1). Log: `results_local/selftest.log`. JSON with the environment: `results_local/selftest_report.json`.

## What was validated

| Group | Check | Result |
|---|---|---|
| spec | held-out cells equal their seeded draw; 22 decision cells; block sizes 6/6/1/1/4/4 | PASS |
| spec | L_F = 17,920 ≥ S1 max prompt (1,030) + 16,384; budgets 11,776 / 15,872 / 24,064 MiB; predicted U-max (3,3)/(5,5)/(9,9); exact parameter count = index total_size | PASS |
| spec | E0 revisions and the vLLM version equal the S1 run's | PASS |
| s1 | sha256 of `q3_scores.jsonl` and `sample_200.jsonl` pinned | PASS |
| s1 | 10 actions × 200 queries × 2 seeds complete; CONTINUE recomputed from per-query data; gaps equal `S1_SUMMARY.md` | PASS |
| s1 | the pre-declared 400-query extension sample: the pinned keys and sample hashes, 120/120/80/80, contains the 200, re-draws byte-identically from the benchmark cache | PASS |
| template | the pinned tokenizer + chat template reproduce **400 / 400** S1 `prompt_tokens` (200 prompts × thinking / non-thinking) | PASS |
| template | the frozen workload rebuilds to ids sha256 `615f0392…3d47`; the frozen Q2 keys reproduce | PASS |
| runtime | teacher-forced split hidden states are **bitwise equal** to HF `Qwen3Model.forward` for U-min, (2,3), (3,3); bf16 in-process and over TCP, fp32 in-process; the wire refuses non-bf16 tensors | PASS |
| runtime | greedy split decoding over TCP is token-identical to HF `generate` with chunked prefill of 7 and 16 tokens and unchunked | PASS |
| runtime | wire bytes = n × hidden × 2 + 72-byte header each way; the server cache length tracks positions; prefill chunking 8/8/8/6 | PASS |
| runtime | failure paths: a wrong position is rejected by the server; illegal splits (k = 0 or j = 0) are refused; a transformers version mismatch is refused at HELLO; cache overflow; peer drop | PASS |
| runtime | the echo stub returns the payload bit-exactly; the app shaper adds RTT + serialisation per round trip (20 ms RTT at 1 Mbps measured within tolerance) | PASS |
| runtime | `Slice` materialises only the owned weights (exact parameter bytes; the rest on `meta`); KV bytes exact | PASS |
| real | **Qwen3-0.6B real weights:** TCP split U(2,3) is bitwise equal to HF on a real S1 chat prompt; greedy output equals HF `generate` | PASS |
| real | **Q pipeline CLIs end to end on real weights:** `cont-gen` → `ref` → `cmp --config hf` and `cmp --config split` (k = 1, 3) over TCP → `q2-ref` → `q2-run` → rules. On one CPU: D = 0, KL < 1e-9, no Q2 divergence | PASS |
| q | the coarse KL is 0 for identical distributions, ≥ 0, and ≤ the full-vocabulary KL (50 random cases) | PASS |
| q | Q1 thresholds incl. both floors 0; Q2 near-tie counting (27/30), degeneration (period, invalid id), "split degenerate only" | PASS |
| f | the U-max search returns the largest passing k from any start (6 truths × 5 starts) and never repeats an attempt | PASS |
| f | edge-fit keeps the S5 profile only from the passing (last) attempt; the others become `.rejected` | PASS |
| f | failure classes: CUDA OOM, cuBLAS allocation and vLLM KV-capacity texts count as memory; anything else as harness/engine; `summarize` marks Stage 1 invalid on a harness defect and blocks a class whose desk-eligible competitor failed for an engine reason | PASS |
| f | the NVML sampler (fake device): peak minus baseline, trapezoid energy | PASS |
| f | `split-fit --cpu-dry` CLI: load, 17,664-token chunked prefill, 256 decode tokens, final context = L_F, JSON record | PASS |
| f | **clean exit (H1):** split-fit (CPU dry) joins its echo thread, leaves no live thread and exits 0 with no libstdc++ terminate (3 runs, one through the exit-status wrapper); the echo thread also stops when no client connected; split-fit's thread layout and exit sequence at Stage-1 wire scale (hidden 5,120, 2,048-token chunks) exits 0 in 5 of 5 runs | PASS |
| f | **exit-status audit (H1):** a child that writes a PASS record and then aborts through the real daemon-thread/torch mechanism (SIGABRT, −6) is a harness defect; so are a wrong exit code and a missing record; split-umax stops at a defective record; summarize marks Stage 1 invalid for records without an exit status, for a signal-killed profile client and for non-zero logged edge-fit / run-split statuses | PASS |
| f | **vLLM server cleanup (H2):** a server that already exited and was reaped (the Stage-1 state; a bare `killpg` raises `ProcessLookupError` there, positive control) is not signalled and keeps its own status; a running one is terminated (SIGTERM, or SIGKILL when it ignores SIGTERM); a SIGSEGV death is recorded as such; a group member orphaned by its leader is found and killed; no process is left in the group | PASS |
| f | **edge-fit with a stand-in vLLM (H2):** a start-up KV-capacity exit (the engine's error followed by a longer front-end traceback) is a memory FAIL with its record written and exit 0, where the old 2,000-character tail would have read "engine"; an engine error, a SIGSEGV death and an orphaned engine core are engine FAILs with the exit status recorded; a running server PASSes and is terminated; a server that dies on its own after serving, or a group member that cannot be killed, is a harness defect and edge-fit exits non-zero; summarize marks Stage 1 invalid for an edge-fit record without a server record, a harness attempt or a missing record; no stray process remains | PASS |
| latency | request plans (2 warm-ups + 30, or 36 with depth probes), seeded order, every cell scheduled once | PASS |
| latency | netem command generation and `tc -s qdisc` state parsing (both directions, delay = RTT/2, rate) | PASS |
| latency | SPLIT cell runner over TCP on the **real frozen workload** (incl. the 17,664-token depth probes) with app shaping: 38 complete records, 256 tokens each, per-step components, the server cache ends at L_F − 1; resume skips completed requests | PASS |
| latency | `linkcheck` over the split protocol against an echo stub | PASS |
| latency | streaming client vs a fake vLLM SSE server: token counting from `continuous_usage_stats` with coalesced chunks; prompt-token check; app RTT adjustment; load-status check | PASS |
| r | `Profile.T` closed form = explicit sum; addition; extrapolation | PASS |
| r | the stratified bootstrap keeps per-benchmark counts; identical actions give Δ = 0 (fail); the no-competitor exclusion | PASS |
| r | the grid passes a strictly better SPLIT, fails a slower one at short D, fails `open` when REMOTE is identical | PASS |
| r | upper-bound monotonicity: a lower-bounded T gives Δ and CI-low ≥ the true values in every cell | PASS |
| r | upper-bound STOP rule on real S1 data. It does not STOP at 16 GB, where the instantaneous bound gives Δ = +3.25 with CI low < 0 (the §8.5 zone; the pre-audit rule would have STOPped). It STOPs at 24 GB, where max Δ = +2.50. Missing profile: INCOMPLETE. No EDGE competitor: open cells bounded with REMOTE at U = 0. The 0.90 discount only shortens times | PASS |
| r | the additive model recovers synthetic coefficients (within 2 %) and contention ratios; held-out validation passes, and fails when held-out cells are perturbed by +20 %, falling back to measured cells only | PASS |
| r | R CLIs on synthetic stage directories: `gate-ub1`, `gate-ub2`, `fit`, `r` (120 cells), `decide` | PASS |
| r | decision table: Q / F / R / errors map to the E0 §8 consequences; the §8.5 trigger gives PENDING once (never at n = 400, never with Q/F failing, not at Δ < 3); a 24 GB-only pass gives PENDING (§8.4 a); a 12 GB pass is labelled EMULATED | PASS |
| docs | every `e0_*.py <subcommand>` named in `RUNBOOK.md` exists | PASS |

## Defect found by Stage 1 run 1 (RTX 3090, 2026-11-09; INVALID)

Passing `split-fit` children aborted at exit (`terminate called without an active exception`), and `split-umax`
did not check exit statuses. Root cause, CPU reproduction (old shutdown 7/20 aborts, fixed 0/20) and the fix are
in `HARNESS_FIXES.md` H1. The local suite had missed it: the tiny test model's activations are small, so the echo
thread's teardown was too fast to race with interpreter shutdown, and no check looked at a child's exit status.

## Defect found by Stage 1 run 2 (RTX 3090, 2026-11-09; INVALID)

`edge-fit` crashed with `ProcessLookupError` in `os.killpg` when a vLLM server had already exited at start-up
(the expected memory FAIL of Qwen3-14B-AWQ and Qwen3-8B at 12 GB), so no record was written. The same path
classified failures from a log tail that missed vLLM's KV-capacity error. Root cause and fix: `HARNESS_FIXES.md`
H2. The local suite had missed it: edge-fit needs NVML and a vLLM binary, so it had never run locally. It now
runs against a stand-in vLLM and a fake NVML.

## Defects found and fixed by the self-test (before any hardware)

1. `Sampler._stop` shadowed `threading.Thread._stop`. Every NVML memory/power measurement would have crashed
   on the GPU host. Renamed.
2. `HostBuffers.d2h` silently **cast** non-bf16 activations to bf16. E0 §7 forbids any dtype change on the cut
   tensor. It now raises.
3. `Profile.pre` failed on scalar input. Fixed (the vectorised R path was unaffected).
4. `from_pretrained(device_map=…)` needs `accelerate`, which is now pinned. Without it, a 65 GB model would be
   staged through host RAM.
5. The lower-bound server profile added one decode step to TTFT, so it could over-count. The bound is now strict.
6. The wire header is 72 bytes, not 64. It is a spec constant checked against the runtime struct.

## Local measurements (CPU loopback; not E0 data)

- `linkcheck` on loopback: per-token round trip p50 0.22 ms (10 KB each way, including Python serialisation);
  bulk ≈ 13.9 Gbps. This is the protocol floor on this host only.

## Informational prior (`frozen/R_prior_s1_roofline_overhead{0,10,20}ms.md`)

The S1 per-query data with spec-sheet roofline profiles (PREREG §12):

| Extra SPLIT overhead per token | Cells passing (of 60, good state) |
|---|---|
| 0 ms | 5, all at 12 GB no-raw-text |
| 10 ms | 2, both at 12 GB no-raw-text |
| 20 ms | 2, both at 12 GB no-raw-text |

16 GB, 24 GB and `open` fail everywhere. This is not a result.

## Not validated locally. Stage 1 is the first CUDA validation (PREREG §9.5)

These checks are encouraging but are CPU-only. They do **not** validate CUDA, the allocator cap, pinned transfers,
NVML or any Qwen3-32B execution. The pod self-test runs the same CPU checks. **Stage 1's first `split-fit` is the
first CUDA execution of the harness**, and a non-memory failure there is a harness defect (PREREG §5.6), not an
F result.


- The CUDA allocator cap (`set_per_process_memory_fraction`), pinned-memory non-blocking copies, CUDA-event
  timing and NVML on a real device.
- Edge memory **measurements**: the analytic model is a prediction only.
- vLLM 0.8.5.post1 itself: the `edge-fit` launch, `min_tokens` / `ignore_eos` / `continuous_usage_stats` on the
  real server (tested only against a fake SSE server), and fp8 KV for Qwen3-32B-AWQ on the edge GPU.
- `tc netem` / `ifb` application (commands and parsing only).
- Cross-host numerics (edge GPU ≠ server GPU). Locally everything is one CPU, hence bitwise equality. The Q floors
  exist to measure exactly this on hardware.
- Pod networking (RTT, bandwidth, reachability) and H200 co-residency of vLLM + the segment server.
