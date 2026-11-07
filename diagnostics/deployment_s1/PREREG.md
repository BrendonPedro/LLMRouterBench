# S1 / E0-Q3 pre-registration: server-only quality gate

Frozen 2026-11-07, before any generation. Amended once on 2026-11-07, also before any generation
(**A1**, §10: `presence_penalty=1.5` for the three AWQ models). **No S1 generation has been run.**

Sources of the design (not changed here):
- `~/dev/cloud_slm-router/deployment-pivot/E0-SPLIT-VIABILITY.md`: §3.1 models and the mode amendment, §4.3 Q3
  settings, §4.4 sample, §8 early stop.
- `~/dev/cloud_slm-router/deployment-pivot/WEEK-EXECUTION-PLAN.md`: S1, the harness-parity rule (Mon 11:00), gate G1.

This file pins the implementation choices E0 leaves open. Every choice is listed in §9 together with its
expected direction of bias.

## 1. Question and decision

S1 asks whether Qwen3-32B bf16 (SPLIT's quality, once Q passes) is at least 3.0 pt more accurate than the best model
that fits fully on the edge at 12, 16 and 24 GB.
- It is measured per benchmark and per mode, on the server, and touches nothing of the split runtime.
- It feeds the pre-declared early stop (E0 §8, gate G1).
- It is not the R test. R needs on-time utility, latency profiles and the paired bootstrap per cell.

## 2. Frozen sample

`sample_200.jsonl`, sha256 `2c4df49360f997c3e35593030da40111c2826f068f03b1401284a394d9716955` (pinned in
`run_s1.py`; `sample_manifest.json` describes the draw).

| Benchmark | n | Frame | Cached Qwen3-8B accuracy on the sample |
|---|---|---|---|
| MMLU-Pro (`test_1000`) | 60 | 1000 (1001 records; index 338 is an exact duplicate of 16 and is dropped) | 0.617 |
| GPQA-Diamond | 60 | 198 | 0.600 |
| LiveMathBench | 40 | 121 | 0.750 |
| LiveCodeBench | 40 | 1055 (release_v6) | 0.625 |

**Procedure.**
1. The frame is the cached LLMRouterBench Qwen3-8B record file of each benchmark (§5). Later copies of a duplicated
   prompt are dropped.
2. The frame is sorted by index and shuffled with `random.Random("deployment-s1:20261107:<dataset>")`.
3. The first n records are taken.

**Stable key.** `<dataset>:<index>`.
- `index` is the 1-based position in the evaluator's load order, the same as in `results/bench`.
- `check` verifies that, for every sampled key, all 109 cached model files of the three answer-keyed benchmarks
  carry the same prompt and gold answer.

**Extension.** The pre-declared extension to 400 (E0 §4.4) takes the first 2n of the same permutation, so it
contains the 200 (`check` verifies this).

## 3. Models and memory classes

Model IDs are the official Qwen repositories. The Qwen3 AWQ cards and the E0 desk check name them. Revisions are the
Hugging Face `main` SHAs on 2026-11-07, and the checkpoint sizes come from the HF tree API.

| Action | Repo @ revision | Serving dtype | Weights | Role |
|---|---|---|---|---|
| Qwen3-32B | `Qwen/Qwen3-32B` @ `9216db5781bf21249d130ec9da846c4624c16137` | bf16 | 65.5 GB | reference (REMOTE / SPLIT quality) |
| Qwen3-8B | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` | bf16 | 16.4 GB | edge at 24 GB; harness parity |
| Qwen3-8B-AWQ | `Qwen/Qwen3-8B-AWQ` @ `4da05a8edb55c6046cce958586c33b61da07bb79` | fp16 (`auto`) | 6.1 GB | edge at 12/16/24 GB |
| Qwen3-14B-AWQ | `Qwen/Qwen3-14B-AWQ` @ `31c69efc29464b6bb0aee1398b5a7b50a99340c3` | fp16 (`auto`) | 10.0 GB | edge at 16/24 GB |
| Qwen3-32B-AWQ | `Qwen/Qwen3-32B-AWQ` @ `0499c3ac83fdef8810b907a23894ba91e95eddd8` | fp16 (`auto`) | 19.3 GB | edge at 24 GB (FP8 KV) |

The AWQ checkpoints declare `torch_dtype: float16` and AWQ 4-bit with group size 128. They are served with
`--dtype auto`.

**Edge eligibility.** The default eligibility is the E0 §1 desk check: checkpoint + bf16 KV for 16k tokens + 1.5 GB.

| Memory class | Eligible |
|---|---|
| 12 GB | Qwen3-8B-AWQ (10.0 GB) |
| 16 GB | + Qwen3-14B-AWQ (14.2 GB) |
| 24 GB | + Qwen3-8B bf16 (20.3 GB), + Qwen3-32B-AWQ (22.9 GB with FP8 KV, as E0 counts it) |

S6's measured fit results replace this table when they exist (`summarize_s1.py --fit-json`). No other change is
allowed.

## 4. Engine, decoding, prompts and scoring

**Engine.**
- vLLM `0.8.5.post1`, run as an OpenAI-compatible server, one model at a time.
- Flags: `--max-model-len 32768 --seed 0 --gpu-memory-utilization 0.90`, plus `--tensor-parallel-size` 1 (or 2 on
  2×80 GB).
- Everything else is the engine default: prefix caching on, `--generation-config auto`, no reasoning parser. With
  no reasoning parser, the thinking trace stays inline in `content`, as in the cached LLMRouterBench outputs.
- The client sends every sampling field explicitly, so no `generation_config.json` default changes a request.

**Requests.**
- One user message holding the exact cached LLMRouterBench prompt string, with no system prompt. This matches
  `DirectGenerator`.
- Each request also carries `chat_template_kwargs.enable_thinking` and a per-request `seed`.

| Setting | enable_thinking | T | top_p | top_k | min_p | max_tokens | Seeds |
|---|---|---|---|---|---|---|---|
| S1 thinking | true | 0.6 | 0.95 | 20 | 0 | 16384 | 0, 1 |
| S1 non-thinking | false | 0.7 | 0.8 | 20 | 0 | 16384 | 0, 1 |
| Harness parity (Qwen3-8B only) | true | 0.2 | 1.0 | 20 | 0 | none (32768 total cap) | 0 |

`presence_penalty` is set per model, not per setting (amendment A1, §10):
- **AWQ models** (Qwen3-8B-AWQ, Qwen3-14B-AWQ, Qwen3-32B-AWQ): `presence_penalty=1.5` is sent with every request,
  in both modes and in the smoke run.
- **bf16 models** (Qwen3-32B, Qwen3-8B, including harness parity): `presence_penalty` is not sent, so it stays at
  the vLLM default 0.

Every other field in the table is identical for AWQ and bf16 models.

**Matrix.** 5 models × 2 modes × 2 seeds × 200 queries = 4,000 generations.

**Client.**
- The client is plain HTTP and records `finish_reason`, `usage` and the latency of each request.
- `DirectGenerator` is not used, for two reasons. It discards `finish_reason`, which the truncation rate needs.
  It also turns failures into "Generation failed" text that would then be scored.
- A failed request is stored as `status: error` and never scored. Only crashed requests are retried, never
  low-scoring ones (at most 5 attempts per key).

**Scoring.**
- Scoring calls each benchmark's unchanged `evaluate(data, output_text)`. The score conversion is the same as in
  `data_collector/runner.py`.
- This includes outputs truncated at 16,384 tokens, exactly as LLMRouterBench scored its capped outputs.

**Data.** The repo's `data/` files are Git-LFS stubs. The LFS budget of both `origin` and `upstream` is exhausted
(HTTP 403, checked 2026-11-07). So the evaluator inputs come from elsewhere:
- **MMLU-Pro, GPQA, LiveMathBench:** these evaluators read only `item["answer"]`. That value is taken from the
  cached record's `ground_truth`. Re-scoring the cached Qwen3-8B outputs with it reproduces 160/160 sampled scores
  and every score in the full files (1001, 198 and 121). LiveMathBench differs from the cached file in 5 `prediction`
  strings, all through the later last-integer fallback, with no score change.
- **LiveCodeBench:** the tests come from `livecodebench/code_generation_lite` @ `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`.
  These are `test.jsonl`…`test6.jsonl`, sha256-pinned, 1055 rows in `sorted(*.jsonl)` order. `check` verifies
  that the rebuilt prompts equal the cached prompts. Re-executing the cached LiveCodeBench code (`check --execute-lcb`)
  runs on the pod only and is reported, not gated, because execution timeouts depend on the host.
- **Cached outputs:** these come from `NPULH/LLMRouterBench` `bench-release.tar.gz`, sha256 `b79f8cde…c8f1`. Its members
  are byte-identical to `results/bench`.

**Per-query record** (`runs/s1/scores.jsonl`; exported as `q3_scores.jsonl`): dataset, index, model, mode, seed,
score, completion_tokens, prompt_tokens, finish_reason, truncated, prediction, ground_truth, and think-block flags.
The raw outputs stay in `generations.jsonl`.

## 5. Harness parity (validation only; decides nothing about SPLIT)

**Run.** Qwen3-8B bf16 on the 200 queries at the cached run's settings (§4 table).

**Rule** (WEEK-EXECUTION-PLAN). Both parts must hold:
- For every benchmark, the new accuracy must lie inside the Wilson 95% CI of the cached accuracy on the same queries.
- Per-query score agreement must be ≥ 80% on MMLU-Pro and GPQA.

**If it fails:**
- Fix the harness (prompt, chat template, answer extraction) and re-run parity. This does not use up E0's one Q fix.
- Or record the residual gap as a stated limitation: `run --accept-parity-limitation "…"`, written to
  `invocations.jsonl`.

`run` refuses to start without parity PASS or that recorded limitation, and without a smoke PASS.

The cached run's engine, revision and top_k are reconstructions (paper: vLLM 0.8.4; generation_config top_k=20).
They are not recorded facts.

## 6. Analysis (`summarize_s1.py`)

**Accuracy.** `acc(model, mode, benchmark)` = mean score over the benchmark's queries and both seeds, in %.

**EDGE-best(M).** For each benchmark, mode and memory class M, EDGE-best(M) is the eligible model with the highest
accuracy in the same mode. Ties go to the smaller model.

**Gap.** `gap(M) = acc(Qwen3-32B) − acc(EDGE-best(M))`, in percentage points. A paired-bootstrap 95% CI over queries
(10,000 resamples, seed 20261107) is reported as description only.

**Early stop** (E0 §8, applied per mode, strict `< 3.0` on point estimates):

| Condition | Decision | Consequence |
|---|---|---|
| some gap(16) ≥ 3.0 | `CONTINUE` | the 16/24 GB split regimes stay; G1 passes |
| every gap(16) < 3.0 (4 benchmarks × 2 modes) | `DROP_16_24` | remove the 16/24 GB regimes; only the 12 GB class and the 6 core E0-L cells |
| every gap(16) < 3.0 **and** every gap(12) < 3.0 | `STOP_NO_HEADROOM` | SPLIT has no quality headroom: R fails by the S1 bound; stop before split-runtime work; `e0/DECISION.md`; branch B |

- **Incomplete data.** No decision is issued unless all 4,000 generations are scored. `--allow-incomplete` gives a
  labelled `PROVISIONAL_*` view only.
- **24 GB.** All gaps at 24 GB < 3.0 is reported for information. It is not a pre-registered rule.

The summary also reports:
- truncation rate and mean completion tokens per model × mode × benchmark (E0 §4.3);
- template sanity: thinking outputs must open `<think>`, and non-thinking outputs must not.

## 7. Order of operations on the pod

`preflight` → `fetch-data` → `check --hash-lcb --execute-lcb` → `smoke` → `parity` → `run` → `score` →
`summarize_s1.py`. Exact commands are in `RUNPOD.md`.

The full run order is Qwen3-32B → 8B-AWQ → 14B-AWQ → 32B-AWQ → 8B. This puts the early-stop-deciding models first.

## 8. Not done here

- No split runtime, Q1/Q2, E0-L latency cells or R computation.
- No Qwen3-30B-A3B (E0 §3.1 adds it only if R passes).
- No FP8-KV variant run.
- No greedy decoding.
- No change to thresholds, sample, models or decoding after data are seen.

## 9. Known deviations and biases (stated before data)

1. **Presence penalty differs by precision (amended by A1, §10).** AWQ models get `presence_penalty=1.5`, as their
   model cards recommend for quantised models. bf16 models get 0. E0 freezes only T, top_p and top_k.
   - *Direction:* see §10. The original bias (all models at 0, which favours SPLIT) is removed. A smaller residual of
     unknown sign remains.
   - *Visibility:* the per-model truncation table and mean completion tokens show it.
2. **FP8 KV is not measured.** Qwen3-32B-AWQ counts at 24 GB under E0's FP8-KV assumption, but S1 serves it with an
   fp16 KV cache.
   - *Direction:* it slightly flatters the 24 GB competitor.
3. **Selection effect.** EDGE-best is selected on the same queries it is scored on.
   - *Direction:* it slightly flatters the edge (towards the early stop). With 1–4 candidates per class the effect is
     small, and the per-model gaps are reported alongside.
4. **Engine version.** The engine is vLLM 0.8.5.post1, the first version the AWQ cards support. The cached run is
   reconstructed as 0.8.4. Harness parity checks whether that difference matters.
5. **Truncated outputs are scored by the unchanged evaluators.** The LiveMathBench last-integer fallback can credit a
   truncated trace.
   - *Visibility:* reported through the truncation rate. The truncated outputs are not re-scored.
6. **Seed reproducibility.** A per-request seed does not make batched vLLM sampling bit-reproducible. The two seeds
   are two samples, not a determinism claim.
   - *Visibility:* the smoke test reports a same-seed probe.
7. **Hardware.** The run uses a rented RunPod server GPU (H200 141 GB, or 2×H100 80 GB), at the user's direction on
   2026-11-07. Accuracy does not depend on GPU type (E0 §4.3), and the engine, versions and `pip freeze` are recorded
   per server launch.
8. **Evaluator inputs** come from cached records and the public LiveCodeBench release instead of the LFS data files
   (§4). This is verified as described there.

## 10. Amendments (pre-data)

### A1 (2026-11-07): `presence_penalty=1.5` for the three AWQ models

**Status.** Made before any S1 generation (smoke, parity or run) and before any S1 output was seen.

**Change.** Every request to Qwen3-8B-AWQ, Qwen3-14B-AWQ and Qwen3-32B-AWQ carries `presence_penalty=1.5`. This
covers both modes, both seeds and the smoke run. The bf16 models (Qwen3-32B, Qwen3-8B, including harness parity) are
unchanged: `presence_penalty` is not sent, so it stays at 0.

**Unchanged.** Every other sampling field (T, top_p, top_k, min_p, max_tokens, seeds, enable_thinking), the sample,
the models and revisions, the engine and flags, the metric, the 3.0 pt threshold and the early-stop rules.

**Reason.** The official Qwen3 AWQ model cards strongly recommend `presence_penalty=1.5` for the quantised models.
The frozen design sent 0 to every model. That serves the AWQ models outside their recommended configuration.

**Direction of the bias removed.** Without the penalty, quantised models are more prone to repetition loops. In
thinking mode with a 16,384-token cap, a loop becomes a truncated, usually wrong answer. That would lower AWQ
accuracy and so lower EDGE-best at every memory class: all 12 and 16 GB candidates are AWQ, and so is one of the
two at 24 GB. The result:
- the measured gap `acc(Qwen3-32B) − acc(EDGE-best)` would widen;
- `CONTINUE` would become more likely and `DROP_16_24` / `STOP_NO_HEADROOM` less likely.

The quality-gap test would therefore be biased in favour of SPLIT, the outcome S1 is meant to be able to rule out.
That is a one-sided bias against the early stop.

**Residual bias after A1 (sign unknown, stated before data).**
- **Towards the early stop (against SPLIT).** The bf16 reference stays at 0. Any repetition or truncation it has at 0
  lowers `acc(Qwen3-32B)` and narrows the gap. The bf16 model cards make the penalty optional, not recommended, so
  0 is kept as frozen.
- **Towards SPLIT.** A presence penalty also discourages tokens a correct answer legitimately repeats (code
  identifiers, mathematical notation, answer formats). Where that costs accuracy, AWQ accuracy falls and the gap
  widens.
- **24 GB comparison.** The candidates are Qwen3-8B bf16 (at 0) and Qwen3-32B-AWQ (at 1.5). Which of them is
  EDGE-best(24) can depend on A1. The 24 GB outcome is information only (§6).
- *Visibility:* the per-model × mode × benchmark truncation rate and mean completion tokens (§6) are reported for
  every model. AWQ-vs-bf16 differences in truncation can be read directly. They are not used to re-score or to
  change any decision.

**Enforcement** (`run_s1.py`).
- `MODELS[...]["presence_penalty"]` is the single source of the value, and `build_request` adds it only for the AWQ
  models.
- `check` verifies every planned request (s1, parity and smoke): 1.5 on AWQ, absent on bf16, and all other fields
  equal to the frozen settings.
- `selftest` checks the request bodies and that the fake server receives 1.5 on every AWQ request.
- Resuming over any existing `ok` record whose stored request differs from the current one (for example, one
  generated before A1) is refused before a request is sent, so pre- and post-amendment generations cannot be mixed.
