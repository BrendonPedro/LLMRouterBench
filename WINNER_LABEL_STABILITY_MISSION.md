# Winner-Label Stability Experiment

## Status

PREPARED, NOT RUN. No inference has been performed and no results exist.
Branch: `winner-label-stability-diagnostic`, based on `c66b9e7`.
Model pair (exact names): `DeepSeek-R1-0528-Qwen3-8B` (class 0) vs `Qwen3-8B` (class 1).

**Reconstructed evaluation, not exact reproduction.** The original serving endpoint, model revision, precision and serving engine are not recorded (see Audit §3–5). Fresh inference will therefore run in a **reconstructed evaluation setup**: the stored prompts verbatim, the known decoding settings (temperature 0.2, top_p 1.0, thinking on, no system prompt, 32,768-token total context) and the unchanged repository evaluators. It is **not** an exact reproduction of the original serving environment. All results must be described that way.

**Reproducibility checkpoint (2026-11-02, no inference):**
- **KorBench evaluator reproduction is complete:** 100% agreement with the stored scores.
- **MBPP prompt reconstruction is complete:** 1,948/1,948 prompts match, and 820/820 non-tie programs are AST-identical.
- **MBPP score reproduction remains blocked** until the stored generated code can be run in a safe sandbox (§7a, "Blockers").
- **The generation cache must be disabled** for every fresh run.
- **No main winner-label stability inference may begin** until the remaining blockers are cleared and the user approves the run.
- **The frozen main experiment design has not changed:**
  - model pair, six benchmarks, original non-ties only, ~20 queries/benchmark;
  - 5 repetitions per query per model, with the 4/5 winner-stable criterion;
  - original decoding and prompt policy, Cases A–D.

  This checkpoint revised only the pre-flight validation, the reproducibility wording and the estimate labels.

---

# Research Question

Is the per-query "winning model" label stable when the same query is evaluated several times? Or is generation/evaluation variability contributing to the near-chance within-benchmark routing results?

Prior evidence (`diagnostics/within_task_routing/WITHIN_TASK_ROUTING_REPORT.md`):

| | Value |
|---|---|
| MiniLM overall AUC | ≈ 0.670 |
| Capability+MiniLM overall AUC | ≈ 0.673 |
| MiniLM within-benchmark macro AUC | ≈ 0.501 |
| Capability+MiniLM within-benchmark macro AUC | ≈ 0.512 |
| Benchmark-identity diagnostic reference Macro-F1 | ≈ 0.613 |

Clear within-task signal appears only on BBH, KorBench and MBPP. MELD, MedQA and FinQA are near chance.

Do not modify any previous diagnostic result.

---

# Phase 1 Audit Findings (done before this design; no inference)

## 1. Where the original query text is stored

- **`results/bench/<benchmark>/test/<model>/<benchmark>-test-<model>-<timestamp>.json`**, one file per model per benchmark. Each record has `index`, `origin_query`, `prompt` (the exact text sent to the model), `raw_output`, `prediction`, `ground_truth`, `score`, `prompt_tokens` and `completion_tokens`.
- The routing pool, `baselines/AvengersPro/data/small_models_seed_42/seed42_split0.7/{train,test}.jsonl` (train ∪ test = full pool; joined to the frozen folds via `diagnostics/scripts/run_cv_eval.load_full_pool_with_folds`), stores `query` = the stored `prompt`.
- Verified on all rows of the six benchmarks:
  - pool `query` == stored `prompt`: 100%;
  - stored `prompt` is identical for both models: 100%;
  - stored prompts begin with the current evaluator `PROMPT` template for BBH, MBPP, MELD, MedQA and FinQA. KorBench builds prompts from per-task YAML templates, so there is no single constant to check.
- The raw benchmark files under `data/<Benchmark>/` are **Git LFS pointer stubs** (`.gitattributes`: `data/** filter=lfs`), and `git-lfs` is not installed. For KorBench/MBPP the required objects were fetched outside the working tree on 2026-11-02 (§7).

## 2. Where the original outputs / correctness labels are stored

- They are in the same `results/bench/...` JSON files: `raw_output`, `prediction` and binary `score`.

| Benchmark | DeepSeek-R1-0528-Qwen3-8B file | Qwen3-8B file |
|---|---|---|
| bbh | `bbh-test-…-20251004_152248.json` | `bbh-test-Qwen3-8B-20250919_071923.json` |
| korbench | `korbench-test-…-20251004_184641.json` | `korbench-test-Qwen3-8B-20250919_102827.json` |
| mbpp | `mbpp-test-…-20251004_235947.json` | `mbpp-test-Qwen3-8B-20250919_132029.json` |
| meld | `meld-test-…-20251019_064225.json` | `meld-test-Qwen3-8B-20251019_064159.json` |
| medqa | `medqa-test-…-20251019_064141.json` | `medqa-test-Qwen3-8B-20251019_064130.json` |
| finqa | `finqa-test-…-20251004_163738.json` | `finqa-test-Qwen3-8B-20250919_081013.json` |

- Routing-pool scores equal the stored `score` on **every non-tie row** of the six benchmarks.
- There is one disagreement in the whole six-benchmark pool, on a tie row: KorBench index 222. The pool has DeepSeek = 1, which makes it a tie; the stored file has DeepSeek = 0. This row is excluded by the sampling rule below.

## 3–4. How the two models were originally invoked

- Both models went through the same path: `data_collector` → `generators/generator.py::DirectGenerator`, which uses an OpenAI-compatible `chat.completions.create` with
  `messages=[{"role": "user", "content": <prompt>}]`. There is **no system prompt** and the model's **default chat template** is used.
- Config: `config/data_collector_small_model_config.yaml`. The `api_model_name` values are exactly `DeepSeek-R1-0528-Qwen3-8B` and `Qwen3-8B`. `base_url` is redacted (`https://url.here/v1`) and `api_key: HUOSHAN_API_KEY`, so **the actual serving endpoint is not recorded**.
- `vllms/run_vllm.sh` (vLLM OpenAI server, `--seed 42`, `--max-model-len 8192`) is **not** what produced these files: the stored outputs reach prompt + completion = **32,768 tokens**. The context limit actually used was 32,768.
- **Thinking mode was on for both models:** 100% of stored outputs contain `<think>`.
  - Qwen3-8B hits the 32,768-token cap with unterminated thinking on 0.4–4.1% of non-tie rows, depending on the benchmark.
  - DeepSeek hits it on ≤ 0.7%.
- Collection dates: Qwen3-8B 2025-09-19 (MELD/MedQA 2025-10-19); DeepSeek 2025-10-04 (MELD/MedQA 2025-10-19).
- Measured original throughput (from the stored `time_taken`, concurrency 32) was about 1,200–1,640 completion tokens/s.

## 5. Original generation settings

| Setting | Value | Source / confidence |
|---|---|---|
| temperature | **0.2** (both models) | collector config; known |
| top_p | **1.0** (both models) | collector config; known |
| max tokens | **not set by the client**, so the server default applied; observed total cap = 32,768 tokens | inferred from stored token counts |
| seed | **no per-request seed**. `run_vllm.sh` sets a server seed of 42, but that script's other settings do not match the data, and a server seed does not make concurrent sampled requests reproducible | known / inferred |
| chat template / system prompt | model-default template, single user turn, no system prompt, thinking enabled | code + stored outputs |
| reasoning_effort / extra_body | not set for either model | collector config |
| model revision / precision / serving engine version | **unknown** | not recorded |

**The original setup was NOT deterministic.** It used temperature 0.2 sampling, and the original label is a single draw from each model's output distribution.

## 6. How answers are evaluated (`evaluation/<Benchmark>/`)

All six evaluators are deterministic and rule-based. None calls an LLM judge.

| Benchmark | Rule |
|---|---|
| BBH | Extract the `Answer:` line (or `\boxed{}`); grade with `grade_answer_mathd` or `grade_answer_sympy` against `target` |
| KorBench | `extract_single_answer` + `evaluate_response_vs_answer`, depending on `question_type`, `rule_id` and `idx` |
| MBPP | Sanitize the code, append `test_list`, execute with `check_correctness` (10 s timeout). **This runs model-generated code** |
| MELD | Extract the `Answer:` letter; exact match |
| MedQA | Extract the `Answer:` letter; exact match |
| FinQA | Extract `\boxed{}`; grade with mathd or sympy, plus a rounded-number fallback |

## 7. Can the exact original evaluator be reused?

- **Yes for BBH, MELD, MedQA and FinQA.** Re-scoring every stored non-tie `raw_output` for both models with the current evaluator and the stored `ground_truth` reproduces the stored score in **100% of cases**:

  | Benchmark | Outputs re-scored |
  |---|---|
  | BBH | 270 |
  | MELD | 502 |
  | MedQA | 378 |
  | FinQA | 254 |

- **KorBench and MBPP need dataset fields the stored records do not keep.** KorBench needs `question_type`, `rule_id` and `idx`; MBPP needs `test_list` and `task_id`. Those fields live in the LFS-stubbed `data/` files.

### 7a. Data dependency resolution (2026-11-02, no inference)

**Exact LFS files required** are the ones the unchanged evaluators' `load_data()` reads (17 files, 2.07 MB). Nothing else under `data/` is needed:

| Benchmark | Files |
|---|---|
| KorBench | `data/KORBench/three-shot.yaml`, plus `data/KORBench/{cipher,operation,puzzle,counterfactual,logic}/{sample,three-shot,rule}.jsonl` |
| MBPP | `data/MBPP/test.json` |

- `data/KORBench/mixed/*` and `cipher/subquestions.json` are not read by the evaluator.
- **How the files were obtained:**
  - Downloaded through the GitHub LFS batch API of `origin` (`BrendonPedro/LLMRouterBench`; `upstream` serves the same objects).
  - Each object was **verified against the committed pointer's sha256 oid and size** (17/17 OK).
  - The copy is stored **outside the working tree**: without `git-lfs`, overwriting the pointer stubs would show 17 modified tracked files. The working tree and index are unchanged.
  - For the actual run, re-fetch these 17 objects the same way, or with `git lfs pull --include=...` once git-lfs is installed, and re-verify the oids.
- **Mapping:** the stored `index` is the 1-based position in the evaluator's `load_data("test")` order (`data_collector/runner.py`). The mapping was confirmed by rebuilding every prompt with the unchanged evaluator code. The only change was pointing `DATA_DIR` at the verified copy.

**KorBench: reproduced exactly.**
- Rebuilt prompt == stored `prompt`: 2,500/2,500 (both models, 1,250 items).
- Re-scoring stored `raw_output` with `KORBenchEvaluator.evaluate` gives an exact match of stored `score` **and** stored `prediction`:
  - **506/506 (100%)** on the 253 pool non-tie rows, both models;
  - 2,496/2,496 on all 1,248 pool rows.

**MBPP: reproduced up to, but not including, code execution.**
- Rebuilt prompt == stored `prompt`: 1,948/1,948.
- The program the evaluator would execute (`extract_raw_answer`: sanitize + imports + `test_list`) was compared with the stored `prediction` field, which is that program:
  - byte-identical in 793/820 non-tie outputs;
  - **AST-identical in 820/820 (100%)**.
- The 27 byte differences are purely `ast.unparse` formatting, e.g. `(a, b) = …` vs `a, b = …`. This is consistent with a different Python version at collection time; the current version is 3.12.3. AST-identical programs execute identically, so this is not a reproducibility problem.
- **Score agreement is NOT verified.** It needs `check_correctness`, i.e. running model-generated code. No appropriate sandbox exists on this machine:
  - Docker Desktop's WSL integration is off;
  - there is no podman, bwrap, firejail, nsjail or gVisor;
  - `unshare -rn` would remove the network but not filesystem access.
- The evaluator's `reliability_guard` is not a security boundary. **Generated code was therefore not executed.** This is a remaining blocker.

## 8. Are the models accessible for fresh inference here?

**No.**
- There is no GPU (no `nvidia-smi`, `torch.cuda.is_available() == False`) and `vllm` is not installed.
- No local weights exist for either model; the HF cache holds only embedding and encoder models.
- No API credentials are set in the environment.
- The machine has 4 CPU cores and 9.7 GB RAM, which is below the ~16.4 GB needed for bf16 8B weights.

## 9. Generation count and resources

See "Estimated Generation Count and Resources" below.

---

# Frozen Sample Strategy

## Sampling frame

1. Start from the full routing pool: `load_full_pool_with_folds()`, 11,467 rows, with the frozen folds attached.
2. Keep rows whose `dataset` ∈ {`bbh`, `korbench`, `mbpp`, `meld`, `medqa`, `finqa`}.
3. Keep **original non-ties only**: `target` ∈ {0, 1} (0 = DeepSeek-only correct, 1 = Qwen-only correct).
4. Keep only rows whose pool scores equal the stored `results/bench` scores for both models. Currently every non-tie row passes; the rule is declared in advance.

Available non-ties (DeepSeek wins / Qwen wins):

| Benchmark | Group | DeepSeek wins | Qwen wins |
|---|---|---|---|
| bbh | signal | 97 | 38 |
| korbench | signal | 140 | 113 |
| mbpp | signal | 109 | 301 |
| meld | chance | 106 | 145 |
| medqa | chance | 110 | 79 |
| finqa | chance | 40 | 87 |

Every benchmark has ≥ 10 of each class, so **the target balance of 10 + 10 is achievable everywhere** and no imbalance fill-in is needed. If the frame ever changes, the fallback rule is: take all of the scarcer class and fill the rest from the other class, then report the imbalance.

## Selection procedure (deterministic)

- Sort the frame by (`dataset`, `index`).
- Create **one** generator, `rng = numpy.random.default_rng(20261101)`.
- Iterate benchmarks in the fixed order `bbh, korbench, mbpp, meld, medqa, finqa`. Within each benchmark, take class 0 (DeepSeek wins) and then class 1 (Qwen wins). For each, draw **10 rows without replacement** with `rng.choice(n_class, size=10, replace=False)` over the class's sorted rows.
- Total = **6 × 20 = 120 queries**: 60 signal, 60 chance; 60 original DeepSeek wins, 60 original Qwen wins.
- Save the sample, with `dataset`, `index`, `query_sha1`, `fold` and original target, **before any generation** to `diagnostics/winner_label_stability/frozen_sample.csv`, and do not change it afterwards.
- Benchmarks, sample size and the sample itself must not change after any stability result is seen.

## Pre-flight sample (separate from the main sample; NOT part of the original stability design)

The pre-flight was added during design preparation. It is **not** part of the pre-declared stability design (main sample, 5 repetitions, 4/5 criterion, Cases A–D), and it does not change any of it. See "Pre-flight validation" below for why it is kept.

- 20 queries per benchmark (120 total) drawn **from all rows including ties**, excluding the main sample, with `numpy.random.default_rng(20261102)`, same benchmark order, by simple random sampling.
- Saved to `diagnostics/winner_label_stability/preflight/preflight_sample.csv` before any generation.
- Purpose: validate the reconstructed setup, not to measure stability.

---

# Repetition Count

- **Main experiment:** 5 fresh evaluations per query per model.
- **Pre-flight:** 1 evaluation per query per model.
- The original stored outputs are **not** counted as one of the 5 repetitions. The original label is kept separately as the reference.

---

# Generation-Setting Policy

This is a **reconstructed evaluation setup**. It matches the recorded prompts, the known decoding settings and the unchanged evaluators. The original serving environment (endpoint, model revision, precision, engine) is unknown, so it **cannot be exactly reproduced**. The settings below that the audit did not recover (weights source, dtype, engine) are reconstruction choices, not known original values. **Do not introduce or raise temperature to manufacture variability.**

- **Prompt:** the stored `prompt` string, verbatim, as a single user message. No system prompt.
- **Chat template:** the model's default template with thinking enabled (as observed). Do not pass `enable_thinking=False`.
- **Sampling:**
  - temperature = 0.2, top_p = 1.0 for both models;
  - no other sampling parameters (no top_k, min_p or repetition penalty beyond engine defaults);
  - no `reasoning_effort`, no `extra_body`.
- **Length:** total context limit 32,768 tokens. Do not set a smaller `max_tokens`. If the client must set one, use `max_tokens = 32768 − prompt_tokens`. An output that hits the cap is scored by the evaluator exactly as in the original run (usually wrong), and its finish reason is recorded.
- **Weights (reconstruction choice; the original revision and precision are unknown):** official checkpoints `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` and `Qwen/Qwen3-8B`, bf16, no quantization, served under the exact names above.
  - Record the model revision hash, engine name and version, dtype and GPU type. For a hosted API, record the provider, the model identifiers and whatever revision/precision the provider discloses.
- **Seeds:** the original used no per-request seed. Each repetition is an independent sample. Setting an explicit per-request seed for audit reproducibility, e.g. `seed = int(sha1(f"{dataset}|{index}|{model}|{rep}")[:8], 16)`, is allowed because it does not change the sampling distribution. Record whichever choice is made.
- **CACHE MUST BE DISABLED.** The collector's cache key is (`model`, `temperature`, `top_p`, `messages`, `reasoning_effort`). With caching on, repetitions 2–5 would return the cached first response, and stability would look like 100% purely as an artefact. Set `cache.enabled: false` (or `force_override_cache: true` with distinct keys) and verify that the 5 raw outputs per query/model are not byte-identical copies coming from a cache.
- Run each repetition as a separate request. The 5 × 2 requests for a query may be interleaved in any order.
- Store every raw output, prompt/completion tokens, finish reason, timestamp and score.

## Pre-flight validation (required before the main run; revised 2026-11-02, before any inference)

**Why it is needed.**
- In the main experiment, original-winner retention compares fresh draws from the *reconstructed* setup against a single draw from the *original, unrecorded* setup.
- If the reconstruction differs materially (wrong weights, thinking off, a shorter length cap, a template change, a quantized provider), per-query accuracy shifts. Those shifts show up as "flips" and "tie emergence", which are indistinguishable from genuine label instability. They could push the result toward Case C or D for reasons that have nothing to do with label noise.
- The main sample cannot detect this. It is selected on the original outcome (exactly one model correct), so regression to the mean already moves fresh accuracy away from the original by an unknown amount.
- An unbiased original-vs-fresh accuracy comparison needs a sample **not** selected on the outcome. The pre-flight sample is that sample.
- The pre-flight is therefore scientifically necessary as a validity check on the reconstruction. It is not a stability measurement.

**What it validates, and what it does not.**
- It validates:
  - that the reconstructed setup runs as specified (operational checks below);
  - that per-model marginal accuracy is not grossly shifted from the original on outcome-unselected queries.
- It does **not** establish equivalence of per-query output distributions. A stack difference that moves *which* queries are solved without changing marginal accuracy cannot be detected by any check available here.
- Main-run results are therefore stability *of the reconstructed setup relative to the original single draw*. The between-repetition `modal agreement` (already defined below) is the stack-independent descriptive companion.

**The previous ±7 pp rule is withdrawn.**
- It was never part of the original design, and it was not calibrated.
- Under a correctly reconstructed stack, its false-stop probability depends on the unknown per-query discordance rate *r* between two independent draws. That rate is itself what this experiment studies.
- Exact enumeration at n = 120 gives a false-stop probability per model of 0.015 / 0.082 / 0.156 / 0.220 for *r* = 0.1 / 0.2 / 0.3 / 0.4. Across two models that is up to roughly 0.39.
- A fixed accuracy-difference threshold therefore cannot be justified before inference.

**Operational checks (hard stop on any failure; no statistics):**
1. Served model identifiers match the two exact names. Revision, dtype and engine (or provider) are recorded.
2. Thinking is active: every output contains `<think>`, as in 100% of the original outputs.
3. The length policy is respected: no output ends with `finish_reason = length` below the 32,768-token total.
4. The generation cache is disabled in the run config, and the run reports no cache reads.
5. The unchanged evaluator scores 100% of pre-flight outputs without error. MBPP runs only inside the sandbox.

**Gross-mismatch test (stop and report if triggered):**
- Per model, pair each pre-flight query's fresh score with its **stored `results/bench` score**. The stored score is the reference because the evaluator reproduces it; the pool differs only at KorBench index 222.
- Apply an **exact two-sided McNemar test** (binomial on discordant pairs), α = 0.05 per model, with no multiplicity correction.
- **The McNemar test uses alpha = 0.05 separately for each model. This is an operational warning check used to pause for review if the reconstructed setup looks different. It is not part of the paper's main statistical evidence and does not prove that the reconstructed setup is equivalent to the original one.**
- Why this test:
  - Its false-stop rate stays ≤ α whatever *r* is (exact enumeration: 0.024–0.036 for *r* = 0.1–0.4), so it needs no assumption about the quantity under study.
  - Leaving α uncorrected is a deliberate choice: a false stop costs only a pause for review, while missing a gross mismatch would contaminate the main run.
- What it can detect (exact enumeration, n = 120): a 15 pp accuracy shift is detected with probability 0.71–0.97 and a 20 pp shift with 0.93–1.00, depending on *r*. A 10 pp shift is detected with only 0.36–0.98.
- It is a **gross-mismatch detector, not an equivalence certificate**. Passing it must not be reported as "the setup is equivalent to the original".
- Per-model accuracy differences with Wilson/paired CIs are reported descriptively in all cases.

**Rules.**
- If any check or the test fails, **stop and report** to the user before the main run.
  - A rerun is allowed only after a change that moves the setup *toward* a recorded original setting (e.g. enabling thinking, or restoring the 32 k limit).
  - Every attempt is logged. No setting may be tuned toward a desired accuracy or stability result.
- **Pre-flight outputs are kept completely separate** under `diagnostics/winner_label_stability/preflight/`. They are never pooled into, counted as repetitions of, or used in any main-run metric, and the pre-flight queries are disjoint from the main sample by construction.

---

# Evaluator-Reuse Policy

- Score every fresh output with the **unchanged** repository evaluator: `EvaluatorFactory().get_evaluator(<benchmark>).evaluate(item, raw_output)`, with `score = float(is_correct)`.
- Do not edit evaluator code, answer patterns, tolerances or timeouts.
- **BBH, MELD, MedQA, FinQA:** `item` = {`target` (BBH) or `answer`: the stored `ground_truth`}. This reproduces 100% of the stored non-tie scores, as verified in the audit.
- **KorBench, MBPP:** `item` = `load_data("test")[index − 1]` from the 17 sha256-verified LFS files (§7a). Only `DATA_DIR` is redirected if the files are not restored in place.
  - **KorBench: gate passed** (2026-11-02). There is 100% score and prediction agreement on all 506 non-tie stored outputs (superset of the sample).
  - **MBPP: gate not yet passed.** Inside the sandbox, re-score all 820 stored non-tie outputs (superset of the sample) and **require 100% score agreement** before scoring any fresh output.
    - If agreement is below 100%, stop and report, including whether any disagreement is a 10 s timeout. Do not change the timeout.
    - The pre-execution part already reproduces: 820/820 AST-identical programs.
- **MBPP executes model-generated code.** Run evaluation only in an isolated sandbox. Minimum requirements:
  - a container or VM with **no network** (`--network none` or equivalent);
  - a non-root user;
  - no mount of the home directory or repo other than read-only evaluator code;
  - a tmpfs working directory and CPU/memory limits;
  - Python with `numpy` (the evaluator prepends `import numpy`).

  None exists on this machine (§7a).
- Prompts are reused verbatim from the stored records, so the evaluator's `format_prompt` is not used to rebuild them.

---

# Outcome Definitions

For each query q, model m ∈ {DS, QW} and repetition r ∈ {1..5}, `c[q,m,r]` ∈ {0, 1} is the evaluator score. The pair outcome is exactly one of:

1. **DeepSeek-only correct**: `c[DS]=1, c[QW]=0`
2. **Qwen-only correct**: `c[DS]=0, c[QW]=1`
3. **Both correct**: `c[DS]=1, c[QW]=1`
4. **Both wrong**: `c[DS]=0, c[QW]=0`

Outcomes 3 and 4 are ties. Repeated inference is allowed to produce ties even though every sampled query was an original non-tie.

Per query:

| Metric | Definition |
|---|---|
| original winner | DeepSeek (target 0) or Qwen (target 1) from the original stored run |
| modal repeated outcome | the most frequent of the four outcomes over the 5 reps. Ties in frequency are reported as `multi-modal` with the tied set listed |
| original-winner retention rate | (# reps whose outcome = the original winner's "only-correct" outcome) / 5 |
| opposite-winner flip rate | (# reps whose outcome = the other model's "only-correct" outcome) / 5 |
| tie-emergence rate | (# reps whose outcome ∈ {both correct, both wrong}) / 5 |
| modal agreement | (# reps matching the modal outcome) / 5 |

Retention + flip + tie-emergence = 1 for every query.

Secondary descriptive metrics (not part of the decision rules):
- per-model repeated accuracy `p̂[q,m] = mean_r c[q,m,r]`;
- the soft winner margin `p̂[q,QW] − p̂[q,DS]`, and whether its sign agrees with the original winner.

---

# Stability Criterion (pre-declared; do not change after seeing results)

A query is **winner-stable** if the **same original non-tie winner occurs in at least 4 of 5 repetitions**, i.e. retention rate ≥ 0.8.

---

# Benchmark-Level and Group-Level Analysis

For each benchmark report:
- number sampled;
- original DeepSeek/Qwen balance;
- winner-stable query percentage, with a Wilson 95% CI;
- mean original-winner retention rate;
- mean opposite-winner flip rate;
- mean tie-emergence rate;
- the distribution of modal outcomes;
- the secondary per-model repeated accuracies.

Report the same quantities for the **signal group** (BBH / KorBench / MBPP, n = 60) and the **chance group** (MELD / MedQA / FinQA, n = 60). Compare them with:
- the difference in winner-stable percentage, and the differences in mean retention, flip and tie-emergence;
- a two-sided permutation test that shuffles group labels over queries (10,000 permutations, `default_rng(20261103)`) on the winner-stable-proportion difference and on the mean-retention difference. This is the only significance test planned, and it is reported as such.

Also report the signal vs chance comparison split by original winner class (DeepSeek-win vs Qwen-win) as a descriptive check.

**Caveats to state with the results:**
- **Selection effect (regression to the mean).** Queries were selected *because* one draw was a non-tie. Some tie emergence is therefore expected even if each model's per-query accuracy is fixed but not 0 or 1. Both groups are selected the same way, so the group comparison stays fair.
- **Small n per benchmark (20).** Benchmark-level percentages carry wide intervals, roughly ±20 pp.

---

# Pre-Declared Interpretation

Operational thresholds are fixed now, before any result:
- "substantially more instability" = winner-stable % lower by **≥ 15 percentage points** with permutation **p < 0.05**;
- "highly stable" = winner-stable **≥ 70%**;
- "substantial instability" = winner-stable **< 50%**.

| Case | Condition | Conclusion |
|---|---|---|
| **A** | The chance group has substantially more winner flips/ties than the signal group (its winner-stable % is ≥ 15 pp lower than the signal group's, permutation p < 0.05) | Label instability is a plausible contributor to poor within-task routing |
| **B** | **Both** groups are highly stable (winner-stable ≥ 70% in each), while routing stays near chance within MELD/MedQA/FinQA | The main problem is more likely that the current query representation does not contain or capture the relevant per-query winner information |
| **C** | **Both** groups show substantial instability (winner-stable < 50% in each) | The original binary single-draw winner label may be too noisy for the current routing formulation |
| **D** | Anything else (mixed) | Identify exactly which benchmarks suffer instability (benchmark winner-stable < 50%), and do not generalize beyond them |

If both A and C hold, report both.

No result may be called a mathematical ceiling.

---

# Estimated Generation Count and Resources

**Status of these numbers:**
- The generation counts follow from the design.
- Every token, time, GPU-hour and cost figure below is a **planning estimate, not a measured fact**. Token figures project the original run's stored token counts, and generation lengths under the reconstructed setup may differ.
- Throughput was measured only for the original, unrecorded endpoint, and transferring it to any new hardware is an assumption.
- No provider or GPU configuration has been chosen, so no figure is tied to a concrete configuration. Re-estimate once one is.

| Item | Count |
|---|---|
| Main: 120 queries × 2 models × 5 reps | **1,200 generations** |
| Pre-flight: 120 queries × 2 models × 1 rep | 240 generations |
| **Total** | **1,440 generations** |

**Token estimate** (planning estimate: stored mean token counts of non-tie rows per benchmark × planned generations):
- Main: about **4.2 M completion tokens** (DeepSeek ≈ 2.25 M, Qwen ≈ 1.98 M) plus about 0.6 M prompt tokens.
- Pre-flight: about 0.9 M completion tokens.
- Total: about **5–5.5 M tokens**.
- Per-benchmark main-run completion tokens: FinQA ≈ 1.18 M, KorBench ≈ 0.87 M, MBPP ≈ 0.79 M, MedQA ≈ 0.54 M, BBH ≈ 0.49 M, MELD ≈ 0.36 M.
- Expect a long tail: individual generations reach 32 k tokens, about 2–4% of Qwen3-8B outputs on KorBench, MBPP and FinQA.

**Self-hosted GPU (preferred for control over settings):**
- One GPU with ≥ 40 GB (A100-40/80GB, L40S, H100) running vLLM with bf16 and `max-model-len 32768`. The 8B weights take ~16.4 GB; the rest is KV cache for long contexts.
- *Assuming* the original endpoint's ~1.2–1.6 k tokens/s aggregate (concurrency 32) carries over, the main run would be about 45–60 min of decode for both models combined. The long tail sets a floor of roughly 5–10 min per 32 k-token sequence. Both figures are assumptions, not measurements on any target GPU.
- Planning budget: **~2–4 GPU-hours** in total, including two model loads, the pre-flight and retries. This is an unvalidated estimate.
- Plus about 30 GB of disk for the two checkpoints.

**Hosted API alternative:**
- The collector config lists per-million prices (DeepSeek $0.03 in / $0.11 out; Qwen $0.035 / $0.138). These belong to the original, unidentified provider and have not been checked against any current provider. At those prices the planning estimate would be **< $1**. The actual cost depends on the provider chosen.
- The provider's weights, precision and engine are not under our control, and the provider must support a 32 k context with thinking enabled. The pre-flight validation is mandatory in this route, as in any route.

**This machine:** **not feasible.** It has no GPU, and 9.7 GB RAM is below bf16 weight size. A quantized CPU run would change the model and would take on the order of weeks for ~5 M tokens.

**MBPP evaluation:** needs a sandboxed Python runtime for executing generated code, at negligible compute.

---

# Blockers To Solve Before Inference

Status as of 2026-11-02:

1. **OPEN: inference access.** This needs one of:
   - a GPU host with vLLM and the two official checkpoints;
   - API credentials for an endpoint serving both exact models with a 32 k context and thinking enabled.
2. **RESOLVED (data only): KorBench/MBPP benchmark data.** The 17 required LFS objects were fetched and sha256-verified outside the working tree (§7a). The KorBench re-score gate passed at 100%. *Housekeeping for the run:* the verified copy is session-local, so re-fetch and re-verify it, or install git-lfs.
3. **OPEN: MBPP sandbox and re-score gate.**
   - There is no no-network sandbox on this machine, so stored MBPP score agreement is unverified and no MBPP output (stored or fresh) has been executed.
   - Needed: a sandbox meeting the evaluator-reuse requirements, then 100% score agreement on the 820 stored non-tie outputs.
4. **ACCEPTED LIMITATION, not a blocker: the original serving stack is not recorded.**
   - The endpoint is redacted, the model revision, precision and engine are unknown, and `run_vllm.sh` does not match the data.
   - The run is a reconstructed evaluation, validated by the pre-flight (operational checks + McNemar gross-mismatch test), with the new stack fully recorded. It must not be called an exact reproduction.
5. **REQUIRED AT LAUNCH: cache off.** Otherwise the repetitions are not independent (see the generation-setting policy). It is checked in pre-flight check 4 and by the main-run non-identity check.
6. **REQUIRED: user approval** of this revised checkpoint, especially the pre-flight change, before any inference.

---

# Planned Outputs (next phase; none exist yet)

Under `diagnostics/winner_label_stability/` (new directory; previous diagnostics untouched):
- `frozen_sample.csv` (written before any generation)
- `preflight/`: kept separate and never pooled with main-run data
  - `preflight_sample.csv` (written before any generation)
  - `preflight_generations.jsonl`
  - `preflight_check.csv` (operational checks, McNemar per model, descriptive accuracy differences)
- `generations.jsonl` (main-run raw outputs plus metadata only)
- `query_stability.csv`
- `benchmark_stability.csv`
- `group_comparison.csv`
- `WINNER_LABEL_STABILITY_REPORT.md`

Scripts under `diagnostics/scripts/`.

---

# Constraints

- No inference until the blockers are resolved and the user approves the run.
- No temperature or other generation-setting changes to manufacture variability.
- No change to benchmarks, sample, repetition count, outcome definitions, stability criterion or interpretation thresholds after results are seen.
- No expansion of the model pool; no router retraining; no Edge/Split/Remote integration.
- Do not modify existing diagnostic results.
- Commit only after user review.
