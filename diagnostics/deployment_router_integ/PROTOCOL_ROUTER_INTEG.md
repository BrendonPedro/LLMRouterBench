# ROUTER-INTEG: the frozen lightweight router connected to Edge and Remote inference. Protocol `router-integ-v1`

**Frozen by `frozen/FREEZE.json`**, which lists the SHA-256 of this text, of every code file and of every frozen
input. No Git commit is involved. Every target command refuses to run without that record, or if a listed file has
changed. A changed rule is a new protocol version with a new record. No GPU has executed any code of this protocol
before the record, and no result exists. Every number below was fixed before any result.

This protocol covers the first of two pieces of work the advisor ordered: the integration of our router with our
execution system. The reproduction of Mix-or-Split under our setting is separate, comes afterwards, and is not
prepared, run or approximated here.

## 1. Question and what is not claimed

**Question.** Can the existing lightweight semantic router select between its original two candidate models, and can
that selection be connected to real Edge or Remote inference under a frozen execution policy?

Three things are measured and reported separately:

1. **Semantic router quality**: how well the router's choice matches the winner of each query.
2. **Execution placement**: how each model behaves under each execution mode (completion time, answers correct by a
   deadline).
3. **End-to-end integrated use**: each model-selection policy followed by the same fixed placement rule.

**Not claimed, whatever the outcome:** that a working integration shows the routing policy is useful; that the router
is better or worse trained than before; that hindsight selection is deployable or an upper limit for future systems;
anything about queueing, concurrency, caching or changing load; anything about a layer split of these two models;
anything about other models, other benchmarks, quantised builds or another machine pair; any comparison with
Mix-or-Split.

## 2. The frozen router and its model pair

The router is the "MiniLM router" of the manuscript (Macro-F1 0.600 under stratified 5-fold cross-validation).

- **Architecture.** `sentence-transformers/all-MiniLM-L6-v2` (22,565,376 parameters without its unused pooler), kept
  fixed; WordPiece tokenisation truncated at 256 tokens; attention-mask mean pooling; L2 normalisation; a multilayer
  perceptron head 384 -> 64 (ReLU) -> 1 with 24,705 parameters. The head was trained by regression on the decision
  score of a logistic-regression teacher on GTE-Qwen2-7B embeddings. The teacher is not needed at inference.
- **Decision.** score > 0 selects Qwen3-8B, otherwise DeepSeek-R1-0528-Qwen3-8B. The threshold 0 is the router
  study's. It was never tuned there and is not tuned here.
- **Arrangement.** Five cross-validation fold models. No model trained on the complete pool exists, and no single
  held-out checkpoint exists. Each fold model was trained on four folds of the 11,467-query pool and scores only the
  fifth.
- **Artifact.** The router study stored no serialised checkpoint. `tools/rint_router_build.py` regenerates the five
  heads with the study's own unmodified training code, cached embeddings, frozen folds and seeds (42 + fold), and
  accepts them only if they reproduce the stored per-query held-out scores behind the manuscript's numbers. Result
  (`router/router_build_report.json`): the epochs of all five heads equal the stored epochs (55, 57, 61, 53, 56); all
  11,467 held-out scores are reproduced with a largest difference of 5.9e-08; no decision differs. The smallest
  absolute held-out score is 2.8e-05, far above that difference. The heads are exported as plain arrays
  (`router/router_heads.npz`, pinned by hash), so inference needs no scikit-learn.
- **Reproducibility of inference.** From raw query text on the desktop, the adapter (`rint_router.py`) reproduces the
  pinned scores of all 126 workload queries (largest difference 6.5e-07, no decision differs). The edge preflight
  repeats this check on the edge device, and the router-timing stage repeats it for every evaluation query.
- **Model pair.** DeepSeek-R1-0528-Qwen3-8B (class 0) and Qwen3-8B (class 1). The router's output is never mapped to
  any other model. The three-model pool of the deployment study (Qwen3-8B, 14B, 32B) is not used. No router is
  trained or retrained.

## 3. The two models

| | DeepSeek-R1-0528-Qwen3-8B | Qwen3-8B |
|---|---|---|
| Repository and revision | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` @ `6e8885a6ff5c1dc5201574c8fd700323f23c25fa` | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` |
| Source of the pin | the revisions of the winner-label stability experiment | the same; also the manifest QWEN-SELECT verified on both hosts |
| Weights | 399 tensors, 16,381,470,720 bytes, bfloat16 as published, 2 shards | 399 tensors, 16,381,470,720 bytes, bfloat16 as published, 5 shards |
| Tensor names, shapes, types | identical in both checkpoints (checked from the safetensors headers) | |
| Quantisation | none | none |
| Tokenizer file | its own (7.0 MB) | Qwen3's (11.4 MB) |
| Chat template | `<BOS><User>` text `<Assistant>` | `<im_start>user` text `<im_end><im_start>assistant` |
| End-of-sequence ids | 151645 (config.json; the checkpoint has no generation_config.json) | 151643, 151645 (generation_config.json) |
| Sampling default of the model | none | top-k 20 (temperature 0.6 and top-p 0.95 are overridden by the request) |
| Position scaling in config.json | yarn, factor 4, `attn_factor` 0.878 | none |

Equal size does not make the two checkpoints interchangeable at run time. Five differences matter, and each is handled
explicitly: token ids are pinned per model; the stop set is per model; top-k is per model; the DeepSeek checkpoint
needs no generation configuration file; and the position scaling needs the alignment of section 7.2.

## 4. Hardware and memory

Edge: the DGX Spark (GB10, unified memory). Server: one RTX A6000 (48 GiB) of the laboratory server, chosen and
pinned by QWEN-SELECT's `select-gpu` rule (fewest other processes, then most free memory, then lowest index; excluded
GPUs stay excluded). No rented GPU. No second server GPU.

Both models are held by one process at the same time on each host, so that the two models of a query can run back to
back and the live pipeline can execute whichever model the router picks.

| Load | Longest context | Weights | KV cache | Other | Required free |
|---|---|---|---|---|---|
| Edge, both models | 18,100 tokens | 30.51 GiB | 4.97 GiB | 7.36 GiB | **42.84 GiB** of available RAM |
| Server, both models | 18,100 tokens | 30.51 GiB | 4.97 GiB | 4.20 GiB | **39.68 GiB** on the pinned GPU |
| Server, one model (fallback of section 13) | 18,100 tokens | 15.26 GiB | 4.97 GiB | 4.20 GiB | 24.43 GiB |

The formula is QWEN-SELECT's (`qsel_models.whole_need`), restated for two models: all weights; one growing KV cache at
the longest reachable context (output cap 16,384 plus the longest workload prompt, 1,716 tokens), counted twice
because a growing cache is copied when it grows; one step's activations; the loading transient; fixed allowances; a
margin of 4 GiB (edge) or 2 GiB (server). The gates are QWEN-SELECT's: before loading, and again inside the process
before any weight is read. Short of memory is BLOCKED: nothing is loaded, nobody is asked to free memory.

**Known evidence.** The whole Qwen3-8B loaded and ran on the edge on 21 and 22 November (peak 15.9 GiB allocated at a
4,096-token context). The edge had between 73 and 81 GiB available at the moments recorded on 18 to 21 November. **Not yet
known:** whether either model loads on the server GPU (no 8B model has run there), whether DeepSeek loads on either
host, and memory at contexts above 4,096 tokens. The server requirement leaves 8.3 GiB of a 48 GiB GPU for other
users; a GPU that others occupy beyond that is BLOCKED.

## 5. Execution configurations and data policies

| Configuration | Model | Execution | Runtime |
|---|---|---|---|
| DS_EDGE | DeepSeek-R1-0528-Qwen3-8B | whole, on the edge | the Hugging Face model loaded by the reused `k16_runtime.hf_model`, its own KV cache |
| QW_EDGE | Qwen3-8B | whole, on the edge | same |
| DS_REMOTE | DeepSeek-R1-0528-Qwen3-8B | whole, on the server GPU; the edge sends token ids and receives output tokens one by one | same loader, on the server; QWEN-SELECT's wire framing |
| QW_REMOTE | Qwen3-8B | same | same |

**Layer Split: not run.** A split is admitted only for a model that has passed the applicable correctness gate
(K16-CORRECT-v1). That gate has been passed only by Qwen3-32B at one cut. No gate result exists for either 8B model,
and the reused split runtime is configured for the 32B checkpoint and its one cut. Building and gating an 8B split would be new GPU work
that this week does not allow. The split is therefore recorded as NOT_RUN. It is not a failed gate and not a negative
result, and nothing is concluded about it.

Our layer split (an edge prefix and suffix of the layers with the middle layers on the server, hidden states on the
wire) is a different mechanism from Mix-or-Split's phase split (cloud prefill, key-value cache transfer, edge
decode). Neither is run here.

**Data policies** (reported separately, never switched after a result):

| | Policy A: no raw query transfer | Policy B: raw query transfer allowed |
|---|---|---|
| EDGE (sends nothing) | feasible | feasible |
| REMOTE (sends the query's token ids, which are the raw query) | **infeasible** | feasible |
| SPLIT (would send hidden states) | not run | not run |

No configuration is called feasible under a policy it violates. Policy A is a data-handling rule and not a privacy
guarantee.

**Placement rule** (`rint_spec.PLACEMENT`, the same for both models, fixed before any execution of this pair):
policy A places every request on EDGE; policy B places every request on REMOTE. Basis: under B the server is used
because, in the archived QWEN-SELECT records, the server GPU produced tokens faster than the edge (Qwen3-14B: 57.5 ms
per token through the wire against 140.1 ms on the edge). That evidence is for another model size. The rule is not
revised after this protocol's records exist, and no placement is fitted to the evaluation queries. All four
configurations are measured for every query, so the alternative placements are reported as description, and the best
configuration per query only as a retrospective reference.

In the live pipeline only: under policy B, if the server does not answer or fails during a request, that request
falls back to EDGE (feasible under B). The fallback, the time lost and the time to load the edge models are recorded
and are inside the request's end-to-end time. The edge models are loaded at the first fallback and stay loaded; their
GPU time is charged to the edge and is confined to what is left of live B's wall-clock limit. Three requests in a row
that needed the fallback stop the pass (the server is down); it is resumed when the server is back. Policy A has no
fallback and needs none.

## 6. Evaluation sample and protection against leakage

- **Frame.** The router study's pool of 11,467 queries from 15 benchmarks. Every query is held out for exactly one
  fold model and is scored only by that model (`router/fold_map.json`; the record of every request carries the fold).
  No query is scored by a model that saw it in training.
- **Removed from the frame**, by rules that look at no router output and no new model output: HumanEval (164) and
  LiveCodeBench (1,055), whose scorers execute model-written code without a validated sandbox here; 14 FinQA rows
  with an empty reference answer (the winner-label stability experiment's data-validity rule). Frame: 10,234 queries
  in 13 benchmarks. MBPP stays, scored only inside that experiment's validated sandbox.
- **Sample.** 120 queries, allocated proportionally (largest remainder) over benchmark x original outcome class, so
  that ties and both winner classes keep their natural shares: 67 both correct, 29 both wrong, 10 only DeepSeek
  correct, 14 only Qwen3-8B correct (frame shares 56.0%, 23.6%, 8.9%, 11.4%). Inside a stratum queries are ranked by
  a seeded SHA-256. The primary evaluation uses all 120. The 24 queries with an original winner are reported
  separately as a diagnostic.
- **Not used for selection:** the router's score or choice, the desired outcome, any output length, any new model
  output. The sample size was fixed from the time budget (section 12) before any execution.
- **Benchmark identity** is used for stratification and analysis only. It is never a feature of the router.
- Three of the 120 queries were also among the 120 of the winner-label stability sample. They are kept, because
  removing them would change the natural mix. They remain held out for the router.
- **Profiling queries:** 6 further queries from the same frame, disjoint from the sample, never analysed as results.

`frozen/workload.json` holds keys, order, hashes, token counts and the pinned decisions, and no prompt text. Prompts,
reference answers and token ids are in a private file pinned by hash. The prompt of every query is the stored
original prompt, verbatim.

## 7. Serving protocol

### 7.1 Starting point and what is kept

The starting point is the reconstructed protocol of the winner-label stability experiment
(`diagnostics/scripts/wls_common.py`, `wls_serve_vllm.sh`, `wls_preflight_generate.py`), which itself reconstructs
the conditions of the released labels: the stored prompt as one user turn, no system prompt, the checkpoint's own chat
template with its defaults (so both models produce reasoning traces), temperature 0.2, top-p 1.0, the model's own
sampling default where one exists (top-k 20 for Qwen3-8B, none for DeepSeek), bfloat16, the unchanged repository
evaluators. All of this is kept. The reconstructed prompts are checked: for all 126 workload queries and both models
the number of prompt tokens equals the count the original server recorded.

### 7.2 Deviations, all forced, all applied to both models alike

| Setting | Original labels / stability experiment | This protocol | Why, and what it may do |
|---|---|---|---|
| Serving engine | vLLM 0.8.4, one A100/A800-class GPU | the Hugging Face model in a single-request loop (QWEN-SELECT's whole-model loop with a sampled token choice) | vLLM has not been validated on either laboratory host. Numerics and speed differ. Outputs are not expected to be identical |
| Position scaling of DeepSeek | vLLM multiplies the rotary terms by (0.1 ln 4 + 1) x `attn_factor` = 1.000 | set to the same 1.000 (`rint_backend.align_position_scaling`) | transformers 4.51.3 does not read the key `attn_factor` and would use 1.139 for DeepSeek only. The inverse frequencies of the two engines are identical. Without the alignment DeepSeek alone would run under a positional scale it was not labelled with. No checkpoint file is changed |
| Concurrency | 32 concurrent requests, continuous batching | one request at a time | Completion time must be that of one request on one device |
| Output limit | 32,768 tokens in total, no other cap | min(16,384, 32,768 minus prompt) new tokens | Memory of two models on one 48 GiB GPU. In the frame's original outputs 2.7% of generations exceed 16,384 tokens (6 of the sample's 240); they are cut and scored as they stand |
| Time limit | none | a request still running after **600 s** is stopped and scored as it stands; it is late at every deadline | GPU budget. At the archived edge speed 600 s is about 6,200 tokens; 26 of the 240 original generations of the sample are longer. Edge correctness is therefore lower than deadline-free correctness, by construction |
| Seeds | no per-request seed | one seed per request from a fixed formula (query, model, mode, phase) | Makes a rerun auditable. It does not change the sampling distribution |
| Sampling arithmetic | on the GPU, vLLM's sampler | temperature, then top-k, then one draw on the CPU from the device's logits | A different implementation of the same distribution |
| Stop condition | the checkpoint's end-of-sequence tokens | the same ids, per model (section 3) | none intended |
| Prefix caching | disabled | not applicable (no cross-request cache exists) | none |

### 7.3 Differences from the later Qwen3 deployment protocol (QWEN-SELECT)

| | QWEN-SELECT | This protocol |
|---|---|---|
| Models | Qwen3-8B, 14B, 32B | DeepSeek-R1-0528-Qwen3-8B, Qwen3-8B |
| Reasoning | off (`enable_thinking=False`) | on (template defaults) |
| Decoding | greedy | sampled: temperature 0.2, top-p 1.0, top-k per model, seeded |
| Context and cap | 4,096 and 3,072 tokens | 32,768 and 16,384 tokens |
| Cutoff and deadlines | 300 s; 30, 60, 120, 300 s | 600 s; 30, 60, 120, 300, 600 s |
| Queries | 160 from MMLU-Pro, GPQA, LiveMathBench | 120 from 13 benchmarks of the router's pool |
| Evaluator | the repository's, unchanged | the repository's, unchanged |
| Precision, loader, wire framing | bfloat16, `k16_runtime.hf_model`, QWEN-SELECT's wire | the same, reused by import |
| Models per process | one | two |

Results of this protocol are never pooled with QWEN-SELECT's, with the winner-label stability experiment's, or with
the released labels, as if the conditions were the same. The only number carried over from QWEN-SELECT is planning
arithmetic (section 12) and the basis of the placement rule (section 5).

## 8. Shared measurement set

GPU time is spent once, on a shared set of records, and every policy is then evaluated on the same records.

- **What is measured.** For each of the 120 queries, each of the four configurations, once: 480 requests.
- **Lanes.** The edge lane (both models on the edge GPU) and the remote lane (both models on the server GPU, driven
  from the edge without an edge GPU) run at the same time on the two machines. Each lane processes the queries in the
  same seeded order.
- **Order.** Inside a query the two models run back to back. Which model runs first is fixed per query by a seeded
  rule that puts each model first for exactly 60 queries. No model owns a time of day or a position.
- **Warm-up.** Each lane starts with one 8-token request per model on a profiling query. Warm-up rows are recorded
  and never analysed. The clock of a request starts when its token ids are handed to the backend, with the models
  loaded and the server connected (QWEN-SELECT's definition), and stops when the last token is on the edge.
- **Record** (one row per request, `rint_record.py`): query id, the fold of the router model that scores it, exact
  model identifier (repository, revision, checkpoint identity hash), execution mode, hardware role and GPU type,
  precision and quantisation, prompt and generation settings and their hash, sampling seed, output tokens and decoded
  text, time to first token, completion time, time stamp of every token, output token count, whether it ended inside
  each deadline, stop reason (end of sequence, cap, cutoff), error, feasibility under each data policy, run identifier,
  phase, attempt, bytes each way and the server's own times for REMOTE. Correctness is added by the scorer in a
  separate file keyed by the row, so no measurement row is ever edited. Deadline success (correct and on time) is
  computed from the two.
- **Failures.** A request that fails is a row with status `error`. It is never dropped. Two kinds are told apart by
  the exception, not by judgment. *A failure of a running model* (for example out of memory) settles the request: it
  is not repeated, it counts as incorrect and late in every analysis, and it is counted as a failure. Three such
  failures in a row stop the lane. *A transport failure* (the server or the wire cannot be reached, or the connection
  is lost during a request) says nothing about a model: its row is kept and marked, the lane stops at once, and the
  request is repeated when the lane is resumed. The later row is the one that counts; the number of repeated requests
  is reported. A transport failure that was never repeated leaves its query without that configuration, and the query
  is then not evaluated (see Completeness).
- **Resumption.** A lane that dies is resumed from the first request without a settling row, in a fresh process after
  the same warm-up. Rows already written are kept. At most three attempts per lane; every row carries its attempt. An
  attempt begins when every check has passed and the lane is about to load or send: a server that does not answer, or
  a BLOCKED gate, is a stop and not an attempt.
- **Completeness.** A query enters the evaluation only if all four configurations have a row. If the wall-clock
  limit ends a lane early, the queries completed in all four configurations form a prefix of the seeded order; they
  are evaluated and the result is labelled PARTIAL_PREFIX with its count. The prefix is decided by time, never by
  correctness.

### Replication and uncertainty

One stochastic draw per (query, configuration). Replicates of the full set are not affordable before Friday (the
edge lane alone is planned at 16 hours). The results are therefore labelled **single-draw**, and conclusions are
limited accordingly:

- Our own stability study found that a single-draw winner label is unreliable for an individual query (26.7% of 120
  selected queries kept their original winner in at least four of five repeats). The new labels B of section 11
  inherit this. No claim is made about the winner of any individual query.
- Each model is in fact drawn twice per query: once on the edge and once on the server, with different seeds. The
  agreement of the two draws (per model, and of the two winner labels) is reported as this protocol's own estimate of
  draw-to-draw stability. The two draws also differ in hardware and in how many answers the cutoff truncates, so the
  agreement is a lower bound on same-condition stability, and it is described, not tested.
- Intervals are paired bootstrap intervals over queries (10,000 resamples, seed 20261123, stratified by benchmark,
  percentile). They cover the choice of queries. They do not cover draw-to-draw variation of the same query.
- **What 120 queries can show.** On the original labels of this sample the fixed models score 64.2% (DeepSeek) and
  67.5% (Qwen3-8B), the router 69.2%, the task lookup 70.0%, and the hindsight choice 75.8%. The router differs in
  outcome from always-Qwen3-8B on 8 queries (5 better, 3 worse) and from the task lookup on 1. Differences of this
  size cannot be separated from zero with 120 queries. The expected outcome of the policy comparison is therefore an
  interval that includes zero. That would be reported as such. It would not show that the policies are equivalent.

## 9. Policies evaluated on the shared records

For each data policy, every policy selects a model per query; the fixed placement rule of that data policy then
selects the configuration; the recorded outcome of that configuration is read.

| Policy | Model choice | Status | Latency counted |
|---|---|---|---|
| A. Always DeepSeek | DeepSeek for every query | deployable | the configuration's completion time |
| B. Always Qwen3-8B | Qwen3-8B for every query | deployable | the same |
| C. Frozen router | the pinned held-out decision of the query's fold model | deployable as a router; the per-fold dispatch is an evaluation device | completion time plus the router's latency measured on the edge for that query |
| D2. Task lookup, predicted | a table benchmark -> model from training folds only, indexed by the benchmark predicted by a recogniser fitted on training folds only | predictive | completion time plus the same encoder latency as C (it shares the router's encoder; assumed equal, not measured separately) |
| D1. Task lookup, true benchmark | the same table indexed by the true benchmark | **non-deployable diagnostic** | completion time |
| E. Hindsight model | the model of the query that was correct (ties: the faster) under the same placement | **retrospective reference, not deployable, not an upper limit** | completion time |
| E'. Hindsight configuration | the best feasible configuration of the query | the same | completion time |

- The table of D1 and D2 assigns a benchmark to Qwen3-8B if Qwen3-8B wins at least half of that benchmark's
  training-fold queries with a winner. It reproduces the stored benchmark-identity reference of the router study on
  all 75 fold-benchmark entries. The recogniser of D2 is one fixed logistic regression on the router's own
  embeddings (held-out benchmark accuracy 92.7% to 94.6% per fold). Both were computed, and their decisions for the
  sample pinned, before any execution (`router/task_lookup.json`, `frozen/workload.json`).
- Routing latency is added to C and D2 only. Fixed-model policies carry no routing overhead. Every policy carries the
  actual completion time of its configuration, which for REMOTE includes the wire.
- Reported per policy and data policy: selection frequencies; agreement of the model choice with the new winner label
  on queries that have one; share correct within the cutoff; share correct by each deadline with its interval;
  end-to-end latency; failures; data-policy violations (zero by construction, and counted); paired differences
  between policies with intervals and the number of discordant queries.
- **Limits of record-based evaluation.** It replays recorded single requests. It does not show how a policy behaves
  under other queueing, concurrency, caching or resource conditions, and it does not replace the live pass.

## 10. Live integration test

After the shared records exist, the real pipeline runs once per data policy on the same 120 queries, in its own seeded
order:

    raw query text -> router on the edge CPU -> model decision -> placement -> feasibility check
                   -> the request built from the raw text -> backend (edge GPU, or the wire) -> output -> record

- The router runs on the edge device's CPU (four threads, as in the router study's timing), with the fold model held
  out for the query.
- Live B (policy B, REMOTE) runs first, then live A (policy A, EDGE). Live B starts only with the server up and
  holding both models.
- Measured per query: router latency with its stages; dispatch overhead (placement, feasibility, tokenisation of the
  request); bytes each way and the time not spent in the server's own clock (REMOTE); time to first token; generation
  time; end-to-end time from raw text to the last token; fallbacks and failures.
- Checked per query and recorded: the router's live decision against the pinned decision; the token ids built live
  against the pinned ids; that the backend ran the model that was selected (the backend returns the identifier of the
  model it ran, and a mismatch is an error).
- **Comparison with the record-based result of policy C:** selection agreement; correctness agreement and
  disagreements (2 x 2 table); absolute and relative difference of generation time; end-to-end time against record
  plus router latency; routing and dispatch overhead; failures and fallbacks; the share correct by each deadline, live
  against record. The record of a query is that of the configuration policy C plans for it, whatever the live pass
  executed. Both sides use the same queries (those with a live row and that record); generation times are compared
  only where the live pass executed the planned configuration.
- The live requests use other seeds than the shared records. Outputs and times are not expected to be equal. One
  live pass shows that the pipeline works. It does not establish statistical stability.
- A separate CPU-only stage (`router-timing`) measures the router on every evaluation query on the edge before the
  live passes. Its latencies are the ones used by the record-based evaluation.

## 11. Semantic metric and the three kinds of label

- **A. Original labels**: the released one-shot correctness of the two models, from which the router study derived
  its winner labels and on which the router was trained and evaluated.
- **B. New labels**: derived from this protocol's executions. Rule, fixed here: for an execution mode, the winner of
  a query is the model that alone is correct within the cutoff; if both or neither are correct, the query is a tie.
  B is computed separately for EDGE and for REMOTE. The number of ties is reported.
- **C. The benchmark's reference answers**, which only decide correctness. They are not winner labels.

These are not interchangeable. Reported: the router's Macro-F1 (primary; the mean of the two per-class F1 scores, on
queries with a winner, as in the router study) and AUC, against A, against B-edge and against B-remote, each with the
number of queries with a winner and an interval. If a label set has winners of one class only, Macro-F1 and AUC are
undefined for it and reported as such; bootstrap resamples that lack a winner class are left out and their share is
reported. Also reported: the 3 x 3 tables A against B-edge, A against B-remote and B-edge
against B-remote; per-model agreement of new correctness with the original. A and B are compared directly because the
query is the same and the prompt is identical; the conditions differ as listed in section 7.2, and the comparison says
so. A change of Macro-F1 between A and B is not evidence about training. No threshold is tuned on any of these
results.

With about 24 queries with a winner, the Macro-F1 of this sample is imprecise: on the original labels it is 0.644 on
this sample against 0.600 on the pool. It is reported with its interval and not interpreted beyond it.

## 12. Deadlines, budget and limits

**Deadlines.** Deadline utility is E0's: the share of queries answered correctly with completion time at most D.
D = 30, 60, 120, 300 and 600 s, which is E0's grid. QWEN-SELECT dropped the 600 s cell for budget reasons and ran
without reasoning; with reasoning traces, at the archived edge speed a 300 s limit would cut about a third of the
generations (73 of the sample's 240 original outputs are longer than 300 s allows), so the 600 s cell is restored and
the cutoff is 600 s. Ties between policies are reported as ties. E0's
3.0-point margin and its selection rule are not applied: this protocol tests no split and claims no selection.

**Timing evidence available.** Qwen3-8B whole on the edge: 96.6 ms per output token (QWEN-SELECT, 160 queries, greedy,
contexts up to 4,096). Qwen3-14B on the server through the wire: 57.5 ms per token, with a slow period in one third
of that block. **Missing:** DeepSeek on either host; any 8B model on the server; time per token at contexts above
4,096 tokens (the stock KV cache is copied as it grows, so a step costs more at a long context); sampled decoding
(one CPU draw per token). The profiling stage measures all four configurations on the 6 profiling queries before the
shared set. Its purpose is to expose a gross error in the arithmetic below: if the median time per output token on a
host is more than 1.5 times the planning value (edge above 0.145 s, server above 0.086 s), the run stops and the
figures are reported before the shared set starts. Profiling changes no rule, and the sample is not resized from it.

**Planning arithmetic** (original output lengths of the sampled queries x archived time per token, capped at 600 s;
the server figure uses the 14B speed and is therefore likely too high). These are not predictions.

| Stage | Edge GPU-hours | Server GPU-hours | Wall-clock limit |
|---|---|---|---|
| Profiling (6 queries, 4 configurations) | 1.0 | 0.8 | 2.5 h each |
| Shared set (120 queries, 4 configurations; lanes concurrent) | 16.2 | 10.8 | edge 22 h, server 16 h |
| Live B (120 queries, REMOTE) | 0 | 5.2 | 8 h |
| Live A (120 queries, EDGE) | 7.9 | 0 | 11.5 h |
| **Planned** | **25.1** | **16.8** | about 31 h of GPU execution in sequence |
| **Allotment (hard)** | **36.0** | **30.0** | **ceiling 66 GPU-hours** |

One GPU held by one of our processes for one hour is one GPU-hour, computing or waiting. The server figures do not
include the idle time before a server releases its GPU (at most 30 minutes after its last connection). Each host
keeps QWEN-SELECT's append-only ledger and refuses a stage outside its allotment. A stage's wall-clock limit covers
all its attempts: what earlier attempts used is subtracted, on the edge and on the server alike. A stage that the
edge drives without an edge GPU (a remote lane, live B) stops 30 minutes before its server's limit, so that the
server is not the first to stop. The worst case of the shared set
(every request reaches 600 s) is 40 edge hours, which exceeds its 22-hour limit: the limit, not the worst case, binds,
and a lane stopped by it yields the PARTIAL_PREFIX result of section 8.

**Friday.** With a start on Tuesday 24 November the planned sequence ends during Thursday 26 November, which leaves
Thursday evening and Friday for scoring, verification and the manuscript revision. There is about one day of slack
and no more. A blocked GPU, a failed lane or a later start removes it. Completion by Friday 27 November is feasible
and is not promised.

**Reductions, in this order, if time runs short** (decided here, never by looking at correctness): live A is run on
the first 60 queries of its seeded order (`live --reduced`); then live B likewise; the shared set is never reduced
below the completed prefix. A reduced pass is marked in its records and in the summary.

## 13. Stopping rules and outcomes

- BLOCKED (memory): nothing is loaded; wait and repeat. Never switch GPU after a server stage has started; never ask
  anyone to free memory.
- **Server cannot hold both models** (BLOCKED twice, at least two hours apart): the server serves one model at a
  time (`--models DS`, then `--models QW`) for the shared set. The within-query order balance is then lost on the
  server and that is disclosed. Live B needs both models and waits.
- A lane has three attempts. Three model failures in a row stop a lane; one transport failure stops it at once
  (section 8). Three fallbacks in a row stop live B.
- Position scaling other than 1.000 after loading, a checkpoint that is not the pinned one, token ids that differ from
  the pinned ids, a server on another revision: STOP before any request.
- A wall-clock limit or an exhausted allotment ends the stage. Completed rows are kept and evaluated as a prefix.
- The router's decisions on the edge differ from the pinned decisions: reported as a finding; the record-based
  evaluation uses the pinned decisions, the live rows show what the edge did.

| Outcome | Condition |
|---|---|
| COMPLETE | all 480 shared rows exist; both live passes complete |
| PARTIAL_PREFIX | a limit ended a lane; the completed prefix is evaluated and labelled with its size |
| REMOTE_NOT_EVALUATED | the server never loaded: policy B is not evaluated; policy A stands by itself |
| NOT_EVALUABLE | the edge could not load the models in two attempts |

All are reportable. A negative or inconclusive result is a result and is reported in full.

## 14. Scoring

On the analysis machine only, after the lanes have ended. The unchanged repository evaluator of each benchmark,
on the decoded text. For every benchmark of the frame, re-scoring every stored original output of both models
reproduces the stored score: eleven benchmarks were checked here on all rows of both models (16,076 scores, no
mismatch); KorBench (all rows) and MBPP (all 820 rows with a winner, inside the sandbox) were checked in the
winner-label stability experiment. `rint_score.py self-test` sends the stored original outputs of the 126 workload
queries through the exact scoring path and must reproduce all 252 stored scores before any new output is scored. An
answer cut off by the cutoff or the cap is scored as it stands. A request that failed is not scored and counts as
wrong.

## 15. Integrity

A target command runs only with the freeze record intact and the reused runtime at tag `qwen-select-prereg-v1`
(`4484a872`), unmodified. Checkpoints are verified by SHA-256 on each host. Token ids are verified against the
workload on the edge. The protocol identity crosses in the wire's HELLO; a server on another revision is refused. The
private site file is hashed into every output. Raw outputs stay outside every repository, because they name hosts.
`--dev` exists for CPU tests with tiny random models; dev and mock rows are refused by the scorer and the evaluator.

Before the freeze, the complete command sequence (server, both lanes, router timing, both live passes, a server that
is down, a resumption) was run on a CPU with tiny random models of the real layer count, one of them carrying
DeepSeek's position-scaling configuration. That shows that the drivers run and that the selected model identifier
reaches the backend. It is not a result, and no timing from it is used.

## 16. Disclosures: what was known when this was written

- The router study's results, including that the router mostly recognises the task and that the original labels are
  unstable for individual queries.
- The outcomes of QWEN-SELECT and GEMMA-SELECT, and the per-token timings used above.
- The original correctness and output length of every query of the pool. They were used to stratify the sample, to
  plan the budget and to state what 120 queries can show. No query was chosen or dropped because of them, beyond the
  proportional stratification.
- The alignment of DeepSeek's position scaling was found by reading the two engines' source code, before any
  DeepSeek output of ours existed.
- Choices that could favour one model: the 600 s cutoff truncates DeepSeek more often than Qwen3-8B on the edge (19
  against 7 of the sampled original generations exceed it at the archived speed). This is a property of the
  deployment, not a correction, and it is reported per configuration.
