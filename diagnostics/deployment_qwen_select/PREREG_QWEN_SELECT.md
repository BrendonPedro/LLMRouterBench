# QWEN-SELECT: Edge, Split or Remote for Qwen3 on the laboratory pair. Pre-registration

**Frozen by the tag `qwen-select-prereg-v1`.** No GPU has executed any code of this experiment before that tag, and
no GPU result exists. Every number below was fixed before any result. The decision files (this text, `qsel_spec.py`,
`qsel_stats.py`, `qsel_workload.py`, `qsel_runtime.py`, `workload/workload.json`, the manifests in `frozen/`) are never
edited after the tag. A change of any rule is a new pre-registration under a new tag.

## 1. Relation to the other experiments

| Experiment | What it was | Relation |
|---|---|---|
| S1 / E0-Q3 (pre-registered quality gate, 8 November) | Answer quality of five Qwen3 builds on 200 queries, one rented GPU, sampled decoding | Its sample defines the sampling frame. Its scores and output lengths were used to design this experiment (section 15). None of its queries is reused |
| E0 (`e0-prereg-v1`) | Staged screen of a Qwen3-32B split against quantised edge models on emulated 12 and 16 GB edges. Stopped at its second bound | Method reference: deadline utility, the 3.0-point margin, the paired stratified bootstrap. Its STOP stands and is not tested again here: other machines, other memory budget |
| E1-OPT (`e1opt-prereg-v1` to `-v3`) | Optimised split runtime. Failed its first GPU stage; its one permitted fix is used up | Not reopened. Nothing of its optimised mode is used. Its protocol and segment code are reused only in the reference mode, through K16-HWFIT |
| K16-HWFIT (exploratory, 18 November) | The k = 16 split of Qwen3-32B on the laboratory pair: agreement with the unsplit model, time per token against the unsplit model | Its runtime, cut and correctness rule are reused unchanged, by import. Its timings motivated this experiment (section 15). Its numbers are not pooled with this experiment's |
| GEMMA-SELECT (`gemma-select-prereg-v2`) | The same decision for another model whose split came from another group. Completed | Harness design reference only. No code is imported. Results are never pooled |

## 2. Question, hypotheses and what is not tested

**Question.** On an edge machine of the DGX Spark class with one RTX A6000 as server, serving Qwen3 in bfloat16 without
reasoning traces, which configuration does the E0 decision rule select at each deadline: the split of Qwen3-32B, or
the best configuration that runs whole on the edge? Does the early decision of the staged screen agree with the
end-to-end measurement? If the query text may be sent to the server, does running a model whole on the server change
the choice?

The staged screen stops at its first stage when the split's model also fits whole on the edge, because the split can
then add no answer quality. K16-HWFIT found that the split nevertheless produced tokens faster than the same model
whole on the edge. A faster split can turn late answers into on-time answers. Whether that is enough to select the
split, against the same model and against smaller and faster models on the same edge, has not been measured.

- **H1 (primary).** Screen decision: Edge, at Stage 1 (section 6). Prediction: under the no-raw-text rule, no deadline
  cell selects Split against the best edge configuration. H1 is contradicted if at least one evaluated cell selects
  Split.
- **H2 (secondary, directional).** Against the same model whole on the edge alone (E32), Split is selected in at least
  one cell. If H1 and H2 both hold, the split helps the 32B model but a smaller model on the edge is the better
  configuration: model choice decides, not placement.
- **H3 (secondary, descriptive).** Under the open rule, with REMOTE admitted, no cell selects Split, and the best
  competitor is reported per cell. No screen decision exists for REMOTE, so nothing is confirmed or contradicted here.

**Not tested and never to be claimed from this experiment:** reasoning (thinking) mode; quantised builds; any memory
budget smaller than the edge machine's own; a split of a model that does not fit on the edge; cuts other than k = 16;
the optimised static mode; privacy of the transmitted activations; cost or energy.

## 3. Configurations

| Id | Execution | Model | Where it runs | Runtime |
|---|---|---|---|---|
| S | SPLIT | Qwen3-32B | edge: embedding, layers 0 to 15 and 48 to 63, norm, LM head, argmax. server: layers 16 to 47 | K16-HWFIT's runtime in reference mode, unchanged |
| E32 | EDGE | Qwen3-32B | whole, on the edge | Hugging Face model loaded by `k16_runtime.hf_model`, its own KV cache |
| E14 | EDGE | Qwen3-14B | whole, on the edge | same |
| E8 | EDGE | Qwen3-8B | whole, on the edge | same |
| R14 | REMOTE | Qwen3-14B | whole, on the server GPU. The edge sends token ids and receives the output tokens one by one | same loader, on the server |

All five use bfloat16 weights as published, pinned by revision and per-file SHA-256 (`frozen/`, and K16-HWFIT's
manifest for the 32B model), the same tokenizer file, the same chat template with thinking off, the same token ids
per query (pinned in the workload), greedy decoding, the same EOS set, the same context window of 4,096 tokens and the
same output cap.

### 3.1 Data rules

Two rules are evaluated and reported separately. They were fixed here and are never switched after a result.

- **no-raw-text (primary).** The query text and its token ids stay on the edge. REMOTE is excluded by definition,
  because it sends the token ids. Competitors of the split: E32, E14, E8.
- **open (secondary).** The token ids may be sent. Competitors of the split: E32, E14, E8 and R14.

The split sends hidden states, which can be partly inverted. No privacy property is claimed for any configuration.

### 3.2 Memory budget

The edge budget is the edge machine's own memory on the day. It is not capped. Every configuration must actually
load and run there (section 11). No smaller budget is emulated: nothing independent of the split's interest
establishes one for this machine, and an emulated cap was a stated limitation of E0.

Consequence: the split's model also fits whole on the edge. This experiment therefore cannot ask whether a split
gives access to a model that the edge cannot hold. It asks whether the split is the better way to run a model the
edge can hold, and whether that model is the right one.

### 3.3 Configurations that are excluded, and why

| Excluded | Reason | Direction of the effect |
|---|---|---|
| REMOTE with Qwen3-32B | Its weights (61.0 GiB) exceed one server GPU (48 GiB). One server GPU is the laboratory's allocation. The server preflight records the arithmetic | Removes a REMOTE option. Open-rule cells may understate REMOTE |
| REMOTE with Qwen3-8B | REMOTE is represented by the largest model that fits the server GPU, to stay inside the GPU budget | Open-rule cells may understate REMOTE at short deadlines |
| Quantised (AWQ) builds on the edge or the server | No runtime for them has been validated on either machine. E0's edge winner was a 4-bit build on another GPU | Removes edge competitors that would be faster. It can only favour the split |
| Thinking mode, for every configuration | At the 18 November speeds the 32B model cannot finish reasoning traces inside these deadlines, and one block per configuration would exceed the GPU budget | For the split: none expected (section 15). For smaller edge models at long deadlines: removes an option, which can only favour the split |
| Other Qwen3 sizes and the mixture-of-experts models | Not among the sizes S1 examined | Removes edge competitors. It can only favour the split |
| The split's optimised static mode | It was slower than reference mode on this pair and failed E1-OPT's check | None on the question asked |
| A second server GPU | Not allocated | Keeps REMOTE with the 32B model infeasible |

Every exclusion except the first two removes a competitor of the split or leaves the split unchanged. A result in
which no cell selects the split is therefore conservative with respect to these exclusions. A result in which a
cell selects the split holds only against the competitors that were run, and must say so.

### 3.4 Fairness and limits

- The whole-model configurations use the stock Hugging Face path, whose KV cache grows, while the split
  preallocates its caches. The summary reports the edge compute per layer of both (section 10). If they differ by
  more than 10%, a selection of the split is labelled as possibly owed to the runtime.
- The machines are shared and the run lasts more than a day, so every primary configuration is measured in two
  halves about half a day apart: S, E32, E14 and E8 on the first 80 queries of the seeded order, then the same four on
  the last 80. No configuration owns one time of day. Inside a half the configurations run one after another. A
  second pass of the split on the sentinel queries checks drift (section 13), and each configuration's time per token
  is reported per half (section 10).
- The link is an SSH forward from the edge to the server's loopback interface, as on 18 November. It is measured as
  it is and never assumed.

## 4. Workload (`workload/workload.json`, pinned)

- **Held-out queries.** 160 queries: MMLU-Pro (60), GPQA (60), LiveMathBench (40). They are ranks n to 2n-1 of the
  S1 permutation, the second half of the 400-query extension that E0 pinned on 8 November and never ran. No
  generation of S1, E0, K16-HWFIT or GEMMA-SELECT used them, and no design choice of this experiment looked at them.
  The benchmark's own cached records hold other models' outputs on these queries; they were not read for this design.
- LiveCodeBench is excluded because its scorer executes model-written code on the analysis machine.
- **Sentinel set:** 24 of them (9, 9, 6), run a second time through the split after its second half.
- **Halves:** the first 80 and the last 80 queries of the seeded order.
- **Correctness set:** the 5 shortest and 5 longest prompts in tokens, each with a 64-token continuation. The longest
  has 2,823 tokens, so the set includes a two-chunk prefill.
- **Format and decoding.** One user turn, the checkpoint's chat template, thinking off. Greedy decoding. Output ends
  at an EOS token, at min(3,072, 4,096 minus the prompt length) new tokens, or at the cutoff.
- **Cutoff.** A query still running after 300 s is stopped and counts as late in every cell.
- **Scoring.** The unchanged benchmark evaluators of this repository, on the decoded text, on the analysis machine.
- The public file holds keys, order and hashes, not prompt text. The prompts and gold answers are rebuilt from the
  public benchmark release by `qsel_workload.py build` and pinned by SHA-256.

## 5. Stages, in order

Each command first checks the tag, the site file, the host role, the checkpoint hashes, the free memory and the
GPU-hour allotment.

0. **GPU selection** on the server (CPU only; section 11).
1. **Preflight** on each host (CPU only): checkpoints, tokenisation against the pinned token ids, memory, ports.
2. **Reference**: E32's model on the correctness set, no timers. Records the screen decision.
3. **Serve**: the split's server half. **Correctness**: S against the reference, no timers. PASS or FAIL.
4. **First halves**, a fresh process each: timed blocks S1a, then E32a, E14a, E8a. The server is released after S1a.
5. **Serve** again on the same GPU. **Second halves**: S1b, then S2 (the sentinel queries through the split), then
   E32b, E14b, E8b. The server is released after S2.
6. The nine blocks of stages 4 and 5 are the primary blocks.
7. **Serve-remote**. **Timed block R14** (all 160 queries). Released.
8. **Scoring and summary** on the analysis machine.

No timing runs without a correctness PASS. No score is computed while the primary blocks can still continue; a run
that has ended incomplete is scored with a flag that records it. The primary result needs stages 0 to 5 and 8. Stage 7
serves only the open rule; if it does not run, the open rule is reported as not evaluated.

## 6. The screen as applied

The staged screen continues past its first stage only if the split's model answers at least 3.0 points better than
the best model that fits on the edge. Here the split's model is E32's model, and it fits on the edge if stage 2
loads it. The lead is zero by identity. **Screen decision: Edge, at Stage 1.** It is recorded by stage 2, before any
timed run. If the model cannot be loaded whole on the edge in two attempts, the outcome is NOT_EVALUABLE.

The timed stages measure what the screen skipped. They cannot change the screen decision. They test it.

## 7. Correctness gate (S against E32's model)

The rule is K16-CORRECT-v1 of K16-HWFIT, unchanged and pinned by hash. Teacher forcing over the prompt and the
reference's 64 greedy tokens, and free-running greedy generation, on the ten correctness prompts:

- every value finite;
- where the reference's top-1 minus top-2 log-probability is at least 0.5 nats, the argmax agrees at 99.5% or more;
- mean coarse KL divergence from the reference at most 0.05 nats;
- greedy output identical, or first different where the reference margin is below 0.5 nats, for at least 9 of 10;
- the edge and server cache positions in step at every round trip.

FAIL is final: outcome CORRECTNESS_FAIL, no timing is run or reported. The split passed this rule on these machines
on 18 November on other prompts; that run does not count here.

## 8. Measurements

- **Completion time** T(c, q): from the moment the token ids are handed to configuration c, model loaded and server
  connected, until the last output token is on the edge. One clock, in the edge process. For S it includes the
  session reset, the transfer of the prompt's hidden states and every round trip. For R14 it includes sending the
  token ids and receiving every token.
- **Time to first token** and **time per output token** ((last minus first) over (tokens minus 1)).
- **Memory:** required bytes computed from the checkpoint headers before loading; allocated peak after. A
  configuration fits if it loads and completes its block.
- **Communication:** for each split request, bytes each way, round-trip time and the server's own compute time, from
  the runtime's counters. For each REMOTE request, bytes each way and the server's own times.
- **Answer quality:** each configuration's own outputs, as produced inside the output cap and the cutoff, scored per
  query. An answer cut off at 300 s is scored as it stands and counts as late in every cell.
- **GPU time:** the ledger of section 12.

## 9. Decision rule and uncertainty (E0's rule)

- U(c, D): share of the 160 queries answered correctly with T(c, q) at most D.
- Deadlines D: 30, 60, 120 and 300 s (E0's first four; section 15 explains the fifth).
- Under a data rule, Δ(D) = 100 × [U(S, D) minus the highest U(c, D) among that rule's competitors].
- 95% interval: paired bootstrap, 10,000 resamples, seed 20261120, stratified by benchmark, the best competitor
  re-selected inside every replicate.
- **Split is selected at D only if Δ(D), rounded to 10⁻⁶ points, is at least 3.0 points and the lower end of the
  interval is above zero.** Otherwise the competitor with the highest utility is selected (equal utilities: an edge
  configuration before the server, and among edge configurations the smaller model). Cells where every configuration
  has U below 0.05 are not evaluated.
- Every cell is reported. No correction for the number of cells is applied to the rule.
- S is represented by its first pass.

With 160 queries the rule can detect only clear advantages. In a planning resample of the S1 queries (`posthoc/`),
an assumed advantage of 6 points was selected in 98% of workloads when the competitor was the same model, and in 45%
when it was another model, whose answers differ on more queries. A cell that does not select the split is therefore
not evidence that the split has no advantage there; the interval is what is reported.

## 10. Secondary analyses (reported, never a decision)

- **Same model:** the rule of section 9 with E32 as the only competitor (H2).
- **Open rule:** the rule with R14 added (H3).
- **Zero-network counterfactual:** the rule with the split's time reduced by everything of each round trip that the
  server did not report as compute. This is what a network-free bound would have assumed.
- **Runtime parity:** median edge compute per layer and decode step, E32 against the split's edge half.
- **Time per token by half:** each primary configuration's median time per output token in its second half over its
  first. The halves hold different queries, so this describes drift and does not test it.
- **Token agreement:** share of queries on which S and E32, and R14 and E14, wrote the same tokens.

## 11. Resource gates and non-interference

- **Before every load:** required free memory is computed from the tensors that will be loaded, plus KV cache for
  the full context, activations, loading transients and a margin (`qsel_models.py memory`). Short of memory:
  BLOCKED, nothing is loaded, try later. BLOCKED is not an attempt and not a result.
- **Again inside the process, before any weight is read.** Server: the device's own free-memory figure. Edge: its
  memory is unified and the device's figure excludes reclaimable page cache, so available RAM is read again and
  decides.
- **The server GPU is chosen by `select-gpu`**, before the first server stage. Eligible: a GPU of the server model
  that the private site file does not exclude and that has the larger of the two server loads free at that moment.
  Among the eligible: the fewest compute processes of other users, then the most free memory, then the lowest
  index. Until a server stage has started, the selection may be made again and the earlier record is kept. From
  then on it is fixed: a selected GPU that later lacks memory is BLOCKED, not replaced.
- Each server process sees exactly one GPU. A server holds the GPU only while a stage needs it: it exits when its
  last edge connection closed 20 minutes ago, or when no edge came for 30 minutes.
- Loopback binds only, on ports of the experiment's own. The REMOTE wire requires a token of the experiment's own.
- No process of any user is signalled, and nobody is asked to free memory. Outputs record how many other compute
  processes shared a GPU and their total memory, never names, owners or command lines.

## 12. GPU budget, wall-clock limits and repeats

**Ceiling: 48 GPU-hours over both machines, everything included.** One GPU held by one of our processes for one
hour is one GPU-hour, whether it computes or waits. The ceiling is not raised under this tag.

| Stage | Edge GPU-hours, planned | Server GPU-hours, planned | Longest a stage can run by itself |
|---|---|---|---|
| reference | 0.4 | 0 | 1 h |
| serve, correctness, S1a | 0.3 + 3.7 | 4.7 | 1 h (correctness); 80 × 300 s + 0.5 h = 7.2 h (a half block) |
| E32a, E14a, E8a | 4.3 + 2.4 + 1.9 | 0 | 7.2 h each |
| serve, S1b, S2 | 3.7 + 1.2 | 5.3 | 7.2 h; 24 × 300 s + 0.5 h = 2.5 h (S2) |
| E32b, E14b, E8b | 4.3 + 2.4 + 1.9 | 0 | 7.2 h each |
| serve-remote, R14 | 0 | 2.6 | 160 × 300 s + 0.5 h = 13.8 h |
| **Planned total** | **26.5** | **12.6** | planned 39.1 of 48 |
| **Allotment (hard)** | **32.5** | **15.5** | 48.0 |

The server figures include the 20 idle minutes after which a server releases its GPU by itself.

The planned figures are arithmetic from S1's sampled output lengths and the 18 November timings (`posthoc/`). They
are not predictions of greedy output lengths.

**Enforcement.** Each host keeps an append-only ledger. Every GPU stage opens an entry before it loads anything and
closes it when it ends. An entry that was never closed is charged to its last heartbeat plus ten minutes. A stage
starts only inside its host's allotment and is stopped by a wall-clock limit equal to the smaller of its own longest
run and what the allotment leaves. The two allotments sum to the ceiling, so the worst case over both machines is
48 GPU-hours by construction. If an allotment is used up before the primary blocks are complete, the outcome is
INCOMPLETE.

**No second half that cannot be finished.** The second halves repeat the first halves' work on other queries. S1b
starts only if the edge has at least 0.9 times the GPU time of the four first-half blocks left. Otherwise nothing more
is loaded and the run is INCOMPLETE. The first halves are then a balanced partial result: all four primary
configurations on the same 80 queries. The rule of section 9 is computed on them and reported as description, never as
the outcome.

**Elapsed time.** The stages are sequential because one edge GPU runs four of the five configurations. Planned GPU
execution is about 27 hours for the primary result and about 29 with REMOTE. This exceeds the project's target of
24 hours. It was not brought under the target by cutting queries, because 160 queries is already the smallest
workload at which the rule can detect a clear advantage (section 9).

**Repeats.**
- A timed block that dies (error, lost forward, limit) is resumed from the first query without a complete row, in a
  fresh process after the same warm-up. Rows already written are kept and never rewritten. A block has at most
  three attempts. Every row carries its attempt number.
- Reference and correctness: an attempt that dies is kept under another name and repeated once.
- An attempt begins when a stage has passed every check and is about to load. A stop before that (BLOCKED, server not
  answering, budget) is not an attempt. An attempt that is killed without a trace still counts.
- A wall-clock limit ends a command wherever it is, including a server that is serving.
- A completed stage is final. Starting it again is a stop and changes nothing.

## 13. Stopping rules and outcomes

- **Stability:** median completion time of the second split pass over the first, on the sentinel queries, must lie
  in [0.90, 1.10]. Otherwise UNSTABLE.
- **Environment:** every edge stage must report the same library versions and the same site file, and every server
  stage the same site file. Otherwise INCOMPLETE.

| Outcome | Condition | What it means |
|---|---|---|
| SCREEN_CONFIRMED | No evaluated no-raw-text cell selects Split | The screen's early decision agrees with the measurement. With H2 true: the split speeds up the 32B model but another edge configuration is better. With H2 false: the speed advantage of 18 November did not carry over to answers on time |
| SCREEN_CONTRADICTED | At least one no-raw-text cell selects Split | Against the configurations that were run, the split is the best configuration at some deadline. The screen's first stage is insufficient when the server is faster per layer than the edge, and must compare speed as well as quality. Quantised and reasoning-mode competitors were not run (section 3.3) |
| CORRECTNESS_FAIL | The gate of section 7 fails | The split does not reproduce the whole model here. No timing exists |
| SPLIT_INFEASIBLE | The server failed to start or the correctness stage did not finish, twice in all, after passed preflights | The split could not be run in this configuration |
| NOT_EVALUABLE | The 32B model could not be loaded whole on the edge in two attempts, or no cell reached the 0.05 floor | The premise of section 6 does not hold on the day, or the workload is too hard for every configuration |
| UNSTABLE | The two split passes differ by more than 10% | The machines drifted during the run. Utilities and what the rule would give are reported and labelled; no selection is claimed |
| INCOMPLETE | A primary stage is missing, a block used its three attempts, the environment or a site file changed, or an allotment ran out | No selection is claimed. Quality and latency of every completed block are reported as partial; if the four first halves are complete, the rule on those 80 queries is reported as description |

All are reportable results. None is rerun under this tag.

## 14. What a result may and may not claim

May claim: the screen's decision; each configuration's measured utility, quality, memory, time to first token, time
per token and completion time; the split's communication cost; which configuration the rule selects per deadline
under each data rule; whether the measurement agrees with the screen.

May not claim: anything about reasoning mode, quantised models, smaller edge budgets or other cuts; that a split is
never or always worthwhile; a privacy property; that REMOTE was evaluated at its best; a comparison with Gemma;
generality beyond one model family, one cut, one machine pair, one run and 160 queries.

## 15. Disclosures: choices informed by earlier outcomes

The PI knew the following before writing this protocol, and each shaped it.

| Choice | Informed by | Alternative, and whether it would change the reading |
|---|---|---|
| Model family and the 32B split | S1: Qwen3-32B is the most accurate of the five builds. K16-HWFIT: its k = 16 split runs on this pair | Another family would be a new experiment. Results here do not transfer |
| Cut k = 16 | It is the only cut that has run on this pair. The server GPU cannot hold the middle layers of E0's cuts | A cut with more layers on the faster server GPU would favour the split and does not fit the GPU. A cut with fewer would be slower |
| Edge competitors 14B and 8B in bfloat16 | S1 examined these sizes (the 14B only as a 4-bit build). The post hoc screen says the 14B model decides the 300 s cell | Without the 14B model the post hoc screen predicts a split selection at 300 s. Leaving it out would have been a choice in the split's favour |
| No quantised builds | No validated runtime on these machines | Section 3.3. Their absence can only favour the split |
| Full edge memory | The whole 32B model loaded on this edge on 18 November | A capped edge would exclude E32 and possibly E14 and make the split look necessary. No independent ground for a cap exists |
| Non-thinking mode only | S1 output lengths: reasoning traces of 2,500 to 8,900 tokens on average. K16-HWFIT: 3.3 to 4.4 tokens per second | With thinking mode the post hoc screen never prefers it for the split up to 600 s. For small edge models it could add utility at long deadlines, against the split |
| Deadlines 30 to 300 s, cutoff 300 s | E0's grid without its 600 s cell. The post hoc screen puts the split 1.5 points ahead of the same model at 600 s, below the margin, and the cell costs about 5 GPU-hours | Dropping a cell cannot produce a split selection. It removes one cell in which H1 could have been contradicted, with a post hoc chance of a few percent |
| Greedy decoding | Deterministic, as in K16-HWFIT's checks and timings. The model card recommends sampling | Greedy output can be longer or repetitive. It affects every configuration; the sign is unknown. Stops at the cap and at the cutoff are reported per configuration |
| Output cap 3,072 and context 4,096 | The split runtime's preallocated cache. In S1, non-thinking answers longer than 3,072 tokens were 1 to 6% and almost never correct | A lower cap would cut correct answers. A higher one needs a larger cache |
| 160 queries, 24 sentinel queries | GEMMA-SELECT's sizes; the GPU budget | Section 9 |
| Margin 3.0 points, interval above zero, bootstrap, utility floor 0.05, stability band | E0 and GEMMA-SELECT, unchanged | None considered |
| Correctness thresholds | K16-HWFIT, unchanged. The split passed them on 18 November | None considered |
| Two halves per primary configuration; order S, E32, E14, E8 in each; REMOTE last | Shared machines, a run of more than a day; the primary blocks first | One block per configuration would give each its own time of day. Halves reduce that and do not remove it: inside a half the order is fixed. REMOTE is measured once, at the end |
| Host allotments 32.5 and 15.5 GPU-hours | Planning arithmetic above | None |

**The post hoc screen** (`posthoc/POSTHOC_SCREEN.md`, written before this tag, exploratory). It combines S1's scores
and output lengths with K16-HWFIT's median timings, and assumes that a smaller model's time on the edge scales with
its weight bytes. It predicts: against the same model on the edge the split is selected at 30 to 300 s; against the
8B model as well, only at 300 s; with the 14B model added, in no cell, with Δ = 0.0 points [−6.6, +5.9] at 300 s. The
14B model's speed on the edge has never been measured; the prediction flips if it needs more than about 0.85 of the
32B model's time. These numbers come from incompatible protocols and are not evidence for H1.

**Also known to the PI:** the outcome of GEMMA-SELECT, and its measured link cost on the same machine pair.

## 16. Integrity

A target command runs only from a checkout whose HEAD is the tag and whose experiment directory and reused runtime
directories are unmodified. Checkpoints are verified by hash on each host. Token ids are verified against the
workload on the edge. The code identity crosses in the HELLO of both wires, and a server refuses another revision.
The private site file is hashed into every output. Raw outputs stay outside the repository, because they name hosts
and devices. The summary carries none of that. `--dev` exists for CPU tests with tiny random models, and dev
outputs are refused as results.

Before this tag the complete command sequence was run on a CPU with tiny random models of the real layer counts,
under the library versions of both hosts, and with the real prompts and tokenizer. That showed that the drivers run
and that the split reproduces the whole model on random weights. It is not a Qwen3 result, and no timing from it is
used.
