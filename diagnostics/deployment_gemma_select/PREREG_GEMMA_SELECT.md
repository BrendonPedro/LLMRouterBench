# GEMMA-SELECT: an Edge-versus-Split deployment decision for Gemma-4 E4B. Pre-registration

**Frozen by the tag `gemma-select-prereg-v2`.** No GPU has executed any code of this experiment before that tag, and
no GPU result exists. Every number below was fixed before any result. The decision files (this text, `gsel_spec.py`,
`gsel_stats.py`, `gsel_workload.py`, `gsel_whole.py`, `gsel_adapter.py`, `workload/workload.json`) are never edited
after the tag. A change of any rule is a new pre-registration under a new tag.

## 0. Version 2: what changed from `gemma-select-prereg-v1`, and why

Version 1 was frozen on 19 November 2026 and was never executed on a GPU. Its tag is unchanged and stays published.
An audit of v1 against records of the same two machines, made before any GPU run, found that v1 could not be executed
as written. Version 2 replaces it. **No decision rule changed:** the question, the hypotheses, the three
configurations, the workload and its order, the correctness thresholds, the measurements, the deadlines, the margin,
the bootstrap, the stability band, the bound and the outcome definitions are those of v1, and `gsel_stats.py`,
`gsel_workload.py`, `gsel_whole.py` and `workload/workload.json` are byte-identical to v1.

| # | Change | Reason |
|---|---|---|
| 1 | Edge, check inside the process before a whole-model load: available RAM is read again, and the device's own free-memory figure is recorded but not applied (section 11) | On unified memory that figure excludes reclaimable page cache. Records of the exploratory Qwen3 run on this edge machine (18 November) show it reporting 860 MiB free with 74,911 MiB of RAM available, after which a model of 62,489 MiB loaded. Under v1 the edge-matched baseline would have been BLOCKED indefinitely |
| 2 | The device's figure is still applied to a split half | The split runtime applies it itself and refuses to load otherwise. The harness asks first, so that a shortfall is BLOCKED and not a failed attempt of the split |
| 3 | The server GPU is chosen by a fixed rule (`select-gpu`), pinned, and fixed for the run once a server stage has run. The site file lists GPUs that are never eligible. A stage run with another site file makes the run INCOMPLETE (sections 11, 12) | The machines are shared and free memory changes by the hour. v1 left the choice to a hand-written site file and did not check that it stayed the same |
| 4 | A reference attempt that died is set aside and may be repeated. A second start of a completed stage is a stop, not a failed attempt (section 12) | In v1 the partial files of a died reference attempt made every repeat fail, which would have ended in NOT_EVALUABLE for a reason unrelated to the model |
| 5 | A split stage starts only if the experiment's server answers through the forward. If it does not, that is a stop before anything is loaded (section 12) | In v1 a forward that was down counted as a failed attempt of the split, and two of them as SPLIT_INFEASIBLE |
| 6 | Wall-clock limits per command (section 12) | v1 bounded a query (600 s) and a socket operation, but not a command that hangs |
| 7 | The private adapter gained the readiness call that change 5 uses. Its pin changed | |
| 8 | Both preflights check the adapter pin. The edge preflight also reports, without reading a weight, whether the checkpoint header holds every key and shape the whole-model class expects under the installed library | So that a load that would fail is seen before any GPU stage. The report is advisory and changes no verdict |

## 1. Relation to the other experiments

| Experiment | What it was | Relation |
|---|---|---|
| E0 (`e0-prereg-v1`) | Staged screen of a Qwen3-32B split against quantised edge models, emulated edge budgets | Method reference only: deadline utility, the 3.0-point margin, the paired stratified bootstrap. No E0 code is imported. Its query sample and the benchmark evaluators are reused unchanged |
| E1-OPT (`e1opt-prereg-v1`) | Optimised Qwen3 split runtime | None. Untouched |
| GEMMA-SPLIT (`gemma-split-prereg-v1`) | Capture of an existing, running Gemma split deployment | Abandoned at its Stage 0 with STOP_INCOMPLETE. It produced no measurement. It asked a different question, about a running service. Nothing of it is imported. Two identity hashes established during its Stage 0 (checkpoint and split source) are reused as pins |
| GEMMA-SELECT v1 (`gemma-select-prereg-v1`) | The first freeze of this experiment | Replaced by this version before any GPU run (section 0). It produced no output |
| Exploratory Qwen3 two-host run of 18 November | Split against the unsplit model | None |

Gemma results are never pooled with, or reported as a continuation of, Qwen3 numbers.

## 2. Question and hypotheses

**Question.** For Gemma-4 E4B with an edge machine of the DGX Spark class and one RTX A6000 as server, which
configuration does the E0 decision rule select at each deadline, Edge or Split, and does the early decision of the
staged screen agree with the full end-to-end measurement?

- **H1 (primary).** The screen's decision is Edge, taken at its first stage (section 6). Prediction: in the full
  measurement no deadline cell selects Split. H1 is contradicted if at least one evaluated cell selects Split.
- **H2 (secondary).** The screen's timing bound is optimistic: for at least 95% of queries the measured completion
  time of the split is at least the bound.

**Not tested and never to be claimed from this experiment:** a model-selection or quality advantage of any
configuration. The split and the edge baseline run the same checkpoint, so answer quality is equal by construction
up to numerical effects. The comparison is a same-model deployment decision.

## 3. Configurations

### 3.1 The three configurations

| Id | Configuration | Runtime | Role |
|---|---|---|---|
| S | Split: layers 0 to 23 on the edge, 24 to 41 on the server | a third party's unpublished split implementation, float32, eager attention, run from a hash-verified copy in our own configuration | under test |
| Em | Edge-matched: whole model on the edge | Hugging Face transformers, float32, eager attention | edge competitor, and the correctness reference |
| Es | Edge-stored: whole model on the edge | Hugging Face transformers, bfloat16 as stored, default attention | edge competitor |

All three use the same checkpoint (seven files pinned by SHA-256 in `gsel_spec.py`), the same token ids, greedy
decoding, the same EOS set (from `generation_config.json`), at most 1,024 new tokens (the split runtime's own limit)
and prompts formatted with the checkpoint's chat template, thinking off. Both edge configurations run on the machine
that holds the split's edge half.

**The edge competitor of the decision rule is the better of Em and Es in each cell**, re-selected inside every
bootstrap replicate.

### 3.2 The split implementation

It is not in this repository and is not redistributed. Its source directory is pinned by a manifest hash
(`SOURCE_MANIFEST_SHA256`, 54 files) that every command recomputes. A private adapter file, pinned by
`ADAPTER_SHA256`, calls it unmodified through the contract in `gsel_adapter.py`. The experiment starts its **own**
server instance on a loopback port of its own, with its own credentials, reached through its own SSH forward. It does
not use, contact or depend on any other running service.

### 3.3 Other edge competitors: none available

A smaller model of the family or a quantised build would make this a model-selection test. **None is available to
the experiment**: no such checkpoint or build is held, captured or hash-pinned by us, and builds belonging to other
users of the machines are not ours to run. None is used, and none may be added under this tag.

### 3.4 Why the comparison is fair, and its limits

- Same weights, tokens, decoding, cap and edge machine for all three.
- Em removes dtype and attention kernel as explanations of a difference. Es is the form an operator would deploy.
- The edge baseline is not handicapped: the better of the two is used.
- Limits: the blocks are not simultaneous and the machines are shared (section 11). The split's server and the
  link are measured as they are. Running the whole model on the server is out of scope and is not evaluated.
- **Data rule.** The server receives intermediate tensors derived from the prompt and produces the output tokens.
  The experiment makes no claim that this split keeps query text on the device, and treats the comparison as the
  case without a data rule.

## 4. Workload (`workload/workload.json`, pinned)

- **Utility set:** the 160 queries of the E0 sample (`diagnostics/deployment_s1/sample_200.jsonl`, pinned) from
  MMLU-Pro (60), GPQA (60) and LiveMathBench (40), with their gold answers, in one seeded order. LiveCodeBench is
  excluded because its scorer executes model-written code.
- **Sentinel set:** 24 of them (9, 9, 6), run a second time through the split.
- **Correctness set:** the 5 shortest and 5 longest prompts, each with a 64-token continuation.
- **Scoring:** the unchanged benchmark evaluators of this repository, on the decoded text, on the analysis host.

## 5. Stages, in order

Each command first checks the tag, the site file, the host role, the source and checkpoint hashes and the free memory.

0. **GPU selection** on the server (CPU only; section 11).
1. **Preflight** on each host (CPU only).
2. **Server profile**: the server half alone (secondary bound).
3. **Serve**: the experiment's server instance, left running.
4. **Reference**: Em on the correctness set, no timers. Records the screen decision.
5. **Correctness**: S against that reference, no timers. PASS or FAIL.
6. **Edge profile**: the edge half alone.
7. **Timed blocks**, fresh process each: S (utility set), Es, Em, S again (sentinel set).
8. **Scoring and summary** on the analysis host, after every block is complete.

No end-to-end timing runs without a correctness PASS, and the segment profiles are used only after one. No score is
computed before the last block is complete.

## 6. The screen as applied

The staged screen of E0 continues past its first stage only if the split's model answers at least 3.0 points better
than the best model that fits on the edge. Here the split's model is Em's model, and Em fits on the edge if stage 4
loads it. The lead is therefore zero by identity. **Screen decision: Edge, at Stage 1.** It is recorded by stage 4,
before any timed run. If Em cannot be loaded, there is no matched competitor and the outcome is NOT_EVALUABLE.

The timed stages measure what the screen skipped. They cannot change the screen decision. They test it.

## 7. Correctness gate (S against Em)

Teacher forcing over the 64 continuation tokens of each prompt (640 positions) and free-running greedy generation.

- Every logit finite.
- At positions where the reference's top-1 margin is at least 0.5 nats, the argmax agrees in at least 99.5%.
- Mean KL divergence from the reference at most 0.05 nats, on the full vocabulary.
- Greedy output identical, or first different at a position whose reference margin is below 0.5 nats, for at least
  9 of 10 prompts.

FAIL is final: outcome CORRECTNESS_FAIL, no timing is run or reported. The thresholds are judgment calls made
without Gemma cross-device data. They are not changed after any result.

## 8. Measurements

- **Completion time** T(c, q): from the moment the token ids are handed to configuration c, models loaded and server
  ready, until the last output token is on the edge. One clock, in the edge process. For S it includes session
  start, the transfer of the prompt's tensors and every step.
- **Time to first token** and **time per output token** ((last minus first) over (tokens minus 1)).
- **Cutoff:** a query still running after 600 s is stopped and counts as late in every cell.
- **Memory:** required bytes estimated from the checkpoint header before loading, and peak allocated memory after.
- **Network:** never assumed. For each split request the adapter reports bytes sent, round-trip time and the
  server's own compute time. The summary reports their difference and the effective rate.
- **Answer quality:** each configuration's own outputs, scored per query.

## 9. Decision rule and uncertainty (E0's rule)

- U(c, D): share of the 160 queries answered correctly with T(c, q) at most D.
- Δ(D) = U(S, D) minus the larger of U(Em, D) and U(Es, D), in points.
- Deadlines: 5, 10, 30, 60, 120, 300 and 600 s. The last five are E0's. The first two are added because this model is
  far smaller than E0's and its output is capped, so that a deadline can bind.
- 95% interval: paired bootstrap, 10,000 resamples, seed 20261119, stratified by benchmark.
- **Split is selected at D only if Δ(D) is at least 3.0 points and the lower end of the interval is above zero.**
  Otherwise Edge. Cells where every configuration has U below 0.05 are not evaluated.
- Every cell is reported. No correction for the seven cells is applied to the rule. The outcome names how many cells
  selected Split.
- S is represented by its first pass.

## 10. Secondary: the timing bound

From the isolated segment profiles (prefill at 64, 256 and 1,024 tokens, 32 decode steps, 3 repetitions): prefill
time is a least-squares line in prompt length, a decode step is the median step. Bound for a query = 0.9 times
(edge segment + server segment), no network, with the split's own output length. Reported: the share of queries with
measured time at least the bound (H2), and the decision the rule would give with bound times in place of measured
ones. E0's first bound (datasheet server speed) is not used, because the server is at hand.

## 11. Resource gates and non-interference

- **Before every load:** the weight bytes of what is loaded are computed from the checkpoint header. Required free
  memory is 1.25 times that plus 1 GiB for a split half (the split runtime's own rule) and 2 times that plus 1 GiB for
  a whole model (it is read into RAM and then moved to the device). The server needs it free on its one pinned GPU and
  in RAM. The edge needs it in available RAM. Short of memory: BLOCKED, nothing is loaded, try later. BLOCKED is not
  a result.
- **Again inside the process, before any weight is read.** Server: the device's own free-memory figure. Edge: its
  memory is unified, and the device's own figure excludes reclaimable page cache (section 0, change 1). So available
  RAM is read again and decides for a whole model, and the device's figure is recorded. For a split half the device's
  figure must pass as well, because the split runtime applies it itself.
- **The server GPU is chosen by `select-gpu`**, before the first server stage. Eligible: a GPU of the server
  model that the private site file does not exclude and that has the server half's required memory free at that
  moment. Among the eligible: the fewest compute processes of other users, then the most free memory, then the lowest
  index. The choice is written into the site file and recorded. Until a server stage that loads the model (the server
  profile or the server itself) has written an output, the selection may be made again, and the earlier record is
  kept. From then on it is fixed: every server command refuses a site file that names another GPU, and a selected GPU
  that later lacks memory is BLOCKED, not replaced. The site file lists the GPUs that carry other services of the
  lab. They are never eligible.
- The server process sees exactly one GPU, the selected one.
- Loopback binds only. Own port, checked against the ports the site file lists as used by others. Own credentials,
  outside the source directory, mode 0600.
- No process of any user is signalled, and nobody is asked to free memory. Outputs record how many other compute
  processes shared a GPU and their total memory, never names, owners or command lines.
- Shared machines are a stated limit, not a gate. Stability is checked instead (section 12).

## 12. Stopping rules, repeats and outcomes

- **Stability:** median completion time of the second split pass over the first, on the sentinel queries, must lie
  in [0.90, 1.10]. Otherwise UNSTABLE.
- **Repeats:** a block that dies is kept under another name and repeated from its start, at most twice. A completed
  block is final. A reference attempt that dies is kept under another name in the same way. Starting a completed
  stage again is a stop and changes nothing.
- **Wall-clock limits:** GPU selection 10 minutes; preflight and each profile 1 hour; reference and correctness
  2 hours each; a timed block 600 s per query plus 1 hour (27.7 hours for 160 queries, 5 hours for the sentinel
  pass). A command that reaches its limit is an attempt that died.
- **Server not answering:** a split stage first asks the experiment's server, through the forward, whether it is
  ready. If it does not answer, the command stops before anything is loaded. That is a fault of the setup, not an
  attempt of the split, and it is not counted.
- **Environment:** every edge stage must report the same library versions, and every stage of a host the same site
  file. Otherwise INCOMPLETE.
- Outputs are written once. Scores and the summary are computed by the frozen code.

| Outcome | Meaning |
|---|---|
| SCREEN_CONFIRMED | No evaluated cell selects Split. The screen's early decision agrees with the measurement |
| SCREEN_CONTRADICTED | At least one cell selects Split. The screen's first stage was wrong for this system |
| CORRECTNESS_FAIL | The split does not reproduce the whole model. No timing |
| SPLIT_INFEASIBLE | The split could not be started or run in our configuration in two attempts after passed preflights |
| NOT_EVALUABLE | Em could not be loaded in two attempts, or no cell reached the 0.05 floor |
| UNSTABLE | The two split passes differ by more than 10% |
| INCOMPLETE | A stage is missing, a block exceeded its repeats, or the environment or a site file changed |

All are reportable results. None is rerun under this tag, except that BLOCKED attempts do not count.

## 13. What a result may and may not claim

May claim: the screen's decision; each configuration's measured utility, quality, memory and latency; whether the
measurement agrees with the screen in each cell; whether the bound was optimistic.

May not claim: a model-selection advantage; that the method chooses correctly between configurations of different
quality; anything about any other running service; anything about server-only execution; any privacy property; any
comparison with Qwen3; generality beyond one model, one cut, one machine pair and one run.

A SCREEN_CONFIRMED result is a second, independent application in which the early decision held. It is an
equal-quality case and says so wherever it is reported.

## 14. Disclosures

- The screen decision is known before the run. What is at risk is whether the measurement agrees with it.
- Before this protocol was written, the PI saw latencies returned by six requests to another running Gemma split
  service, and the split's own documents state CPU timings and a same-host parity result. None is used here.
- The split's authors report CPU tests on one host only. Its behaviour on these GPUs is unknown to us.
- The GPUs are shared, the link is an SSH forward over a wireless interface on the edge side, and blocks are not
  simultaneous.
- Before this tag the complete command sequence was run on a CPU with the unmodified split runtime on both sides and
  a random model of the runtime's wire dimensions, under the library versions of both hosts. It showed that the
  drivers run and that the split reproduces the whole model on random weights. It is not a Gemma result, and no
  timing from it is used.

## 15. Integrity

A target command runs only from a checkout whose HEAD is the tag and whose experiment directory is unmodified.
The checkpoint, the split source and the adapter are verified by hash. The private site file is hashed into every
output. Raw outputs stay outside the repository, because they name hosts and devices. The summary carries none of
that. `--dev` exists for CPU tests with a test double, and dev outputs are refused as results.
