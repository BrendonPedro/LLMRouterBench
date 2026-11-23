# ROUTER-INTEG `router-integ-v1`: clarification 1

Written on 23 November 2026 and approved by the PI, with the additions marked below, on 24 November 2026. Both dates
are before any GPU stage: no GPU had executed any code of this protocol and no result existed.

This note changes no frozen file and no rule, sample, threshold or setting. The freeze record `frozen/FREEZE.json`
is untouched. The note states how passages of the frozen protocol are to be read, and declares four descriptive
counts in advance. If any of it should instead be a rule, that is a new protocol version.

## 0. In plain terms

- The router is five cross-validation heads. Each head scores only queries outside its own training folds.
- This protocol is an out-of-fold experimental integration. It does not produce or test a deployable checkpoint for
  arbitrary new queries.
- Macro-F1 against the new, hardware-derived labels is not comparable with the earlier 0.600.
- Truncation (the 600 s cutoff, the 16,384-token cap) and execution on this hardware can change which model is the
  winner of a query. The winner target of this protocol is therefore not the router study's target.
- Placement is fixed by the data-transfer rule: no raw query transfer means EDGE, raw query transfer allowed means
  REMOTE. It is not a learned optimiser and is not fitted to anything.

## 1. The router heads are selected by the known evaluation fold (protocol sections 2, 6, 9, 10)

Every query of the sample belongs to the router study's pool, and the pipeline scores it with the one fold head whose
training folds did not contain it (`router/fold_map.json`; the adapter refuses a query that is not in the pool or
whose text differs). This is out-of-fold research evaluation. It is not inference on an unseen query without a fold:
for such a query no head is designated, and this protocol does not choose one (a single fold head, an average of the
five, or a head trained on the whole pool would each be a new, unevaluated router).

Supported by `router-integ-v1`: out-of-fold decisions of the router study's five-fold procedure, each made by a head
that never saw the query, can be connected to real execution; a router of this architecture runs on the edge device
at the measured latency; the pipeline executes the model the router selected.

Not supported: that one deployable router checkpoint was evaluated; anything about queries outside the pool.

In section 9, "deployable as a router" for policy C, and the label "frozen router (deployable)" in generated tables,
mean a router of deployable size whose decisions are out-of-fold. Reports must say "out-of-fold decisions of the
five-fold router", not "the deployed router".

## 2. Labels B are a time-limited target (sections 7.2, 8, 11, 14)

- Correctness of a row is the unchanged evaluator's verdict on the text as it stood when the request ended, whatever
  ended it (end of sequence, the 16,384-token cap, the 600 s cutoff). A request that failed counts as wrong.
- A row ended by the cutoff has a completion time above 600 s, so it is never correct by any deadline, including
  600 s. It can still be scored correct "as it stands"; then it counts in "correct within the cutoff" and in labels B.
- Labels B are derived separately from the EDGE rows and from the REMOTE rows of this protocol. No other reference is
  used. Labels A (the released one-shot labels) remain the router's training target.
- The original labels had no time limit. Labels B therefore differ in kind: "alone correct within 600 s on this
  hardware", not "alone correct". A model that reasons longer is cut more often. Macro-F1 against B is not comparable
  with the router study's 0.600 and is not evidence about the router's training.

Declared before any GPU run, reported with the results, descriptive only (no rule depends on them):

a. per configuration: rows ended by the cutoff or the cap, and how many of those were scored correct;
b. on the queries whose four rows all ended at an end-of-sequence token: the label counts per execution mode and the
   router's Macro-F1 and AUC with the number of queries, without an interval;
c. per execution mode: queries whose label B has a winner only because the other model's request failed;
d. (added 24 November) for each model and each placement, that is for each of the four configurations: among the
   answers that were not cut by the 600 s cutoff or the 16,384-token cap, the number of queries whose correctness
   equals the correctness of their stored original output, with the denominator. The denominator is the rows of that
   configuration that ended at an end-of-sequence token and were scored; failed requests are not answers and are
   left out, and their number is reported beside it. This is a serving-consistency diagnostic for the change of
   backend and for the position-scale setting of DeepSeek (protocol, section 7.2). The original and the new answer
   are different stochastic draws, so a disagreement is not proof of a fault. The count is not used to tune,
   select or change anything.

## 3. Placement and the profiling rule (sections 5, 12)

The placement rule is fixed: policy B runs on REMOTE whatever the profiling stage shows. If the server is not faster
than the edge, the run continues as frozen and the result reports it; all four configurations are recorded for every
query, so the other placement is visible as description. Changing the placement after profiling would be a new
protocol version, disclosed as informed by profiling.

The profiling stop (median time per output token above 1.5 times the planning value) is applied by the operator
from the printed rows; the code does not enforce it. It guards the budget arithmetic. It is not a performance
expectation: the edge reference (0.0966 s) was measured for Qwen3-8B only, greedy, without reasoning, at contexts up
to 4,096 tokens; the server reference (0.0575 s) was measured for Qwen3-14B, a different model. No compatible measured
reference exists for DeepSeek on either host or for any 8B model on the server. A server that is slow for an 8B model
but below 0.086 s per token would pass the rule.

## 4. Wording

- "All four configurations are measured for every query" (section 5) and "What is measured" (section 8) describe the
  design. When this note was written no configuration of this protocol had been measured. The only archived physical
  measurement of either model on either host is Qwen3-8B whole on the edge under QWEN-SELECT's protocol.
- "No Git commit is involved" (protocol header) describes the freeze mechanism, which is the hash record. Publishing
  the frozen files under a Git tag afterwards does not change them; the tag's annotation carries the hash of
  `frozen/FREEZE.json`.
