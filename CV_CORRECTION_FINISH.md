# Finish Corrected CV Diagnosis

## Completed Work — DO NOT RERUN

Task 1 is complete.

Actual nonlinear MiniLM MLP-64 score distillation under frozen stratified
5-fold CV:

Macro-F1 = 0.6003 ± 0.0154.

The previous 0.5922 result was confirmed to be the old linear Ridge student.

Task 2 is complete.

GTE classifier-capacity results:

- LogReg default: 0.5980
- LogReg + training-only threshold: 0.6153
- MLP-64 default: 0.5911
- MLP-64 + threshold: 0.6063
- RBF-SVM default: 0.6128
- RBF-SVM + threshold: 0.6260 ± 0.0296

No method reached 0.70.
Best single fold was approximately 0.666.

Linear LogReg AUC = approximately 0.683
RBF-SVM AUC = approximately 0.674
MLP-64 AUC = approximately 0.665.

Do not rerun Tasks 1 or 2.
Reuse their saved results.

## Task 3 — Correct Efficiency Evidence

Create:

diagnostics/stratified_cv/corrected_efficiency.csv

Clearly separate:

1. representation/encoder parameters
2. router-head parameters
3. approximate/verified total deployed parameters
4. trainable router parameters
5. router-head disk size
6. cached-embedding head training time
7. cached-embedding head inference time
8. online encoder requirement
9. offline teacher requirement
10. whether teacher is required at deployment

Do not describe cached-embedding measurements as end-to-end latency.

For MiniLM:
- report encoder as all-MiniLM-L6-v2
- approximately 22.7M encoder parameters, but verify from loaded model config
  where practical
- nonlinear MLP head = 24,705 parameters

For GTE:
- deployment requires Alibaba-NLP/gte-Qwen2-7B-instruct
- obtain parameter count from local/config/model metadata if possible
- do not pretend the 3,585 LogReg parameters are the total deployed model

Use explicit labels such as:
"router-head-only cached-embedding latency"

## Task 4 — Raw-Query End-to-End Timing

Measure MiniLM end-to-end inference on the CURRENT machine.

Environment already observed:
- CPU only
- torch has no CUDA
- approximately 9 GB RAM
- 4 CPU cores
- sentence_transformers is not installed

Use transformers directly if necessary.

End-to-end path must include:

raw query
→ tokenizer
→ MiniLM encoder
→ correct sentence pooling
→ normalization if required
→ router head
→ routing decision

Use a reproducible subset of real benchmark queries.

Include warm-up runs.

Report at minimum:
- number of timed queries
- mean latency/query
- median latency/query
- queries/second
- hardware/environment
- batch size

Keep cached-head-only timing separately.

For GTE-7B:
do NOT attempt loading if it cannot safely fit in the approximately 9 GB RAM
environment.

If it cannot be fairly measured, report:

"End-to-end GTE timing was not measured on this machine because the 7B
encoder cannot be fairly loaded within available memory."

Do not estimate fake latency.

## Task 5 — Final Report

Create:

diagnostics/stratified_cv/CV_CORRECTION_REPORT.md

The report must synthesize the completed evidence and answer:

1. What is the corrected nonlinear MiniLM 5-fold result?
2. Was the old 0.5922 result Ridge rather than nonlinear MLP?
3. Does nonlinear classifier capacity on GTE substantially improve F1?
4. Does any tested method reach 0.70?
5. What efficiency evidence is actually end-to-end?
6. Which earlier efficiency claims must be relabelled?
7. What bottleneck is supported by the current evidence?
8. What exactly ONE next experiment should be run?

## Interpretation Requirements

Do NOT claim that 0.626 is a mathematically proven ceiling.

Use careful wording:

"The tested classifiers do not reveal substantial additional winner signal
from the current query-only GTE representation."

Distinguish:
- classifier-capacity evidence
- representation evidence
- intrinsic pair difficulty
- downstream routing headroom

The approximately 79.7% tie rate limits downstream routing opportunity,
but it does NOT mathematically limit non-tie Macro-F1 to approximately 0.62.

## Required ONE Next Research Direction

Recommend one materially different experiment, not another classifier tweak.

The recommendation should investigate richer, deployable query-level
model-suitability information beyond a single generic semantic embedding.

Prefer an interpretable capability-aware or task-conditional routing
formulation if supported by the diagnostic evidence.

Do not use true dataset labels at deployment.

Do not expand the model pool yet.
Do not begin Edge/Split/Remote integration yet.

## Constraints

- Do not rerun Tasks 1 or 2.
- Do not regenerate folds.
- Do not overwrite previous result files.
- No broad hyperparameter sweep.
- No model-pool expansion.
- No system integration.
- No test leakage.
- No unsupported claims of a proven performance ceiling.
