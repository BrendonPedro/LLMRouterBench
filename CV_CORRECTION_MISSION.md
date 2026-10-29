# Corrected CV and Representation-Ceiling Mission

## Context

The existing stratified 5-fold CV infrastructure and frozen fold assignments
are valid and MUST be reused.

Existing best 5-fold result:
GTE + training-only threshold = 0.6153 Macro-F1.

Two issues were identified in the previous CV harness:

1. The method labelled "Frozen-MiniLM distillation" actually uses Ridge:
   Ridge(alpha=1.0)
   This is the OLD LINEAR score-distillation student.

2. Efficiency measurements use cached embeddings, so reported inference
   times and parameter counts describe only the classifier/head, NOT
   raw-query end-to-end deployment.

Do not overwrite the existing result files.

## Task 1 — Correct nonlinear distillation CV

Implement the actual previously-defined best lightweight student:

Frozen all-MiniLM-L6-v2 embeddings (384-D)
→ MLPRegressor hidden layer 64
→ ReLU
→ scalar teacher routing score

Use the SAME fixed hyperparameters as the existing
train_minilm_nonlinear_score_distillation.py experiment.

Teacher:
GTE direct logistic router trained only on the outer training portion.

Teacher produces continuous scores for ALL outer-training queries.

No outer-test labels may be used for:
- teacher fitting
- student fitting
- threshold selection
- epoch/model selection
- hyperparameter selection

Evaluate Macro-F1 on outer held-out non-tie samples.

Report all 5 folds plus mean/std.

Do not tune the nonlinear architecture.

## Task 2 — GTE classifier-capacity diagnostic

Using the SAME fixed GTE embeddings and SAME frozen folds, compare:

1. Logistic regression default threshold
2. Logistic regression + training-only threshold
3. One fixed nonlinear MLP classifier
4. One fixed tree/kernel nonlinear classifier if computationally practical

Do not perform broad hyperparameter sweeps.

Purpose:

Determine whether the existing GTE representation contains substantially
more winner-predictive information than the ~0.615 linear router extracts.

The important question is whether a nonlinear classifier can move performance
meaningfully toward or beyond 0.70 Macro-F1.

## Task 3 — Correct efficiency terminology

Preserve cached-embedding timings, but rename them clearly as:

- head training time
- head inference time on precomputed embeddings

Do NOT describe these as end-to-end inference latency.

For each method separately report:

A. Encoder / representation model parameters
B. Router-head parameters
C. Total deployed parameter count
D. Trainable parameters
E. Router-head disk size
F. Whether an external teacher is required at inference

For MiniLM:
total online model includes the ~22.7M encoder plus router head.

For GTE:
total online model includes the GTE-Qwen2-7B encoder plus router head.

Do not claim exact encoder disk size unless measured from the actual checkpoint.

## Task 4 — End-to-end timing feasibility

Determine whether raw-query end-to-end inference can be measured fairly
on the current machine.

End-to-end means:

raw query
→ tokenizer
→ encoder forward pass
→ pooling
→ router head
→ routing decision

Do not use cached embeddings for this measurement.

If the 7B GTE model cannot be loaded or fairly timed on current hardware,
state this explicitly rather than estimating.

For MiniLM, measure:
- warm-up runs
- repeated raw-query inference
- median latency/query
- mean latency/query
- queries/second
- peak memory if practical

Keep classifier-only cached timings separately.

## Outputs

Create:

diagnostics/stratified_cv/corrected_nonlinear_cv.csv
diagnostics/stratified_cv/gte_classifier_capacity.csv
diagnostics/stratified_cv/corrected_efficiency.csv
diagnostics/stratified_cv/CV_CORRECTION_REPORT.md

Final report must answer:

1. What is the actual nonlinear MiniLM distilled 5-fold Macro-F1?
2. Was the previous 0.5922 result only the linear Ridge student?
3. Can nonlinear classification on GTE substantially exceed 0.615?
4. Does any tested method reach 0.70?
5. What efficiency evidence is truly measured end-to-end?
6. Which previous efficiency claims must be withdrawn or relabelled?
7. Based on this evidence, what ONE next research direction is justified?

## Constraints

- Reuse existing frozen fold assignments.
- No new fold generation.
- No model-pool change.
- No broad hyperparameter sweep.
- No test-fold leakage.
- No system integration.
- Do not overwrite existing results.
- Do not report cached-embedding latency as end-to-end latency.
