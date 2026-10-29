# Stratified K-Fold Router Evaluation Mission

## Motivation

Advisor feedback requires a stronger generalization protocol.

Existing repeated-holdout experiments plateau around Macro-F1 0.59–0.61.
The evaluation audit has already verified that the existing pipeline is correct.
Do not redo that audit.

Completed diagnostic evidence:
- TF-IDF: ~0.588 Macro-F1
- AvengersPro: ~0.599
- GTE Direct: ~0.603
- GTE threshold: ~0.610
- approximately 79.7% of examples are ties
- winner preference is strongly dataset-conditional
- no leakage or label inversion was found

The advisor considers F1 below 0.70 inadequate and suggested stratified
K-fold cross-validation to improve the generalizability of model and
hyperparameter evaluation.

## Primary Goal

Implement a reproducible STRATIFIED 5-FOLD CROSS-VALIDATION protocol and
determine whether the existing ~0.60 Macro-F1 plateau persists.

Do NOT invent a new routing method in this mission.

## Fold Construction

Use K=5.

Stratify the full dataset using the raw three-way outcome category:
1. DeepSeek-only correct
2. tie
3. Qwen-only correct

Use a fixed random seed and save the fold assignment for reproducibility.

Verify and report:
- total samples per fold
- DeepSeek-win count
- Qwen-win count
- tie count
- proportions
- dataset composition

No query may occur in more than one outer test fold.

Every query must be evaluated exactly once as outer held-out data.

## Evaluation

Router Macro-F1 must continue to be computed on held-out NON-TIE examples.

Also report:
- balanced accuracy
- per-class precision
- per-class recall
- per-class F1
- confusion matrix
- mean across folds
- standard deviation across folds

Do not call downstream task performance "router accuracy."

## Model / Hyperparameter Selection

Do not tune hyperparameters using the outer held-out fold.

If epoch selection or early stopping is required:
- create a stratified validation subset from the four training folds
- use that validation subset for epoch selection / early stopping
- evaluate the outer fold only after model selection

For models with already-fixed hyperparameters, keep those hyperparameters
fixed for the first K-fold comparison.

## Required Baselines

Evaluate as many as can be reproduced fairly:

1. TF-IDF + Logistic Regression
2. GTE Direct Logistic Regression
3. GTE + training-only threshold selection
4. MiniLM Direct
5. Frozen MiniLM nonlinear routing-score distillation

Include AvengersPro only if its fold-specific model can be retrained fairly
without using held-out labels.

Do not approximate unavailable results.

## Efficiency Evidence

For every evaluated method, measure on the SAME hardware where possible:

- total model parameter count
- trainable parameter count
- model/checkpoint size on disk
- training time per fold
- total 5-fold training time
- inference time on held-out fold
- milliseconds/query
- queries/second
- peak memory where practical
- representation model required at deployment

Clearly distinguish:
- offline teacher requirements
- online inference requirements

## Comparison With Existing Protocol

Create a table comparing:

Repeated holdout mean Macro-F1
vs
5-fold stratified CV Macro-F1

Determine whether the old evaluation protocol materially affected the
estimated generalization performance.

## Required Outputs

Create:

diagnostics/stratified_cv/fold_assignments.csv
diagnostics/stratified_cv/fold_statistics.csv
diagnostics/stratified_cv/router_cv_results.csv
diagnostics/stratified_cv/efficiency_results.csv
diagnostics/stratified_cv/STRATIFIED_CV_REPORT.md

The final report must answer:

1. Does stratified 5-fold CV materially change Macro-F1?
2. Does any existing method exceed 0.70?
3. Which method generalizes best?
4. How variable is performance across folds?
5. What is the training/inference cost of each method?
6. Is the lightweight claim supported by measured evidence?
7. Does the ~0.60 plateau remain after stronger evaluation?

## Constraints

- No test-fold leakage.
- No hyperparameter tuning on outer folds.
- No new model pool.
- No Edge/Split/Remote system work.
- No broad hyperparameter sweeps.
- No new routing architecture.
- Preserve all previous results.
- Commit scripts, fold assignments, and outputs.
