# Capability-Profile Router Experiment

## Elevator Pitch

My research is about building a lightweight router that can decide which AI
model is better suited to answer a particular question.

So far, stronger embeddings, nonlinear classifiers, and stratified 5-fold
cross-validation have not pushed the router much beyond about 0.63 Macro-F1.

The next question is whether the router needs more useful information about
what the query actually requires.

This experiment describes each query using simple, interpretable capabilities
such as math, code, multiple-choice structure, reasoning cues, answer-format
requirements, and query length.

The goal is to test whether these explicit query signals improve model
selection beyond a generic semantic embedding.

Long term, the aim is a small, fast router that can later combine model
suitability with execution cost and latency.

---

# Research Question

Can simple, deployable, interpretable query-capability features provide
winner-prediction information that generic semantic embeddings currently miss?

This is NOT another classifier-capacity experiment.

The classifier family should remain simple.

The experiment changes the ROUTING SIGNAL.

---

# Existing Evidence

Do not rerun previous experiments.

Use these established results as references:

- MiniLM MLP-64 score-distillation:
  Macro-F1 = 0.6003 ± 0.0154

- GTE LogReg + training-only threshold:
  Macro-F1 = 0.6153

- GTE RBF-SVM + training-only threshold:
  Macro-F1 = 0.6260 ± 0.0296

- Best existing GTE LogReg AUC:
  approximately 0.683

- No tested method reaches 0.70 Macro-F1.

- Approximately 79.7% of queries are ties.

- Model preference is strongly task-conditional.

- MiniLM truncates approximately 55.9% of benchmark queries at its
  256-token input limit.

Do not claim that 0.626 is a proven performance ceiling.

---

# Frozen Evaluation Protocol

Reuse the EXISTING frozen stratified 5-fold assignments:

diagnostics/stratified_cv/fold_assignments.csv

Do not regenerate folds.

Outer evaluation:
- Macro-F1 on held-out NON-TIE examples is the primary metric.
- Also report:
  - balanced accuracy
  - AUC
  - per-class precision
  - per-class recall
  - per-class F1
  - confusion matrix

All fitting, threshold selection, preprocessing, and feature normalization
must use outer-training data only.

No outer-test labels may influence any model decision.

---

# Frozen Capability Feature Set

Define these features BEFORE running any experiment.

The extractor must be deterministic and use RAW QUERY TEXT ONLY.

It must NOT use:
- dataset name
- dataset ID
- benchmark metadata
- model outcomes
- winner labels
- test-fold information

Use this fixed feature set:

1. character_count
   Total query characters.

2. whitespace_word_count
   Simple whitespace-separated word count.

3. minilm_token_count
   Number of tokens produced by the MiniLM tokenizer before truncation.

4. exceeds_256_tokens
   Binary flag: query is longer than MiniLM's 256-token window.

5. contains_digits
   Binary flag for numeric digits.

6. numeric_density
   Fraction of query characters that are digits.

7. contains_math_expression
   Binary heuristic for mathematical symbols/equations such as
   =, +, -, *, /, ^, %, sqrt-style notation, fractions, or LaTeX math.

8. contains_code
   Binary heuristic for code blocks or programming syntax.

9. requests_code
   Binary heuristic for instructions such as write/code/program/function/
   implement/debug.

10. multiple_choice_structure
    Binary heuristic for answer choices such as A/B/C/D, (A), (B),
    numbered alternatives, or similar structured options.

11. explicit_answer_format
    Binary heuristic for instructions such as:
    "answer only", "letter only", "JSON", "boxed answer",
    "one word", "yes/no", "output format", etc.

12. reasoning_cue
    Binary heuristic for phrases such as:
    explain, reason, derive, prove, calculate, step by step,
    why, justify, analyze.

13. constraint_count
    Count of explicit requirement words/phrases such as:
    must, only, exactly, do not, at least, at most, required,
    without, include, exclude.

14. has_embedded_context
    Binary heuristic indicating a substantial passage, document,
    conversation, table, or context is supplied before the final task.

15. structural_line_count
    Number of non-empty lines in the query.

Do NOT add or remove features after seeing fold results.

Version this extractor as capability-profile-v1.

---

# Experiment A — Capability-Only Router

Input:
the 15 frozen capability features only.

Model:
Logistic Regression.

Use sensible fixed preprocessing:
- standardize continuous/count features using outer-training statistics
- leave binary features as binary

No broad hyperparameter search.

Evaluate on the same frozen 5 folds.

Report:
- Macro-F1
- AUC
- balanced accuracy
- per-class metrics
- fold mean/std
- learned coefficients
- coefficient sign stability across folds

Also evaluate a training-only threshold chosen without outer-test labels.

---

# Experiment B — Existing MiniLM Reference

Do NOT retrain or redesign this method unnecessarily.

Reference:

Frozen MiniLM
→ MLP-64 routing-score distillation

Existing 5-fold Macro-F1:
0.6003 ± 0.0154

Use its existing saved fold results and/or reproduce only the fold-level
scores needed for alignment.

This is the lightweight semantic baseline.

---

# Experiment C — Capability + MiniLM Score

Combine:

15 capability-profile-v1 features
+
1 MiniLM MLP-64 routing score

Final model:
Logistic Regression.

IMPORTANT LEAKAGE RULE:

For OUTER-TRAIN rows, the MiniLM routing score used to train the final
combiner must be generated OUT-OF-FOLD within the outer-training data.

Do not use an in-sample student score as a training feature.

For the OUTER-TEST fold:
- fit the MiniLM student using outer-training data only
- produce routing scores for the held-out outer-test fold
- combine those scores with its capability features
- apply the final combiner trained only from outer-training information

All threshold selection must remain inside outer-training data.

---

# Required Comparisons

Compare:

A. Capability-only LogReg

B. Existing MiniLM MLP-64 student score

C. Capability profile + MiniLM student score

Also show the existing GTE references separately:

- GTE LogReg + threshold = 0.6153
- GTE RBF-SVM + threshold = 0.6260

Do not present those GTE values as newly rerun results.

---

# Pre-Declared Interpretation

Do not move the goalposts after seeing results.

A capability signal is considered meaningfully useful if:

1. Experiment C improves mean Macro-F1 over the MiniLM reference
   by at least 0.03,

AND

2. Experiment C beats the MiniLM reference on at least 4 of 5 folds.

A stronger result is achieved if C also exceeds the current best
tested mean Macro-F1 of approximately 0.626.

A highly promising result would approach or exceed 0.70 Macro-F1,
which remains the advisor's expected performance region.

Also inspect AUC.

If capability features improve AUC and Macro-F1 consistently:
conclude that generic semantic embeddings are missing useful
query-level capability information.

If capability-only and capability+MiniLM produce little or no improvement:
the evidence shifts toward the winner being difficult to predict from
query text alone for this closely related model pair.

Do not call either outcome a mathematical proof.

---

# Truncation Analysis

Because approximately 55.9% of queries exceed MiniLM's 256-token window,
report performance separately for:

- queries <= 256 MiniLM tokens
- queries > 256 MiniLM tokens

This is a DIAGNOSTIC breakdown only.

Do not change the encoder or context length in this experiment.

Report whether routing Macro-F1 and/or AUC differ materially between
truncated and non-truncated queries.

Do not use the test-fold truncation result to tune the router.

---

# Efficiency

The capability extractor should be lightweight.

Measure:

- capability extraction time/query
- capability-only router inference time/query
- capability + MiniLM additional overhead
- number of learned router parameters
- total deployed MiniLM + combiner parameters where relevant

Reuse the existing measured MiniLM end-to-end CPU reference:

approximately 38 ms/query on the current CPU.

Do not rerun GTE end-to-end timing.

---

# Required Outputs

Create:

diagnostics/capability_profile/capability_features_v1.csv
diagnostics/capability_profile/fold_results.csv
diagnostics/capability_profile/coefficients_by_fold.csv
diagnostics/capability_profile/truncation_analysis.csv
diagnostics/capability_profile/efficiency.csv
diagnostics/capability_profile/CAPABILITY_PROFILE_REPORT.md

Create scripts under:

diagnostics/scripts/

Do not overwrite previous diagnostic files.

---

# Final Report Questions

CAPABILITY_PROFILE_REPORT.md must answer:

1. Does the capability-only router contain useful winner signal?
2. Does capability + MiniLM beat MiniLM alone?
3. Does it beat the existing 0.626 best tested router?
4. Does anything reach or approach 0.70?
5. Which capability features are stable across folds?
6. Is performance different for queries above vs below 256 tokens?
7. What is the inference overhead?
8. What do the results imply about the current bottleneck?
9. What exactly ONE next research step is justified?

---

# Constraints

- No dataset labels as router features.
- No benchmark IDs as router features.
- No test-label leakage.
- No fold regeneration.
- No broad hyperparameter sweep.
- No new model pool.
- No Edge/Split/Remote integration.
- No GTE classifier tweaking.
- No changing the feature set after results are observed.
- Preserve all existing results.
- Commit scripts and result files only after review.
