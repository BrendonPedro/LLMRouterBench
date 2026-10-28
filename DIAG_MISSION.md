# LLMRouterBench Router Performance Diagnostic Mission

## Context

This repository contains controlled LLMRouterBench routing experiments.

Current controlled model pair:
- DeepSeek-R1-0528-Qwen3-8B
- Qwen3-8B

Controlled seeds:
42, 43, 44, 45, 46, 47, 48, 49, 50, 51

Primary advisor-requested router metric:
Macro-F1 on non-tie model-selection examples.

Important existing results:
- Tiny BERT direct selector: ~0.5779 Macro-F1
- BERT-Mini direct selector: ~0.5819
- MiniLM direct selector: ~0.5822
- GTE direct selector: ~0.6032
- GTE threshold-optimized selector: ~0.6104
- Frozen MiniLM nonlinear routing-score distillation: ~0.5975
- Pair Oracle downstream Sample-Avg: ~75.95%
- Best Fixed downstream Sample-Avg: ~67.41%
- Roughly 80% of training examples are model-outcome ties.

Advisor feedback:
The F1 scores remain too low. Current work can remain as baseline work, but the main method must achieve substantially stronger model-selection performance before system integration.

## Mission

DO NOT start by inventing another router or sweeping hyperparameters.

The primary goal is to determine WHY model-selection Macro-F1 is stuck around 0.60.

Work evidence-first.

### 1. Audit evaluation correctness

Verify from raw data and code:

- exact construction of DeepSeek-win / Qwen-win / tie labels
- train/test separation for every seed
- duplicate-query leakage
- class balance among non-tie train and test examples
- per-dataset non-tie counts
- whether all methods are evaluated on identical non-tie test instances
- Macro-F1 implementation
- balanced accuracy implementation
- threshold handling
- any accidental label inversion
- any dataset or query normalization inconsistency

Produce:
- diagnostics/evaluation_audit.md
- diagnostics/label_statistics.csv

Do not silently fix anything. Document first.

### 2. Establish an F1 reference ladder

Using the EXACT SAME pair, seeds, train/test splits, and non-tie F1 definition, calculate Macro-F1 for as many existing LLMRouterBench routing baselines as can be reproduced fairly from this repository or their saved predictions.

Prioritize:
- Random
- trivial / Best Single selector where meaningful
- TF-IDF
- Avengers / AvengersPro
- EmbedLLM
- RouterDC
- MODEL-SAT
- GraphRouter
- any other existing benchmark router that can be evaluated fairly

The question is:

"What Macro-F1 do established routers obtain on this exact pairwise winner-prediction task?"

Produce:
- diagnostics/official_router_f1_comparison.csv
- diagnostics/official_router_f1_analysis.md

If a method cannot be reproduced fairly, state why. Do not approximate or invent results.

### 3. Diagnose pair learnability

For each seed and dataset compute:

- both correct
- both wrong
- DeepSeek only correct
- Qwen only correct
- tie percentage
- non-tie percentage
- outcome agreement
- model preference balance
- pair-oracle improvement
- number of informative winner labels

Identify datasets where:
- routing appears learnable
- one model nearly always dominates
- outcomes are highly correlated
- winner labels are sparse or unstable

Produce:
- diagnostics/pair_learnability.csv
- diagnostics/pair_learnability.md

Do NOT change the model pool during this mission.

### 4. Diagnose model-recall failures

For the strongest existing routers, analyze cases where exactly one candidate model is correct but the router selects the wrong model.

Break failures down by:
- dataset
- true winning model
- query length
- math cues
- code cues
- logic cues
- knowledge cues
- predicted confidence / margin where available

Determine whether failures appear to come from:

A. representation failures  
B. decision-boundary or calibration failures  
C. class imbalance  
D. dataset/domain confusion  
E. queries where the winner has little obvious query-side signal

Produce:
- diagnostics/model_recall_failure_analysis.md
- diagnostics/model_recall_failures.csv

### 5. Test whether GTE contains more usable routing signal

Keep the existing GTE embeddings fixed.

This is a diagnostic, NOT a broad hyperparameter search.

Compare a small pre-specified classifier set:

- Logistic regression with fixed threshold
- Logistic regression with TRAIN/OOF F1 threshold selection
- one nonlinear MLP classifier
- one nonlinear kernel or tree-based diagnostic if computationally practical

Use identical train/test splits.

Never select settings using test labels.

Question:

"Is ~0.61 F1 mainly a linear-classifier limitation, or does the GTE representation itself lack enough winner-predictive information?"

Produce:
- diagnostics/gte_classifier_diagnostic.csv
- diagnostics/gte_classifier_diagnostic.md

### 6. Measure simple structural query signal

Test whether observable query features contain complementary routing information:

- query length
- digit density
- mathematical symbols
- code markers
- multiple-choice structure where detectable
- question-style indicators

Compare:

- GTE only
- structural features only
- GTE + structural features

No test-label feature selection.

Produce:
- diagnostics/query_feature_signal.csv
- diagnostics/query_feature_signal.md

### 7. GPU / compute assessment

Based on the diagnostic evidence, state explicitly:

- which bottleneck would actually benefit from GPU access
- which current experiments would give the same scientific conclusion on CPU but run slower
- whether GPU access is necessary for the next justified experiment
- what stronger neural router is justified only if the evidence supports it

Produce:
- diagnostics/compute_recommendation.md

## Required final report

Create:

diagnostics/ROUTER_PERFORMANCE_DIAGNOSIS.md

It must answer with evidence:

1. Is the current F1 implementation correct?
2. What F1 do established comparable routers obtain under the same definition?
3. Is this two-model pair intrinsically difficult to route?
4. Is the main bottleneck representation, classifier capacity, calibration, label sparsity, or something else?
5. Are the roughly 80% ties contributing to weak supervision?
6. Is the current GTE teacher fundamentally too weak?
7. Would GPU access address the actual bottleneck?
8. What ONE next experiment is most scientifically justified?

## Constraints

- Do not change the research question.
- Do not begin Edge / Split / Remote system integration.
- Do not expand the model pool yet.
- Do not use test labels for training, threshold selection, feature selection, or hyperparameter selection.
- Do not run broad hyperparameter sweeps.
- Do not claim statistical significance without proper testing.
- Do not overwrite existing experimental result files.
- Preserve all existing branches and results.
- Prefer diagnostic scripts and reports over altering existing experimental code.
- Commit reproducible diagnostic scripts and final diagnostic outputs.
- Recommend exactly ONE next research experiment at the end.

The mission is complete only when the likely cause of the low F1 has been narrowed using reproducible evidence.
