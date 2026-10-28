# Router Performance Diagnostic — Continuation

Continue the diagnostic mission defined in DIAG_MISSION.md from the CURRENT repository state.

## IMPORTANT: Section 1 is COMPLETE

Do NOT redo the evaluation audit.

The completed audit is committed at 30d6e3e and produced:

- diagnostics/evaluation_audit.md
- diagnostics/label_statistics.csv
- diagnostics/scripts/audit_evaluation.py

The audit established:

- label construction is correct
- train/test separation is correct
- no duplicate-query leakage exists
- Macro-F1 and balanced accuracy implementations are correct
- all compared methods use identical non-tie test instances
- no label inversion exists
- GTE threshold selection uses OOF TRAINING data only
- mean test tie rate is approximately 79.7%
- non-tie winner classes are reasonably balanced
- strong dataset-level preference heterogeneity exists

The approximately 0.60 Macro-F1 ceiling is therefore NOT an evaluation artifact.

## Continue the mission in this priority order

### Priority 1 — Comparable F1 reference ladder

Complete Section 2 of DIAG_MISSION.md.

Determine Macro-F1 under the EXACT SAME pairwise non-tie evaluation for as many existing LLMRouterBench routers as can be reproduced fairly.

This is now the most important question:

"Is approximately 0.60 actually poor relative to established routing methods on this exact pair and split definition?"

Do not substitute downstream Sample-Avg for router F1.

Do not invent results for methods that cannot be reproduced.

Produce:
- diagnostics/official_router_f1_comparison.csv
- diagnostics/official_router_f1_analysis.md

### Priority 2 — Pair learnability

Complete Section 3.

Use the existing label_statistics.csv where possible rather than recomputing Section 1.

Quantify:
- outcome agreement
- informative non-tie rate
- per-dataset winner balance
- pair-oracle gain
- winner-label sparsity
- task-specific dominance
- train/test stability of winner distributions

Determine whether this particular DeepSeek/Qwen pair creates an unusually difficult routing target.

Produce:
- diagnostics/pair_learnability.csv
- diagnostics/pair_learnability.md

### Priority 3 — GTE representation/classifier ceiling

Complete Section 5.

Keep GTE embeddings fixed.

Compare only the pre-specified small diagnostic set:
- logistic fixed threshold
- logistic + OOF F1 threshold
- one nonlinear MLP
- one practical nonlinear kernel/tree diagnostic

NO broad hyperparameter sweep.

The key question is:

"Can a more expressive classifier extract substantially more winner signal from the SAME GTE representation?"

If nonlinear classifiers remain near ~0.61, this is strong evidence that the bottleneck is not simply the logistic head.

Produce:
- diagnostics/gte_classifier_diagnostic.csv
- diagnostics/gte_classifier_diagnostic.md

### Priority 4 — Model-recall failure analysis

Complete Section 4.

Analyze cases where exactly one model succeeds but the router chooses the other model.

Determine whether failures are associated with:
- dataset/task
- true winning model
- query length
- math/code/logic/knowledge cues
- router confidence
- apparently unpredictable winner outcomes

Produce:
- diagnostics/model_recall_failures.csv
- diagnostics/model_recall_failure_analysis.md

### Priority 5 — Structural query signal

Complete Section 6.

Compare:
- GTE only
- structural features only
- GTE + structural features

Use only observable query features and no test-label feature selection.

Produce:
- diagnostics/query_feature_signal.csv
- diagnostics/query_feature_signal.md

### Priority 6 — GPU / compute diagnosis

Complete Section 7 only after the evidence above.

Explicitly answer whether GPU access would address the demonstrated bottleneck or merely make experiments faster.

Produce:
- diagnostics/compute_recommendation.md

## Final synthesis

Complete:
diagnostics/ROUTER_PERFORMANCE_DIAGNOSIS.md

The final report must clearly answer:

1. Is evaluation correct? Use the completed Section 1 evidence.
2. How does ~0.60 F1 compare with established routers under the same evaluation?
3. Is this model pair intrinsically difficult?
4. How much useful winner signal exists in GTE?
5. Is classifier capacity a bottleneck?
6. What causes model-recall failures?
7. Are ties / sparse winner labels a major limitation?
8. Would GPU access solve the demonstrated problem?
9. What ONE next research experiment is most justified?

Do not propose multiple competing next experiments.

## Constraints

- Preserve completed Section 1.
- Do not change model pool.
- Do not begin Edge/Split/Remote integration.
- Do not use test labels for training or selection.
- Do not run broad sweeps.
- Do not overwrite prior results.
- Commit reproducible scripts and outputs.
- Evidence first, method invention later.
