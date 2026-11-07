# Task-Centered Residual 3-Class Router Result

Date: 2026-11-07

Question:
Does adding a task-centered residual head to the predicted-task prior improve the 3-class router?

Baseline reproduction:
- Within-benchmark Macro-F1: 0.367981
- Overall Macro-F1: 0.414997
- Baseline predictions reproduced exactly.

Treatment:
- Within-benchmark Macro-F1: 0.3622
- Overall Macro-F1: 0.4188
- Cell-AUC: 0.6003

Delta vs baseline:
- Within-benchmark Macro-F1: -0.0058
- Overall Macro-F1: +0.0038
- Cell-AUC: +0.0026

Fold within-F1 deltas:
- Fold 0: -0.0292
- Fold 1: +0.0051
- Fold 2: -0.0058
- Fold 3: -0.0090
- Fold 4: -0.0039

Only 1/5 folds improved.

Pre-registered verdict: FAIL.

Interpretation:
The task-centered residual slightly improves ranking and overall Macro-F1, but it does not improve within-task 3-class routing. The residual signal is too weak to support the proposed factored semantic router under the current supervision.

No follow-up experiment was run.
