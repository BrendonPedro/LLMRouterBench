# Capability-Profile Router: Report

Mission: `CAPABILITY_PROFILE_MISSION.md`. Repository: `/home/brendonpedro/research/LLMRouterBench`, branch `capability-profile-router`.
This is a fresh run. The scripts and outputs here were all written and generated in this run. Untracked outputs from an earlier session were set aside first and were not used.

Pair: DeepSeek-R1-0528-Qwen3-8B (class 0) vs Qwen3-8B (class 1). The primary metric is Macro-F1 on **held-out non-tie** rows from the **existing frozen stratified 5-fold** assignment (`diagnostics/stratified_cv/fold_assignments.csv`, not regenerated). There are 11,467 queries, of which 2,333 are non-tie and 79.7% are ties.

## Scripts

| Script | Role |
|---|---|
| `diagnostics/scripts/capability_profile_v1.py` | Frozen **capability-profile-v1** extractor: 15 features from raw query text only, written before any experiment ran |
| `diagnostics/scripts/build_capability_features_v1.py` | Writes `capability_features_v1.csv`. Label-free; keys are kept for joins only |
| `diagnostics/scripts/run_capability_profile_cv.py` | Runs A / B / C on the frozen folds and writes all other outputs |
| `diagnostics/scripts/run_capability_profile_b_threshold.py` | Fairness correction: adds the **B-threshold** calibration comparator (appended to `fold_results.csv`) |

The extractor sees no dataset name or ID, no benchmark metadata, no outcomes and no fold information. `minilm_token_count` is the untruncated all-MiniLM-L6-v2 token count, including [CLS]/[SEP]. Under this definition 56.3% of queries exceed 256 tokens, close to the 55.9% reference measured on fold 0. Before any model was fit, I checked only feature prevalence, with no labels. No feature was changed after that.

## Protocol

- **A**: the 15 capability features go into `LogisticRegression(C=1, L2, lbfgs)`. The 6 continuous/count features are standardized with outer-train statistics; the 9 binary features are left as 0/1. It is trained on outer-train non-tie rows.
- **B**: frozen MiniLM, then the MLP-64 routing-score distillation student (GTE LogReg teacher), decision `score > 0`. I regenerated its fold-level held-out scores with the exact `run_corrected_nonlinear_cv.py` config so they align row by row with C. **The per-fold Macro-F1 matches the saved `corrected_nonlinear_cv.csv` exactly on all 5 folds (asserted in the script), giving 0.6003 ± 0.0154.** B was not redesigned.
- **B-threshold** (fairness correction, added after the main run): the *same* MiniLM student scores, with a training-only cutoff chosen by the same protocol used for A and C. The details:
  - The outer-train out-of-fold student scores are exactly the ones C uses: same inner splits and seeds. The script re-checks them against C's saved OOF-student Macro-F1/AUC, which match to 1e-9.
  - The cutoff maximizes OOF Macro-F1 on outer-train non-tie rows.
  - The student emits an unbounded regression score rather than a probability, so the 0.05–0.95 probability grid used for A/C does not apply. The candidates are instead all midpoints between consecutive distinct OOF scores.
  - Before anything was written, the regenerated held-out B scores were verified against `heldout_predictions.csv` (max |Δ| = 6e-8, which is CSV rounding) and against the saved per-fold Macro-F1 (exact).
  - Outer-test labels were used only for the final metrics. The model is unchanged, and the B default-cutoff row is kept as-is.
- **C**: the 15 capability features plus 1 MiniLM student score go into the same LogReg, with the student score standardized.
  - **Leakage control:** the outer-train student scores are **out-of-fold**. An inner StratifiedKFold(5) runs over the outer-train rows. In each inner split, *both* the GTE teacher (inner-train non-ties) and the MLP-64 student (inner-train queries) are refit, then they score the inner held-out rows.
  - The outer-test scores come from the B teacher/student fit on the full outer-train fold.
  - As a check, the OOF student scores reach Macro-F1 0.594–0.608 and AUC 0.656–0.673 on outer-train. That matches out-of-sample behaviour, so these are not in-sample scores.
- **Decision rules (declared in the script docstring before running):**
  - PRIMARY: `P(Qwen) >= 0.5`.
  - SECONDARY: a training-only threshold. It is the cutoff that maximizes Macro-F1 of out-of-fold `cross_val_predict` probabilities on outer-train non-tie rows, searched over a grid from 0.05 to 0.95. This is the same procedure as the GTE-threshold baseline. For C, these OOF probabilities are built on the OOF student scores.
- No hyperparameter search, no fold regeneration, no GTE retraining or tuning, no new models in the pool.

Existing references, **not rerun**: GTE LogReg + threshold = 0.6153; GTE RBF-SVM + threshold = 0.6260 ± 0.0296; best GTE LogReg AUC ≈ 0.683.

## Results (held-out non-tie, 5-fold)

| Condition | Decision rule | Macro-F1 mean ± std | AUC mean ± std | Bal. acc | Folds > B (Macro-F1) | Mean Δ vs B |
|---|---|---|---|---|---|---|
| A capability-only LogReg | p ≥ 0.5 (primary) | 0.5969 ± 0.0137 | 0.6477 ± 0.0238 | 0.6021 | 1/5 | −0.0034 |
| A capability-only LogReg | training-only threshold | 0.6053 ± 0.0219 | 0.6477 ± 0.0238 | 0.6098 | 3/5 | +0.0050 |
| **B MiniLM MLP-64 student (reference)** | score > 0 | **0.6003 ± 0.0154** | 0.6699 ± 0.0167 | 0.6049 | — | — |
| B-threshold (calibration comparator) | training-only threshold | 0.6009 ± 0.0149 | 0.6699 ± 0.0167 | 0.6023 | 3/5 | +0.0006 |
| C capability + MiniLM score LogReg | p ≥ 0.5 (primary) | 0.5968 ± 0.0181 | 0.6725 ± 0.0224 | 0.6012 | 2/5 | −0.0035 |
| C capability + MiniLM score LogReg | training-only threshold | 0.6082 ± 0.0196 | 0.6725 ± 0.0224 | 0.6119 | 4/5 | +0.0079 |
| *GTE LogReg + threshold (existing)* | | *0.6153* | *≈0.683* | | | |
| *GTE RBF-SVM + threshold (existing)* | | *0.6260 ± 0.0296* | | | | |

Per-fold Macro-F1:

| Fold | A (0.5) | A (thr) | B | B (thr) | C (0.5) | C (thr) | C(thr) − B | C(thr) − B(thr) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5729 | 0.5750 | 0.5830 | 0.5850 | 0.5671 | 0.5843 | +0.0013 | −0.0007 |
| 1 | 0.6049 | 0.5918 | 0.6207 | 0.6082 | 0.6142 | 0.5987 | −0.0220 | −0.0095 |
| 2 | 0.6019 | 0.6209 | 0.6030 | 0.6179 | 0.6070 | 0.6230 | +0.0200 | +0.0051 |
| 3 | 0.6062 | 0.6101 | 0.5872 | 0.5853 | 0.6008 | 0.6019 | +0.0147 | +0.0166 |
| 4 | 0.5988 | 0.6286 | 0.6078 | 0.6083 | 0.5947 | 0.6329 | +0.0250 | +0.0246 |
| **mean** | | | | | | | **+0.0079** | **+0.0072** |

B-threshold cutoffs (score units): +0.268, +0.163, +0.092, +0.131, +0.234. B-threshold predicts Qwen on 59.5% of rows, against 72.5% at score > 0.

C − B AUC per fold: −0.0078, +0.0045, +0.0090, +0.0035, +0.0036, a mean of +0.0026 (4/5 positive).

Per-class metrics and confusion matrices, pooled over the 5 held-out folds (rows = true class, cols = predicted class [DS, Qwen]):

| Condition | P(DS) | R(DS) | F1(DS) | P(Qwen) | R(Qwen) | F1(Qwen) | CM [[DS→DS, DS→Q],[Q→DS, Q→Q]] |
|---|---|---|---|---|---|---|---|
| A 0.5 | 0.608 | 0.389 | 0.474 | 0.644 | 0.815 | 0.720 | [[385, 605], [248, 1095]] |
| A thr | 0.534 | 0.617 | 0.572 | 0.681 | 0.602 | 0.639 | [[611, 379], [534, 809]] |
| B | 0.611 | 0.396 | 0.480 | 0.646 | 0.814 | 0.721 | [[392, 598], [250, 1093]] |
| B thr | 0.548 | 0.523 | 0.535 | 0.660 | 0.681 | 0.670 | [[518, 472], [428, 915]] |
| C 0.5 | 0.601 | 0.396 | 0.478 | 0.644 | 0.806 | 0.716 | [[392, 598], [260, 1083]] |
| C thr | 0.545 | 0.582 | 0.563 | 0.676 | 0.642 | 0.658 | [[576, 414], [481, 862]] |

Per-fold values for every metric, with fold mean/std and pooled rows, are in `fold_results.csv`.

### Pre-declared criterion

| Criterion | C (0.5) vs B | C (thr) vs B (score > 0) | C (thr) vs B-threshold (calibrated, fair) |
|---|---|---|---|
| Mean Macro-F1 gain ≥ 0.03 | **No** (−0.0035) | **No** (+0.0079) | **No** (+0.0072) |
| Beats reference on ≥ 4/5 folds | No (2/5) | Yes (4/5; fold 0 margin +0.0013) | **No (3/5)** |
| **Meaningfully useful (both)** | **Not met** | **Not met** | **Not met** |
| Exceeds 0.626 | No | No | No |
| Reaches ~0.70 | No | No | No |

**Threshold attribution (corrected after adding B-threshold).** An earlier version of this report said C's thresholded gain over B was "largely a threshold effect". **That was wrong.**
- Giving B the same training-only threshold barely changes it: 0.6009 vs 0.6003.
- C-threshold still leads B-threshold by +0.0072 mean Macro-F1, on 3/5 folds.
- The threshold-free AUC gain of C over B is +0.0026 (4/5 folds positive).
- So the small edge C has is a real but tiny contribution from the capability features. It is not a calibration artefact.
- It is far below the +0.03 bar and is not consistent across folds once the comparison is calibrated.

## Answers to the report questions

**1. Does the capability-only router contain useful winner signal?**
Yes, but nothing beyond what we already had.
- With 16 learned parameters and no encoder, A reaches AUC 0.648 and Macro-F1 0.597 (0.605 with a training-only threshold). That is close to MiniLM (AUC 0.670, F1 0.600) and far above a constant predictor (Macro-F1 ≈ 0.37).
- Its AUC is below both MiniLM and GTE (≈0.683).

**2. Does capability + MiniLM beat MiniLM alone?**
Not meaningfully.
- At the primary 0.5 cutoff, C equals B: 0.5968 vs 0.6003, 2/5 folds better.
- With a training-only threshold, C reaches 0.6082 and beats B (score > 0) on 4/5 folds by +0.008.
- In the fair calibrated comparison, B-threshold reaches 0.6009. C-threshold beats it by only **+0.0072 mean, on 3/5 folds**.
- AUC rises by only +0.003 (0.6699 → 0.6725).
- The pre-declared "meaningfully useful" criterion is **not met** under any comparison.

**3. Does it beat the existing 0.626 best tested router?**
No. The best result here is 0.6082 (C, thresholded). That is also below the existing GTE LogReg + threshold (0.6153).

**4. Does anything reach or approach 0.70?**
No. The highest single fold of any condition is 0.6329 (C thr, fold 4).

**5. Which capability features are stable across folds?**
Sign-stable means the same sign on 5/5 folds. Coefficients are on the standardized scale for continuous features and on 0/1 for binary features; positive favours Qwen.
- Strongest in A, ranked by |mean|/std:
  - `multiple_choice_structure` −0.85 (DeepSeek)
  - `contains_digits` −0.45 (DeepSeek)
  - `reasoning_cue` +0.97 (Qwen)
  - `requests_code` +1.17 (Qwen)
  - `character_count` −0.59
  - `explicit_answer_format` +0.27
  - `numeric_density` +0.17
  - `exceeds_256_tokens` +0.04
  - `minilm_token_count` +0.27
  - `contains_math_expression` −0.32
- Stable in C next to the student score (+0.57, |mean|/std 13.7): `reasoning_cue` +0.65, `contains_digits` −0.24, `contains_code` +0.28, `requests_code` +0.26, `multiple_choice_structure` −0.27, `character_count` −0.55, `minilm_token_count` +0.34, `numeric_density` +0.10, `exceeds_256_tokens` +0.08, `contains_math_expression` −0.17.
- Unstable: `has_embedded_context` (both models), `contains_code` (in A), and in C also `explicit_answer_format`, `constraint_count`, `structural_line_count`.
- The three length counts are highly collinear (opposite signs), so read them as a joint length effect, not one by one.
- In plain terms, the stable signals encode **task type**: code requests and reasoning/explanation prompts favour Qwen; multiple-choice and digit-bearing prompts favour DeepSeek. That matches the known task-conditional preference.

**6. Is performance different above vs below 256 tokens?**
Yes, but in the opposite direction from a truncation story. **Truncated (>256) queries are routed *better*:**

| Pooled held-out | ≤256 tokens (n=1,144 non-tie) | >256 tokens (n=1,189 non-tie) |
|---|---|---|
| Tie rate / Qwen share (non-tie) | 0.772 / 0.618 | 0.816 / 0.535 |
| B Macro-F1 / AUC | 0.556 / 0.652 | 0.626 / 0.677 |
| A (0.5) Macro-F1 / AUC | 0.546 / 0.627 | 0.621 / 0.651 |
| C (0.5) Macro-F1 / AUC | 0.559 / 0.654 | 0.618 / 0.678 |
| C (thr) Macro-F1 | 0.589 | 0.616 |

B is better on >256 than on ≤256 queries on all 5 folds (per-fold values are in `truncation_analysis.csv`). The truncated group's AUC is also higher, and the capability-only model, which sees the full length, shows the same gap. So MiniLM's 256-token truncation is **not** what limits routing. The short queries are the harder ones. Their class mix also differs (more Qwen-favoured, fewer ties), which moves the default-cutoff Macro-F1, so the AUC gap (~0.025) is the cleaner comparison. This breakdown is diagnostic only; nothing was tuned on it.

**7. What is the inference overhead?**
Measured in a single process on CPU over the 2,294 fold-0 held-out queries (`efficiency.csv`):
- **Capability extraction:** 0.83 ms/query for the regex features plus 0.55 ms/query for the untruncated MiniLM tokenizer, **1.39 ms total**.
- **A inference:** 0.004 ms/query as a numpy dot product (1.74 ms per single-row sklearn call, which is almost all sklearn call overhead). End to end, A takes about **1.4 ms/query with 16 learned parameters** (+12 scaler statistics) and a 2.9 KB pickle, with no neural encoder.
- **C overhead over MiniLM:** about **0.84 ms/query**, or 1.39 ms if the token count is recomputed rather than reused from the MiniLM tokenizer pass. That is roughly +2% over the existing 38 ms/query MiniLM end-to-end reference (not rerun), for an estimated ~38.8 ms/query.
- **C parameters:** +17 learned parameters (+14 scaler statistics), for **22,590,112 deployed parameters** in total (encoder 22,565,376 + MLP-64 head 24,705 + combiner).
- **Training cost per fold:** C takes 16 s with cached embeddings, because of the 5 inner teacher/student refits. A takes 0.4 s.

**8. What do the results imply about the current bottleneck?**
- Explicit capability signals recover most of the winner signal that a 22.6M-parameter semantic encoder finds (AUC 0.648 vs 0.670), using 16 weights. Added to MiniLM, they contribute almost nothing new: +0.003 AUC, and +0.007 Macro-F1 against the calibrated B-threshold.
- So the embeddings are not missing query-level capability information. Both representations capture the same coarse task-type signal, which is real but limited (AUC ≈ 0.65–0.68).
- Truncation is ruled out as the main limit.
- Together with the earlier capacity results (nonlinear heads ≤ 0.626), this moves the evidence further toward this view: beyond task type, the winner between these two closely related 8B models is hard to predict from query text alone. Contributing factors are binary single-run correctness labels, 80% ties and about 2.3k non-tie examples.
- This is evidence, not proof. One limitation: the capability features were standardized but not log-transformed (as the mission specifies), so heavy-tailed length counts may be modelled suboptimally.

**9. What exactly ONE next research step is justified?**
**A within-task discrimination diagnostic on the existing held-out predictions.**
- **What:** stratify `heldout_predictions.csv` by source benchmark and compute within-benchmark AUC / Macro-F1 for A, B and C. Compare them against a training-fold-only "benchmark prior" reference (per-benchmark Qwen share, used as an *evaluation-only oracle diagnostic*, never as a router feature).
- **Question it answers:** does any router discriminate winners *within* a task, or does all of the ~0.65–0.68 AUC come from identifying the task?
- **Why this one:** it needs no new model, no tuning and no new data, and it directly tests the bottleneck hypothesis from Q8.
- **How to read it:**
  - If within-task AUC is near 0.5, more query-text representation work is unlikely to help, and the project should move to label quality/quantity (for example, repeated sampling for outcome reliability) or to cost-aware routing.
  - If within-task AUC is clearly above 0.5, instance-level representation work is still justified.

## Output files

- `capability_features_v1.csv`: 15 features per query (11,467 rows; dataset/index/query_sha1 are join keys only)
- `fold_results.csv`: per-fold, mean, std and pooled metrics for A / B / C × decision rule, with paired Δ vs B
- `coefficients_by_fold.csv`: per-fold coefficients for A and C, plus a sign-stability summary
- `truncation_analysis.csv`: ≤256 vs >256 MiniLM tokens, per fold, mean/std and pooled
- `efficiency.csv`: timings, parameter counts, training cost
- `heldout_predictions.csv`: every held-out row (ties included, flagged) with A/B/C scores and predictions
