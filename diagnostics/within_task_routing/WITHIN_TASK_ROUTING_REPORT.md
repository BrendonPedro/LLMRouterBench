# Within-Task Routing Diagnostic: Report

Repository `/home/brendonpedro/research/LLMRouterBench`, branch `within-task-routing-diagnostic`, built on commit `c5b0d7a`.
Script: `diagnostics/scripts/within_task_routing_diagnostic.py`. This is analysis only:
- no router was trained or retrained;
- no thresholds were tuned;
- folds were not regenerated;
- `diagnostics/capability_profile/` was not modified.

**Benchmark identity (`dataset`) is used ONLY for this diagnostic analysis. It is never a router feature.** The "benchmark-identity diagnostic reference" is explanatory only. It is not a model and not a proposed routing method.

**Question:** do MiniLM (B) and Capability+MiniLM (C) predict which model wins *for individual queries*, or do they mainly recognise which benchmark a query comes from?

## Part 1: Verification of the saved held-out predictions

`diagnostics/capability_profile/heldout_predictions.csv` columns used:

| Role | Column |
|---|---|
| Query identifier | (`dataset`, `index`), plus `query_sha1` |
| Benchmark identifier | `dataset` (15 benchmarks) |
| Frozen outer fold | `fold` |
| True winner | `target` (0 = DeepSeek, 1 = Qwen, −1 = tie); `is_nontie` |
| B held-out score / prediction | `B_student_score` / `B_pred` (score > 0) |
| C held-out score / prediction | `C_prob_qwen` / `C_pred_threshold` (training-only threshold) |

The script checks all of the following before writing anything:
- The file has 11,467 rows, exactly one per query, with no duplicate (`dataset`, `index`).
- Every row's `fold` and `query_sha1` match `diagnostics/stratified_cv/fold_assignments.csv`.
- `is_nontie` agrees with `target`.
- `B_pred == (B_student_score > 0)`.
- There are 2,333 held-out non-tie rows.

Reproduced fold-mean metrics match the saved `capability_profile/fold_results.csv` exactly (to 1e-9, per fold):

| | Macro-F1 | AUC |
|---|---|---|
| B (score > 0) | 0.6003 | 0.6699 |
| C (training-only threshold) | 0.6082 | 0.6725 |

## Pre-declared definitions

These were fixed in the script docstring before any per-benchmark result was seen.
- **Within-benchmark AUC:** AUC over one benchmark's held-out non-tie rows, pooled across the 5 outer folds. Each row is scored by its own fold's model. It is computed only when both winner classes are present.
- **Near chance:** 0.45 ≤ AUC ≤ 0.55.
- **Clear within-task signal:** AUC > 0.55 **and** the 95% class-stratified bootstrap CI lower bound > 0.50 (2,000 resamples).
- **C vs B per benchmark:** paired bootstrap CI of (C − B) AUC; a winner is named only if the CI excludes 0.
- **Weighted mean:** weights = number of non-tie rows in the benchmark.

The bootstrap CIs are uncertainty estimates. They are **not** multiple-comparison-corrected significance tests. With 15 benchmarks, one interval or so may exclude 0.5 by chance.

## Part 2: Within-benchmark AUC

All 15 benchmarks have both winner classes, so **15 are valid** and none had to be dropped. Three have a minority class under 10 rows (aime 4 DS wins, humaneval 3 DS wins, mathbench 6 DS wins); their AUCs are very noisy.

| Benchmark | Non-tie | DS wins | Qwen wins | Qwen share | B AUC [95% CI] | C AUC [95% CI] | C − B | Verdict |
|---|---|---|---|---|---|---|---|---|
| aime* | 13 | 4 | 9 | 0.692 | 0.556 [0.22, 0.86] | 0.528 [0.22, 0.83] | −0.028 | undetermined (n tiny) |
| **bbh** | 135 | 97 | 38 | 0.281 | **0.713** [0.62, 0.80] | **0.712** [0.62, 0.80] | −0.001 | **clear signal (B and C)** |
| emorynlp | 120 | 60 | 60 | 0.500 | 0.429 [0.33, 0.53] | 0.452 [0.34, 0.55] | +0.023 | chance / below |
| finqa | 127 | 40 | 87 | 0.685 | 0.480 [0.38, 0.59] | 0.488 [0.39, 0.60] | +0.008 | near chance |
| gpqa | 50 | 28 | 22 | 0.440 | 0.617 [0.46, 0.77] | 0.604 [0.44, 0.76] | −0.013 | > 0.60 but CI includes 0.5 |
| humaneval* | 67 | 3 | 64 | 0.955 | 0.385 [0.05, 0.92] | 0.365 [0.17, 0.58] | −0.021 | undetermined (3 DS wins) |
| kandk | 315 | 101 | 214 | 0.679 | 0.398 [0.33, 0.47] | 0.388 [0.32, 0.46] | −0.009 | **below chance (inverted)** |
| **korbench** | 253 | 140 | 113 | 0.447 | **0.681** [0.62, 0.74] | **0.690** [0.63, 0.75] | +0.010 | **clear signal (B and C)** |
| livecodebench | 187 | 78 | 109 | 0.583 | 0.487 [0.40, 0.57] | 0.501 [0.42, 0.58] | +0.014 | near chance |
| math500 | 23 | 13 | 10 | 0.435 | 0.162 [0.00, 0.35] | 0.308 [0.09, 0.55] | +0.146 (C better†) | inverted (n tiny) |
| mathbench* | 20 | 6 | 14 | 0.700 | 0.476 [0.24, 0.74] | 0.452 [0.21, 0.71] | −0.024 | near chance (n tiny) |
| **mbpp** | 410 | 109 | 301 | 0.734 | **0.669** [0.61, 0.73] | **0.680** [0.62, 0.74] | +0.011 | **clear signal (B and C)** |
| medqa | 189 | 110 | 79 | 0.418 | 0.505 [0.42, 0.59] | 0.511 [0.42, 0.59] | +0.006 | near chance |
| meld | 251 | 106 | 145 | 0.578 | 0.456 [0.39, 0.53] | 0.499 [0.43, 0.58] | +0.043 | near chance |
| mmlupro | 173 | 95 | 78 | 0.451 | 0.495 [0.41, 0.58] | 0.494 [0.41, 0.58] | −0.001 | near chance |

\* Minority class < 10 rows. † math500 is the only benchmark where the paired bootstrap CI of C − B excludes 0. It has n = 23, and both AUCs are far below 0.5.

Aggregates over the 15 valid benchmarks:

| | B | C |
|---|---|---|
| Macro (unweighted) mean AUC [95% CI] | **0.501** [0.457, 0.548] | **0.512** [0.473, 0.552] |
| Weighted mean AUC [95% CI] | **0.533** [0.507, 0.561] | **0.543** [0.520, 0.566] |
| Median AUC | **0.487** | **0.499** |
| Benchmarks with AUC < 0.45 | 4 | 3 |
| 0.45–0.55 | 6 | 8 |
| > 0.55 | 5 | 4 |
| ≥ 0.60 | 4 | 4 |
| Clear within-task signal (pre-declared rule) | 3 (bbh, korbench, mbpp) | 3 (bbh, korbench, mbpp) |

**Sensitivity check (pooling across fold models).** Pooled scores come from 5 different fold models, so the check was repeated two other ways. The same three benchmarks remain the only clear ones, and the aggregates stay near 0.50–0.56:

| Method | B macro / weighted | C macro / weighted |
|---|---|---|
| Mean of per-fold AUCs | 0.529 / 0.561 | 0.548 / 0.564 |
| Within-fold percentile ranks, then pooled | 0.505 / 0.542 | 0.507 / 0.545 |

## Part 3: Benchmark-identity diagnostic reference

For each outer fold, score = the **outer-train** non-tie Qwen share of the row's benchmark. Predict Qwen if the share is ≥ 0.5. No held-out label enters the score. **No benchmark ever had zero outer-train non-tie rows, so 0 held-out rows used the global-share fallback.** The per-fold, per-benchmark shares are in `benchmark_identity_reference.csv`. The predicted side is the same in all 5 folds for 13 benchmarks; it flips for emorynlp and math500, which sit near 0.5.

| Fold | Macro-F1 | AUC | Bal. acc |
|---|---|---|---|
| 0 | 0.6076 | 0.6275 | 0.6072 |
| 1 | 0.6204 | 0.6485 | 0.6195 |
| 2 | 0.6115 | 0.6551 | 0.6108 |
| 3 | 0.6025 | 0.6671 | 0.6024 |
| 4 | 0.6241 | 0.6851 | 0.6235 |
| **mean ± std** | **0.6132 ± 0.0089** | **0.6567 ± 0.0215** | 0.6127 |
| pooled | 0.6134 | 0.6511 | 0.6127 |

Pooled confusion matrix (rows = true [DS, Qwen], cols = predicted [DS, Qwen]): [[501, 489], [377, 966]]. Precision/recall: DS 0.571 / 0.506, Qwen 0.664 / 0.719.

By construction, its within-benchmark AUC is 0.5: the score is constant within each benchmark in each fold.

## Part 4: Between-task vs within-task comparison

| | Overall Macro-F1 (fold mean ± std) | Overall AUC (fold mean) | Overall AUC (pooled) | Within macro AUC | Within weighted AUC | Within median AUC |
|---|---|---|---|---|---|---|
| benchmark-identity diagnostic reference | **0.6132 ± 0.0089** | 0.6567 | 0.6511 | 0.500 (by construction) | 0.500 | 0.500 |
| B: MiniLM MLP-64 student | 0.6003 ± 0.0154 | 0.6699 | 0.6685 | 0.501 | 0.533 | 0.487 |
| C: capability + MiniLM | 0.6082 ± 0.0196 | 0.6725 | 0.6708 | 0.512 | 0.543 | 0.499 |
| C − B | +0.0078 | +0.0026 | +0.0023 | +0.011 [−0.019, +0.040] | +0.010 [−0.006, +0.026] | +0.011 |

The within-benchmark AUCs are pooled across folds, so the pooled overall AUC is the like-for-like column to compare them with.

What this shows:
- **Benchmark identity alone matches or beats both routers on Macro-F1.** The reference is higher than B on 4/5 folds and higher than C on 3/5 folds, with lower variance.
- B and C rank slightly better overall (AUC +0.013 to +0.016 over the reference). That small margin is roughly what the three benchmarks with real within-task signal would contribute.
- **Both B and C are near chance within the benchmark** (0.45–0.55) on finqa, livecodebench, medqa, meld, mmlupro and mathbench. emorynlp is near chance for C (0.452) and just below for B (0.429). aime and humaneval are uninformative because the minority class is too small.
- **Both are systematically below chance** on kandk: AUC ≈ 0.39–0.40, with both CIs entirely below 0.5 (inverted ranking). math500 shows the same pattern but has only 23 rows. Here the routers' scores order queries *opposite* to the true winner within the task. That is consistent with the routers relying on a cue that separates benchmarks but runs the other way inside kandk. It is an observation, not a tested mechanism.
- **Clear within-task ranking signal** (B and C both, under the pre-declared rule) appears only on **bbh (≈0.71), korbench (≈0.68–0.69) and mbpp (≈0.67–0.68)**. Together they hold 798 of the 2,333 non-tie rows (34%). gpqa reaches about 0.60–0.62 but its CI includes 0.5 (n = 50).
- **C vs B within tasks:** C's within-benchmark gains (+0.010 to +0.011 in macro, weighted and median) are about as large as its overall AUC gain (+0.002 to +0.003). But every aggregate CI includes 0, and no benchmark with adequate n shows a CI-excluding difference. The only CI-excluding case is math500 (n = 23, both AUCs < 0.5). **C does not add measurable within-task information beyond B.** It also adds essentially no between-task separation (overall AUC +0.003).

## Interpretation against the pre-declared rules

- **Rule 1 applies.** Overall AUC is 0.67, yet within-benchmark AUC is near or below 0.5 for most benchmarks: B has 10/15 at ≤ 0.55, C has 11/15, and macro AUC is ≈ 0.50 with median ≈ 0.49–0.50. Benchmark identity alone, used purely as a diagnostic, reaches higher Macro-F1 than either router. **So much of the apparent routing signal is associated with task/benchmark identity rather than individual-query winner prediction.**
- **Rule 2 does not apply.** Within-benchmark macro AUC is ≈ 0.50–0.51, not ≈ 0.60. Only 4 benchmarks are ≥ 0.60, and only 3 pass the CI rule.
- **Rule 3 (where within-task signal exists):** genuine per-query ranking signal is confined to **bbh, korbench and mbpp**. Nothing in these results supports generalizing it to the other benchmarks.
- **Rule 4:** benchmark identity remains an explanatory diagnostic only. It must not become a deployed router feature.
- **Rule 5:** none of this is a mathematical ceiling. Different labels (e.g. outcomes averaged over repeated samples), different features or more data could change it.

**Main answer: task recognition.** The routers' overall AUC of about 0.67 comes mostly from recognising which benchmark a query belongs to. Within a benchmark, ranking is at chance for most tasks (and inverted for kandk). The exceptions, with real within-task signal, are bbh, korbench and mbpp.

## One next research step

**Measure per-query winner reliability.** Re-sample both models several times (for example k = 5–10 generations each, same decoding settings) on a fixed sample of non-tie queries drawn from the signal benchmarks (bbh, korbench, mbpp) and the chance-level ones (kandk, meld, medqa, finqa). Estimate how often the single-run "winner" label flips.
- **Why this one:** with binary single-run correctness and 80% ties, near-chance within-task AUC has two explanations that this analysis cannot separate.
  - The per-query winner may be largely stochastic. Then no text router can rank it, and routing should rely on task-level priors and cost.
  - The labels may be stable, and the representation is what fails. Then per-query routing work is still justified, starting from the three benchmarks where signal already exists.
- This needs generations from the two existing models only. It does not expand the model pool, use benchmark identity as a feature, or touch Edge/Split/Remote.

## Output files

- `benchmark_auc.csv`: per-benchmark counts, B/C AUC with bootstrap CIs, paired C − B CIs, verdict flags, sensitivity AUCs
- `benchmark_identity_reference.csv`: per fold × benchmark outer-train Qwen share, fallback flag, predicted side, held-out counts
- `fold_results.csv`: per-fold / pooled / mean / std overall metrics (incl. confusion matrices) for the reference, B and C
- `comparison_summary.csv`: the Part 4 table plus within-benchmark aggregates, AUC-band counts, bootstrap CIs and sensitivity aggregates
