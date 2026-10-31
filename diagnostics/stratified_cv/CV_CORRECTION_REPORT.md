# CV Correction Report — DeepSeek-R1-0528-Qwen3-8B vs Qwen3-8B router

Scope: synthesis of the completed corrected-CV evidence (Tasks 1–2, not re-run)
plus the corrected efficiency table (Task 3) and a new raw-query end-to-end
MiniLM timing (Task 4). All accuracy numbers are held-out **non-tie Macro-F1**
on the frozen stratified 5-fold split (`fold_assignments.csv`), mean ± sample
std over folds.

Artefacts:

| File | Content |
|---|---|
| `corrected_nonlinear_cv.csv` | Task 1: Ridge vs MLP-64 MiniLM distillation students (saved, not re-run) |
| `gte_classifier_capacity.csv` | Task 2: GTE LogReg / MLP-64 / RBF-SVM (saved, not re-run) |
| `minilm_end_to_end_timing.csv` | **new**: raw-query end-to-end CPU timing (`scripts/time_minilm_end_to_end.py`) |
| `corrected_efficiency.csv` | **new**: corrected parameter / timing / requirement table (`scripts/build_corrected_efficiency.py`) |

---

## 1. What is the corrected nonlinear MiniLM result?

**Frozen all-MiniLM-L6-v2 → MLP-64 score-distillation student: Macro-F1 = 0.6003 ± 0.0154**
(AUC 0.670 ± 0.017).

| Student (same fold-local GTE LogReg teacher) | Macro-F1 | AUC | Head params |
|---|---|---|---|
| Ridge (linear) | 0.5922 ± 0.0054 | 0.659 | 385 |
| MLP-64 (nonlinear) | **0.6003 ± 0.0154** | 0.670 | 24,705 |

The MLP-64 student beats Ridge on 4 of 5 folds (per-fold Δ: −0.013, +0.025,
+0.006, +0.002, +0.020; mean +0.008). That gain is small and lies within one
fold-std of the MLP result. The student tracks its teacher closely: its score
correlates 0.94 with the teacher's score, and 90% of routing choices agree. It
matches the GTE LogReg teacher's own default-cutoff F1 (0.598) while running no
7B model online.

## 2. Was the previous 0.5922 result the Ridge student?

**Yes.** `run_cv_eval.run_distillation` uses `Ridge(alpha=1.0)`. Task 1 re-ran
that Ridge student on the same teacher scores and reproduced 0.5922 exactly. The
old "Frozen-MiniLM distillation" row is therefore the **linear** student, not the
nonlinear MLP-64.

## 3. Did nonlinear classifier capacity substantially improve GTE routing?

**No.** On the same cached GTE-Qwen2-7B embeddings:

| GTE classifier | default cutoff | + training-only threshold | AUC | train F1 (default) |
|---|---|---|---|---|
| LogReg (linear) | 0.5980 | 0.6153 ± 0.0184 | **0.683** | 0.671 |
| MLP-64 | 0.5911 | 0.6063 ± 0.0034 | 0.664 | 0.622 |
| RBF-SVM | 0.6128 | **0.6260 ± 0.0296** | 0.674 | 0.721 |

- The best nonlinear result, RBF-SVM with a training-only threshold, beats linear
  LogReg with a threshold by only **+0.011** F1. That is smaller than the SVM's own
  fold-std (0.030) and is not consistent: the per-fold Δ is +0.023, +0.009,
  +0.043, +0.016 and −0.036.
- On threshold-free ranking quality (AUC), **neither nonlinear model beats linear
  LogReg** (0.674 and 0.664 vs 0.683). The SVM's small F1 edge therefore comes from
  its operating point, not from better winner ranking.
- The SVM fits the training folds better (train F1 0.721 vs test 0.613), but that
  extra fit does not carry over to held-out queries.

Interpretation: **the tested classifiers do not reveal substantial additional
winner signal from the current query-only GTE representation.** This is evidence
about *classifier capacity on this representation at this data scale* (~1,866
non-tie training queries per fold). It is not a proof that no other function of
these embeddings could do better, and **0.626 is not a mathematically proven
ceiling**.

## 4. Did anything reach 0.70?

**No.** No tested method reaches 0.70 mean Macro-F1. The best mean is 0.6260
(GTE RBF-SVM + threshold), and the best single fold is 0.666 (same method,
fold 2).

## 5. Which efficiency measurements are genuinely end-to-end?

**Only one neural-router measurement is end-to-end:** the new MiniLM MLP-64 timing.

It covers the full path: raw query → WordPiece tokenizer (truncation at 256
tokens) → all-MiniLM-L6-v2 encoder → attention-mask mean pooling → L2
normalisation → MLP-64 head → routing decision. It runs on CPU in one process,
with the model already in memory.

| | batch 1 | batch 32 |
|---|---|---|
| Timed queries | 2,294 (all fold-0 test queries; real benchmark text) | 2,294 (72 calls) |
| Warm-up | 32 fold-0 *train* queries per batch size, untimed | same |
| Mean latency / query | **38.0 ms** | 37.9 ms amortized (1,207 ms per 32-query call) |
| Median latency / query | **38.0 ms** | 1,181 ms per call |
| p95 latency / query | 53.6 ms | 1,413 ms per call |
| Throughput | **26.3 queries/s** | 26.4 queries/s |
| Stage breakdown (ms/query) | tokenize 1.39, **encoder 35.9**, pool+norm 0.27, head+decision 0.39 | encoder 37.2 |
| Peak RSS of online process | **1,195 MB** (188 MB at start, 830 MB after torch/transformers import, 854 MB after model load) | |

Environment: AMD Ryzen 5 8500G, 4 logical CPUs, 4 torch threads, CPU only
(CUDA unavailable), 9.7 GiB RAM, WSL2, torch 2.14.1, transformers 5.19.0.
sentence-transformers is not installed; pooling was implemented directly.

Validation:
- The end-to-end embeddings match the cached CV embeddings exactly: min cosine
  1.000000 over all 2,294 queries.
- Routing decisions agree 100% with the cached-embedding path (max |score diff|
  1.1e-6).
- The rebuilt head is Task 1's fold-0 MLP-64. It uses the same code and seed, and
  its 55 epochs equal Task 1's fold-0 epoch count.
- No metric was recomputed.

Notes:
- The encoder is ~94% of latency; the router head is negligible (~0.4 ms).
- Batching to 32 gives no throughput gain on this CPU. Most queries fill the
  256-token window (55.9% are truncated at 256), so the encoder is compute-bound.
- A first timing run measured 41.2 ms/query at batch 1. It was discarded because
  its memory measurement was invalid (`ru_maxrss` is inherited across fork+exec).
  Treat the 38–41 ms range as run-to-run variation on this machine.

**TF-IDF + LogReg** is *partially* end-to-end. Its timer includes the vectorizer
transform on raw text, but only as one batch over the whole non-tie test fold.
The resulting 0.29 ms/query is batch-amortized throughput, not single-query
latency.

**GTE-Qwen2-7B: end-to-end GTE timing was not measured on this machine because
the 7B encoder cannot be fairly loaded within available memory.** The local
config gives 7,069,121,024 encoder parameters (Qwen2Model, no lm_head). That is
~28.3 GB of weights in fp32 and ~14.1 GB in bf16, against 9.7 GiB of RAM. The
weights are also not present locally; only `config.json` is. No GTE latency is
estimated anywhere.

## 6. Which earlier efficiency measurements must be relabelled?

All of the following are from `efficiency_results.csv`. Corrected values are in
`corrected_efficiency.csv`.

1. **GTE Direct LogReg, GTE + train-only threshold, MiniLM Direct, Frozen-MiniLM
   distillation: `ms_per_query` / `queries_per_sec`.**
   - These timed only the router head on **cached embeddings**. No tokenizer or
     encoder was run, so they must be labelled *router-head-only cached-embedding
     latency (batch-amortized)*, never end-to-end latency.
   - They also divided the head time for *all* ~2,293 test rows (ties included) by
     the ~467 non-tie rows, which inflates the per-query figure 4.9×. Example:
     GTE LogReg 0.134 → 0.027 ms per scored row.
   - The "7,466 q/s" for GTE describes a 3,585-parameter dot product, not a
     7B-parameter router.
2. **Frozen-MiniLM distillation: `total_params = 3,970`.**
   - This adds the offline GTE teacher head (3,585) to the online Ridge student
     (385). The deployed router head is **385** parameters, and the deployed total
     is 22,565,761 (MiniLM encoder + head).
   - This row is also the **linear Ridge** student (0.5922), not the nonlinear
     MLP-64.
3. **GTE rows: `total_params = 3,585`.**
   - This counts the head only. The **total deployed** model is ~7.07B
     (7,069,124,609 with the LogReg head). The approximation is noted because the
     count is derived from config, not from loaded weights.
4. **MiniLM "22.7M param encoder".**
   - The loaded encoder used for routing has **22,565,376** parameters. 22.7M
     (22,713,216) includes a 147,840-parameter pooler that mean pooling never uses.
5. **`train_sec_per_fold`** for all neural rows is *cached-embedding head training
   time*. It excludes embedding the training queries, and for distillation it
   excludes producing the GTE teacher embeddings.

Also new in the corrected table:
- GTE RBF-SVM stores ~5.76M values (~1,606 support vectors × 3,584 dims) but has
  only ~1,607 learned dual coefficients plus an intercept. Its cached head
  inference is ~4.6 ms per row, two to three orders of magnitude slower than the
  LogReg head (0.008–0.03 ms per row).
- The MLP-64 MiniLM head is 24,705 parameters (502,611-byte sklearn pickle
  including Adam state; 197,640 bytes of raw float64 weights). Its total deployed
  model is 22,590,081 parameters.
- The GTE teacher is needed offline only, never at deployment.

## 7. What bottleneck is supported by the evidence?

Four kinds of evidence point the same way:

- **Classifier capacity.** This is *not* the bottleneck on this representation:
  three classifier families with different inductive biases land within ~0.02 F1
  of each other, and nonlinearity does not improve AUC. The same pattern appears
  on MiniLM, where MLP-64 vs Ridge gives +0.008.
- **Representation scale.** This buys little. A 22.6M-parameter MiniLM student
  (0.600) sits within ~0.015–0.026 F1 of the 7.07B GTE classifiers (0.615–0.626),
  with ~300× fewer encoder parameters.
- **Representation type.** Both encoders are generic semantic sentence embeddings
  of the query alone. The supported reading is that **the query-only generic
  semantic embedding carries only modest information about which of these two
  models will win**, whichever encoder or classifier reads it. MiniLM also sees
  only the first 256 tokens of 55.9% of queries.
- **Intrinsic pair difficulty.** The two candidates are closely related 8B models.
  The current evidence **cannot separate** "the winner signal exists but generic
  embeddings miss it" from "the per-query winner is largely unpredictable from the
  query". Both remain open.
- **Downstream routing headroom.** The ~79.7% tie rate means that only ~20% of
  queries are ones where the routing choice changes the outcome. That bounds how
  much any router can improve end-task accuracy or cost, but it **does not**
  mathematically impose a ~0.62 Macro-F1 ceiling on the non-tie subset, where
  Macro-F1 is computed.

Conclusion: the limiting factor supported by the evidence is the
**winner-relevant information content of the query-only generic embedding**, not
classifier capacity. Whether richer query-level information exists is the open
question.

## 8. Exactly ONE next experiment

**Interpretable query capability-profile routing (frozen 5-fold CV).**

Test whether a *materially different, deployable* query-level signal adds winner
information beyond the generic embedding. The signal is an explicit,
interpretable profile of what each query demands, computed from the raw query
text alone.

1. **Signal.** Define a fixed, versioned capability extractor *before* looking at
   any fold results. It produces ~10–15 interpretable attributes from raw query
   text, for example:
   - math / numeric computation present
   - multiple-choice options present
   - code present or requested
   - explicit answer-format constraint (boxed answer, letter only, JSON, …)
   - number of explicit instructions or constraints
   - multi-step reasoning cues
   - embedded context passage present
   - query token length and the >256-token truncation flag

   The extractor must not use the dataset name/ID or any outcome label, and it is
   deterministic, so it can be used at deployment.
2. **Models.** One fixed configuration each, with no sweep:
   - (A) capability-only LogReg (the interpretable router)
   - (B) the existing MiniLM MLP-64 student score (reference, 0.6003)
   - (C) capability profile + MLP-64 student score → LogReg. The student score
     enters as an inner out-of-fold score inside outer-train, so there is no
     leakage.
3. **Protocol.** Same frozen folds. Fit and threshold selection use outer-train
   only. Report non-tie Macro-F1 (default cutoff and training-only threshold),
   AUC, and per-attribute coefficients with their fold-to-fold sign stability.
4. **Pre-declared reading.**
   - If (C) improves AUC over the best existing AUC (GTE LogReg, 0.683) by more
     than one fold-std on ≥4/5 folds, query-level capability information carries
     winner signal that generic embeddings miss. That would also give
     interpretable routing rules, at negligible online cost beyond MiniLM.
   - If (A) and (C) add nothing, the evidence shifts toward intrinsic pair
     difficulty, i.e. winners weakly predictable from the query alone.
   - Caveat: some attributes may partly proxy dataset identity. That is
     acceptable for deployment, since no dataset label is used, but it must be
     stated when interpreting coefficients.

Out of scope for this experiment: true dataset labels at deployment, model-pool
expansion, Edge/Split/Remote integration, and further classifier tweaks on
GTE/MiniLM embeddings.
