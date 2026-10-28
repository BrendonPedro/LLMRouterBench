# Evaluation Correctness Audit

Section 1 of the Router Performance Diagnostic Mission.

Read-only reproduction of label construction, train/test separation, leakage checks, and class balance directly from the raw pair data at `baselines/AvengersPro/data/small_models_seed_<seed>/seed<seed>_split0.7/`.

- Model pair: **DeepSeek-R1-0528-Qwen3-8B** (label 0) vs **Qwen3-8B** (label 1)
- Seeds audited: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

## 1. Label construction (DeepSeek-win / Qwen-win / tie)

All experiment scripts build the model-selection target with the same rule (verified by reading `train_gte_direct_selector.py`, `train_gte_threshold_selector.py`, and the `*_direct_selector` scripts):

```python
selector_target = np.select(
    [deepseek_score > qwen_score, qwen_score > deepseek_score],
    [0, 1],            # 0 = DeepSeek wins, 1 = Qwen wins
    default=-1,        # -1 = tie (equal scores)
)
# non-tie evaluation set = rows where selector_target != -1
```

Raw per-model scores take exactly the value set `[0.0, 1.0]`. Because the LLMRouterBench small-model records are **binary correctness** (0/1), a *tie* is precisely `both correct` **or** `both wrong`, and a *non-tie* winner label exists only when exactly one of the two models is correct.

Training label `y_train = (qwen_score > deepseek_score)` is identical to `selector_target` restricted to non-tie rows, so training and evaluation use the same label definition. No inversion: class 1 is Qwen in both `classifier.classes_ == [0,1]` and the `direct_selector_choice` mapping.

## 2. Train/test separation and duplicate-query leakage

| seed | train/test query overlap | dup queries (train) | dup queries (test) |
|---|---:|---:|---:|
| 42 | 0 | 0 | 0 |
| 43 | 0 | 0 | 0 |
| 44 | 0 | 0 | 0 |
| 45 | 0 | 0 | 0 |
| 46 | 0 | 0 | 0 |
| 47 | 0 | 0 | 0 |
| 48 | 0 | 0 | 0 |
| 49 | 0 | 0 | 0 |
| 50 | 0 | 0 | 0 |
| 51 | 0 | 0 | 0 |

**No train/test leakage**: zero queries appear in both the train and test split of any seed (SHA-1 exact-text match).

No duplicate query texts within any split.

## 3. Class balance among non-tie examples

| seed | split | total | tie | tie% | non-tie | DeepSeek-win | Qwen-win | Qwen share of non-tie |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | train | 8021 | 6407 | 79.88 | 1614 | 673 | 941 | 0.583 |
| 42 | test | 3446 | 2727 | 79.14 | 719 | 317 | 402 | 0.5591 |
| 43 | train | 8021 | 6417 | 80.0 | 1604 | 680 | 924 | 0.5761 |
| 43 | test | 3446 | 2717 | 78.85 | 729 | 310 | 419 | 0.5748 |
| 44 | train | 8021 | 6365 | 79.35 | 1656 | 728 | 928 | 0.5604 |
| 44 | test | 3446 | 2769 | 80.35 | 677 | 262 | 415 | 0.613 |
| 45 | train | 8021 | 6388 | 79.64 | 1633 | 703 | 930 | 0.5695 |
| 45 | test | 3446 | 2746 | 79.69 | 700 | 287 | 413 | 0.59 |
| 46 | train | 8021 | 6399 | 79.78 | 1622 | 696 | 926 | 0.5709 |
| 46 | test | 3446 | 2735 | 79.37 | 711 | 294 | 417 | 0.5865 |
| 47 | train | 8021 | 6380 | 79.54 | 1641 | 705 | 936 | 0.5704 |
| 47 | test | 3446 | 2754 | 79.92 | 692 | 285 | 407 | 0.5882 |
| 48 | train | 8021 | 6359 | 79.28 | 1662 | 710 | 952 | 0.5728 |
| 48 | test | 3446 | 2775 | 80.53 | 671 | 280 | 391 | 0.5827 |
| 49 | train | 8021 | 6381 | 79.55 | 1640 | 696 | 944 | 0.5756 |
| 49 | test | 3446 | 2753 | 79.89 | 693 | 294 | 399 | 0.5758 |
| 50 | train | 8021 | 6380 | 79.54 | 1641 | 696 | 945 | 0.5759 |
| 50 | test | 3446 | 2754 | 79.92 | 692 | 294 | 398 | 0.5751 |
| 51 | train | 8021 | 6395 | 79.73 | 1626 | 672 | 954 | 0.5867 |
| 51 | test | 3446 | 2739 | 79.48 | 707 | 318 | 389 | 0.5502 |

- Mean tie rate: train 79.6%, test 79.7% — consistent with the mission's '~80% ties' note.
- Mean Qwen share of non-tie test labels: 0.580 (near-balanced; a constant-Qwen predictor would get ~50% of non-tie test rows right but Macro-F1 ~0.33 because it never predicts DeepSeek).
- Mean non-tie test rows per seed: 699 of 3446 total (20.3%).

Per-dataset non-tie counts, both_correct, and both_wrong breakdowns are in `label_statistics.csv` (one row per seed x split x dataset).

## 4. Methods evaluated on identical non-tie test instances

For every saved method prediction file, the stored `selector_target` was recomputed from that file's own stored scores and compared; the non-tie (dataset,index) set was also compared against the set recomputed from the raw test split.

| seed | method | label mismatches | rows | non-tie set symmetric-diff vs raw |
|---|---|---:|---:|---:|
| 42 | gte_direct | 0 | 3446 | 0 |
| 42 | bertmini_direct | 0 | 3446 | 0 |
| 42 | minilm_direct | 0 | 3446 | 0 |
| 43 | gte_direct | 0 | 3446 | 0 |
| 43 | bertmini_direct | 0 | 3446 | 0 |
| 43 | minilm_direct | 0 | 3446 | 0 |
| 44 | gte_direct | 0 | 3446 | 0 |
| 44 | bertmini_direct | 0 | 3446 | 0 |
| 44 | minilm_direct | 0 | 3446 | 0 |
| 45 | gte_direct | 0 | 3446 | 0 |
| 45 | bertmini_direct | 0 | 3446 | 0 |
| 45 | minilm_direct | 0 | 3446 | 0 |
| 46 | gte_direct | 0 | 3446 | 0 |
| 46 | bertmini_direct | 0 | 3446 | 0 |
| 46 | minilm_direct | 0 | 3446 | 0 |
| 47 | gte_direct | 0 | 3446 | 0 |
| 47 | bertmini_direct | 0 | 3446 | 0 |
| 47 | minilm_direct | 0 | 3446 | 0 |
| 48 | gte_direct | 0 | 3446 | 0 |
| 48 | bertmini_direct | 0 | 3446 | 0 |
| 48 | minilm_direct | 0 | 3446 | 0 |
| 49 | gte_direct | 0 | 3446 | 0 |
| 49 | bertmini_direct | 0 | 3446 | 0 |
| 49 | minilm_direct | 0 | 3446 | 0 |
| 50 | gte_direct | 0 | 3446 | 0 |
| 50 | bertmini_direct | 0 | 3446 | 0 |
| 50 | minilm_direct | 0 | 3446 | 0 |
| 51 | gte_direct | 0 | 3446 | 0 |
| 51 | bertmini_direct | 0 | 3446 | 0 |
| 51 | minilm_direct | 0 | 3446 | 0 |

**No label inversion**: every method's stored `selector_target` matches the canonical rule recomputed from its own scores.
**Identical non-tie test set**: each method's non-tie test instances exactly match the set derived from the raw test split (symmetric difference = 0), so cross-method Macro-F1 numbers are computed on the same instances and are directly comparable.

## 5. Macro-F1 / balanced-accuracy / threshold implementation

- **Macro-F1**: `sklearn.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)` on the two-class non-tie set. Correct, unweighted mean of per-class F1.
- **Balanced accuracy**: `sklearn.metrics.balanced_accuracy_score` (mean of per-class recall). Correct.
- **Default threshold** (direct selectors): `argmax`/`predict`, i.e. Qwen chosen when `P(qwen) >= 0.5`.
- **Threshold selector**: candidate thresholds scored by `f1_score` on **out-of-fold training predictions** (`oof_qwen_probability`), ties broken toward 0.5. Test labels are **not** used to pick the threshold — methodologically clean.

Sanity re-computation on saved GTE seed-42 predictions: Macro-F1 = 0.6024, balanced-accuracy = 0.6086 (matches the reported ~0.60 regime).

## Verdict

The evaluation pipeline is **correct**. Label construction, train/test separation, non-tie filtering, metric implementations, and threshold selection are all sound; no leakage and no label inversion were found. The low Macro-F1 is therefore not an evaluation bug — it reflects the difficulty of the task itself. Two structural facts stand out for later sections: (a) ~80% of rows are ties and are **dropped** from supervision, and (b) identical query texts can appear with different winner labels, capping any text-only router's attainable F1.
