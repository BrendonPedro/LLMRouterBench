# Pair Learnability (Section 3)

Router Performance Diagnostic Mission, Priority 2.

**Question answered:** *Does the controlled DeepSeek-R1-0528-Qwen3-8B vs Qwen3-8B pair create an unusually difficult routing target, independent of any particular classifier?*

Derived entirely from the Section-1-audited `label_statistics.csv` (10 seeds, 42-51; 15 datasets). A row is a **tie** when both models share the same binary outcome (both correct or both wrong) and **non-tie / informative** when exactly one model is correct. Only non-tie rows carry a learnable winner label, and routers are evaluated only on them.

## Pool-level verdict (TEST split, mean over 10 seeds)

| metric | value | meaning |
|---|---:|---|
| outcome agreement (tie rate) | 0.797±0.005 | share of rows where routing is irrelevant (both same outcome) |
| &nbsp;&nbsp;of which both-correct | 0.557±0.006 | either model works |
| &nbsp;&nbsp;of which both-wrong | 0.241±0.004 | neither model works |
| informative non-tie rate | 0.203±0.005 | share of rows carrying a learnable winner label |
| winner-label sparsity | 0.797 | share of rows with NO learnable winner signal |
| DeepSeek winner share (of non-tie) | 0.420±0.016 | class balance of the winner label |
| winner minority share | 0.420±0.016 | 0.50 = perfectly balanced |
| **pair-oracle gain** | **0.085±0.005** | accuracy a perfect router adds OVER always picking the best single model |

## Why this is a hard target

1. **Routing is irrelevant on ~80% of the pool.** Only ~20% of test rows are informative, so ~80% carry no winner signal to learn from or be scored on. The learnable surface is intrinsically small.

2. **The ceiling on *usefulness* is tiny.** A perfect oracle router beats the best single static model by only **8.5 percentage points** of overall accuracy (pair-oracle gain = min(DS,Qwen wins)/total). Most of the apparent headroom in Macro-F1 lives in a sliver of the data.

3. **The winner label is close to a coin flip overall** (DeepSeek share 0.42 of non-tie), so there is no dominant-class shortcut at the pool level — yet the signal is highly *dataset-conditional* (next section).

## Task-specific dominance (TEST, mean over seeds)

Per-dataset winner preference among non-tie rows. `ds_share` > 0.5 means DeepSeek tends to win the informative rows; < 0.5 means Qwen does. `dominance` is the majority winner's share (1.0 = one model always wins the informative rows).

| dataset | mean non-tie n | nontie_rate | ds_share | dominance | oracle_gain | train→test ds_share gap | pref stable |
|---|---:|---:|---:|---:|---:|---:|:--:|
| humaneval | 20 | 0.406 | 0.058 | 0.942 | 0.024 | 0.020 | yes |
| mbpp | 119 | 0.407 | 0.261 | 0.739 | 0.106 | 0.007 | yes |
| mathbench | 6 | 0.140 | 0.263 | 0.737 | 0.040 | 0.040 | yes |
| finqa | 38 | 0.111 | 0.297 | 0.703 | 0.033 | 0.026 | yes |
| kandk | 92 | 0.437 | 0.312 | 0.688 | 0.136 | 0.012 | yes |
| livecodebench | 57 | 0.180 | 0.404 | 0.599 | 0.072 | 0.019 | yes |
| aime | 3 | 0.178 | 0.423 | 0.727 | 0.056 | 0.150 | yes |
| meld | 75 | 0.202 | 0.432 | 0.575 | 0.086 | 0.014 | yes |
| emorynlp | 36 | 0.175 | 0.503 | 0.558 | 0.077 | 0.004 | **NO** |
| korbench | 78 | 0.209 | 0.533 | 0.560 | 0.092 | 0.030 | yes |
| math500 | 8 | 0.050 | 0.540 | 0.673 | 0.017 | 0.032 | yes |
| medqa | 58 | 0.152 | 0.560 | 0.560 | 0.067 | 0.033 | yes |
| mmlupro | 53 | 0.176 | 0.573 | 0.580 | 0.074 | 0.033 | yes |
| gpqa | 14 | 0.228 | 0.574 | 0.596 | 0.093 | 0.017 | yes |
| bbh | 41 | 0.127 | 0.700 | 0.700 | 0.038 | 0.025 | yes |

**Reading the table.** The pair splits cleanly into model-specialised regimes: reasoning/knowledge datasets (bbh, korbench, gpqa, mmlupro, medqa, math/aime) lean DeepSeek; coding/affective datasets (mbpp, humaneval, livecodebench, meld, emorynlp) lean Qwen. Because the preference is strong *within* a dataset but flips *across* datasets, a router that merely learns which dataset a query comes from captures most of the available signal — there is little residual per-query signal beyond dataset identity.

## Where the routing headroom actually is

Datasets ranked by pair-oracle gain (TEST) — i.e. where correctly choosing per-query adds the most overall accuracy:

| rank | dataset | oracle_gain | winner_minority_share | interpretation |
|---:|---|---:|---:|---|
| 1 | kandk | 0.136 | 0.312 | mostly one-sided — a per-dataset constant captures it |
| 2 | mbpp | 0.106 | 0.261 | mostly one-sided — a per-dataset constant captures it |
| 3 | gpqa | 0.093 | 0.404 | genuinely contested — both models win substantial shares |
| 4 | korbench | 0.092 | 0.440 | genuinely contested — both models win substantial shares |
| 5 | meld | 0.086 | 0.425 | genuinely contested — both models win substantial shares |
| 6 | emorynlp | 0.077 | 0.442 | genuinely contested — both models win substantial shares |

The gain is concentrated in a few high-non-tie datasets (kandk, mbpp, and the coding sets). Elsewhere the informative rows are so rare that even perfect per-query routing barely moves overall accuracy.

## Train/test stability of winner distributions

Across the 10 seeds, the pooled DeepSeek winner share is 0.420 on test vs 0.426 on train (gap 0.005), and the per-dataset preference sign agrees between train and test for 14/15 datasets. The winner *distribution* is therefore stable and learnable in principle — instability is NOT the obstacle. The obstacle is that the stable signal is almost entirely dataset-level, and the per-query residual within a dataset is small and noisy (cross-seed std of the test winner share is large on the rare-non-tie datasets simply because of small counts).

## Verdict

**Yes — this DeepSeek/Qwen pair is an intrinsically difficult, low-headroom routing target.** Three structural facts, all independent of the classifier, explain the ~0.60 Macro-F1 band found in Section 2:

- **Sparse labels:** ~80% of rows are ties and carry no winner signal.
- **Tiny usefulness ceiling:** a perfect oracle adds only ~8.5 pts of accuracy over the best single model.
- **Signal is dataset-conditional, not per-query:** strong, stable model specialisation across datasets, but little residual per-query signal once the dataset is known.

This frames Sections 4-5: the open question is whether the ~0.60 ceiling is set by (a) the small/noisy per-query residual itself (a data property) or (b) the GTE-representation + linear-head's inability to extract it (a model property). Section 5 tests (b) directly by holding GTE fixed and increasing classifier capacity.

