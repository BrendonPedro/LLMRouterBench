# Comparable Macro-F1 Reference Ladder (Section 2)

Router Performance Diagnostic Mission, Priority 1.

**Question answered:** *Is ~0.60 Macro-F1 actually poor relative to established routing methods on this EXACT DeepSeek/Qwen pair, split, and non-tie winner-prediction definition?*

All methods are scored on the **identical** non-tie test instances from the canonical audited split files (`small_models_seed_<seed>/seed<seed>_split0.7/`), across all 10 seeds (42-51), with Macro-F1 = `f1_score(target, pred, average='macro')` where target in {0=DeepSeek, 1=Qwen} on rows where exactly one model is correct. Downstream Sample-Avg accuracy is deliberately NOT used as the router metric.

## Ladder (10-seed mean, ascending Macro-F1)

| method | n seeds | Macro-F1 mean | std | min | max | balanced acc | non-tie acc | pred Qwen frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Best-Single const (Qwen) | 10 | 0.3668 | 0.0069 | 0.3549 | 0.3800 | 0.5000 | 0.5795 | 1.000 |
| Random (train-prior) | 10 | 0.5019 | 0.0171 | 0.4681 | 0.5244 | 0.5022 | 0.5139 | 0.575 |
| Random (uniform) | 10 | 0.5054 | 0.0163 | 0.4721 | 0.5339 | 0.5092 | 0.5087 | 0.499 |
| TF-IDF + LogReg | 10 | 0.5878 | 0.0165 | 0.5651 | 0.6134 | 0.5910 | 0.6193 | 0.696 |
| AvengersPro (pair) | 10 | 0.5985 | 0.0225 | 0.5626 | 0.6322 | 0.5997 | 0.6230 | 0.664 |
| GTE direct | 10 | 0.6032 | 0.0167 | 0.5793 | 0.6305 | 0.6050 | 0.6320 | 0.689 |
| GTE threshold | 10 | 0.6104 | 0.0145 | 0.5849 | 0.6278 | 0.6154 | 0.6173 | 0.536 |

## What the ladder shows

- **Degenerate floor.** A constant Best-Single predictor scores Macro-F1 ~0.367 and a uniform random router ~0.505. The constant predictor never names the minority class, so one per-class F1 is 0 and the macro average collapses to ~0.33 even though its *accuracy* looks reasonable. This is the real baseline the learned routers must beat, and they do.
- **TF-IDF + Logistic Regression** (lexical text router) reaches Macro-F1 ~0.588 on the identical split.
- **AvengersPro** (the established embedding + K-means semantic router, restricted to this pair) reaches Macro-F1 ~0.599.
- **GTE direct** ~0.603 and **GTE threshold** ~0.610 sit at the top of the reproducible ladder.

**Interpretation.** On this exact pairwise winner-prediction task, ~0.60 is NOT poor relative to the established routers that can be run on the same split: the GTE selectors are the *strongest* reproducible methods here, and every learned router — lexical, semantic-cluster, or embedding-logistic — lands in a narrow ~0.59-0.61 band well above the ~0.33 degenerate floor but far below a perfect 1.0. The ceiling behaves like a property of the task, not of one classifier. Sections 3-5 test that hypothesis directly (pair learnability, and whether a more expressive classifier on the same GTE features can break out of the band).

## Methods that could NOT be reproduced fairly

Per the mission, results are not invented for methods that cannot be run on this exact pair/split. The following established benchmark routers ship **no saved predictions on the controlled DeepSeek/Qwen pair and 10-seed 0.7 split**, and reproducing them fairly would require a full retrain (explicitly out of scope for this evidence-first diagnostic):

- **RouterDC, GraphRouter, EmbedLLM, MODEL-SAT** — `baselines/<name>/` contains no seed/prediction artifacts for this pair; their training pipelines target the full multi-model pool.
- **RouteLLM** — the only saved artifacts (`baselines/RouteLLM/results/mf_results_seed*.json`) are matrix-factorization selection results on a DIFFERENT model pool and a DIFFERENT dataset set (arc-agi, hle, swe-bench, tau2, arenahard, simpleqa, ...) with seeds {42, 999, 2024, 2025, 3407}. They are not on the controlled 15-dataset DeepSeek/Qwen pair, so scoring them here would be an apples-to-oranges comparison and is omitted.
- **FrugalGPT / Best-route-llm** — cascade / cost-routing methods whose objective is cost-quality trade-off, not pairwise winner classification; no comparable per-query winner predictions exist for this pair.

The reproducible ladder above is therefore the fair answer to the Section 2 question. AvengersPro is the one bona-fide *established* router that could be scored on the exact pair, and it does not beat the GTE selectors.

Per-seed numbers for every method: `official_router_f1_comparison.csv`.
