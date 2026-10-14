# LLMRouterBench Baseline Experiments

This directory contains my baseline training and evaluation work built on top of LLMRouterBench.

## Purpose

The goal of this work is to gain hands-on experience with LLM routing before developing the proposed edge/split/remote routing method.

The current experiments focus on:

1. Understanding routing opportunity between candidate models.
2. Building a simple TF-IDF + logistic-regression routing baseline.
3. Reproducing the AvengersPro semantic-clustering baseline.
4. Comparing the methods under controlled model-pool and train/test settings.
5. Measuring robustness across multiple random seeds.

## Benchmark

The experiments use the LLMRouterBench small-model setting.

The benchmark data itself is not included in this repository.

## Main Experiments

### 1. Pair headroom analysis

`pair_headroom.py`

Measures how much query-level routing opportunity exists between pairs of models.

For the learning baseline, the selected pair is:

- DeepSeek-R1-0528-Qwen3-8B
- Qwen3-8B

The pair was selected because both are strong models while still showing meaningful complementary performance.

### 2. TF-IDF + Logistic Regression

`train_tfidf_logreg.py`

A deliberately simple lexical routing baseline.

It uses TF-IDF word/bigram features and logistic regression to predict which of the two candidate models is more suitable for a query.

`repeat_tfidf_logreg.py` evaluates this baseline over multiple random splits.

### 3. AvengersPro reproduction

The official AvengersPro implementation from LLMRouterBench was reproduced using:

- `Alibaba-NLP/gte-Qwen2-7B-instruct`
- 3584-dimensional query embeddings
- K-means clustering
- 30 clusters

The original 20-model experiment achieved:

- Dataset-Avg: 71.44%
- Sample-Avg: 70.08%

The best overall fixed model was Qwen3-8B.

### 4. Controlled two-model comparison

To make the comparison more controlled, AvengersPro was restricted to the same two models used by the TF-IDF baseline:

- DeepSeek-R1-0528-Qwen3-8B
- Qwen3-8B

On the seed-42 AvengersPro train/test split:

| Method | Sample-Avg | Dataset-Avg |
|---|---:|---:|
| TF-IDF + Logistic Regression | 67.38% | 67.96% |
| AvengersPro | 67.44% | 67.78% |
| Pair Oracle | 75.48% | 75.98% |

The two learned baselines are therefore very close on this controlled split, while a substantial gap remains to the pair oracle.

This remaining gap motivates further investigation into better query-conditioned routing.

### 5. AvengersPro multi-seed evaluation

AvengersPro was evaluated over seeds 42-51 using the two-model setting.

Summary:

- Mean Sample-Avg: 68.30%
- Mean Dataset-Avg: 69.53%
- Mean Dataset-Avg gain over the best fixed model: +0.46 percentage points

See:

- `results/avengerspro_pair_multiseed_results.csv`
- `results/avengerspro_pair_multiseed_summary.csv`

## Important Notes

The initial TF-IDF multi-seed experiment and the AvengersPro multi-seed experiment do not yet use identical generated splits for every seed, so they should not be treated as a final direct algorithm comparison.

The seed-42 controlled experiment does use the same AvengersPro train/test files for the two-model comparison.

The benchmark cost fields are incomplete for several models, so cost results should be interpreted cautiously.

## Embedding Cache

`run_avengerspro_cached.py` adds persistent local caching of query embeddings.

This does not modify the AvengersPro routing algorithm. It only prevents identical deterministic embeddings from being requested repeatedly from the hosted embedding endpoint.

The local cache database is excluded from Git.

## Key Result Files

- `results/baseline_overview.csv`
- `results/pair_headroom.csv`
- `results/tfidf_exact_avengers_split_metrics.csv`
- `results/avengerspro_overall_metrics.csv`
- `results/avengerspro_pair_multiseed_results.csv`
- `results/avengerspro_pair_multiseed_summary.csv`

## Current Status

The baseline pipeline is operational.

The next step is to use these baseline observations to inform the design of the proposed routing method rather than simply adding increasingly complex baseline models.
