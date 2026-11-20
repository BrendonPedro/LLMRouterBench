# Post hoc screen of the laboratory pair (exploratory; not a result)

Written 2026-11-20 from archived records, before the QWEN-SELECT tag. It combines S1 quality scores and output lengths
(sampled decoding, another engine, another machine) with K16-HWFIT timings (greedy, 256 forced tokens, ten prompts).
Times of smaller models on the edge are assumed, never measured. Nothing here validates anything.

## A. Same model: split against the whole 32B model on the edge

| D (s) | U split, non-thinking | U edge, non-thinking | U split, thinking | U edge, thinking | Delta (pt) | 95% CI | rule |
|---|---|---|---|---|---|---|---|
| 30 | 9.00 | 3.25 | 0.00 | 0.00 | +5.75 | [+3.25, +8.75] | SPLIT |
| 60 | 18.25 | 13.75 | 0.00 | 0.00 | +4.50 | [+2.25, +7.00] | SPLIT |
| 120 | 32.75 | 25.25 | 0.75 | 0.00 | +7.50 | [+4.75, +10.50] | SPLIT |
| 300 | 53.25 | 48.25 | 20.50 | 13.75 | +5.00 | [+2.75, +7.75] | SPLIT |
| 600 | 57.50 | 56.00 | 37.75 | 29.25 | +1.50 | [+0.25, +3.25] | edge |

Reading: with the K16 medians the split's speed advantage per token turns into a utility advantage over the same model
on the edge at 30 to 300 s. Thinking mode is never the split's better mode at these speeds.

## B. Smaller bf16 models on the edge, time assumed = unsplit 32B time x weight-bytes ratio

Ratios: 14B 0.45, 8B 0.25. Quality of the 14B model: its 4-bit build's S1 scores (a proxy).

| Edge competitors | D=30 | D=60 | D=120 | D=300 | D=600 | cells selecting SPLIT |
|---|---|---|---|---|---|---|
| 32B and 8B | -7.75 (8B) | -15.00 (8B) | -9.50 (8B) | +5.00 (32B) | +1.50 (32B) | [300] |
| 32B, 14B and 8B | -7.75 (8B) | -15.00 (8B) | -9.50 (8B) | -0.25 (14B) | +1.50 (32B) | [] |

Each entry is Delta in points against the best competitor (named in brackets).

Break-even: with the 8B model at its ratio, the split is first selected in some cell when the 14B model's time ratio reaches **0.85** (central assumption 0.45). Below that, no cell selects the split.

## C. The cells of the QWEN-SELECT design (prediction recorded before the run)

Non-thinking mode, MMLU-Pro, GPQA and LiveMathBench of the S1 sample (160 queries, two sampled seeds), output cap and 300 s cutoff of the
protocol. The experiment itself uses other queries, greedy decoding and measured times.

| D (s) | U S | U E32 | U E14 | U E8 | U R14 | no-raw-text: Delta, CI, rule | open: Delta, CI, rule | same model: Delta, CI, rule |
|---|---|---|---|---|---|---|---|---|
| 30 | 4.06 | 0.94 | 6.88 | 11.56 | 27.81 | -7.50 [-12.19, -3.12] E8 | -23.75 [-30.00, -17.81] R14 | +3.12 [+1.25, +5.31] SPLIT |
| 60 | 13.44 | 9.38 | 22.50 | 31.87 | 44.38 | -18.44 [-25.00, -11.87] E8 | -30.94 [-38.12, -24.06] R14 | +4.06 [+1.88, +6.56] SPLIT |
| 120 | 30.63 | 21.25 | 40.94 | 43.12 | 55.62 | -12.50 [-19.06, -7.19] E8 | -25.00 [-31.87, -18.44] R14 | +9.38 [+5.94, +13.13] SPLIT |
| 300 | 55.62 | 49.69 | 55.62 | 47.50 | 56.25 | +0.00 [-6.56, +5.94] E14 | -0.62 [-7.19, +5.62] R14 | +5.94 [+3.12, +9.06] SPLIT |

Predicted outcome under these assumptions: no cell selects the split against the best edge configuration, and the split beats the
same model on the edge in several cells. Both statements are at risk in the experiment.

## D. Planning arithmetic for the GPU budget

| Block | expected hours for 160 queries | share of queries cut off at 300 s |
|---|---|---|
| S | 7.1 | 0.17 |
| E32 | 8.4 | 0.30 |
| E14 | 4.5 | 0.04 |
| E8 | 3.5 | 0.00 |
| R14 | 2.0 | 0.00 |

The split block holds two GPUs. These figures rest on sampled output lengths; greedy lengths may differ.

## E. Sensitivity of the rule at 160 queries, one pass (resampled S1 queries; planning only)

| Competitors | D=30 | D=60 | D=120 | D=300 |
|---|---|---|---|---|
| E32 only (same model) | 0.58 | 0.82 | 1.00 | 0.98 |
| E32 and E8 | 0.00 | 0.00 | 0.00 | 0.45 |
| E32, E14 and E8 | 0.00 | 0.00 | 0.00 | 0.03 |

Each entry is the share of resampled workloads in which the rule selects the split.

