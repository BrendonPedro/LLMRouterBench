# S1 quality gate: CONTINUE

Generated 2026-11-08T06:57:41.754603+00:00 from 4000 scored generations. Eligibility: E0 §1 desk check (default). Threshold: gap < 3.0 pt (point estimate).

**Decision.** 32B-bf16 is >= 3.0 pt better than EDGE-best(16) on at least one benchmark in at least one mode: the 16/24 GB split regimes stay. Proceed with the full E0 (G1 passes).

rule16 (all gaps at 16 GB < 3.0): False; rule12 (rule16 and all gaps at 12 GB < 3.0): False. Informational, not a pre-registered rule: all gaps at 24 GB < 3.0: False.

## Gap: Qwen3-32B bf16 minus EDGE-best(M), percentage points (within mode)

| benchmark | mode | 32B acc | 12 GB: best, gap [95% CI] | 16 GB: best, gap [95% CI] | 24 GB: best, gap [95% CI] |
|---|---|---|---|---|---|
| mmlupro | thinking | 77.5 | Qwen3-8B-AWQ, **+15.8** [+6.7, +25.0] | Qwen3-14B-AWQ, **+5.8** [-0.8, +12.5] | Qwen3-32B-AWQ, **+2.5** [-2.5, +8.3] |
| mmlupro | nonthinking | 68.3 | Qwen3-8B-AWQ, **+27.5** [+16.7, +38.3] | Qwen3-14B-AWQ, **+5.0** [-4.2, +15.0] | Qwen3-14B-AWQ, **+5.0** [-5.0, +15.0] |
| gpqa | thinking | 69.2 | Qwen3-8B-AWQ, **+10.0** [+0.8, +19.2] | Qwen3-14B-AWQ, **+5.0** [-1.7, +12.5] | Qwen3-14B-AWQ, **+5.0** [-2.5, +12.5] |
| gpqa | nonthinking | 53.3 | Qwen3-8B-AWQ, **+15.0** [+4.2, +25.8] | Qwen3-14B-AWQ, **+7.5** [-5.0, +20.0] | Qwen3-32B-AWQ, **+5.8** [-3.3, +15.0] |
| livemathbench | thinking | 76.2 | Qwen3-8B-AWQ, **+2.5** [-2.5, +8.8] | Qwen3-14B-AWQ, **-2.5** [-8.8, +5.0] | Qwen3-32B-AWQ, **-6.2** [-12.5, -1.2] |
| livemathbench | nonthinking | 65.0 | Qwen3-8B-AWQ, **+3.8** [-5.0, +12.5] | Qwen3-8B-AWQ, **+3.8** [-5.0, +12.5] | Qwen3-8B-AWQ, **+3.8** [-5.0, +12.5] |
| livecodebench | thinking | 77.5 | Qwen3-8B-AWQ, **+17.5** [+8.8, +28.7] | Qwen3-14B-AWQ, **+2.5** [-6.2, +12.5] | Qwen3-14B-AWQ, **+2.5** [-7.5, +12.5] |
| livecodebench | nonthinking | 42.5 | Qwen3-8B-AWQ, **+10.0** [+0.0, +21.2] | Qwen3-14B-AWQ, **-2.5** [-13.8, +8.8] | Qwen3-14B-AWQ, **-2.5** [-13.8, +8.8] |

## Accuracy (%), mean over queries and seeds

| benchmark | mode | Qwen3-32B | Qwen3-8B-AWQ | Qwen3-14B-AWQ | Qwen3-32B-AWQ | Qwen3-8B |
|---|---|---|---|---|---|---|
| mmlupro | thinking | 77.5 | 61.7 | 71.7 | 75.0 | 65.8 |
| mmlupro | nonthinking | 68.3 | 40.8 | 63.3 | 62.5 | 51.7 |
| gpqa | thinking | 69.2 | 59.2 | 64.2 | 64.2 | 60.8 |
| gpqa | nonthinking | 53.3 | 38.3 | 45.8 | 47.5 | 37.5 |
| livemathbench | thinking | 76.2 | 73.8 | 78.8 | 82.5 | 77.5 |
| livemathbench | nonthinking | 65.0 | 61.2 | 61.2 | 60.0 | 57.5 |
| livecodebench | thinking | 77.5 | 60.0 | 75.0 | 70.0 | 62.5 |
| livecodebench | nonthinking | 42.5 | 32.5 | 45.0 | 41.2 | 38.8 |

## Truncation rate at max_new_tokens=16384 and mean completion tokens

| model | mode | mmlupro | gpqa | livemathbench | livecodebench |
|---|---|---|---|---|---|
| Qwen3-32B | thinking | 0.02 / 2550 | 0.01 / 4934 | 0.12 / 6106 | 0.16 / 8922 |
| Qwen3-32B | nonthinking | 0.01 / 593 | 0.03 / 1274 | 0.04 / 1699 | 0.00 / 268 |
| Qwen3-8B-AWQ | thinking | 0.03 / 3428 | 0.03 / 6542 | 0.11 / 7278 | 0.28 / 9584 |
| Qwen3-8B-AWQ | nonthinking | 0.00 / 649 | 0.01 / 1278 | 0.04 / 1833 | 0.00 / 319 |
| Qwen3-14B-AWQ | thinking | 0.00 / 2452 | 0.01 / 4661 | 0.05 / 5919 | 0.15 / 8694 |
| Qwen3-14B-AWQ | nonthinking | 0.01 / 609 | 0.01 / 930 | 0.00 / 985 | 0.00 / 305 |
| Qwen3-32B-AWQ | thinking | 0.00 / 2057 | 0.00 / 4210 | 0.07 / 5752 | 0.21 / 8860 |
| Qwen3-32B-AWQ | nonthinking | 0.00 / 442 | 0.01 / 1098 | 0.00 / 1046 | 0.00 / 258 |
| Qwen3-8B | thinking | 0.03 / 3659 | 0.04 / 7006 | 0.11 / 7217 | 0.25 / 9464 |
| Qwen3-8B | nonthinking | 0.01 / 684 | 0.07 / 2196 | 0.04 / 1940 | 0.01 / 493 |

Template sanity: thinking outputs without `<think>`: 0; non-thinking outputs with `<think>`: 0.

## Coverage

Expected scored generations per (model, mode): 400.

- Qwen3-32B|thinking: 400
- Qwen3-32B|nonthinking: 400
- Qwen3-8B-AWQ|thinking: 400
- Qwen3-8B-AWQ|nonthinking: 400
- Qwen3-14B-AWQ|thinking: 400
- Qwen3-14B-AWQ|nonthinking: 400
- Qwen3-32B-AWQ|thinking: 400
- Qwen3-32B-AWQ|nonthinking: 400
- Qwen3-8B|thinking: 400
- Qwen3-8B|nonthinking: 400

## Eligibility per memory class

- 12 GB: Qwen3-8B-AWQ
- 16 GB: Qwen3-8B-AWQ, Qwen3-14B-AWQ
- 24 GB: Qwen3-8B-AWQ, Qwen3-14B-AWQ, Qwen3-8B, Qwen3-32B-AWQ

Notes: EDGE-best is chosen per benchmark and mode on the same queries it is scored on (a small selection effect that favours the edge, i.e. the early stop). The 24 GB class counts Qwen3-32B-AWQ under the E0 FP8-KV assumption; S1 serves it with a bf16/fp16 KV cache. The CI is a paired bootstrap over queries against the selected EDGE-best, descriptive only.
