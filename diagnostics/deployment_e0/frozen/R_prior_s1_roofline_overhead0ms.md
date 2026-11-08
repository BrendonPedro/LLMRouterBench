# E0-R prior from S1 per-query data + spec-sheet roofline (informational; NOT a result)

Spec-sheet roofline at 0.7 bandwidth / 0.5 compute efficiency for every action (edge RTX 3090 (capped), server H200); SPLIT link = RTT per token plus 0 ms runtime/serialisation overhead per token; predicted U-max; desk-check eligibility; quality and output lengths are the measured S1 per-query values. Good state only. With 0 ms overhead the prior favours SPLIT (no serialisation, no HF-eager overhead); the same efficiency for the edge competitors favours neither. Decides nothing.

Summary: 5/60 evaluated cells would pass; best cell Δ = +13.0 pt at M=12, no-raw-text, D=60 s.
# Cells (good state)

| M | privacy | state | RTT | D s | SPLIT best (U) | best competitor (U) | Δ pt | 95% CI | pass | Bonferroni lo |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | no-raw-text | good | native | 30 | SPLIT|nonthinking (0.497) | EDGE:Qwen3-8B-AWQ|nonthinking (0.425) | +7.3 | [+1.7, +13.0] | PASS | -2.5 |
| 12 | no-raw-text | good | native | 60 | SPLIT|nonthinking (0.570) | EDGE:Qwen3-8B-AWQ|thinking (0.440) | +13.0 | [+7.3, +17.5] | PASS | +3.7 |
| 12 | no-raw-text | good | native | 120 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-8B-AWQ|thinking (0.557) | +2.2 | [-3.2, +7.8] | - | -7.3 |
| 12 | no-raw-text | good | native | 300 | SPLIT|thinking (0.670) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | +4.0 | [-0.7, +8.8] | - | -4.0 |
| 12 | no-raw-text | good | native | 600 | SPLIT|thinking (0.748) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | +11.8 | [+7.3, +16.5] | PASS | +4.5 |
| 12 | no-raw-text | good | 20 | 30 | SPLIT|nonthinking (0.405) | EDGE:Qwen3-8B-AWQ|nonthinking (0.425) | -2.0 | [-7.7, +3.8] | - | -12.0 |
| 12 | no-raw-text | good | 20 | 60 | SPLIT|nonthinking (0.527) | EDGE:Qwen3-8B-AWQ|thinking (0.440) | +8.7 | [+3.0, +13.3] | PASS | +0.0 |
| 12 | no-raw-text | good | 20 | 120 | SPLIT|nonthinking (0.575) | EDGE:Qwen3-8B-AWQ|thinking (0.557) | +1.7 | [-3.7, +7.3] | - | -8.0 |
| 12 | no-raw-text | good | 20 | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | -5.0 | [-10.3, +0.8] | - | -13.5 |
| 12 | no-raw-text | good | 20 | 600 | SPLIT|thinking (0.700) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | +7.0 | [+2.5, +11.7] | PASS | -0.8 |
| 12 | open | good | native | 30 | SPLIT|nonthinking (0.497) | REMOTE:Qwen3-32B|nonthinking (0.550) | -5.3 | [-7.8, -3.0] | - | -9.8 |
| 12 | open | good | native | 60 | SPLIT|nonthinking (0.570) | REMOTE:Qwen3-32B|nonthinking (0.580) | -1.0 | [-2.0, -0.2] | - | -3.0 |
| 12 | open | good | native | 120 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|nonthinking (0.580) | +0.0 | [-3.5, +0.0] | - | -7.3 |
| 12 | open | good | native | 300 | SPLIT|thinking (0.670) | REMOTE:Qwen3-32B|thinking (0.733) | -6.2 | [-9.0, -3.7] | - | -11.5 |
| 12 | open | good | native | 600 | SPLIT|thinking (0.748) | REMOTE:Qwen3-32B|thinking (0.748) | +0.0 | [+0.0, +0.0] | - | +0.0 |
| 12 | open | good | 20 | 30 | SPLIT|nonthinking (0.405) | REMOTE:Qwen3-32B|nonthinking (0.550) | -14.5 | [-18.5, -10.7] | - | -21.5 |
| 12 | open | good | 20 | 60 | SPLIT|nonthinking (0.527) | REMOTE:Qwen3-32B|nonthinking (0.580) | -5.2 | [-8.0, -2.8] | - | -10.2 |
| 12 | open | good | 20 | 120 | SPLIT|nonthinking (0.575) | REMOTE:Qwen3-32B|nonthinking (0.580) | -0.5 | [-4.0, +0.0] | - | -8.0 |
| 12 | open | good | 20 | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-19.8, -10.0] | - | -22.5 |
| 12 | open | good | 20 | 600 | SPLIT|thinking (0.700) | REMOTE:Qwen3-32B|thinking (0.748) | -4.8 | [-7.3, -2.5] | - | -9.3 |
| 16 | no-raw-text | good | native | 30 | SPLIT|nonthinking (0.477) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -5.8 | [-11.5, +0.0] | - | -16.0 |
| 16 | no-raw-text | good | native | 60 | SPLIT|nonthinking (0.560) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | +2.0 | [-3.5, +7.5] | - | -7.5 |
| 16 | no-raw-text | good | native | 120 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | +1.5 | [-4.2, +6.0] | - | -8.2 |
| 16 | no-raw-text | good | native | 300 | SPLIT|thinking (0.635) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -8.0 | [-13.2, -3.0] | - | -17.2 |
| 16 | no-raw-text | good | native | 600 | SPLIT|thinking (0.748) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | +3.3 | [-0.5, +7.0] | - | -3.0 |
| 16 | no-raw-text | good | 20 | 30 | SPLIT|nonthinking (0.360) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -17.5 | [-23.5, -11.8] | - | -27.7 |
| 16 | no-raw-text | good | 20 | 60 | SPLIT|nonthinking (0.507) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | -3.3 | [-8.8, +2.5] | - | -13.0 |
| 16 | no-raw-text | good | 20 | 120 | SPLIT|nonthinking (0.575) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | +1.0 | [-4.5, +5.5] | - | -8.7 |
| 16 | no-raw-text | good | 20 | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -13.5 | [-19.2, -7.8] | - | -23.2 |
| 16 | no-raw-text | good | 20 | 600 | SPLIT|thinking (0.690) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -2.5 | [-7.0, +1.8] | - | -10.3 |
| 16 | open | good | native | 30 | SPLIT|nonthinking (0.477) | REMOTE:Qwen3-32B|nonthinking (0.550) | -7.3 | [-11.8, -4.7] | - | -16.0 |
| 16 | open | good | native | 60 | SPLIT|nonthinking (0.560) | REMOTE:Qwen3-32B|nonthinking (0.580) | -2.0 | [-4.2, -0.5] | - | -7.5 |
| 16 | open | good | native | 120 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|nonthinking (0.580) | +0.0 | [-4.2, +0.0] | - | -8.2 |
| 16 | open | good | native | 300 | SPLIT|thinking (0.635) | REMOTE:Qwen3-32B|thinking (0.733) | -9.8 | [-14.0, -6.2] | - | -17.2 |
| 16 | open | good | native | 600 | SPLIT|thinking (0.748) | REMOTE:Qwen3-32B|thinking (0.748) | +0.0 | [-0.5, +0.0] | - | -3.0 |
| 16 | open | good | 20 | 30 | SPLIT|nonthinking (0.360) | REMOTE:Qwen3-32B|nonthinking (0.550) | -19.0 | [-24.0, -15.3] | - | -27.7 |
| 16 | open | good | 20 | 60 | SPLIT|nonthinking (0.507) | REMOTE:Qwen3-32B|nonthinking (0.580) | -7.3 | [-10.5, -4.5] | - | -13.5 |
| 16 | open | good | 20 | 120 | SPLIT|nonthinking (0.575) | REMOTE:Qwen3-32B|nonthinking (0.580) | -0.5 | [-4.7, +0.0] | - | -8.7 |
| 16 | open | good | 20 | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-20.5, -10.3] | - | -23.7 |
| 16 | open | good | 20 | 600 | SPLIT|thinking (0.690) | REMOTE:Qwen3-32B|thinking (0.748) | -5.8 | [-8.8, -3.2] | - | -11.3 |
| 24 | no-raw-text | good | native | 30 | SPLIT|nonthinking (0.435) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -10.0 | [-15.5, -4.5] | - | -19.5 |
| 24 | no-raw-text | good | native | 60 | SPLIT|nonthinking (0.542) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | +0.2 | [-5.3, +5.0] | - | -9.5 |
| 24 | no-raw-text | good | native | 120 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | +1.5 | [-4.2, +5.8] | - | -8.2 |
| 24 | no-raw-text | good | native | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -13.5 | [-17.8, -7.7] | - | -21.5 |
| 24 | no-raw-text | good | native | 600 | SPLIT|thinking (0.718) | EDGE:Qwen3-32B-AWQ|thinking (0.723) | -0.5 | [-4.7, +2.5] | - | -7.0 |
| 24 | no-raw-text | good | 20 | 30 | SPLIT|nonthinking (0.282) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -25.3 | [-31.5, -19.2] | - | -36.5 |
| 24 | no-raw-text | good | 20 | 60 | SPLIT|nonthinking (0.482) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | -5.8 | [-11.5, -1.0] | - | -16.0 |
| 24 | no-raw-text | good | 20 | 120 | SPLIT|nonthinking (0.562) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | -0.2 | [-5.8, +3.8] | - | -9.7 |
| 24 | no-raw-text | good | 20 | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -13.5 | [-19.5, -7.8] | - | -23.2 |
| 24 | no-raw-text | good | 20 | 600 | SPLIT|thinking (0.645) | EDGE:Qwen3-32B-AWQ|thinking (0.723) | -7.8 | [-12.8, -3.8] | - | -16.5 |
| 24 | open | good | native | 30 | SPLIT|nonthinking (0.435) | REMOTE:Qwen3-32B|nonthinking (0.550) | -11.5 | [-16.0, -8.5] | - | -19.5 |
| 24 | open | good | native | 60 | SPLIT|nonthinking (0.542) | REMOTE:Qwen3-32B|nonthinking (0.580) | -3.7 | [-6.5, -1.7] | - | -9.5 |
| 24 | open | good | native | 120 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|nonthinking (0.580) | +0.0 | [-4.2, +0.0] | - | -8.2 |
| 24 | open | good | native | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-19.0, -10.3] | - | -22.2 |
| 24 | open | good | native | 600 | SPLIT|thinking (0.718) | REMOTE:Qwen3-32B|thinking (0.748) | -3.0 | [-5.0, -1.5] | - | -7.0 |
| 24 | open | good | 20 | 30 | SPLIT|nonthinking (0.282) | REMOTE:Qwen3-32B|nonthinking (0.550) | -26.8 | [-32.5, -22.5] | - | -36.9 |
| 24 | open | good | 20 | 60 | SPLIT|nonthinking (0.482) | REMOTE:Qwen3-32B|nonthinking (0.580) | -9.7 | [-13.3, -6.5] | - | -16.7 |
| 24 | open | good | 20 | 120 | SPLIT|nonthinking (0.562) | REMOTE:Qwen3-32B|nonthinking (0.580) | -1.7 | [-5.8, -0.7] | - | -9.7 |
| 24 | open | good | 20 | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-20.7, -10.3] | - | -24.2 |
| 24 | open | good | 20 | 600 | SPLIT|thinking (0.645) | REMOTE:Qwen3-32B|thinking (0.748) | -10.3 | [-14.2, -6.8] | - | -17.0 |
