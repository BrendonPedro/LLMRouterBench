# E0-R prior from S1 per-query data + spec-sheet roofline (informational; NOT a result)

Spec-sheet roofline at 0.7 bandwidth / 0.5 compute efficiency for every action (edge RTX 3090 (capped), server H200); SPLIT link = RTT per token plus 20 ms runtime/serialisation overhead per token; predicted U-max; desk-check eligibility; quality and output lengths are the measured S1 per-query values. Good state only. With 0 ms overhead the prior favours SPLIT (no serialisation, no HF-eager overhead); the same efficiency for the edge competitors favours neither. Decides nothing.

Summary: 2/60 evaluated cells would pass; best cell Δ = +8.5 pt at M=12, no-raw-text, D=60 s.
# Cells (good state)

| M | privacy | state | RTT | D s | SPLIT best (U) | best competitor (U) | Δ pt | 95% CI | pass | Bonferroni lo |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | no-raw-text | good | native | 30 | SPLIT|nonthinking (0.400) | EDGE:Qwen3-8B-AWQ|nonthinking (0.425) | -2.5 | [-8.2, +3.3] | - | -12.8 |
| 12 | no-raw-text | good | native | 60 | SPLIT|nonthinking (0.525) | EDGE:Qwen3-8B-AWQ|thinking (0.440) | +8.5 | [+2.8, +13.0] | PASS | -0.3 |
| 12 | no-raw-text | good | native | 120 | SPLIT|nonthinking (0.575) | EDGE:Qwen3-8B-AWQ|thinking (0.557) | +1.7 | [-3.7, +7.3] | - | -8.0 |
| 12 | no-raw-text | good | native | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | -5.0 | [-10.3, +0.8] | - | -14.0 |
| 12 | no-raw-text | good | native | 600 | SPLIT|thinking (0.698) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | +6.8 | [+2.2, +11.5] | PASS | -1.0 |
| 12 | no-raw-text | good | 20 | 30 | SPLIT|nonthinking (0.263) | EDGE:Qwen3-8B-AWQ|nonthinking (0.425) | -16.2 | [-22.3, -10.2] | - | -26.0 |
| 12 | no-raw-text | good | 20 | 60 | SPLIT|nonthinking (0.477) | EDGE:Qwen3-8B-AWQ|thinking (0.440) | +3.7 | [-2.0, +8.3] | - | -5.5 |
| 12 | no-raw-text | good | 20 | 120 | SPLIT|nonthinking (0.560) | EDGE:Qwen3-8B-AWQ|thinking (0.557) | +0.3 | [-5.2, +5.8] | - | -9.3 |
| 12 | no-raw-text | good | 20 | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | -5.0 | [-10.8, +0.8] | - | -14.7 |
| 12 | no-raw-text | good | 20 | 600 | SPLIT|thinking (0.630) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | +0.0 | [-5.0, +5.0] | - | -8.0 |
| 12 | open | good | native | 30 | SPLIT|nonthinking (0.400) | REMOTE:Qwen3-32B|nonthinking (0.550) | -15.0 | [-19.0, -11.0] | - | -21.8 |
| 12 | open | good | native | 60 | SPLIT|nonthinking (0.525) | REMOTE:Qwen3-32B|nonthinking (0.580) | -5.5 | [-8.3, -3.0] | - | -10.5 |
| 12 | open | good | native | 120 | SPLIT|nonthinking (0.575) | REMOTE:Qwen3-32B|nonthinking (0.580) | -0.5 | [-4.0, +0.0] | - | -8.0 |
| 12 | open | good | native | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-19.8, -10.0] | - | -22.7 |
| 12 | open | good | native | 600 | SPLIT|thinking (0.698) | REMOTE:Qwen3-32B|thinking (0.748) | -5.0 | [-7.8, -2.7] | - | -10.0 |
| 12 | open | good | 20 | 30 | SPLIT|nonthinking (0.263) | REMOTE:Qwen3-32B|nonthinking (0.550) | -28.8 | [-34.0, -23.8] | - | -37.2 |
| 12 | open | good | 20 | 60 | SPLIT|nonthinking (0.477) | REMOTE:Qwen3-32B|nonthinking (0.580) | -10.2 | [-13.8, -7.0] | - | -17.0 |
| 12 | open | good | 20 | 120 | SPLIT|nonthinking (0.560) | REMOTE:Qwen3-32B|nonthinking (0.580) | -2.0 | [-5.5, -0.7] | - | -9.3 |
| 12 | open | good | 20 | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-20.5, -10.0] | - | -24.0 |
| 12 | open | good | 20 | 600 | SPLIT|thinking (0.630) | REMOTE:Qwen3-32B|thinking (0.748) | -11.8 | [-15.7, -8.0] | - | -18.7 |
| 16 | no-raw-text | good | native | 30 | SPLIT|nonthinking (0.355) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -18.0 | [-24.0, -12.2] | - | -28.5 |
| 16 | no-raw-text | good | native | 60 | SPLIT|nonthinking (0.507) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | -3.3 | [-8.8, +2.5] | - | -13.0 |
| 16 | no-raw-text | good | native | 120 | SPLIT|nonthinking (0.575) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | +1.0 | [-4.5, +5.5] | - | -8.7 |
| 16 | no-raw-text | good | native | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -13.5 | [-19.2, -7.8] | - | -23.2 |
| 16 | no-raw-text | good | native | 600 | SPLIT|thinking (0.688) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -2.7 | [-7.3, +1.5] | - | -10.5 |
| 16 | no-raw-text | good | 20 | 30 | SPLIT|nonthinking (0.255) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -28.0 | [-34.2, -21.7] | - | -39.3 |
| 16 | no-raw-text | good | 20 | 60 | SPLIT|nonthinking (0.460) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | -8.0 | [-13.7, -2.2] | - | -18.2 |
| 16 | no-raw-text | good | 20 | 120 | SPLIT|nonthinking (0.555) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | -1.0 | [-6.5, +3.5] | - | -10.5 |
| 16 | no-raw-text | good | 20 | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -13.5 | [-19.5, -7.8] | - | -23.2 |
| 16 | no-raw-text | good | 20 | 600 | SPLIT|thinking (0.615) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -10.0 | [-15.0, -5.0] | - | -18.2 |
| 16 | open | good | native | 30 | SPLIT|nonthinking (0.355) | REMOTE:Qwen3-32B|nonthinking (0.550) | -19.5 | [-24.7, -15.8] | - | -28.5 |
| 16 | open | good | native | 60 | SPLIT|nonthinking (0.507) | REMOTE:Qwen3-32B|nonthinking (0.580) | -7.3 | [-10.5, -4.5] | - | -13.5 |
| 16 | open | good | native | 120 | SPLIT|nonthinking (0.575) | REMOTE:Qwen3-32B|nonthinking (0.580) | -0.5 | [-4.7, +0.0] | - | -8.7 |
| 16 | open | good | native | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-20.5, -10.3] | - | -23.7 |
| 16 | open | good | native | 600 | SPLIT|thinking (0.688) | REMOTE:Qwen3-32B|thinking (0.748) | -6.0 | [-9.0, -3.5] | - | -11.5 |
| 16 | open | good | 20 | 30 | SPLIT|nonthinking (0.255) | REMOTE:Qwen3-32B|nonthinking (0.550) | -29.5 | [-35.5, -25.0] | - | -39.9 |
| 16 | open | good | 20 | 60 | SPLIT|nonthinking (0.460) | REMOTE:Qwen3-32B|nonthinking (0.580) | -12.0 | [-15.8, -8.5] | - | -19.2 |
| 16 | open | good | 20 | 120 | SPLIT|nonthinking (0.555) | REMOTE:Qwen3-32B|nonthinking (0.580) | -2.5 | [-6.8, -1.0] | - | -10.5 |
| 16 | open | good | 20 | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-20.7, -10.3] | - | -24.2 |
| 16 | open | good | 20 | 600 | SPLIT|thinking (0.615) | REMOTE:Qwen3-32B|thinking (0.748) | -13.3 | [-17.2, -9.3] | - | -20.0 |
| 24 | no-raw-text | good | native | 30 | SPLIT|nonthinking (0.278) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -25.8 | [-32.0, -19.7] | - | -37.4 |
| 24 | no-raw-text | good | native | 60 | SPLIT|nonthinking (0.480) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | -6.0 | [-11.8, -1.0] | - | -16.2 |
| 24 | no-raw-text | good | native | 120 | SPLIT|nonthinking (0.562) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | -0.2 | [-5.8, +3.8] | - | -9.7 |
| 24 | no-raw-text | good | native | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -13.5 | [-19.5, -7.8] | - | -23.2 |
| 24 | no-raw-text | good | native | 600 | SPLIT|thinking (0.645) | EDGE:Qwen3-32B-AWQ|thinking (0.723) | -7.8 | [-12.8, -3.8] | - | -16.5 |
| 24 | no-raw-text | good | 20 | 30 | SPLIT|nonthinking (0.217) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -31.8 | [-38.0, -25.5] | - | -43.4 |
| 24 | no-raw-text | good | 20 | 60 | SPLIT|nonthinking (0.435) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | -10.5 | [-16.0, -5.7] | - | -20.2 |
| 24 | no-raw-text | good | 20 | 120 | SPLIT|nonthinking (0.550) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | -1.5 | [-7.3, +2.7] | - | -11.2 |
| 24 | no-raw-text | good | 20 | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -13.5 | [-19.5, -7.8] | - | -23.2 |
| 24 | no-raw-text | good | 20 | 600 | SPLIT|thinking (0.585) | EDGE:Qwen3-32B-AWQ|thinking (0.723) | -13.8 | [-18.0, -8.8] | - | -21.3 |
| 24 | open | good | native | 30 | SPLIT|nonthinking (0.278) | REMOTE:Qwen3-32B|nonthinking (0.550) | -27.3 | [-33.0, -22.8] | - | -37.7 |
| 24 | open | good | native | 60 | SPLIT|nonthinking (0.480) | REMOTE:Qwen3-32B|nonthinking (0.580) | -10.0 | [-13.7, -6.8] | - | -17.2 |
| 24 | open | good | native | 120 | SPLIT|nonthinking (0.562) | REMOTE:Qwen3-32B|nonthinking (0.580) | -1.7 | [-5.8, -0.7] | - | -9.7 |
| 24 | open | good | native | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-20.7, -10.3] | - | -24.2 |
| 24 | open | good | native | 600 | SPLIT|thinking (0.645) | REMOTE:Qwen3-32B|thinking (0.748) | -10.3 | [-14.2, -6.8] | - | -17.0 |
| 24 | open | good | 20 | 30 | SPLIT|nonthinking (0.217) | REMOTE:Qwen3-32B|nonthinking (0.550) | -33.2 | [-39.2, -28.5] | - | -43.5 |
| 24 | open | good | 20 | 60 | SPLIT|nonthinking (0.435) | REMOTE:Qwen3-32B|nonthinking (0.580) | -14.5 | [-18.7, -10.8] | - | -21.5 |
| 24 | open | good | 20 | 120 | SPLIT|nonthinking (0.550) | REMOTE:Qwen3-32B|nonthinking (0.580) | -3.0 | [-7.3, -1.5] | - | -11.2 |
| 24 | open | good | 20 | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-20.7, -10.3] | - | -24.2 |
| 24 | open | good | 20 | 600 | SPLIT|thinking (0.585) | REMOTE:Qwen3-32B|thinking (0.748) | -16.3 | [-19.5, -11.3] | - | -23.0 |
