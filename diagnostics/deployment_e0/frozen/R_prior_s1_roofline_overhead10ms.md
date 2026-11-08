# E0-R prior from S1 per-query data + spec-sheet roofline (informational; NOT a result)

Spec-sheet roofline at 0.7 bandwidth / 0.5 compute efficiency for every action (edge RTX 3090 (capped), server H200); SPLIT link = RTT per token plus 10 ms runtime/serialisation overhead per token; predicted U-max; desk-check eligibility; quality and output lengths are the measured S1 per-query values. Good state only. With 0 ms overhead the prior favours SPLIT (no serialisation, no HF-eager overhead); the same efficiency for the edge competitors favours neither. Decides nothing.

Summary: 2/60 evaluated cells would pass; best cell Δ = +11.2 pt at M=12, no-raw-text, D=60 s.
# Cells (good state)

| M | privacy | state | RTT | D s | SPLIT best (U) | best competitor (U) | Δ pt | 95% CI | pass | Bonferroni lo |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | no-raw-text | good | native | 30 | SPLIT|nonthinking (0.448) | EDGE:Qwen3-8B-AWQ|nonthinking (0.425) | +2.3 | [-3.3, +8.0] | - | -7.7 |
| 12 | no-raw-text | good | native | 60 | SPLIT|nonthinking (0.552) | EDGE:Qwen3-8B-AWQ|thinking (0.440) | +11.2 | [+5.5, +16.0] | PASS | +2.0 |
| 12 | no-raw-text | good | native | 120 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-8B-AWQ|thinking (0.557) | +2.2 | [-3.2, +7.8] | - | -7.3 |
| 12 | no-raw-text | good | native | 300 | SPLIT|thinking (0.603) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | -2.7 | [-7.5, +2.5] | - | -10.8 |
| 12 | no-raw-text | good | native | 600 | SPLIT|thinking (0.738) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | +10.8 | [+6.0, +15.5] | PASS | +3.3 |
| 12 | no-raw-text | good | 20 | 30 | SPLIT|nonthinking (0.320) | EDGE:Qwen3-8B-AWQ|nonthinking (0.425) | -10.5 | [-16.2, -4.8] | - | -20.0 |
| 12 | no-raw-text | good | 20 | 60 | SPLIT|nonthinking (0.495) | EDGE:Qwen3-8B-AWQ|thinking (0.440) | +5.5 | [+0.0, +10.0] | - | -3.7 |
| 12 | no-raw-text | good | 20 | 120 | SPLIT|nonthinking (0.570) | EDGE:Qwen3-8B-AWQ|thinking (0.557) | +1.2 | [-4.2, +6.8] | - | -8.5 |
| 12 | no-raw-text | good | 20 | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | -5.0 | [-10.8, +0.8] | - | -14.7 |
| 12 | no-raw-text | good | 20 | 600 | SPLIT|thinking (0.670) | EDGE:Qwen3-8B-AWQ|thinking (0.630) | +4.0 | [-0.7, +8.8] | - | -4.0 |
| 12 | open | good | native | 30 | SPLIT|nonthinking (0.448) | REMOTE:Qwen3-32B|nonthinking (0.550) | -10.3 | [-13.8, -7.0] | - | -16.5 |
| 12 | open | good | native | 60 | SPLIT|nonthinking (0.552) | REMOTE:Qwen3-32B|nonthinking (0.580) | -2.7 | [-5.0, -1.0] | - | -6.7 |
| 12 | open | good | native | 120 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|nonthinking (0.580) | +0.0 | [-3.5, +0.0] | - | -7.3 |
| 12 | open | good | native | 300 | SPLIT|thinking (0.603) | REMOTE:Qwen3-32B|thinking (0.733) | -13.0 | [-16.5, -9.0] | - | -19.9 |
| 12 | open | good | native | 600 | SPLIT|thinking (0.738) | REMOTE:Qwen3-32B|thinking (0.748) | -1.0 | [-2.0, -0.2] | - | -2.8 |
| 12 | open | good | 20 | 30 | SPLIT|nonthinking (0.320) | REMOTE:Qwen3-32B|nonthinking (0.550) | -23.0 | [-27.7, -18.5] | - | -30.5 |
| 12 | open | good | 20 | 60 | SPLIT|nonthinking (0.495) | REMOTE:Qwen3-32B|nonthinking (0.580) | -8.5 | [-11.8, -5.5] | - | -14.8 |
| 12 | open | good | 20 | 120 | SPLIT|nonthinking (0.570) | REMOTE:Qwen3-32B|nonthinking (0.580) | -1.0 | [-4.5, -0.2] | - | -8.5 |
| 12 | open | good | 20 | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-20.5, -10.0] | - | -24.0 |
| 12 | open | good | 20 | 600 | SPLIT|thinking (0.670) | REMOTE:Qwen3-32B|thinking (0.748) | -7.8 | [-11.0, -4.7] | - | -13.7 |
| 16 | no-raw-text | good | native | 30 | SPLIT|nonthinking (0.432) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -10.3 | [-15.8, -4.7] | - | -19.8 |
| 16 | no-raw-text | good | native | 60 | SPLIT|nonthinking (0.542) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | +0.2 | [-5.3, +6.0] | - | -9.5 |
| 16 | no-raw-text | good | native | 120 | SPLIT|nonthinking (0.578) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | +1.3 | [-4.5, +5.8] | - | -8.5 |
| 16 | no-raw-text | good | native | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -13.5 | [-17.8, -7.5] | - | -21.5 |
| 16 | no-raw-text | good | native | 600 | SPLIT|thinking (0.718) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | +0.3 | [-3.8, +4.2] | - | -6.5 |
| 16 | no-raw-text | good | 20 | 30 | SPLIT|nonthinking (0.278) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -25.8 | [-32.0, -19.7] | - | -37.4 |
| 16 | no-raw-text | good | 20 | 60 | SPLIT|nonthinking (0.480) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | -6.0 | [-11.8, -0.2] | - | -16.2 |
| 16 | no-raw-text | good | 20 | 120 | SPLIT|nonthinking (0.562) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | -0.2 | [-5.8, +4.0] | - | -9.7 |
| 16 | no-raw-text | good | 20 | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -13.5 | [-19.5, -7.8] | - | -23.2 |
| 16 | no-raw-text | good | 20 | 600 | SPLIT|thinking (0.645) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -7.0 | [-12.0, -2.2] | - | -16.2 |
| 16 | open | good | native | 30 | SPLIT|nonthinking (0.432) | REMOTE:Qwen3-32B|nonthinking (0.550) | -11.8 | [-16.3, -8.7] | - | -20.0 |
| 16 | open | good | native | 60 | SPLIT|nonthinking (0.542) | REMOTE:Qwen3-32B|nonthinking (0.580) | -3.7 | [-6.5, -1.7] | - | -9.5 |
| 16 | open | good | native | 120 | SPLIT|nonthinking (0.578) | REMOTE:Qwen3-32B|nonthinking (0.580) | -0.2 | [-4.5, +0.0] | - | -8.5 |
| 16 | open | good | native | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-19.0, -10.3] | - | -22.2 |
| 16 | open | good | native | 600 | SPLIT|thinking (0.718) | REMOTE:Qwen3-32B|thinking (0.748) | -3.0 | [-5.0, -1.5] | - | -6.5 |
| 16 | open | good | 20 | 30 | SPLIT|nonthinking (0.278) | REMOTE:Qwen3-32B|nonthinking (0.550) | -27.3 | [-33.0, -22.8] | - | -37.7 |
| 16 | open | good | 20 | 60 | SPLIT|nonthinking (0.480) | REMOTE:Qwen3-32B|nonthinking (0.580) | -10.0 | [-13.7, -6.8] | - | -17.2 |
| 16 | open | good | 20 | 120 | SPLIT|nonthinking (0.562) | REMOTE:Qwen3-32B|nonthinking (0.580) | -1.7 | [-5.8, -0.7] | - | -9.7 |
| 16 | open | good | 20 | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-20.7, -10.3] | - | -24.2 |
| 16 | open | good | 20 | 600 | SPLIT|thinking (0.645) | REMOTE:Qwen3-32B|thinking (0.748) | -10.3 | [-14.0, -6.8] | - | -17.0 |
| 24 | no-raw-text | good | native | 30 | SPLIT|nonthinking (0.357) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -17.8 | [-23.8, -12.0] | - | -28.0 |
| 24 | no-raw-text | good | native | 60 | SPLIT|nonthinking (0.507) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | -3.3 | [-8.8, +1.5] | - | -13.0 |
| 24 | no-raw-text | good | native | 120 | SPLIT|nonthinking (0.575) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | +1.0 | [-4.5, +5.0] | - | -8.7 |
| 24 | no-raw-text | good | native | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -13.5 | [-19.2, -7.8] | - | -23.2 |
| 24 | no-raw-text | good | native | 600 | SPLIT|thinking (0.688) | EDGE:Qwen3-32B-AWQ|thinking (0.723) | -3.5 | [-8.0, -0.2] | - | -11.0 |
| 24 | no-raw-text | good | 20 | 30 | SPLIT|nonthinking (0.255) | EDGE:Qwen3-14B-AWQ|nonthinking (0.535) | -28.0 | [-34.2, -21.7] | - | -39.3 |
| 24 | no-raw-text | good | 20 | 60 | SPLIT|nonthinking (0.460) | EDGE:Qwen3-14B-AWQ|nonthinking (0.540) | -8.0 | [-13.7, -3.0] | - | -18.2 |
| 24 | no-raw-text | good | 20 | 120 | SPLIT|nonthinking (0.555) | EDGE:Qwen3-14B-AWQ|thinking (0.565) | -1.0 | [-6.5, +3.0] | - | -10.5 |
| 24 | no-raw-text | good | 20 | 300 | SPLIT|nonthinking (0.580) | EDGE:Qwen3-14B-AWQ|thinking (0.715) | -13.5 | [-19.5, -7.8] | - | -23.2 |
| 24 | no-raw-text | good | 20 | 600 | SPLIT|thinking (0.615) | EDGE:Qwen3-32B-AWQ|thinking (0.723) | -10.8 | [-16.0, -6.5] | - | -19.2 |
| 24 | open | good | native | 30 | SPLIT|nonthinking (0.357) | REMOTE:Qwen3-32B|nonthinking (0.550) | -19.3 | [-24.3, -15.5] | - | -28.0 |
| 24 | open | good | native | 60 | SPLIT|nonthinking (0.507) | REMOTE:Qwen3-32B|nonthinking (0.580) | -7.3 | [-10.5, -4.5] | - | -13.5 |
| 24 | open | good | native | 120 | SPLIT|nonthinking (0.575) | REMOTE:Qwen3-32B|nonthinking (0.580) | -0.5 | [-4.7, +0.0] | - | -8.7 |
| 24 | open | good | native | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-20.5, -10.3] | - | -23.7 |
| 24 | open | good | native | 600 | SPLIT|thinking (0.688) | REMOTE:Qwen3-32B|thinking (0.748) | -6.0 | [-9.0, -3.5] | - | -11.5 |
| 24 | open | good | 20 | 30 | SPLIT|nonthinking (0.255) | REMOTE:Qwen3-32B|nonthinking (0.550) | -29.5 | [-35.5, -25.0] | - | -39.9 |
| 24 | open | good | 20 | 60 | SPLIT|nonthinking (0.460) | REMOTE:Qwen3-32B|nonthinking (0.580) | -12.0 | [-15.8, -8.5] | - | -19.2 |
| 24 | open | good | 20 | 120 | SPLIT|nonthinking (0.555) | REMOTE:Qwen3-32B|nonthinking (0.580) | -2.5 | [-6.8, -1.0] | - | -10.5 |
| 24 | open | good | 20 | 300 | SPLIT|nonthinking (0.580) | REMOTE:Qwen3-32B|thinking (0.733) | -15.3 | [-20.7, -10.3] | - | -24.2 |
| 24 | open | good | 20 | 600 | SPLIT|thinking (0.615) | REMOTE:Qwen3-32B|thinking (0.748) | -13.3 | [-17.2, -9.3] | - | -20.0 |
