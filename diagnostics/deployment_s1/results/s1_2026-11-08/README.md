# Deployment S1 quality gate — 2026-11-08

## Result

Pre-registered verdict: **CONTINUE**

- 4,000 / 4,000 planned generations scored.
- 0 unfinished jobs.
- Qwen3-32B bf16 exceeded EDGE-best(16) by at least 3.0 percentage points in at least one benchmark/mode cell.
- Therefore the 16 GB and 24 GB split regimes remain eligible for full E0.
- rule16 = false
- rule12 = false

This result establishes quality headroom only. It does not establish split feasibility, latency benefit, memory feasibility, or end-to-end usefulness.

## Runtime anomaly

During scoring, one background thread emitted:

`ValueError: not enough values to unpack (expected 2, got 0)`

The exception occurred after approximately 2,700 / 4,000 items had been scored. Scoring continued afterward through 4,000 / 4,000, the runner reported 0 unfinished jobs, and the S1 summary was generated successfully.

The anomaly is therefore retained as a run limitation / implementation note. S1 was not rerun.

## Raw evidence

The full raw RunPod result archive is retained locally as:

`s1_results_final.tgz`

SHA-256:

`aad80e5b0ca3584bb0ca921e1130e4bf71ad393ac1860d80f728f8d412aff04a`

Large raw generation files and server logs are intentionally not committed to Git.
