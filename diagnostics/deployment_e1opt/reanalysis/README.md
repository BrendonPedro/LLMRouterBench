# CPU reanalysis of the archived E0 profiles (2026-11-10/11; corrected 2026-11-11)

Inputs: `e0_stage1_VALID_UB1_CONTINUE.tgz` (sha256 3984d73a…b84f) and `e0_stage2_VALID_UB2_STOP.tgz`
(sha256 f0df5369…fdae9), extracted to a scratch directory; `diagnostics/deployment_s1/results/s1_2026-11-08/q3_scores.jsonl`.
Code: E0's `e0_r.py` at commit 81461b6, imported unchanged. No inference, no GPU, no protocol change.

```
python server_speedup_curve.py <stage1 dir> <stage2 dir> out   # f = 1 reproduces gates/gate_ub2.json exactly
python breakdown.py            <stage1 dir> <stage2 dir> out   # per-query deadline view; GB10 bound
python bw_threshold.py         <stage1 dir> out                # server bandwidth roofline scan (archived edge profiles only)
python threshold_audit.py      <stage1 dir> <stage2 dir> out   # correction audit: strict predicate, monotonicity grid, float rule
```
Outputs in `out/`: `profiles.json` (fitted a, b, prefill points), `curve.json` (f -> every no-raw-text cell),
`thresholds.json` (bisection), `link_whatif.json` (per-step RTT added, what-if), `gb10_bound.json`, `bw_threshold.json`.
Run with `.venv-wls-score/bin/python -I` (numpy 2.5.4, python 3.12). Regression tests: `tests/test_reanalysis.py`.

Correction (2026-11-11). The first version of `server_speedup_curve.py` bisected the "full pass" on
`min(delta - 3, ci_lo) >= 0`, which accepts a CI lower bound of exactly 0.0; the gate uses `ci_lo > 0`. Because the
bootstrap statistic is quantised to 0.25 pt, the 12 GB / 60 s cell has ci_lo == 0.0 for f in (0.6888, 0.7054], so
the script reported f = 0.705 (1.42×) and a hand-filled table row showed "1 passing cell" at 0.705 while the gate
showed 0 at 0.700. Corrected: full pass at f = 0.6888 (1.45×); Δ ≥ 3.0 at f = 0.8297 (1.21×) as the gate computes
it. `bw_threshold.py` had the same predicate: pass bandwidth 1,357 GB/s (was 1,322); the Δ ≥ 3 figure (1,127) is
unchanged. Per-cell U_SPLIT, Δ, CI low and pass are monotone in f on a 0.005 grid (`out/fine_grid.json`).
Second finding: `e0_r.py` compares `delta_pt >= 3.0` on floats; (0.485 − 0.455) × 100 = 2.9999999999999973, so an
exact 3.00 is not counted. In exact arithmetic Δ ≥ 3 is reached at f = 0.8410 (1.19×). E1-OPT pre-specifies
rounding to 1e-6 pt (`e1_spec.decide_cell`), subject to the reviewer's approval.
