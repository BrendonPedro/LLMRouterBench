# deployment_e1opt: E1-OPT, the optimised-runtime split follow-up (2026-11-11, revision 3)

**Status: frozen by tag `e1opt-prereg-v1` (preserved); amended before any measurement by PREREG §16, frozen by tag
`e1opt-prereg-v2` (preserved; failed Stage-1b check B); amended after that failure by PREREG §17, frozen by tag
`e1opt-prereg-v3`.** No timing measurement exists. No E0 file is modified (`tests/test_e1_cpu.py` checks `git diff --quiet 81461b6 --
diagnostics/deployment_e0` and the tag `e0-prereg-v1`).

- **Amendment v3 (PREREG §17, 2026-11-17, POST-FAILURE, not pre-registered):** v2 failed Stage-1b check B on the
  GB10 (static max abs 80.8125 > 2⁻⁶, argmax 409/416; A and C passed). Check B is now E0's Q1 rule against an
  SDPA-backend floor (`e1_equivalence.static_floor_rule`); the runtime code is unchanged. Frozen by `e1opt-prereg-v3`.
  Record `frozen/runtime_freeze_v3.json`.
- Amendment v2 (PREREG §16, 2026-11-16, before any E1 measurement): GPU memory capacity falls back to PyTorch's
  physical capacity when `nvidia-smi` reports none (GB10); the competitor vLLM is an explicitly supplied executable
  checked in its own environment. Tag `e1opt-prereg-v1` is preserved; v2 is frozen by `e1opt-prereg-v2`.
- `PREREG_E1OPT.md` draft protocol, revision 3: five approved decisions (rounding rule, one server GPU, fixed cuts,
  compatibility fallback, inventory-decided roles) are in; §14 lists the facts only Stage 0 can supply; §15 maps
  every executable component to its tests.
- `RUNBOOK.md` exact commands per stage with machine labels: tag → 0 inventory → 1a freeze → 1b equivalence (no
  timing fields) → 2 server segment → 3 fresh edge profiles + G-UB-E1 → 4 link gate, joint cells, G-R.
- `IMPLEMENTATION_PLAN.md` why vLLM cannot express the layout; candidate table; chosen path.
- `e1_bench.py` command hub (`inventory`, `freeze`, `check-freeze`, `equivalence`, `profile-server-segment`,
  `fit-split`, `fit-competitor`, `profile-edge-segment`, `summarize-fit`, `gate-ub`, `serve`, `run-cell`,
  `run-remote-cell`, `linkcheck`, `gate-r`). Every GPU command requires `--freeze`.
- Modules: `e1_spec.py` (approved rules, `FIXED_CUTS`, `validate_cut`, `decide_cell`), `e1_static_cache.py`,
  `e1_segment.py`, `e1_attention.py`, `e1_memory.py`, `e1_protocol.py` (HELLO whitelist + freeze hash, fail
  closed), `e1_freeze.py` (record writer/checker), `e1_inventory.py` (Stage 0; feasibility; deterministic roles),
  `e1_provenance.py` (target/screening rows; gate rejection rules), `e1_equivalence.py` (Stage 1b; no timing
  fields), `e1_profile.py` (Stages 2 and 3), `e1_joint.py` (Stage 4), `e1_gate.py` (provenance-checked gates +
  rounded decision sidecar).
- Tests (CPU only; run each with the pinned venv's `python -I`):

| File | What | Tests | Wall time (2026-11-11) |
|---|---|---|---|
| `tests/test_e1_cpu.py` | runtime: prefix bitwise = HF = E0; static tolerance/argmax; greedy; graph control flow; wire; E0 immutability; Qwen3-0.6B; v3: kernel switch passes the floor rule, a removed causal mask fails it | 10 | 105 s |
| `tests/test_e1_freeze.py` | freeze schema, hash set, overwrite refusal, edit detection, file/version/host/role mismatches, CLI exit 2, branch rule; v2 competitor executable and memory source in the record | 10 | 5 s |
| `tests/test_e1_rules.py` | rounding rule cases, fixed-cut validation, inventory parsers, feasibility, one-server-GPU, role rules, provenance rejections; v2: numeric nvidia-smi kept, GB10 N/A -> torch total_memory, never free memory or a wrong device, competitor vLLM probe fails closed, unchanged runtime checks | 17 | 1 s |
| `tests/test_e1_equivalence.py` | no-timing-field schema, in-host checks, segments-only, joint Q1/Q2 in E0 formats, E0 decision rules, CLI; v3: floor rule (pass, fail, envelope, fail closed) and both `pass` predicates enforce it | 13 | 10 s |
| `tests/test_e1_joint.py` | HELLO refusals, malformed messages fail closed, wire content audit, frozen cell through `run-cell`, cut resolution | 5 | 30 s |
| `tests/test_e1_profile_gate.py` | fit-split child/exit status, edge and server profiles readable by E0 fitters, summarize-fit, gate provenance rejections and labelled screen | 4 | 33 s |
| `tests/test_reanalysis.py` | archived-profile reanalysis regression (`.venv-wls-score` python) | 7 | 12 s |

- `reanalysis/` CPU reanalysis of the archived E0 profiles; `threshold_audit.py` is the correction audit (full pass
  at f = 0.6888, 1.45×; the first-pass 0.705 / 1.42× used CI low ≥ 0 and is withdrawn).
- `frozen/` is empty on this branch until Stage 1a writes `runtime_freeze_v3.json`; `env/` is created by Stage 0.
