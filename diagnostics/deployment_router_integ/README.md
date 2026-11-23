# ROUTER-INTEG

The frozen lightweight router (all-MiniLM-L6-v2 + a 24,705-parameter head, five cross-validation fold models)
connected to real Edge and Remote inference of its original model pair, DeepSeek-R1-0528-Qwen3-8B and Qwen3-8B.

Start with `PROTOCOL_ROUTER_INTEG.md`. Nothing here has run on a GPU. No result exists.

| File | What it is |
|---|---|
| `PROTOCOL_ROUTER_INTEG.md` | the frozen evaluation protocol |
| `rint_spec.py` | every fixed parameter |
| `rint_router.py` | adapter of the frozen router; exact model-identifier mapping |
| `rint_policy.py` | placement rule; data-policy feasibility of every execution mode |
| `rint_backend.py` | backend dispatch: Edge (in process), Remote (wire), mock (tests); the sampled generation loop |
| `rint_record.py` | measurement rows, failure and fallback events, run headers |
| `rint_workload.py` | the frozen sample, orders, pinned token ids |
| `rint_models.py` | checkpoint manifests, verification, download, required memory |
| `rint_run.py` | launcher: select-gpu, preflight, serve-remote, shared lanes, router-timing, live pipeline, ledger |
| `rint_score.py` | scorer (analysis machine): unchanged benchmark evaluators |
| `rint_eval.py` | record-based policy evaluator and live comparison |
| `router/` | the five verified fold heads, the fold map, the training-only task lookup, the build report |
| `frozen/` | workload (no prompt text), checkpoint manifests, the freeze record |
| `tools/rint_router_build.py` | regenerates and verifies the router from the router study's own code and data |
| `tools/rint_freeze.py` | writes and checks the freeze record |
| `tests/` | CPU tests with tiny random models, a loopback wire and fixture workloads |
| `RUNBOOK.md` | the command sequence, with placeholders for everything host-specific |
| `CLARIFICATION_1.md` | how passages of the frozen protocol are to be read; four descriptive counts declared before any GPU run |

Reused, by import and unchanged: QWEN-SELECT at tag `qwen-select-prereg-v1` (loader, wire framing, memory gates,
GPU selection, GPU-hour ledger), and the benchmark evaluators of the router study's repository.

Private and never placed here: site files, the workload's prompts and reference answers, raw outputs, run sheets.

Tests (CPU only, about two minutes; needs torch and transformers 4.51.3):

    RINT_QSEL_REPO=<clone of the reused runtime at its tag> python -I -B tests/test_rint.py
