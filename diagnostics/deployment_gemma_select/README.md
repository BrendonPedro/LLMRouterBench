# deployment_gemma_select: GEMMA-SELECT

A same-model Edge-versus-Split deployment decision for Gemma-4 E4B on a DGX Spark and one RTX A6000, using the
deadline-utility rule of E0. Protocol: `PREREG_GEMMA_SELECT.md`. Execution: `RUNBOOK.md`.

It is not E0, not E1-OPT and not GEMMA-SPLIT v1 (`../deployment_gemma_split`, abandoned at STOP_INCOMPLETE, untouched).

| File | Role |
|---|---|
| `gsel_spec.py` | every fixed parameter and pin |
| `gsel_stats.py` | correctness gate, deadline utility, decision rule, bound, outcome |
| `gsel_workload.py`, `workload/workload.json` | the fixed query sets |
| `gsel_whole.py` | the whole model on the edge (transformers) and the shared tokenizer call |
| `gsel_adapter.py` | contract for the split implementation; the adapter itself is private and pinned by hash |
| `gsel_common.py` | hashes, identity checks, resource gates, write-once outputs |
| `gsel_bench.py` | the commands |
| `tests/` | CPU tests with a test double |

The split implementation, the adapter, the site files and all raw outputs are private and are not in this repository.
