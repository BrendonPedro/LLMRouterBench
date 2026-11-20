# deployment_qwen_select: QWEN-SELECT

A pre-registered measurement of which configuration the E0 decision rule selects for Qwen3 on the laboratory pair
(an edge machine of the DGX Spark class and one RTX A6000 as server): the k = 16 split of Qwen3-32B, a whole model on
the edge (Qwen3-32B, 14B or 8B), or, if the query may leave the edge, Qwen3-14B whole on the server.

- `PREREG_QWEN_SELECT.md`: the protocol. Frozen by the tag `qwen-select-prereg-v1`.
- `RUNBOOK.md`: the order of commands. Exact host values are in a private site file and a private execution sheet.
- `qsel_spec.py`, `qsel_stats.py`, `qsel_workload.py`, `qsel_runtime.py`: decision files, never edited after the tag.
- `qsel_bench.py`, `qsel_common.py`, `qsel_models.py`: drivers, gates, the GPU-hour ledger, checkpoint manifests.
- `frozen/`: manifests of Qwen3-14B and Qwen3-8B (the 32B manifest is K16-HWFIT's).
- `workload/workload.json`: the 160 held-out queries by key and hash. No prompt text.
- `posthoc/`: the post hoc screen from archived records that motivated the experiment. Exploratory, not a result.
- `tests/`: CPU tests with tiny random models. `python -I -B tests/test_qsel.py` (torch and transformers 4.51.3).

Every primary configuration is measured in two halves about half a day apart, so that no configuration owns one time
of day on the shared machines, and a run that is cut short still leaves a balanced half.

Reused unchanged, by import: `deployment_k16fit/` (runtime glue, correctness rule), `deployment_e1opt/` (segments and
wire, reference mode only), `deployment_e0/` (weight loading). A test checks that nothing outside this directory
differs from the base commit.

No result exists until `summary.json` has been produced from a run under the tag.
