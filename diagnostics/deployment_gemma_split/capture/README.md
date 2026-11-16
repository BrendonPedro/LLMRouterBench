# capture/ (Stage 0 and Stage 1 inputs; empty of lab facts until Stage 0 runs)

Written after the tag `gemma-split-prereg-v1`: the Stage-0 capture (with `capabilities.json`) is committed in commit A0
before any adapter file exists, the Stage-1 files with the adapter in commit A, and everything is hashed by the freeze
record. Templates: `split_capture.template.json`, `capabilities.template.json`, `candidates.template.json`
(every value is a `FILL` placeholder, booleans included; none of them validates as it is); questions: `LAB_QUESTIONS.md`.
No file here may contain a credential value: authentication is recorded as mechanism + variable / field NAMES only
(`split_capture.json` -> `authentication`, Q17); the freeze and the archive refuse credential-like content.

| File | Written by | Where |
|---|---|---|
| `edge/inventory.json`, `.txt` | `gs_bench.py inventory --role edge` | DGX Spark |
| `server/inventory.json`, `.txt` | `gs_bench.py inventory --role server` | asus-ESC4000-E11 |
| `roles.json` | `gs_bench.py roles` | analysis host |
| `edge/checkpoint.json`, `server/checkpoint.json` | `gs_bench.py checkpoint` | each host |
| `edge/checkpoint_reference.json` | `gs_bench.py checkpoint` on a verified full copy | DGX Spark, only if the edge's own directory is a subset |
| `edge/deployment.json`, `server/deployment.json` | `gs_bench.py deployment` | each host |
| `lab_answers.md` | the operator, from Brian's and the lab's dated written answers; also the log of every `adapter-smoke` run | — |
| `split_capture.json` | the operator, from code, launch files and answers | validated by `gs_bench.py split-validate` |
| `capabilities.json` | the operator, from code reading and Q11, BEFORE any adapter is written | checked by `gs_bench.py stage0`; the adapter must equal it |
| `candidates.json`, `selected_config.json` | operator; `gs_bench.py select` | analysis host |
| `stage0_verdict.json` | `gs_bench.py stage0` | analysis host |
| `workload_tokens.json` | `gs_bench.py tokenize` (Stage 1) | a host with the tokenizer |
| `edge/snapshot.json`, `server/snapshot.json` | `gs_bench.py snapshot` (Stage 1) | each host |
