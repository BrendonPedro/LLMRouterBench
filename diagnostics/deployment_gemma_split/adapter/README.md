# adapter/ (empty until Stage 1)

The adapter for the selected configuration is written here AFTER Stage 0 (when the deployment's code is known and its
capabilities are declared in `capture/capabilities.json`) against the contract in `../gs_adapter.py`, and BEFORE any
correctness or timing run. It is committed in commit A together with capture/, and the freeze record hashes it.
Rules (PREREG §5):

- Call the deployment's own code from the hash-verified snapshot (`capture/<role>/snapshot.json`); never from the
  live tree (in target runs an import hook refuses modules from it), never modified. Use the generation path the
  captured launch commands use. No symlinks and no subpackages outside this directory's hashed files. Put the snapshot
  first on `sys.path`; an editable install of the deployment's package resolves to the live tree and is refused. Anything
  you run (scripts, configs) must lie under a captured source root: launch files outside the roots are guarded too.
- Load no weights in `make_adapter()`: loading happens in `connect()` (edge) / `serve()` (server), inside the Stage-2
  probes. The experiment's own server instance listens on `transport.experiment_endpoint` (never Brian's live port); a
  credential, if needed, is read at run time from the environment variable named in `authentication` and is never
  written, logged or returned.
- Do not change the checkpoint, dtype, layer ownership, cut, kernels, KV policy, transport, serialisation or any
  computation. Force greedy decoding and the frozen token ids only. Return the stopping EOS id in `tokens`.
- No timers: the harness times. No caching across requests that the deployment does not do itself.
- `ADAPTER_INFO["capabilities"]` (and any `missing_capability_justification`) must equal `capture/capabilities.json`;
  the freeze refuses any difference. `teacher_forced_logprobs` is mandatory whenever the split path computes logits the
  adapter can reach without changing the computation.
- Write it only after commit A0 (capture/ with `capabilities.json`, no adapter file). Before the freeze, it runs only
  through `gs_bench.py adapter-smoke` (one short request, types checked, no timer, nothing written; refused before A0);
  `--dev` mode refuses it. Log every smoke run in `capture/lab_answers.md`.
- `ADAPTER_INFO["test_double"]` must be `false`. Check with `gs_bench.py adapter-check --adapter adapter/<file>.py`.

Worked example of the contract: `../tests/doubles/gemma4_split_double.py` (a test double, refused for target runs).
