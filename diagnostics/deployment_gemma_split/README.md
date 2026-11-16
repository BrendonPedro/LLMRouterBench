# deployment_gemma_split: GEMMA-SPLIT, the two-host split with the existing Gemma deployment (2026-11-16, revision 2)

**Status: DRAFT protocol, not frozen, not pre-registered.** The tag `gemma-split-prereg-v1` does not exist yet.
Nothing here has run on a lab machine, no model was downloaded, no GPU was used. Preparation and CPU tests only.

GEMMA-SPLIT is a **second, independent** infrastructure / model-generalisation experiment: does the two-host
split idea run correctly and efficiently with the lab's existing Gemma split deployment, DGX Spark = edge/client,
`asus-ESC4000-E11` (3 x RTX A6000) = server/backend? It is not a replacement for the Qwen3 experiments. The frozen
Qwen3 reference (E1-OPT, tag `e1opt-prereg-v1` -> `d9ba549f`) is not modified, imported, read or hashed by anything
here, and Gemma numbers are never compared with Qwen3 numbers (PREREG §0, §17). `tests/test_gs_protocol.py` checks
both: no GEMMA-SPLIT code refers to the E0 / E1-OPT directories or modules, and the E1-OPT directory is byte-identical
to its tag.

- `PREREG_GEMMA_SPLIT.md` the pre-registration draft, revision 2: question, fixed roles, Stage-0 capture, the
  deterministic configuration-selection rule, the Stage-0 capability declaration, the adapter contract, the freeze
  record and its chain, workload, the four separate questions (correctness, memory feasibility, component runtime,
  end-to-end runtime), STOP / CONTINUE rules, repeats and outcomes, archive, unresolved values (§18) and refused
  assumptions (§19).
- `RUNBOOK.md` exact commands per stage with machine labels (EDGE, SERVER, DESK), gate hand-off and repeats.
- Stage sequence: tag -> 0 capture (facts only) -> 1 adapter + freeze record -> 2 memory (no timers) -> 3 correctness
  (no timers; FAIL = STOP, no performance interpretation) -> 4 component runtime -> 5 end-to-end (split_A, edge_only,
  server_only, [native], split_B) -> 6 checkpoint re-hash, outcome, archive.

| File | Kind | Role |
|---|---|---|
| `gs_spec.py` | decision | every frozen parameter (thresholds, workload sizes, seeds, block order, repeat caps, decision-file list); the Qwen3 design as a written description only |
| `gs_rules.py` | decision | every metric and every verdict rule: correctness (hardware floor, EOS-normalised generation rule, coverage, binding), memory status, runtime summaries and the efficiency endpoint |
| `gs_gate.py` | decision | Stage-0 verdict, gates 2-5 (mode, hash, repeatable parts, caps, final verdicts), `require`, `invalidate`, recomputing `outcome` |
| `gs_freeze.py` | decision | freeze record writer / checker: hashes every file, refuses bytecode and symlinks, decision files pinned to the tag, documented orchestration fixes, executing-code check, record names and append-only evidence, record chain (three cases, at most three records) |
| `gs_split.py`, `gs_results.py`, `gs_stats.py`, `gs_adapter.py`, `gs_common.py`, `gs_checkpoint.py`, `gs_workload.py` | decision | split capture validation / differences / boundary audit / selection; layout, parts, provenance, no-timing guard, exit events; statistics; adapter contract; hashing, write-once IO, redaction; checkpoint identity; workload |
| `gs_reference.py`, `gs_memory.py`, `gs_correctness.py`, `gs_profile.py` | decision (measurement drivers) | HF reference runtime; Stage-2/3/4/5 drivers |
| `gs_inventory.py`, `gs_deployment.py`, `gs_archive.py`, `gs_bench.py` | orchestration | inventory and role verification; deployment capture and snapshot; archive; command hub (preflight, live-tree import guard, serve records, smoke test, retire, exit events) |
| `workload/` | decision | `correctness_prompts.json` (30 prompts), `timing_spec.json` (44 timing + 4 coverage items, 170-text pool) |
| `capture/` | — | Stage-0 templates (split capture, capability declaration, candidates) and questions for Brian / the lab; lab facts go here after the tag |
| `adapter/`, `frozen/` | — | empty until Stage 1 (rules in their READMEs) |
| `tests/doubles/gemma4_split_double.py` | test | a TEST DOUBLE: a real split of a tiny random Gemma 4 model over local TCP implementing the adapter contract (refused for target runs) |

Decision files are byte-identical to the tag in every freeze record; orchestration files, tests and the adapter may
change only through a documented fix in a new record, which is accepted only in the three cases of PREREG §6.4.

## CPU tests (run 2026-11-16 on this WSL box; no GPU)

| File | Interpreter | What | Tests | Result | Wall time |
|---|---|---|---|---|---|
| `tests/test_gs_protocol.py` | system `python3 -I -B` (3.12.3, standard library only) | independence from E0 / E1-OPT; redaction; inventory parsers (GB10 `[N/A]`, power limit) and a live collector smoke run; role verification; checkpoint manifest / features / KV-sharing map / quick and deep checks; deployment capture and snapshot verification; split validation (placeholders, booleans, UUIDs), templates never validate, differences, boundary audit (undeclared, dtype, rank, never crossed), selection rule; workload reproduction and validation; adapter info rules; provenance and the no-timing guard; statistics; correctness rules (hardware floor only, EOS normalisation, margins); memory rules (device vs host OOM, unified memory, allocator OOM events, contention); end-to-end summary (categories, drift, failures, contention, clock-tolerant order, serve windows, native blocks secondary); gates (mode, integrity, recompute), Stage-2 confirmation, stale files (the gate waits), gate-driven repeats, Stage-5 whole-sequence repeats, caps over the record chain incl. a supersede counted as a repeat, attempt accounting from the exit-status logs (crash -> repeat, deleted outputs re-run -> final INCOMPLETE with the surviving attempt identified by digest and its would-be verdict, unaccounted crash before a re-run, missing file of a completed run, refusals are not attempts), would-be results of invalidated attempts, `retire`; Stage-0 verdicts incl. capability declaration and clock sync; archive (all runs); freeze writer / checker in temporary git repositories (commits A0 / A / B, stray files, bytecode, symlinks, decision files pinned even if documented, executing-code check, supersede cases incl. committed crash evidence, a forged gate refused by recomputation, the no-output case read from the logs, revision cap, record names, deleted / renamed evidence, append-only logs, capability declaration committed first, `CUDA_VISIBLE_DEVICES` and `LD_PRELOAD` checks, an adapter naming the live tree, an adapter fix documented and both adapter versions archived, dev/target separation); the live-tree audit guard (import, open, `spec_from_file_location`, subprocess, `os.system`, `chdir`, launch files outside the roots; checkpoint, snapshot and the interpreter's environment allowed) | 59 | OK | 33 s |
| `tests/test_gs_pipeline.py` | `~/research/LLMRouterBench/.venv/bin/python -I -B` (torch 2.14.1 CPU build, transformers 5.19.0 with Gemma 4) | every stage through `gs_bench.py` with the Gemma 4 test double in dev mode (server half as a subprocess, stopped by SIGTERM and by Ctrl-C): reference runtime = HF `generate`; tokenisation of the frozen workload; dev guards (target record, model size); memory, correctness PASS (T1, hardware floor), isolated components, six end-to-end blocks incl. native, checkpoint re-hash, recomputed outcome, archive; the minimal-capability case (in-path trace, reference baselines only); `verify-workload` and `adapter-smoke` (`--dev` refuses a non-test-double adapter); faults caught: missing per-layer inputs, skipped layer, shifted positions, reduced-precision wire -> FAIL; the same defect in the in-host pipeline still FAILs; missing shared KV -> INVALID, after two repeats FAIL; undeclared token ids / bfloat16 wire -> INVALID; T2 tier; FAIL closes Stages 4-5 and cannot be invalidated; timing field -> INVALID; rows outside the serve window -> INVALID; reference infeasible -> NOT_EVALUABLE | 12 | OK | 226 s |

The pipeline test's Stage 5 ends UNSTABLE or COMPLETE depending on CPU timing noise (the drift rule at 10 %); both are
accepted there because CPU timings are not a measurement. Run both with `TMPDIR` pointing at a scratch directory if
`/tmp` is not wanted; the tests write nothing outside temporary directories.

## Independent audit (four rounds, 2026-11-16)

A separate reviewer audited revision 1 for leakage, post-result changes, stop-rule gaps and coupling. Coupling to
the Qwen3 experiments: clean in both rounds. Round 1 found six blockers and several gaps, fixed in revision 2:
dev-mode gates could open target stages (now separate record kinds, gate mode + hash); final verdicts could be
invalidated and repeat caps reset by rewording (now gate-driven repeats, caps per part and cause, finals immutable);
a new freeze record could escape a verdict and rule code could change after results (now a record chain accepted only
after INVALID, decision files pinned to the tag, all runs archived, outcome recomputed from raw files); the in-host
pipeline floor absorbed shared defects (now the hardware floor only); development runs could reveal outcomes before
capabilities were declared (now Stage-0 declaration, test-double-only dev mode, timer-free `adapter-smoke`, primary
comparator always the reference runtime); `--gs-dir` bypassed the freeze (now dev-only, and the executing harness is
checked). Also fixed: device placement checks, OOM classification and confirmation, EOS convention, teacher-forced
coverage, gate hand-off across hosts, serve-window binding, clock-tolerant ordering, Ctrl-C records, checkpoint
re-hash in the outcome, broader hashing and version / environment checks, template booleans, crash logging.

Round 2 re-checked revision 2 (findings 1, 4, 6, 8, 9, 12 confirmed fixed) and found two remaining blockers, both fixed:
a new freeze record could still escape a verdict (records recognised by file name only; any INVALID unlocked a new record
and reset caps; measurement drivers could change; evidence could be deleted) -> every record-shaped file is a record,
fixed names, git-history evidence checks, three supersede cases only, caps over the whole chain, at most three records,
measurement drivers pinned; and `invalidate --incomplete` could discard complete local data (round 3 removed the flag;
see below). Also fixed: per-row timestamps in Stage 3 (now one run-level window), stale host copies
(`retire` + provenance-time check), bytecode / symlinks / extra snapshot files / live-tree imports / `LD_*` variables,
`CUDA_VISIBLE_DEVICES` consistency between inventory and capture, clock synchronisation required, server records
device-checked, the capability declaration committed before any adapter exists, Stage-5 repeats covering the whole
sequence, and invalidated attempts reported with the result they would have shown.

Round 3 confirmed D, F, G and H closed and found one new blocker and two should-fix items, all fixed: the
orchestration-only supersede basis was a reset button (delete uncommitted Stage-5 outputs, re-freeze) -> it needs both
hosts' committed exit-status logs showing a crash while measuring and no completed command of that stage; stale files
were an uncapped redraw -> the gate waits for the fresh file; a supersede after exceptions escaped the caps and hid the
completed items' result -> a supersede counts as a repeat, every INVALID Stage-3 decision records `would_be`, and a
generation-rule failure over the completed items is a FAIL (more items cannot reverse it). Partly closed items: B ->
`invalidate --incomplete` removed; crashes, re-runs and deleted outputs are read from START / RUN / END events by every
gate (deleted outputs re-run = final INCOMPLETE); E -> an audit hook guards open / exec / subprocess / chdir and the
freeze refuses an adapter naming the live tree; I -> `adapter-smoke --dev` runs test doubles only. What software cannot
prevent (coarse durations visible to the operator, uncommitted evidence, local history before a push, child processes,
induced crashes) is stated with its mitigation in PREREG §21.

Round 4 confirmed N1, N2, N3, B and I closed and found two new blockers, both fixed: deleting an output after reading
it turned a seen verdict into a bare INCOMPLETE -> each completed writer's END line carries a digest of its outputs, and
the final INCOMPLETE now reports which attempt's outputs are present and the verdict they would give; the live-tree
guard refused a deployment interpreter installed inside Brian's repository -> the running interpreter's environment is
allowed (Brian's code must still come from the snapshot). Two statements of PREREG §21 were corrected: an adapter
change in a new record now needs its own `HARNESS_FIXES.md` entry, the record lists every changed file and the archive
keeps every record's adapter (`adapter_history/`); launch files outside the source roots are guarded like the live tree
(everything the adapter runs must be a snapshotted source root), and a non-editable installed copy of the deployment's
package is stated as unhashed. The round-4 fixes are covered by CPU tests; no fifth audit round was run.
