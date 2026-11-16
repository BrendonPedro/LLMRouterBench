# GEMMA-SPLIT runbook (revision 2, 2026-11-16; nothing has run on a lab machine; protocol not frozen)

Order: review -> **tag** -> Stage 0 capture -> Stage 1 adapter + freeze -> Stage 2 memory -> Stage 3 correctness
(no timers) -> Stage 4 components -> Stage 5 end-to-end -> Stage 6 checkpoint re-hash, outcome, archive. Each stage
runs only if the previous gate opened (PREREG §13); the drivers enforce it (exit 3 "GATE CLOSED").

Machines: **EDGE** = the DGX Spark (client); **SERVER** = asus-ESC4000-E11 (backend); **DESK** = the analysis host
(this WSL box: no GPU; tagging, gates, archive).
Conventions: `GS=<repo checkout>/diagnostics/deployment_gemma_split` on each machine (the experiment's own checkout,
not Brian's tree); `PY=<the deployment's own python on that machine>` (Stage 0 records which); every command is run
as `cd $GS && $PY -I -B gs_bench.py ...` from that checkout (a target command refuses any other `--gs-dir`; never run
python without `-B` there: bytecode and symlinks in the directory are refused) inside `tmux`; `F=frozen/freeze_record.json` (or the newest `frozen/freeze_record.rN.json`); `ID=<run_id>` printed by
`freeze`. Every command with `--freeze` appends START / RUN / END lines to `runs/$ID/<host>.exit_status.log` (the gates
use them for attempt accounting: never edit or delete them); for Stage 0/1 commands append
`; echo "rc=$? <cmd>" >> capture/<role>/exit_status.log`.

**Commit after every command (EDGE, SERVER) and every gate (DESK):** `git add diagnostics/deployment_gemma_split/runs &&
git commit -m "GEMMA-SPLIT <stage> <cmd> <host>"` (commits touching only `runs/`), and push those commits (the PI pushes,
PREREG §21). Never delete an output or a log line; a deleted-and-re-run attempt makes the stage INCOMPLETE (final).

**Gate hand-off, after every gate (DESK):** `rsync -a` the stage directory of `runs/$ID/` from both hosts to DESK
before gating, together with both hosts' `runs/$ID/<host>.exit_status.log` (role and block sub-directories never
collide); after `gate`, copy `runs/$ID/<stage>/gate.json` to both hosts (the next stage reads it locally) and commit the
gate, the decision / summary and any `_invalidated/LOG.jsonl` (`git add diagnostics/deployment_gemma_split/runs &&
git commit -m "GEMMA-SPLIT <stage> gate"`: a commit touching only `runs/`). `gate` exits 0 if the stage opened, 3 if
not, 4 if raw outputs or logs are still missing or a file is stale (nothing written).

**Repeats:** only the parts a gate lists as repeatable (Stage 5: always the whole block sequence):
`$PY -I -B gs_bench.py invalidate --freeze $F --stage N --all --note "<what happened>"` on DESK (or `--part P` per part
in Stages 2-4). The cause is taken from the gate; notes do not change it; caps count over the whole record chain; final
verdicts are refused. Then copy `runs/$ID/<stage>/_invalidated/LOG.jsonl` to both hosts and run
`$PY -I -B gs_bench.py retire --freeze $F --stage N` on each, which moves the hosts' copies of the same parts aside;
only then repeat. **A crash** (a command killed or failing while it measures): copy the outputs and both hosts' logs
to DESK and compute the gate; it lists the crashed part (Stage 5: the whole sequence) as repeatable. Never re-run a
crashed command without that gate and `invalidate` (an unaccounted crash before a completed run makes the stage
INVALID). A missing file of a command that completed means: copy it. Moved parts stay in
`runs/$ID/<stage>/_invalidated/`.

## Stage T: tag (DESK; by the PI)
```
git add diagnostics/deployment_gemma_split && git commit -m "GEMMA-SPLIT protocol and harness"   # after review
git tag -a gemma-split-prereg-v1 -m "GEMMA-SPLIT pre-registration" && git push origin gemma-split-prereg-v1
```
Never move or recreate the tag. Do not touch `e1opt-prereg-v1` / `e0-prereg-v1` or their directories.

## Stage 0: capture (EDGE, SERVER, DESK; no model, no GPU kernel, no timer; about 1 h plus answers)
On each lab machine: `git fetch --tags && git checkout gemma-split-prereg-v1` in the experiment's checkout.
```
EDGE:    $PY -I -B gs_bench.py inventory --role edge   --peer <SERVER IP or name> --out-dir capture/edge     # in the deployment's environment
SERVER:  $PY -I -B gs_bench.py inventory --role server --peer <EDGE IP or name>   --out-dir capture/server   # (same CUDA_VISIBLE_DEVICES etc.)
EDGE:    $PY -I -B gs_bench.py checkpoint --role edge   --model-dir <dir the edge part loads> --out capture/edge/checkpoint.json
SERVER:  $PY -I -B gs_bench.py checkpoint --role server --model-dir <.../gemma-4-E4B-it>      --out capture/server/checkpoint.json
EDGE:    $PY -I -B gs_bench.py deployment --role edge   --source-root <Brian's code dir> [--launch-file <script/config> ...] --out capture/edge/deployment.json
SERVER:  $PY -I -B gs_bench.py deployment --role server --source-root <Brian's code dir> [--launch-file ...] --out capture/server/deployment.json
```
Name as `--source-root` every directory the adapter will run code or configs from (repeat the flag): only source roots
are snapshotted, and launch files outside them are recorded but guarded like the live tree, so they cannot be run.
If the edge's checkpoint directory is not `identical` to the server's (`python -I -B gs_checkpoint.py compare --edge
... --server ...`, or the Stage-0 verdict, reports it) and the lab permits a copy (Q13): copy the server directory to an
experiment-owned directory on the EDGE, then `checkpoint --role edge --model-dir <copy> --out capture/edge/checkpoint_reference.json`.
Ask Brian and the lab every question of `capture/LAB_QUESTIONS.md`; record dated answers in `capture/lab_answers.md`.
Then on DESK (after `rsync` of `capture/edge` and `capture/server`):
```
$PY -I -B gs_bench.py roles --capture-dir capture                         # -> capture/roles.json (exit 2 = mismatch)
cp capture/split_capture.template.json capture/split_capture.json        # fill from code, launch files, answers (every FILL)
cp capture/capabilities.template.json capture/capabilities.json          # every capability the code supports (Q11), with the basis
cp capture/candidates.template.json capture/candidates.json              # every configuration found
$PY -I -B gs_bench.py split-validate --capture capture/split_capture.json --checkpoint capture/server/checkpoint.json
$PY -I -B gs_bench.py select --candidates capture/candidates.json --roles capture/roles.json --out capture/selected_config.json
$PY -I -B gs_bench.py stage0 --capture-dir capture                        # -> capture/stage0_verdict.json
```
GPU use: none. Stopping rule: any `STOP_*` -> record the missing or corrected facts and re-run the capture step; never
change the deployment (clocks must be NTP-synchronised on both hosts). `split-validate` prints the differences from the
Qwen3 design. On CONTINUE, commit the capture **before any adapter file exists** (commit A0):
```
git add diagnostics/deployment_gemma_split/capture && git commit -m "GEMMA-SPLIT Stage 0 capture and capability declaration"   # commit A0
```
The freeze checks in git history that `capture/capabilities.json` was first committed without any adapter file and never
changed afterwards; `adapter-smoke` refuses to run before A0.

## Stage 1: adapter and freeze (no timer)
```
SERVER:  $PY -I -B gs_bench.py tokenize --tokenizer-dir <server checkpoint dir> --out capture/workload_tokens.json
EDGE:    $PY -I -B gs_bench.py snapshot --role edge   --dest <experiment dir>/gemma_split_snapshot/edge     # -> capture/edge/snapshot.json
SERVER:  $PY -I -B gs_bench.py snapshot --role server --dest <experiment dir>/gemma_split_snapshot/server   # -> capture/server/snapshot.json
```
Write `adapter/gs_adapter_<impl>.py` against `gs_adapter.py` (rules: `adapter/README.md`; example:
`tests/doubles/gemma4_split_double.py`). The only sanctioned way to run it before the freeze is the timer-free smoke
test (no output kept, no token ids printed); log every such run in `capture/lab_answers.md`:
```
SERVER:  $PY -I -B gs_bench.py adapter-smoke --adapter adapter/gs_adapter_<impl>.py --role server        # until Ctrl-C
EDGE:    $PY -I -B gs_bench.py adapter-smoke --adapter adapter/gs_adapter_<impl>.py --role edge          # "contract OK"
```
Then on DESK:
```
$PY -I -B gs_bench.py adapter-check --adapter adapter/gs_adapter_<impl>.py
git add diagnostics/deployment_gemma_split/capture diagnostics/deployment_gemma_split/adapter && git commit -m "GEMMA-SPLIT Stage 1 capture and adapter"   # commit A
$PY -I -B gs_bench.py freeze --adapter adapter/gs_adapter_<impl>.py --out frozen/freeze_record.json      # prints run_id
git add diagnostics/deployment_gemma_split/frozen && git commit -m "GEMMA-SPLIT freeze record"           # commit B: frozen/ only
```
On EDGE and SERVER, after `git fetch && git checkout <commit B>` in the experiment's checkout:
```
EDGE:    $PY -I -B gs_bench.py check-freeze    --freeze $F --role edge      # exit 0 required
EDGE:    $PY -I -B gs_bench.py verify-workload --freeze $F --role edge      # token ids re-derived with this host's tokenizer
SERVER:  $PY -I -B gs_bench.py check-freeze    --freeze $F --role server
SERVER:  $PY -I -B gs_bench.py verify-workload --freeze $F --role server
```
Copy `runs/$ID/stage1_checks/*` to DESK. Stopping rule: a check that cannot pass means the machine changed since Stage 0:
fix the environment (never the record) or redo Stage 0 for that host and write a new record (Stage N below).

## Stage 2: memory feasibility (both hosts; no timer; the experiment's GPUs must be quiet)
```
SERVER:  $PY -I -B gs_bench.py serve  --freeze $F --stage 2                         # wait for READY; leave running
EDGE:    $PY -I -B gs_bench.py memory --freeze $F --what split --role edge
SERVER:  Ctrl-C (or kill -TERM) the serve; wait for "server split memory: ..."
SERVER:  $PY -I -B gs_bench.py memory --freeze $F --what reference --role server
EDGE:    $PY -I -B gs_bench.py memory --freeze $F --what reference --role edge
DESK:    $PY -I -B gs_bench.py gate   --freeze $F --stage 2
```
Stopping rule: STOP_INFEASIBLE or INCOMPLETE -> final (Stage 6). INFEASIBLE_UNCONFIRMED -> `invalidate --stage 2 --all`,
`retire` on both hosts, and repeat Stage 2 once with fresh processes. BLOCKED_CONTENTION -> wait for quiet GPUs, then
the same. INVALID -> the same (twice at most).

## Stage 3: correctness (no timer anywhere)
```
SERVER:  $PY -I -B gs_bench.py reference --freeze $F --role server          # then rsync stage3_correctness/reference_server to EDGE
EDGE:    $PY -I -B gs_bench.py reference --freeze $F --role edge
SERVER:  $PY -I -B gs_bench.py serve --freeze $F --stage 3                  # READY
EDGE:    $PY -I -B gs_bench.py correctness --freeze $F --what split
SERVER:  Ctrl-C the serve; wait for "server record written"                # binds the split rows to this server process
SERVER:  $PY -I -B gs_bench.py correctness --freeze $F --what pipeline      # only if in_host_split was declared (a diagnostic)
DESK:    $PY -I -B gs_bench.py gate --freeze $F --stage 3                   # -> decision.json, gate.json
```
Stopping rule: PASS -> Stage 4. FAIL, NOT_EVALUABLE or INCOMPLETE -> **STOP**: no runtime stage runs (the drivers
refuse); go to Stage 6. INVALID -> repeat the listed parts (exceptions twice at most, then FAIL), or, for a harness /
adapter defect or a boundary difference, Stage N.

## Stage 4: component runtime (each host; descriptive)
```
EDGE:    $PY -I -B gs_bench.py components --freeze $F --role edge
SERVER:  $PY -I -B gs_bench.py components --freeze $F --role server
DESK:    $PY -I -B gs_bench.py gate --freeze $F --stage 4
```
If the adapter declares `step_trace` but not `segment_step`: SERVER `serve --freeze $F --stage 4`, EDGE `components
--role edge`, SERVER Ctrl-C. Stopping rule: INVALID -> `invalidate` the listed role, repeat (twice at most).

## Stage 5: end-to-end (frozen block order; fresh processes per block; quiet GPUs on both hosts)
```
SERVER:  $PY -I -B gs_bench.py serve --freeze $F --stage 5 --block split_A ;  EDGE: ... e2e --freeze $F --block split_A ;  SERVER: Ctrl-C
EDGE:    $PY -I -B gs_bench.py e2e --freeze $F --block edge_only            # reference runtime; server part stopped
SERVER:  $PY -I -B gs_bench.py e2e --freeze $F --block server_only          # reference runtime
EDGE:    $PY -I -B gs_bench.py e2e --freeze $F --block edge_only_native     # only if unsplit_same_runtime was declared
SERVER:  $PY -I -B gs_bench.py e2e --freeze $F --block server_only_native   # only if unsplit_same_runtime was declared
SERVER:  $PY -I -B gs_bench.py serve --freeze $F --stage 5 --block split_B ;  EDGE: ... e2e --freeze $F --block split_B ;  SERVER: Ctrl-C
DESK:    $PY -I -B gs_bench.py gate --freeze $F --stage 5
```
Stopping rule: COMPLETE -> Stage 6. UNSTABLE (drift, first time) -> `invalidate --stage 5 --all` and repeat every block
once; a second UNSTABLE is final. INVALID -> `invalidate` the listed blocks and repeat them (caps: PREREG §11.7).

## Stage N: a new freeze record (three cases only; PREREG §6.4)
(a) the run never produced output (e.g. `check-freeze` failed after an environment change: redo Stage 0 for that host);
(b) the latest gate is INVALID with `supersede_eligible: true` (a boundary difference or workload mismatch: correct the
capture fact; or exceptions: fix the adapter); (c) a command of the next stage crashed while measuring because of an
orchestration defect, before that stage produced any output (both hosts' exit-status logs copied and committed; only
`gs_bench.py`, `gs_inventory.py`, `gs_deployment.py`, `gs_archive.py` or tests may change). Fix in a new commit with a
CPU regression test, add `path | fix commit | regression test | reason` to `frozen/HARNESS_FIXES.md` for every changed
non-decision file, the adapter included (decision files, including the measurement drivers, can never change), then on DESK:
```
$PY -I -B gs_bench.py freeze --adapter adapter/gs_adapter_<impl>.py --out frozen/freeze_record.r2.json --supersede-reason "<defect, INVALID gate>"
git add diagnostics/deployment_gemma_split/frozen && git commit -m "GEMMA-SPLIT freeze record r2"
```
It is refused if any run of the chain has a final verdict, if three records already exist, if any stage of the
previous run is ungated, if evidence was removed, renamed or rewritten, or if none of (a)-(c) holds. The new record
counts as a repeat of the causes that justified it, and repeat caps carry over.
Restart at Stage 1's checks and Stage 2 under the new run_id; the old run stays in `runs/` and in the archive.

## Stage 6: checkpoint re-hash, outcome and archive
```
EDGE:    $PY -I -B gs_bench.py verify-checkpoint --freeze $F --role edge     # full re-hash vs the Stage-0 manifest(s)
SERVER:  $PY -I -B gs_bench.py verify-checkpoint --freeze $F --role server   # exit 2 = checkpoint changed: the run is INVALID
DESK (after rsync of runs/$ID/final from both hosts):
$PY -I -B gs_bench.py outcome --freeze $F                                    # recomputes every gate -> runs/$ID/final/outcome.json
$PY -I -B gs_bench.py archive --run-id $ID --out <archive dir>/gemma_split_${ID}_<OUTCOME>.tgz [--include-snapshot <dir>]   # snapshot only with consent
$PY -I -B gs_bench.py verify-archive --archive <archive dir>/gemma_split_${ID}_<OUTCOME>.tgz
```
Record the archive SHA-256 in the results note. Never report a Gemma number next to a Qwen3 number (PREREG §17).

## Dev / test mode (never a decision)
`--dev` on any command: a dev freeze record (`freeze --dev`), a test-double adapter only, a model below 512 MiB,
`--device` / `--endpoint` / `--model-dir` / `--gs-dir` overrides; outputs and gates are labelled `dev` and no target
command accepts them. `tests/test_gs_pipeline.py` runs every stage this way on CPU with the Gemma 4 test double.
