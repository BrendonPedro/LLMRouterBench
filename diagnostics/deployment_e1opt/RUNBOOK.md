# E1-OPT runbook (revision 3, 2026-11-11; nothing has run on a GPU; protocol not frozen)

Order: review → **tag** → Stage 0 inventory → Stage 1a freeze record → Stage 1b equivalence (no timers) →
Stage 2 server segment → Stage 3 fresh edge profiles and G-UB-E1 → Stage 4 link gate, joint cells, G-R.
Machines: LOCAL DESKTOP (this WSL box, no GPU: tagging, freeze record, gates); LAB MACHINE A and LAB MACHINE B
(the Asus server and the DGX Spark; which is edge and which is server is decided at Stage 0, so the labels below
are "SERVER MACHINE" / "EDGE MACHINE" after Stage 0 and "A / B" before it). Every GPU command takes `--freeze`.

Conventions: `E1=~/research/LLMRouterBench/diagnostics/deployment_e1opt`, `PY=<python of the pinned venv on
that machine>`, `MODEL=<local Qwen3-32B checkpoint dir at revision 9216db57>`. All lab runs happen inside
`tmux`; every command's exit status is appended to `exit_status.log` of its stage directory (`st` wrapper of E0
RUNBOOK §3, or `; echo "rc=$? <cmd>" >> exit_status.log`). Back up each stage directory to the LOCAL DESKTOP
(`rsync -a`) before the next stage starts; tar it with sha256 at the end (§8).

## Stage T: tag (LOCAL DESKTOP; no GPU)

```
cd ~/research/LLMRouterBench && git add diagnostics/deployment_e1opt && git commit -m "E1-OPT protocol and drivers"   # by you, after review
git tag e1opt-prereg-v1 && git push --tags        # by you
```
Expected runtime: minutes. Stopping rule: do not tag while any item of PREREG §14 that is a code constraint is open.

## Stage 0: inventory and roles (LAB MACHINE A and B; no model; no GPU kernels; no timing)

On each lab machine (the script enumerates CUDA devices only):
```
cd $E1 && $PY -I e1_bench.py inventory collect --peer <IP or hostname of the other machine> --out env/$(hostname).json
```
Output: `env/<hostname>.json` + `.txt`. Copy both to LOCAL DESKTOP `$E1/env/`. Then:
```
cd $E1 && $PY -I e1_bench.py inventory roles --inventory env/A.json env/B.json --out env/roles.json --merged-out env/inventory.json \
    [--lab-server HOST --lab-edge HOST]      # only if the lab assigns roles (PREREG §6.3 rule 1)
```
Ask the lab (record the answers in `env/lab_answers.md`): link between the machines; `tc netem` permitted?;
roles or a fixed server device assigned?; is the 2 × RTX 3090 machine the Asus server? Expected runtime: 30 min.
GPU use: none. Safe unattended: yes. Stopping rule: `env/roles.json` says `feasible: false` → INCOMPLETE; stop.
Check PREREG §14 item 5 (edge GPU must be NVML index 0 on a multi-GPU edge).

## Stage 1a: freeze record (LOCAL DESKTOP; then verify on each lab machine)

```
cd $E1 && $PY -I e1_bench.py freeze --inventory env/inventory.json --roles env/roles.json --tag e1opt-prereg-v1 --out frozen/runtime_freeze.json
git add diagnostics/deployment_e1opt/frozen diagnostics/deployment_e1opt/env && git commit -m "E1-OPT freeze record"   # touches frozen/ and env/ only
```
(If `env/` must not be committed, keep it out; the checker only requires that nothing outside `frozen/` of the
E1 directory changed since the tag.) On each lab machine, after `git fetch && git checkout <that commit>`:
```
cd $E1 && $PY -I e1_bench.py check-freeze --freeze frozen/runtime_freeze.json --role server    # SERVER MACHINE
cd $E1 && $PY -I e1_bench.py check-freeze --freeze frozen/runtime_freeze.json --role edge      # EDGE MACHINE
```
Exit 0 on both, else fix the environment (never the record) and re-check. Expected runtime: minutes. GPU: none.
Output: `frozen/runtime_freeze.json` (its `freeze_sha256` appears in every later file). Stopping rule: a check
that cannot pass means the machine does not match the inventory it was frozen from: redo Stage 0 → new record.

## Stage 1b: equivalence (correctness only; no timing fields are produced)

SERVER MACHINE (holds the whole model; needs about 70 GB free on the server GPU for the HF reference):
```
cd $E1 && $PY -I e1_bench.py equivalence reference --freeze frozen/runtime_freeze.json --model-dir $MODEL --device cuda:<server index> --out-dir e1_runs/stage1b/reference \
    [--cont <E0 Q1 continuations jsonl> --limit <n>]      # optional E0-format Q1 references for the joint check
cd $E1 && $PY -I e1_bench.py equivalence local --freeze frozen/runtime_freeze.json --model-dir $MODEL --device cuda:<server index> --role server --class 12 \
    --reference-dir e1_runs/stage1b/reference --out e1_runs/stage1b/local_server_c12.json
cd $E1 && $PY -I e1_bench.py equivalence local ... --class 16 --out e1_runs/stage1b/local_server_c16.json
```
EDGE MACHINE (cannot hold the model under the cap):
```
cd $E1 && $PY -I e1_bench.py equivalence local --freeze frozen/runtime_freeze.json --model-dir $MODEL --device cuda:<edge index> --role edge --class 12 --segments-only --out e1_runs/stage1b/local_edge_c12.json
cd $E1 && $PY -I e1_bench.py equivalence local ... --class 16 --segments-only --out e1_runs/stage1b/local_edge_c16.json
```
JOINT (SERVER MACHINE runs `serve` for the class; EDGE MACHINE runs `joint`), per class 12 and 16:
```
SERVER:  cd $E1 && $PY -I e1_bench.py serve --freeze frozen/runtime_freeze.json --model-dir $MODEL --device cuda:<server index> --class 12 --port 29500 --log e1_runs/stage1b/server_c12.log.jsonl
EDGE:    cd $E1 && $PY -I e1_bench.py equivalence joint --freeze frozen/runtime_freeze.json --model-dir $MODEL --device cuda:<edge index> --class 12 --server <SERVER IP>:29500 \
            --cont <Q1 continuations> --ref-dir e1_runs/stage1b/reference/q1_ref --q2-ref <E0 q2 reference jsonl> --out-dir e1_runs/stage1b/joint_c12
```
Floors (E0 §4.1, unchanged procedures of `deployment_e0/e0_q.py`): pipeline floor = `equivalence local --role server`
Q1 directory of the static in-host split (produced by adding `--cont/--ref-dir` support in the same run, or by
`e0_q.py cmp --config split` against an in-host E1 server); hardware floor = Qwen3-8B `e0_q.py cmp --config hf`
on the edge GPU vs the reference on the server GPU. Then on LOCAL DESKTOP:
```
cd $E1 && $PY -I e1_bench.py equivalence decide --local e1_runs/stage1b/local_*.json --split-dirs e1_runs/stage1b/joint_c12/q1_split e1_runs/stage1b/joint_c16/q1_split \
    --pipeline-dirs <pipe_c12> <pipe_c16> --hw-dir <hw floor dir> --q2-ref <q2 ref> --q2-runs e1_runs/stage1b/joint_c12/q2_split.jsonl e1_runs/stage1b/joint_c16/q2_split.jsonl \
    --out e1_runs/stage1b/stage1b_decision.json
```
Expected runtime: unknown until first measured run (E0's Q1 over 200 continuations took hours on an H200; the
local checks take minutes). GPU use: yes, both. Safe unattended: yes for `reference` and `local`; `joint` needs
both machines up. Outputs: the JSON files above (correctness fields only; `assert_no_timing_fields` is applied).
Stopping rule: `pass: false` → one fix (PREREG §4.4), new freeze record, rerun all of Stage 1b; a second
failure is STOP. A `graph_capture_failed` entry → new freeze record with `off_capture_failed`, rerun.

## Stage 2: server segment (SERVER MACHINE; first timing of the experiment)

```
cd $E1 && $PY -I e1_bench.py profile-server-segment --freeze frozen/runtime_freeze.json --model-dir $MODEL --device cuda:<server index> --class 12 --out-dir e1_runs/stage2
cd $E1 && $PY -I e1_bench.py profile-server-segment --freeze frozen/runtime_freeze.json --model-dir $MODEL --device cuda:<server index> --class 16 --out-dir e1_runs/stage2
```
Expected runtime: unknown until first measured run (E0's Stage 2 on an H200 took about 3 h for four cuts; two cuts
here). GPU use: server only. Safe unattended: yes. Outputs: `e1_runs/stage2/P-server-U(3,3)-load0.jsonl`,
`P-server-U(5,5)-load0.jsonl`. **No decision.** Report the fitted step (E0 `fit_component`) against E0's 66.9 /
61.3 ms and the device roofline in the stage notes.

## Stage 3: edge machine (capped) and the bound

```
cd $E1 && $PY -I e1_bench.py fit-split --freeze frozen/runtime_freeze.json --model-dir $MODEL --device cuda:<edge index> --class 12 --out-dir e1_runs/stage3
cd $E1 && $PY -I e1_bench.py fit-split ... --class 16 --out-dir e1_runs/stage3
for M in 12 16 24; do for model in Qwen3-8B-AWQ Qwen3-14B-AWQ Qwen3-32B-AWQ Qwen3-8B; do
  $PY -I e1_bench.py fit-competitor --freeze frozen/runtime_freeze.json --device cuda:<edge index> --class $M --model $model --vllm-bin <vllm> --out-dir e1_runs/stage3; done; done
cd $E1 && $PY -I e1_bench.py profile-edge-segment --freeze frozen/runtime_freeze.json --model-dir $MODEL --device cuda:<edge index> --class 12 --out-dir e1_runs/stage3
cd $E1 && $PY -I e1_bench.py profile-edge-segment ... --class 16 --out-dir e1_runs/stage3
cd $E1 && $PY -I e1_bench.py summarize-fit --freeze frozen/runtime_freeze.json --dir e1_runs/stage3
```
Then on LOCAL DESKTOP (after `rsync` of `e1_runs/stage2` and `e1_runs/stage3`):
```
cd $E1 && $PY -I e1_bench.py gate-ub --stage3-dir e1_runs/stage3 --stage2-dir e1_runs/stage2 --freeze frozen/runtime_freeze.json --roles env/roles.json --out e1_runs/gates/gate_e1_ub.json
```
Expected runtime: unknown until first measured run (E0's Stage 1 on an RTX 3090: about 6 h including 12 competitor
fits). GPU use: edge only. Safe unattended: the fits and profiles yes; the whole sequence should run under the
`st` wrapper so exit statuses are logged. Outputs: `split_fit_c<M>_k<k>.json`, `edge_fit_<model>_c<M>.json`,
`P-edge-<model>-cap<M>.jsonl`, `P-edge-U(k,k)-cap<M>.jsonl`, `f_results.json`, `exit_status.log`;
`gates/gate_e1_ub.json` + `.md` + `.e1decision.json`. Stopping rule: `gate_e1_ub.e1decision.json` says STOP →
E1-OPT ends (Remove SPLIT by bound); the gate refuses to run if any row lacks target provenance.

## Stage 4: link gate and joint cells (both machines)

```
SERVER:  cd $E1 && $PY -I e1_bench.py serve --freeze frozen/runtime_freeze.json --model-dir $MODEL --device cuda:<server index> --class 16 --port 29500 --log e1_runs/stage4/server.log.jsonl
EDGE:    cd $E1 && $PY -I e1_bench.py linkcheck --freeze frozen/runtime_freeze.json --server <SERVER IP>:29500 --out e1_runs/stage4/linkcheck.json
```
(For `linkcheck` the server may be an echo stub: `serve` with `--dev-mode` is NOT allowed here; use the real
server; its reply timing is subtracted per step.) Proceed only if `joint_allowed: true` (p50 ≤ 5 ms, ≥ 900 Mbps).
Then, in E0's block order (`deployment_e0/e0_latency.py plan`), for every SPLIT cell (the server must serve the
cell's class: U-min and U-max(12) cells → `--class 12`; U-max(16) → `--class 16`; U-max(24) only if evaluable):
```
EDGE:    cd $E1 && $PY -I e1_bench.py run-cell --freeze frozen/runtime_freeze.json --cell <id> --model-dir $MODEL --device cuda:<edge index> --server <SERVER IP>:29500 \
            --f-results e1_runs/stage3/f_results.json --out-dir e1_runs/stage4 [--shaping netem --dev <iface> | --shaping app] [--load-status http://<SERVER IP>:8099/status]
```
REMOTE cells: start E0's vLLM Qwen3-32B and load generator on the SERVER MACHINE as in E0 RUNBOOK (unchanged
commands), then on the EDGE: `run-remote-cell --cell <id> --base-url http://<SERVER IP>:8000/v1 ...`. Shaping:
E0's `e0_latency.py netem --dev <iface> --rtt-ms .. --bw-mbps .. --action apply` on the edge when permitted;
otherwise `--shaping app`. Finally on LOCAL DESKTOP:
```
cd $E1 && $PY -I e1_bench.py gate-r --lat-dir e1_runs/stage4 --stage3-dir e1_runs/stage3 --freeze frozen/runtime_freeze.json --roles env/roles.json --out-dir e1_runs/gates/r
```
Expected runtime: unknown until first measured run (E0 planned about 2 days for 22 cells). GPU use: both. Safe
unattended: one cell at a time, yes; shaping and load changes between cells are manual. Outputs: `<cell>.jsonl`
per cell, `linkcheck.json`, `server.log.jsonl`, `gates/r/profiles.json`, `R_cells.json`, `R_cells.md`,
`R_cells.e1decision.json`. Stopping rule: G-link fails → the pairing is BLOCKED (hardware result); G-R decides
retain/remove; PENDING only by E0 §8.4(a) / §8.5.

## Archive

Tar each stage with sha256 as in E0 (`e1_stage<N>_<VERDICT>.tgz`), including `frozen/runtime_freeze.json` and
`env/`. Gates recompute from the archive with E0's `e0_r.py` unchanged plus the `.e1decision.json` sidecar.

## Dev / test mode (never a decision)

`--dev-mode` on any driver: no tag required, `--host` allowed, non-frozen cuts allowed, tiny models allowed; every
row is `profile_role = screening` and the gates refuse it (or, with `--allow-screening`, write `*.screening.json`).
The CPU test suite uses it: `tests/test_e1_*.py`, `tests/test_reanalysis.py` (see README).

## Public datasheet values for reference (not lab-verified, not decision inputs)

DGX Spark / GB10: 128 GB LPDDR5X unified, 273 GB/s, Blackwell sm_121, aarch64, 10 GbE + 2 × ConnectX-7 QSFP.
With the **archived** RTX 3090 edge profiles a 273 GB/s server fails the bound in every cell; with the actual
target edge the bound is recomputed at Stage 3. Under PREREG §6.3 rule 3 a 128 GB machine is the server over an
80 GB GPU unless the lab assigns roles.
