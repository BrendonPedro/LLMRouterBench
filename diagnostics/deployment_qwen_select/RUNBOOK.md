# QWEN-SELECT runbook (public; exact host values are in the private site files and the private execution sheet)

Frozen tag `qwen-select-prereg-v1`. Everything here follows `PREREG_QWEN_SELECT.md`; if the two differ, the protocol
wins. Run every long command inside `tmux` on its host. Nothing starts, stops or signals a process that is not ours.

`PY` is a Python with CUDA torch and transformers 4.51.3 on each host (the interpreters K16-HWFIT ran under), used
read-only. `QS` is this directory in a checkout whose HEAD is the tag. Every command is `$PY -I -B $QS/qsel_bench.py ...`.

## Roles

| Host | Role | Holds |
|---|---|---|
| analysis machine | CPU only | tests, data preparation, scoring, summary |
| edge machine (DGX Spark class) | `edge` | S (edge half), E32, E14, E8, the R14 client; the clock; the tokenizer; the workload |
| server (one RTX A6000) | `server` | S (server half), R14 |

## 0. Data preparation (no GPU)

1. Clone the tag from the bundle on both hosts. Check `git rev-parse HEAD`.
2. Checkpoints: `$PY -I -B $QS/qsel_models.py fetch --model 14B --dest DIR` (edge and server) and `--model 8B` (edge).
   The download is resumable and verified against the frozen manifest. Qwen3-32B is already on both hosts.
3. Install the private site files, the private workload file (edge) and the environment file.
4. Optional: `CUDA_VISIBLE_DEVICES= $PY -I -B $QS/tests/test_qsel.py` (CPU, about 10 minutes) must end with `OK`.

## 1. Server: GPU selection and preflight (CPU only)

```
select-gpu --site TEMPLATE --out SITE
preflight  --role server --site SITE
```
`preflight` creates the REMOTE token. Copy it to the edge (mode 0600). `BLOCKED` means no eligible GPU has the
memory now: wait and repeat. Never edit the site file.

## 2. Edge: forward and preflight (CPU only)

Open one SSH forward from the edge for both ports and leave it running. Then `preflight --role edge --site SITE`.
It hashes the three checkpoints once (minutes), checks that every query tokenises to the pinned ids, and applies
the memory gates. **Both preflights must say GO before any GPU stage.**

## 3. Correctness (first GPU use)

```
edge:    reference   --site SITE          # whole 32B model; records the screen decision
server:  serve       --site SITE          # leave running; prints {"ready": true, ...}
edge:    correctness --site SITE          # PASS (exit 0) or FAIL (exit 2, final)
```

## 4. Timed blocks

Every primary configuration runs in two halves (the first and the last 80 queries), about half a day apart.
```
edge:    timed --site SITE --block S1a    # then the server exits by itself after 20 idle minutes, or Ctrl-C it once
edge:    timed --site SITE --block E32a
edge:    timed --site SITE --block E14a
edge:    timed --site SITE --block E8a
server:  serve --site SITE                # again, same site file, same GPU
edge:    timed --site SITE --block S1b    # refused (BUDGET) if the second halves cannot fit the edge's allotment
edge:    timed --site SITE --block S2     # the sentinel queries; then the server exits by itself, or Ctrl-C it once
edge:    timed --site SITE --block E32b
edge:    timed --site SITE --block E14b
edge:    timed --site SITE --block E8b
server:  serve-remote --site SITE         # after `serve` has exited
edge:    timed --site SITE --block R14
```
`ledger --role edge|server --site SITE` prints the GPU-hours used and left on a host at any time.

## 5. What to do when something stops

| Message | Meaning | Action |
|---|---|---|
| `BLOCKED` | not enough free memory now; nothing was loaded | wait, repeat the same line. Never another GPU or site file |
| `STOP: ... server does not answer` | server or forward is down; nothing was loaded | look at the server terminal and the forward, repeat. Not an attempt |
| a timed block dies | error, lost forward, wall-clock limit | repeat the same line: it resumes after the last complete row. Three attempts at most |
| `reference` or `correctness` dies | error | repeat once. A second failure ends the experiment |
| `correctness: FAIL` | the gate failed | final. Go to step 6 |
| `BUDGET` | the host's GPU-hour allotment is used up | final for that host. Go to step 6 |
| `... is complete: outputs are written once` | the stage already finished | nothing to do |

## 6. Evidence, cleanup, scoring

1. Make sure both server commands have exited (`server record written`). Close the forward.
2. On each host: `tar` the run directory, `sha256sum` it, copy both archives to the analysis machine, compare sums.
3. Delete the token files on both hosts. Leave run directories, checkouts and checkpoints.
4. Analysis machine, from a checkout of the tag:
```
score   --edge-run RUN_EDGE --workload-private FILE
summary --edge-run RUN_EDGE --server-run RUN_SERVER --out OUT
```
`score` refuses to run before the primary blocks are complete. For a run that has ended incomplete, `score --partial`
scores what exists and the summary records it. `OUT/summary.json` and `SUMMARY.md` hold the outcome and no host detail. Nothing goes into the paper before that
file exists.
