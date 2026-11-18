# GEMMA-SELECT runbook

Three machines: **EDGE** (DGX Spark), **SERVER** (one RTX A6000), **DESK** (analysis host, no GPU).
Host-specific values (paths, host names, GPU identifier, port) are in a private site file per host, never in this
repository. `$GS` is this directory in a checkout at the tag. `$PY` is a Python with torch, CUDA and a transformers
version that has the Gemma 4 classes, the same one for every command on a host. Always `-I -B`.

Rules for every step:
- Read each command's last line. `STOP:` or a non-zero exit means stop there. Do not work around it.
- `BLOCKED` means not enough free memory now. Nothing was loaded. Wait and run the same command again later.
  Never stop, signal or ask about anyone else's job.
- Never start a second copy of a command that is running. Never delete or edit an output.

## 0. Once per host

```
git clone --branch gemma-select-prereg-v1 --depth 1 <repository> <checkout>      # HEAD is the tag
export GS=<checkout>/diagnostics/deployment_gemma_select
export PY=<python>      SITE=<private site file for this host>
```
Place the private adapter file where the site file says. Fill the site file from `site.example.json`.

Credentials: generate one pair of the experiment's own on SERVER with the split source's credential generator, into a
new directory outside the source (exact command in the private runbook). It prints two file names and no secret. Copy
the client key file to EDGE over SSH, mode 0600. Both files are deleted at cleanup.

## 1. SERVER, terminal S1

```
$PY -I -B $GS/gsel_bench.py preflight --role server --site $SITE     # expect: "preflight server: GO ... port free"
$PY -I -B $GS/gsel_bench.py profile   --role server --site $SITE     # expect: "profile_server.json written"
$PY -I -B $GS/gsel_bench.py serve     --site $SITE                   # stays in the foreground; expect a line with "ready": true
```
Leave `serve` running until step 4. Preflight hashes the 16 GB checkpoint, which takes a few minutes.

## 2. EDGE, terminal E1: the SSH forward

```
ssh -N -L 127.0.0.1:<port>:127.0.0.1:<port> <server>                 # no output; leave it running
```

## 3. EDGE, terminal E2

```
$PY -I -B $GS/gsel_bench.py preflight   --role edge --site $SITE     # expect: "preflight edge: GO ... port listening"
$PY -I -B $GS/gsel_bench.py reference   --site $SITE                 # expect: "screen decision recorded: EDGE at Stage 1"
$PY -I -B $GS/gsel_bench.py correctness --site $SITE                 # expect: "correctness: PASS ..."   (exit 2 = FAIL: stop for good)
$PY -I -B $GS/gsel_bench.py profile     --role edge --site $SITE     # expect: "profile_edge.json written"
$PY -I -B $GS/gsel_bench.py timed --site $SITE --block S1            # one line per query; "block S1 complete"
$PY -I -B $GS/gsel_bench.py timed --site $SITE --block Es
$PY -I -B $GS/gsel_bench.py timed --site $SITE --block Em
$PY -I -B $GS/gsel_bench.py timed --site $SITE --block S2
```
Run them in this order, one at a time, inside `tmux`. Each is a fresh process. The harness refuses any other order.

Stop conditions:
- `correctness: FAIL`: final. Go to step 4. No timing.
- A command dies with a Python error after its checks passed: a `failure_*.json` is written. Run the same command once
  more, later. Two failures of `reference` end the experiment (NOT_EVALUABLE). Two failures of starting or running the
  split end it (SPLIT_INFEASIBLE).
- A timed block dies: run the same block again. Its partial file is kept. After two repeats the harness refuses.

## 4. Cleanup

- SERVER S1: Ctrl-C once on `serve`. Wait for "server record written".
- EDGE E1: Ctrl-C on the SSH forward.
- Delete the two credential files. Leave everything else.

## 5. Evidence

On each host:
```
tar -C <parent of run_dir> -cf gsel_<role>.tar <run_dir name> && sha256sum gsel_<role>.tar
```
Copy both archives to DESK into a private directory. Raw outputs name hosts and devices and stay out of every
repository. The ten `reference_logits_*.pt` files (about 670 MB together) are hashed in `reference.json`.

## 6. DESK

```
$PY -I -B $GS/gsel_bench.py score   --edge-run <edge run_dir>
$PY -I -B $GS/gsel_bench.py summary --edge-run <edge run_dir> --server-run <server run_dir> --out <new directory>
```
`summary.json` and `SUMMARY.md` hold the outcome and carry no host detail.

## Time

No Gemma timing on these GPUs exists, so these are arithmetic, not predictions.
- Preflight: minutes (one SHA-256 pass over 16 GB per host).
- Reference and correctness: 10 prompts; 640 teacher-forced steps through the split, each returning 1 MB of logits.
- A full timed block is 160 queries of at most 1,024 tokens. At 20, 50 or 100 ms per token and every answer at the
  cap, that is 0.9, 2.3 or 4.6 hours. The 600 s cutoff bounds any block at 26.7 hours.
- S2 is 24 queries.
The first lines of block S1 show the real pace.

## CPU tests (any machine, no GPU)

```
<python with torch and transformers> -I -B $GS/tests/test_gsel.py
```
