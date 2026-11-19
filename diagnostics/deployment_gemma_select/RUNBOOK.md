# GEMMA-SELECT runbook (version 2)

Three machines: **EDGE** (DGX Spark), **SERVER** (one RTX A6000), **DESK** (analysis host, no GPU).
Host-specific values (paths, host names, GPU identifiers, port) are in a private site file per host, never in this
repository. `$GS` is this directory in a checkout at the tag. `$PY` is a Python with torch, CUDA and a transformers
version that has the Gemma 4 classes, the same one for every command on a host. Always `-I -B`.

Rules for every step:
- Read each command's last line. `STOP:` or a non-zero exit means stop there. Do not work around it.
- `BLOCKED` means not enough free memory now. Nothing was loaded. Wait and run the same command again later.
  Never stop, signal or ask about anyone else's job.
- Never start a second copy of a command that is running. Never delete or edit an output.
- The machines are shared. Every command measures free memory itself, immediately before it loads. An earlier
  reading is never reused.

## 0. Once per host

```
git clone --branch gemma-select-prereg-v2 <repository or bundle file> <checkout>      # HEAD is the tag
export GS=<checkout>/diagnostics/deployment_gemma_select
export PY=<python>      SITE=<private site file for this host>
```
Place the private adapter file where the site file says. Fill the site files from `site.example.json`. The SERVER
file starts as a template with `"gpu_uuid": null` and with `excluded_gpu_uuids` naming every GPU that carries another
service of the lab.

Credentials: generate one pair of the experiment's own on SERVER with the split source's credential generator, into a
new directory outside the source (exact command in the private runbook). It prints two file names and no secret. Copy
the client key file to EDGE over SSH, mode 0600. Both files are deleted at cleanup.

## 1. SERVER, terminal S1

```
$PY -I -B $GS/gsel_bench.py select-gpu --site <template> --out $SITE   # prints every GPU and "selected GPU index N"
$PY -I -B $GS/gsel_bench.py preflight --role server --site $SITE     # expect: "preflight server: GO ... port free"
$PY -I -B $GS/gsel_bench.py profile   --role server --site $SITE     # expect: "profile_server.json written"
$PY -I -B $GS/gsel_bench.py serve     --site $SITE                   # stays in the foreground; expect a line with "ready": true
```
`select-gpu` is run immediately before the preflight. If the selected GPU fills up before the profile has run, run
`select-gpu` again with the same arguments. Once the profile or the server has started, the GPU is fixed for the run:
if it later lacks memory, the answer is BLOCKED and waiting, never another GPU.
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

The EDGE preflight also prints a whole-model class check, made without reading a weight. It must show 0 missing
keys and 0 shape mismatches. If it shows anything else, or could not run, do not start `reference`: the edge baseline
would fail to load, and that has to be settled before any GPU stage.

Memory on EDGE. The preflight prints available and free RAM. A whole model (reference, Es, Em) needs its amount in
available RAM. A split half (correctness, profile, S1, S2) also needs its amount on the device's own count, which
follows free RAM and excludes page cache. If a split stage says BLOCKED on the device figure, wait. Do not drop
caches and do not ask anyone to free memory.

Stop conditions:
- `correctness: FAIL`: final. Go to step 4. No timing.
- `STOP: the experiment's server does not answer ...`: check terminal S1 and the forward, then run the same command
  again. It is not counted as an attempt.
- A command dies with a Python error after its checks passed, or reaches its wall-clock limit: a `failure_*.json` is
  written. Run the same command once more, later. After a split command died, wait 20 minutes first, so that the
  server has dropped the dead session. Two failures of `reference` end the experiment (NOT_EVALUABLE). Two failures
  of starting or running the split end it (SPLIT_INFEASIBLE).
- A timed block dies: run the same block again. Its partial file is kept. After two repeats the harness refuses.
- Wall-clock limits are built in: preflight and profiles 1 hour, reference and correctness 2 hours, a timed block
  600 s per query plus 1 hour. As a last resort against a hang inside a device call, start each command under
  `timeout --signal=INT --kill-after=120 <limit plus 10 minutes>`.

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
