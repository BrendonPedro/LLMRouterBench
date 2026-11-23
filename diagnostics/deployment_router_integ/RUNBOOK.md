# ROUTER-INTEG run sheet (public form: every host-specific value is a placeholder)

Protocol `router-integ-v1`. Nothing below starts, stops, signals or reads a service that is not ours. No GPU is used
before step 5. Every long command runs inside `tmux` on its host. The private run sheet holds the real paths.

Variables: `$PY` the host's interpreter (torch, transformers 4.51.3; never `pip`); `$RI` this directory on the host;
`$SITE` the private site file; `$TPL` the server's site template; `$QSEL` the clone of the reused runtime at tag
`qwen-select-prereg-v1`. `rint` = `timeout --signal=INT --kill-after=300 <seconds> $PY -I -B $RI/rint_run.py`.

| Step | Host | GPU held | Planned | Hard limit |
|---|---|---|---|---|
| 1 deploy, verify | both | none | 15 min | n/a |
| 2 checkpoints | both | none | 1 to 3 h (network), resumable | n/a |
| 3 self-test | both | none | 3 min each | n/a |
| 4 select-gpu, preflights | both | none | 20 min | n/a |
| 5 profiling: serve, both lanes | both | GB10 about 31 GiB; A6000 about 31 GiB | 1 h | 2.5 h each |
| 6 shared set: serve, both lanes at once | both | same | 16 h | edge 22 h, server 16 h |
| 7 router timing | edge | none | 2 min | n/a |
| 8 live B: serve, pipeline | server (edge: CPU) | A6000 | 5 h | 8 h |
| 9 live A: pipeline | edge | GB10 | 8 h | 11.5 h |
| 10 evidence, backup, cleanup | both | none | 15 min | n/a |
| 11 score, evaluate | analysis machine | none | 10 min | n/a |

## 1. Deploy (no GPU)
The code and protocol of the run are the Git tag `router-integ-prereg-v1`. Copy the deployment set (a bundle of the tag
and the private files) to each host, check `sha256sum -c SHA256SUMS`, then on each host:

    git clone --branch router-integ-prereg-v1 <deploy>/rint_v1.bundle <clone>
    git -C <clone> rev-parse HEAD                    # the tag's commit, as printed when the tag was made
    git -C <clone> status --short                    # nothing
    RI=<clone>/diagnostics/deployment_router_integ
    $PY -I -B $RI/tools/rint_freeze.py check         # "changed": []
    git -C $QSEL rev-parse HEAD                      # 4484a872d1007f8cf40a579aad14440dcbd75ad0
    git -C $QSEL status --short                      # nothing

`$QSEL` stays the separate clone of the reused runtime at its own tag: the new clone is used only for `$RI`.

## 2. Checkpoints (no GPU; resumable: repeat the same line after an interruption)
    $PY -I -B $RI/rint_models.py fetch --model DS --dest <models>/DeepSeek-R1-0528-Qwen3-8B     # 16.4 GB, both hosts
    $PY -I -B $RI/rint_models.py fetch --model QW --dest <models>/Qwen3-8B                      # 16.4 GB, where it is not yet present
Each ends with `"ok": true`. Where Qwen3-8B is already present from QWEN-SELECT, `verify --model QW --dir ...` instead.

## 3. Self-test (CPU only)
    CUDA_VISIBLE_DEVICES= RINT_QSEL_REPO=$QSEL RINT_TEST_ENCODER_DIR=<encoder dir> $PY -I -B $RI/tests/test_rint.py
Must end with `OK`. Tests that need files a host does not have are skipped and say so. Send any failure before going on.

## 4. GPU selection and preflights (CPU only)
Server:
    rint 900  select-gpu --site $TPL --out $SITE
    export CUDA_VISIBLE_DEVICES=<the "export" value select-gpu printed>
    rint 7500 preflight --role server --site $SITE
Edge (the SSH forward of the wire port to the server's loopback must be up, in its own tmux window):
    <copy the token file from the server, mode 600>
    rint 7500 preflight --role edge --site $SITE
Both must print `"ok": true`. Edge: `tokenisation.n_mismatches` 0, `router.choices_differing` 0.
`BLOCKED` or `"ok_now": false`: not enough free memory now. Wait and repeat. Never edit a site file, never ask anyone
to free memory. Once a server stage has started, the GPU is fixed.

**Everything from here uses a GPU.**

## 5. Profiling (6 queries, never analysed as results)
Server:  `rint 9300 serve-remote --site $SITE --stage profile`   (wait for `"ready": true`)
Edge, two tmux windows:
    rint 9300 shared --site $SITE --lane edge   --phase profile
    rint 9300 shared --site $SITE --lane remote --phase profile
Median time per output token of each lane (the profiling rule is applied by hand; the code does not enforce it):

    $PY -I -B -c "import json,sys,statistics as st; [print(f, round(st.median(r['per_token_s'] for r in map(json.loads, open(f)) if r['phase']=='profile' and r['status']=='ok' and r['per_token_s']), 4)) for f in sys.argv[1:]]" $RUN/rows_profile_edge.jsonl $RUN/rows_profile_remote.jsonl

If it is above 0.145 s on the edge lane or above 0.086 s on the remote lane (1.5 times the planning values), the plan
of the protocol (section 12) does not hold: stop and report before step 6. A server slower than the edge is not a stop:
the placement rule is fixed (CLARIFICATION_1.md, section 3). Nothing else is decided from these rows. The server exits by itself 30 minutes after its
last connection.

## 6. Shared measurement set (both lanes at the same time)
Server:  `rint 58000 serve-remote --site $SITE --stage shared`   (wait for `"ready": true`)
Edge, two tmux windows, started within a few minutes of each other:
    rint 79500 shared --site $SITE --lane edge
    rint 56500 shared --site $SITE --lane remote
Each prints one line per request and ends with `"ended": "complete"` and `[exit 0]`.
- A lane that died (error, lost forward, limit): repeat its line. It resumes after the last settled row. Three attempts
  in all, and one wall-clock limit per stage over all its attempts. `not an attempt`: the server or the forward is
  not up; bring it up and repeat.
- `the server or the wire was lost`: check the server and the forward windows, then repeat the remote lane's line. The
  request that was lost is repeated; its failed row is kept.
- `BLOCKED`: wait and repeat. `STOP: ... requests in a row ended in an error`: send the last rows before repeating.
- Server cannot hold both models (BLOCKED twice, two hours apart): the fallback of the protocol, section 13:
  `select-gpu --models DS`, `serve-remote --models DS`, `shared --lane remote --models DS`, then the same with `QW`.
- `rint 60 ledger --role edge --site $SITE` shows the GPU-hours used.

## 7. Router timing (edge, CPU only, after the edge lane has ended)
    rint 1800 router-timing --site $SITE
Expected: `"n": 120`, `"choices_agreeing": 120`.

## 8. Live B (policy B: the router on the edge, execution on the server)
Server:  `rint 29500 serve-remote --site $SITE --stage live_B`   (wait for `"ready": true`)
Edge:    `rint 27500 live --site $SITE --policy B`
`needed the fallback`: the server is down. Bring it up (`serve-remote --stage live_B`) and repeat the edge line.
If time is short, the pre-declared reduction is `--reduced` (the first 60 queries), for live A first.

## 9. Live A (policy A: everything on the edge)
Edge:    `rint 42000 live --site $SITE --policy A`

## 10. Evidence and backup (no GPU)
On each host: `tar -C <runs> -cf rint_<role>.tar run_<role> && sha256sum rint_<role>.tar > rint_<role>.tar.sha256`.
Copy both archives and their checksum files to the analysis machine, verify with `sha256sum -c`, extract, and keep
the archives. Nothing is deleted on a host until the checksums have been verified on the analysis machine.

## 11. Scoring and evaluation (analysis machine)
    <study venv>/python -B rint_score.py self-test --private <private workload> --repo <router study repository>     # 252 of 252
    <study venv>/python -B rint_score.py score --edge-run <run_edge> --private <private workload> --repo <router study repository> --out <out>/scores.jsonl
    python3 -I -B rint_eval.py --edge-run <run_edge> --scores <out>/scores.jsonl --out-dir <out>/summary
The self-test must reproduce every stored score before any new output is scored.

## Stop conditions, in one place
Any `STOP:` line. A checksum that is not `OK`. A self-test failure. A preflight that is not `"ok": true`. Position
scaling, checkpoint, token ids or protocol revision refused by a command. An exhausted allotment (`BUDGET`). A lane's
third failed attempt. After any of these: keep everything, run step 10, report.
