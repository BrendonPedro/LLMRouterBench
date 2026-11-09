# E0 harness fixes after the pre-registration tag

The tag `e0-prereg-v1` (commit `85faaf6`) is the pre-data pre-registration and is never moved. PREREG §5.6 asks
that a Stage-1 harness defect be fixed, committed and recorded, and that Stage 1 then be re-run from scratch.
This file is that record. `PREREG.md`, `frozen/` and every protocol, measurement and runtime module
(`e0_spec.py`, `e0_memory.py`, `e0_instrument.py`, `e0_r.py`, `e0_q.py`, `e0_latency.py`, `split_runtime.py`) are
byte-identical to the tag. RUNBOOK §2 checks this before any run.

## H1. split-fit aborted at exit; exit statuses were not checked (Stage 1 run 1, 2026-11-09, INVALID)

### Observation

- **Run:** RTX 3090, commit `85faaf6`. Archive `e0_stage1_INVALID_20261109.tgz`, sha256
  `7a0d464e…8ffabf`.
- **Output:** every passing `split-fit` child printed `terminate called without an active exception`.
- **The mandatory 12 GB k = 1 run** wrote `{"pass": true, "error": null, …}` and then ended with
  `Aborted (core dumped)`. This is a non-memory failure, so Stage 1 is INVALID (PREREG §5.6).

### Root cause

It is the `split-fit` child process, in interpreter shutdown, in the daemon thread that runs the in-process
echo segment server. It is not the parent, the NVML sampler, the TCP peer process, multiprocessing or the CUDA
teardown.

1. **The thread was never joined.**
   - `split-fit` runs the echo `SegmentServer` in a daemon thread.
   - At the end it sent BYE, wrote the record and called `sys.exit`. It never waited for that thread.
   - A probe of `split-fit --cpu-dry` finds the thread still alive at `sys.exit`, inside its teardown.
2. **Leaving `handle()` frees the thread's tensors.** These are `HostBuffers.inp`/`.out` (2 × 20 MiB once a
   2,048-token prefill chunk has passed) and their numpy/memoryview views.
   - A torch tensor deallocation releases the GIL.
   - If the main thread finalises the interpreter meanwhile, CPython 3.12 makes the returning thread call
     `pthread_exit()`.
   - Its forced unwind passes through a `noexcept` pybind11 GIL guard, so libstdc++ calls `std::terminate`.
     That prints exactly `terminate called without an active exception` and raises SIGABRT.
3. **Only passing attempts are exposed.** The OOM attempts failed while loading weights, before the echo thread
   existed. That is why only passing children printed the message.
4. **`split-umax` ignored the exit status.** It called `subprocess.run(cmd)` and read only the JSON. An abort
   after the record (or a crash before it, read as FAIL) therefore went unnoticed. `summarize` had no exit
   status to check: run on the INVALID archive, the tagged code reports `valid: true`.

### Reproduced on CPU (same Python 3.12.3 and torch 2.6.0 as the pod; no CUDA needed)

| Probe | Aborts (SIGABRT, same message) |
|---|---|
| Daemon thread in a torch matmul loop at `sys.exit` | 13/13 |
| Daemon thread only freeing tensor views, with `sys.exit` while it frees | 17/17 (0/10 once the freeing has finished) |
| Daemon thread blocked in a socket call, or sleeping after torch use | 0/6 |
| `split-fit`'s exact thread layout and exit sequence at wire scale (hidden 5,120, two 2,048-token chunks, 16 decode steps), **old** shutdown | 7/20 |
| The same, with the echo thread joined (the fix) | **0/20** |
| The same layout with 1-token messages only (10 KB buffers) | 0/80 |

The small-buffer row explains why the local self-test did not catch the defect. The tiny test model's
activations are small, so the echo thread's teardown is fast. The CUDA frame itself was not observed, because
this box has no GPU: the 3090 runs had `core_pattern=|/bin/false`, and no stack was dumped. If the abort recurs,
the `faulthandler` dump below names the thread.

### Fix (`e0_f.py` only)

1. **`split-fit` stops its echo server and joins the thread** (`_stop_echo`).
   - On success, the client's BYE ends `handle()`.
   - On an error before the client connected, a throwaway connection unblocks `accept()`.
   - It then records `shutdown = {echo_thread_joined, live_threads}`. A PASS with a thread still alive becomes a
     harness error.
   - This runs **after `sampler.stop()`, the last memory sample.**
2. **`faulthandler.enable(all_threads=True)`** at the start of `split-fit`. A fatal signal now dumps every
   thread's Python stack to stderr.
3. **The exit status is recorded and audited.**
   - `run_attempt` stores each child's exit status in its record.
   - Exit 0 must go with PASS and exit 3 with FAIL. Any other status (SIGABRT is −6), or a missing record, is
     `error_kind: harness`. The measured values are kept, but are not F evidence.
   - `split-umax` stops at the first such attempt.
   - The 12 GB k = 1 run now goes through the new `split-attempt` subcommand, so its exit status is recorded the
     same way.
4. **`summarize` marks Stage 1 invalid** for any of:
   - a split-fit record without an exit status, or with one that contradicts its verdict;
   - a U-max search stopped by a defect;
   - an `edge-fit` profile subprocess that died on a signal;
   - a non-zero status in the RUNBOOK's `exit_status.log`. `run-split` may exit 1, which means failed requests;
     those fall under the F error rule in the records. A missing log is also a defect.
5. **RUNBOOK §3** runs every Stage-1 command through `st`. It logs the exit status, tees stderr to `console.log`,
   and requires an empty `$RUNS/stage1`. RUNBOOK §2 checks out the fix commit and verifies that it descends
   from the tag with the protocol and measurement code unchanged.

### Why this cannot change a memory measurement

The following are untouched: the weights loaded, KV size, L_F, chunk size, the attention implementation,
dtypes, the budget, `apply_cap`, the baseline, the sampler and its interval, and the PASS formula.

- The echo thread runs exactly as before during load, prefill and decode.
- Its join happens after the last NVML/allocator sample.
- `faulthandler` reserves a small host-memory signal stack before the baseline is taken. It never touches the
  GPU.

The only verdict change is in the conservative direction: a PASS can now become a harness defect, and never the
reverse.

### The INVALID run's records

- **Not admissible as F evidence.** Stage 1 is re-run completely into an empty directory, and all six U-max
  attempts plus U-min are re-measured.
- **Debugging evidence only.** The abort happened after each record was complete, and the measurement code is
  unchanged, so the numbers remain readable as a sanity check of the memory model.
  - k = 3, 5, 9 passed at device peaks of 11,100 / 14,850 / 22,292 MiB.
  - k = 4, 6, 10 hit genuine CUDA OOMs while loading.
  - U-max = (3,3) / (5,5) / (9,9), as predicted.
  - U-min (k = 1) peaked at 7,440 MiB.
- The re-run is not expected to differ, but its numbers are the ones that count.

## H2. edge-fit crashed cleaning up a vLLM server that had already exited (Stage 1 run 2, 2026-11-09, INVALID)

### Observation

- **Run:** RTX 3090, commit `e17492d` (H1 fix). Archive `e0_stage1_INVALID_EDGEFIT_20261109.tgz`, sha256
  `0e546bdd…3a5c1d`.
- **H1 held.** All of split-umax (12/16/24) and the k = 1 `split-attempt` exited 0, every record has the
  expected exit status, and every `shutdown` reads `{echo_thread_joined: true, live_threads: []}`.
- **edge-fit, Qwen3-8B-AWQ at 12 GB, passed.** It printed `{"pass": true}` and exited 0.
- **The next two fits crashed.** Qwen3-14B-AWQ at 12 GB and Qwen3-8B at 12 GB each ended with
  `ProcessLookupError: [Errno 3] No such process` at `os.killpg(proc.pid, signal.SIGTERM)` and exited 1.
  - They wrote no record.
  - `exit_status.log` holds both `rc=1` lines, so the H1 audit would already have marked Stage 1 invalid.
- The run was stopped during Qwen3-32B-AWQ at 12 GB, before the 16/24 GB fits, the edge-segment profiles, the
  final `summarize` and G-UB1.

### Root cause

1. **The server exits on its own when its fit fails at start-up.** Both vLLM logs end with the engine's
   `ValueError: No available memory for the cache blocks`, then `RuntimeError: Engine core initialization
   failed`. The front end then exits with status 1, and so does the engine core.
2. **`_wait_ready` sees the exit and reaps the server.** It calls `proc.poll()`, which collects the exit status
   of the group leader.
3. **The cleanup assumed a running server.** The `finally` block sent `os.killpg(proc.pid, SIGTERM)`
   unconditionally. Once every member of the server's process group has exited, the kernel has no such group,
   and `killpg` raises ESRCH (`ProcessLookupError`).
4. **The exception escaped `edge_fit`.** It skipped `sampler.stop()`, the classification and the record, and
   the process died with a traceback (exit 1).
5. **Only failing fits are exposed.** A passing fit leaves the server running until cleanup, so `killpg`
   succeeds. That is why 8B-AWQ at 12 GB, on both attempts, was unaffected.

### Was the vLLM exit expected?

Yes. Both are **memory-capacity failures** of models that are not desk-eligible at 12 GB (PREREG §3.2: only
Qwen3-8B-AWQ is). They are not engine faults.

- **Qwen3-8B (bf16).** The weights alone take 15.27 GiB, against an 11.50 GiB budget
  (`--gpu-memory-utilization 0.4792`).
- **Qwen3-14B-AWQ.** 9.36 GiB of weights leave no room for the KV cache at L_F.

vLLM's message is a KV-capacity error, which PREREG §5.6 lists as a memory failure.

### A second defect on the same path: the classification text

- **What the old code classified.** On a start-up failure, it classified `failure_kind` on the last 2,000
  characters of the vLLM log.
- **vLLM's output order.** vLLM prints the engine's exception first, then a longer front-end traceback that
  ends in `RuntimeError: Engine core initialization failed`.
- **The result.** In both Stage-1 logs the 2,000-character tail does not contain "No available memory". The old
  code would have recorded these as `fail_kind: engine`. (Checked on the archived logs.)
- **Why it matters.** At 12 GB these two models are not desk-eligible, so the mistake changes nothing there. At
  16 GB, Qwen3-14B-AWQ *is* desk-eligible, and the same misclassification would block every SPLIT pass at
  16 GB (§8.4 (b)). That would happen for a failure PREREG §5.6 defines as memory.

### Fix (`e0_f.py` only)

1. **`stop_server(proc)` replaces the unconditional `killpg`.**
   - **Status read first.** It reads the exit status with `poll()` before sending any signal.
   - **Server already exited.** It is not signalled, and its own status is kept (`exited_on_its_own: true`).
   - **Server still running.** It gets SIGTERM to its group, then SIGKILL after 120 s, as before. A
     `ProcessLookupError` from a server that exits in between is tolerated and recorded as an exit on its own.
   - **Leftover processes.** Any live (non-zombie) process left in the group, read from `/proc`, gets 30 s and
     then SIGKILL, so no orphaned engine core holds GPU memory into the next fit.
   - **The record.** Each attempt stores `server = {exited_on_its_own, stopped_by, exit_status, signal,
     stray_pids, group_clear}`.
2. **The failure text includes the server's own exception lines.**
   - When an attempt has an error, or the server exited on its own, the server's exit status (and signal) and
     its exception lines are appended to `error` (`server_exceptions`). Examples: `ValueError: No available
     memory …` and `RuntimeError: Engine core …`.
   - INFO lines such as vLLM's "Available KV cache memory" are not exception lines, so they cannot match.
   - `failure_kind` and the memory patterns are unchanged. The text is a superset of the old one, so a
     classification can only move from engine to memory, and only when the server itself raised a
     memory-pattern exception.
   - Anything unrecognised still counts as engine.
3. **Genuine failures stay failures.**
   - A start-up exit, a non-zero status or a signal death (e.g. SIGSEGV) is a FAIL. It is classified memory or
     engine by the PREREG §5.6 rule, and its status is in the record.
   - A server that exits on its own *after* serving the depth request is a **harness defect**
     (`error_kind: harness`). So is a process that survives SIGKILL.
   - On a harness defect, edge-fit writes its record and then exits non-zero.
4. **`summarize` marks Stage 1 invalid for:**
   - an edge-fit attempt without a `server` record (every pre-fix record);
   - a harness attempt;
   - a missing `edge_fit_<model>_c<M>.json`. A crashed fit must not read as "not eligible".
5. **RUNBOOK §3 checks for:**
   - `Traceback` in `console.log`, as well as the abort strings;
   - exactly 12 edge-fit records.

### Why this cannot change a memory or latency measurement

- **Unchanged:**
  - the vLLM command line and its utilisation;
  - the single pre-declared retry;
  - the depth request and the S5 profile;
  - the idle baseline, the sampler and its interval;
  - the PASS formula and the memory patterns.
- **Ordering.** The cleanup still runs after the depth request and the profile, and before `sampler.stop()`,
  as before. For a running server it sends the same signals.
- **New work.** The only new steps are a `poll()`, `/proc` reads, and a log read after the server is down.
- **The fit's NVML window.** Only when processes are left over can it now extend by up to 30 s of grace. Those
  extra samples see only memory that the server's own processes still hold while shutting down, and no new
  load.
- **Latency profiles** are taken before cleanup, so they are unaffected.
- **The classification change** affects `fail_kind` of a failed attempt only, in the direction PREREG §5.6
  already defines.

### The INVALID run's records

- **Not admissible as F evidence.** Stage 1 is re-run completely into an empty directory, starting again with
  the split-fit search.
- **Debugging evidence only.**
  - **Split-fit reproduced run 1 exactly.**
    - U-max = (3,3) / (5,5) / (9,9).
    - Peaks 11,100 / 14,850 / 22,292 MiB.
    - k = 4 / 6 / 10 failed on memory at 11,532 / 15,192 / 23,980 MiB.
    - U-min peaked at 7,440 MiB.
  - **Qwen3-8B-AWQ at 12 GB.**
    - Attempt 0 peaked at 12,313.5 MiB, over the 11,776 MiB budget.
    - The retry at utilisation 0.4473 passed at 11,492 MiB.
    - All 38 rows of its S5 profile are `ok`.
