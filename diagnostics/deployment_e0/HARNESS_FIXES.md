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
