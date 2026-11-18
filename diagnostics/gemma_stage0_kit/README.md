# gemma_stage0_kit: evidence for Stage 0 of the frozen GEMMA-SPLIT protocol (2026-11-18)

**Not part of the protocol.** The protocol is `diagnostics/deployment_gemma_split` at tag `gemma-split-prereg-v1`
and this kit changes none of its files (a test checks it). Stage 0 has not passed: its last verdict was
STOP_INCOMPLETE. This kit gathers the missing facts and shows, with the frozen rules themselves, what is still owed.
It measures nothing. Gemma results are never mixed with the Qwen3 experiments.

What Stage 0 still needs (PREREG §4.4 to §4.6): `capture/split_capture.json` from code reading, `capabilities.json`,
`candidates.json` + `selected_config.json`, and `lab_answers.md` with Brian's dated written answers (Q1 to Q18) and
the log of development runs (the six API requests of 17 November).

| step | where | command | does |
|---|---|---|---|
| 1 | DGX Spark | `bash stage0_host.sh edge` | read-only evidence from `/home/brian/splitllm-edge/gemma-split`, processes, ports 18888 / 18080, GPU placement, one GET of the API's `/openapi.json` |
| 2 | Asus | `bash stage0_host.sh server` | the same for `.../experiments/distributed_cached_split_gemma` |
| 3 | analysis host | `GS=... bash stage0_desk.sh <edge hostname>` | drafts pre-filled with the stated facts (every other field still `FILL`), then the frozen `stage0` verdict on a scratch copy of the capture |
| 4 | analysis host | fill the three files from the evidence and Brian's answers, copy them into `capture/`, then the frozen `gs_bench.py split-validate`, `select`, `stage0` (RUNBOOK Stage 0) | the real verdict |

Safety: no model is loaded, nothing of Brian's is started, stopped, restarted or written, no generation request is
sent, no other user's process environment is read, and every output passes the protocol's own redaction and
credential scan (a file that still looks like it holds a credential is deleted, not kept).

Two facts the stated setup does not change:
- The frozen protocol never measures Brian's running services. It runs its OWN instance of his code from a
  hash-verified snapshot on `transport.experiment_endpoint`, a port other than 18080 and 18888, in windows the lab
  grants (Q12, Q18). The live endpoints are recorded as facts only.
- The split-versus-unsplit quality comparison is Stage 3 and the latency is Stage 5. Both come after Stage 0
  CONTINUE, commit A0, the adapter (written against Brian's code) and the freeze. Stage 2 is the memory gate for the
  unsplit reference and reports BLOCKED_CONTENTION rather than loading beside a job that leaves too little memory.

Tests: `python3 -I -B tests/test_gs0_kit.py` (standard library; 4 tests: frozen directory untouched, collection is
read-only and redacts, drafts never validate, the status preview equals the frozen verdict and leaves the capture alone).
