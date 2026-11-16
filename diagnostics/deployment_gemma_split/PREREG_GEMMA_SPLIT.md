# GEMMA-SPLIT: the two-host split with the existing Gemma deployment. Pre-registration DRAFT, revision 2 (2026-11-16)

Status: **DRAFT, NOT FROZEN.** Nothing has run on a lab machine; no model has been downloaded or executed on a GPU
for this experiment. The protocol is frozen when the PI has reviewed this file and the code it names and tags the
commit `gemma-split-prereg-v1`. Until that tag exists, no document may call GEMMA-SPLIT "pre-registered".
Revision 2 incorporates four rounds of an independent audit (leakage, post-result changes, stop-rule gaps, coupling)
and a final pre-tag audit with an independent review (configuration lock, total repeat caps, outputs bound to their
logged run, contention during memory probes and correctness runs, comparator availability, timing, credentials,
non-interference with the live deployment); every finding is addressed below and covered by a CPU test (§20); what
software cannot prevent is stated in §21.

## 0. Relation to the other deployment experiments

- GEMMA-SPLIT is a **second, independent** experiment: an infrastructure / model-generalisation check of the
  two-host split idea with the Gemma setup that already exists in the lab. It is **not** a replacement, re-run or
  extension of the Qwen3 experiments. The frozen Qwen3 reference stays exactly as it is: E1-OPT, tag
  `e1opt-prereg-v1` -> `d9ba549f006bd18f960b385953acbce8259b9dff` (and E0 before it).
- No file under `diagnostics/deployment_e0/` or `diagnostics/deployment_e1opt/` is imported, read, hashed or
  modified by GEMMA-SPLIT code. The E1-OPT freeze checker only inspects its own directory, so commits here do not
  touch its freeze. `tests/test_gs_protocol.py` checks that the E1-OPT directory is byte-identical to its tag and
  that no GEMMA-SPLIT code refers to the E0 / E1-OPT directories or modules.
- Gemma results are never pooled with, compared with, or reported as a continuation of Qwen3 numbers (§17).
- One deliberate data reuse: the prompt **texts** of the workload were copied once, at preparation, from the
  repository's S1 sample (`diagnostics/deployment_s1/sample_200.jsonl`, sha256 `2c4df493…6955`): key, dataset,
  prompt, prompt hash. No score, answer or model output was copied. The copied files (`workload/*.json`) are the
  workload; the sample is not read by any stage.

## 1. Question, definitions, non-goals

### 1.1 Question
> Can the same two-host split-LLM idea run correctly and efficiently with the existing Gemma setup on the DGX Spark
> client/edge and the main lab server backend?

### 1.2 Four separate questions (none compensates for another)
1. **Correctness**: does the deployed split compute the same function as the unsplit checkpoint, within the
   cross-hardware numerical floor (Stage 3, §9)?
2. **Memory feasibility**: does each host's part of the frozen configuration load and complete the protocol's
   longest request (Stage 2, §8)?
3. **Component runtime**: how long do the edge part and the server part take per prefill and per decode step (Stage
   4, §10)? Descriptive.
4. **End-to-end runtime**: what are TTFT and TPOT of the split over the lab link, relative to running the whole
   checkpoint on the edge alone and on the server alone with the reference runtime, on the same machines (Stage 5,
   §11)?

"Efficiently" is answered only by the pre-stated Stage-5 primary endpoint (§11.4), and only if correctness PASSED
and memory FITS. A fast runtime never offsets a failed correctness gate: the drivers of Stages 4 and 5 refuse to run
unless the previous gate opened (`gs_gate.require`), and no runtime number of a non-equivalent split is reported.

### 1.3 System under test
The existing Gemma split deployment (Brian's implementation on both machines) in **one** configuration, captured at
Stage 0 and frozen at Stage 1 exactly as found (§4.5). Its split structure is **recorded, not assumed and not
changed**. If it differs from the Qwen3 first-k / last-k edge + middle-server design, the difference is listed (§4.4,
§17) and the experiment evaluates the deployment as it is.

### 1.4 Non-goals
No quality or utility gate and no S1 scores; no emulated edge memory budget; no search over cuts, dtypes, kernels,
batch sizes, transports or any other setting (no hyperparameter sweep); no tuning or repair of the deployment; no
network shaping; no Llama run; no comparison with Qwen3; no claim about other models or machines.

## 2. Stage sequence and freeze points

| Stage | Where | What | Timers? | Verdicts |
|---|---|---|---|---|
| T tag | repository | PI review, then `git tag gemma-split-prereg-v1` | — | nothing below runs before the tag |
| 0 capture | DGX Spark, server, analysis host | inventory, roles, checkpoint identity, deployment capture, split capture, **capability declaration**, configuration selection (§3-§5) | no | CONTINUE / STOP_* (§4.6) |
| 1 freeze | each host + analysis host | commit A0 (capture/ with the capability declaration, no adapter yet), tokenize workload, snapshot the deployment source, write the adapter (contract checked with `adapter-smoke`, timer-free), commit A (adapter/ + Stage-1 capture), freeze record, commit B (frozen/), `check-freeze` and `verify-workload` on both hosts (§5-§7) | no | record exists and checks |
| 2 memory | both hosts | split probe + server record; unsplit reference probes (§8) | no | CONTINUE / STOP_INFEASIBLE / INFEASIBLE_UNCONFIRMED / BLOCKED_CONTENTION / INVALID / INCOMPLETE |
| 3 correctness | server, edge, analysis host | references, hardware floor, split runs, boundary audit, decision (§9) | **no** | PASS / FAIL (STOP) / NOT_EVALUABLE (STOP) / INVALID / INCOMPLETE |
| 4 components | each host | edge and server segment profiles (§10) | yes | COMPLETE / INVALID / INCOMPLETE |
| 5 end-to-end | both hosts | split_A, edge_only, server_only, [native blocks], split_B (§11) | yes | COMPLETE / UNSTABLE / INVALID / INCOMPLETE |
| 6 archive | each host + analysis host | checkpoint re-hash, outcome, reproducible archive + hashes (§15) | — | — |

## 3. Roles (fixed by the PI; verified, never derived)

- **Edge / client = the DGX Spark. Server / backend = the main lab server `asus-ESC4000-E11` (3 x RTX A6000 48 GB).**
- Stage 0 verifies identity markers (`gs_inventory.verify_role`): server hostname `asus-ESC4000-E11`, exactly 3
  GPUs whose name contains "RTX A6000", each >= 45,000 MiB; edge architecture `aarch64` with a GPU whose name contains
  "GB10" (public DGX Spark facts used as markers only) and a hostname other than the server's; torch with CUDA on
  both; a route to the peer recorded on both. The DGX hostname is captured, not assumed.
- A failed marker is **STOP_ROLE_MISMATCH**: the operator records the facts and the PI decides whether the machine
  is the intended one. Roles are never swapped by the harness; using a different machine is a protocol amendment.
- Which server GPU(s) the deployment uses is part of the captured configuration (`gpu_placement`: devices, GPU UUIDs
  and `CUDA_VISIBLE_DEVICES`); it is not chosen by this protocol. Stage 0 checks that every captured UUID is a GPU of
  the verified host. Before anything is loaded, every command checks `CUDA_VISIBLE_DEVICES` and that each frozen
  device carries its frozen UUID; after loading, it checks that the adapter uses only the frozen devices; the gates
  re-check the devices recorded in every output (`gs_bench.preflight`, `gs_rules.devices_ok`).

## 4. Stage 0: capture (facts only; no model execution, no GPU kernel, no timer)

### 4.1 Machine inventory (`gs_bench.py inventory --role R --peer P`)
Hostname, FQDN, `uname`, architecture, OS release, `/etc/dgx-release`, CPU, `/proc/meminfo`; every GPU from
`nvidia-smi` (index, name, UUID, memory total/used, driver, compute capability, PCI bus, power limit; unified-memory
values that report `[N/A]` stay null), `nvidia-smi -q` and topology; CUDA toolkit; Python (version, executable,
prefix), full `pip freeze`, versions of torch, transformers, accelerate, safetensors, tokenizers, flash-attn, vLLM
and the other packages in `gs_inventory.PACKAGES` (from metadata, without importing them); torch CUDA facts and
devices; NICs with link speed and MTU, addresses, the peer's resolved address and the route to it (lookup only, no
traffic); listening TCP sockets; clock synchronisation (`timedatectl`, `chronyc`); current foreign compute
processes. Environment variables of the families in `gs_inventory.ENV_PREFIXES` / `ENV_CONTAINS`; values whose
**name** looks secret are replaced by `<redacted>`, and command lines, URLs and `pip freeze` are passed through
`gs_common.redact_text`.

**Credentials (all stages).** GEMMA-SPLIT never requires, records or commits a credential: no API key, token, password,
private key or other secret may appear in `capture/` (including `lab_answers.md` and every `*.json`), `adapter/`,
`runs/`, a freeze record, git history or the archive. If the deployment authenticates, the split capture's
`authentication` section records only the mechanism, the NAMES of the environment variables / configuration fields that
carry the credential, and whether it is supplied out of band (Q17); the adapter reads the value at run time from that
variable and never writes, logs or returns it. `gs_common.secret_findings` (URLs with credentials, private-key blocks,
Hugging Face / API / AWS tokens, authorization / API-key headers, credential command-line options, `sshpass -p`,
secret-named assignments with credential-like values) is run
over the split capture by `split-validate`, over `capture/` by the Stage-0 verdict (before commit A0), by
`gs_bench.py scan-secrets` before every commit, over every file of `capture/` and `adapter/` by the freeze, and over
every lab-data member (`capture/`, `adapter/`, `frozen/`, `runs/`, snapshot) by the archive; any finding blocks the step
and names only the file, line and rule, never the value. Every string the harness writes into `runs/` (exception
messages included) passes through `redact_text` first.

### 4.2 Model / checkpoint identity (`gs_bench.py checkpoint`)
The identity of the model is the SHA-256 manifest (path, size, sha256 of every file; `.git/` and `.cache/` excluded)
of the directory each host uses. A Hugging Face repository and revision are recorded when the directory reveals them
(cache snapshot path or download metadata) and are otherwise null: never assumed. `config.json` is summarised and
these split-relevant features are derived: per-layer input embeddings (PLE), KV sharing between layers (with the
consumer -> source map as transformers' Gemma 4 modules define it), mixed sliding/full attention, logit softcapping,
tied embeddings (null if absent from the config), MoE, multimodal towers, quantisation. `generation_config.json`,
the tokenizer files and the chat template are hashed. Rules:
- The server checkpoint directory must be named `gemma-4-E4B-it` or `split_capture.model.checkpoint_note` must
  explain the identity; else STOP_CHECKPOINT_MISMATCH.
- The edge's deployment files must be byte-identical to the server's (`identical`, or `edge_subset` if the edge holds
  only the files it needs); else STOP_CHECKPOINT_MISMATCH.
- The edge reference and the edge-only baseline need the full checkpoint on the DGX Spark: either the edge's own
  directory is `identical`, or a hash-verified full copy is captured as `edge/checkpoint_reference.json` and is
  identical to the server's manifest; else STOP_CHECKPOINT_MISMATCH.
- Before every command that loads the deployment or a reference, sizes and mtimes are re-checked; a difference
  triggers a full re-hash; a hash difference stops the command (nothing written). At Stage 6 every checkpoint
  directory used is fully re-hashed on its host (`gs_bench.py verify-checkpoint`); the outcome requires both results,
  and a difference makes every stage of the run INVALID.

### 4.3 Deployment capture (`gs_bench.py deployment`)
Per host: each implementation source root (git commit, branch, dirty state, porcelain status, redacted remotes and a
SHA-256 manifest of tracked files plus untracked source-like files, all within the source root; without git, a
manifest of source-like files), the launch scripts and config files the operator names (hash + redacted text), related
processes (redacted), listening sockets and containers. Nothing of Brian's is modified. At Stage 1 `gs_bench.py
snapshot` copies the manifest files to a run-local directory outside Brian's tree, verifies every hash and records the
snapshot (`capture/<role>/snapshot.json`); the adapter runs the snapshot, and every command re-verifies the snapshot
against the capture before loading (a changed, missing or added file stops it). In a target run an import hook refuses
any module that would be loaded from the live deployment tree, and an audit hook (PEP 578) refuses opening, executing or
spawning anything there (including `exec(open(...))`, `spec_from_file_location`, `runpy`, `subprocess`, `os.system`,
`chdir`); the hash-verified checkpoint directories, the snapshot and the running interpreter's own environment (a venv
may sit inside the source root) stay allowed, and launch files outside the source roots are guarded like the tree, so
everything the adapter runs must lie under a captured, snapshotted source root (name it at Stage 0). The freeze refuses
an adapter file that names a live source root. A later edit of the live deployment therefore cannot leak into a measurement (a child process
that reads the live tree by itself is the remaining gap, §21).

**Non-interference with the live deployment.** GEMMA-SPLIT never overwrites, restarts, stops, reconfigures, `chmod`s or
writes into Brian's files, processes or configuration: capture is read-only, the snapshot is written outside his tree
(refused inside it), and the experiment runs its OWN instance of his code from the snapshot, whose server half listens on
`transport.experiment_endpoint` (same protocol and serialisation, a port his running deployment does not use) on the GPUs
and in the windows the lab grants (Q12, Q18). If his running deployment occupies an experiment GPU, the stage waits
(BLOCKED_CONTENTION); only Brian may stop his processes, and he does so himself, recorded with the date in
`lab_answers.md`. The harness runs under the experiment's own account or with `HF_HOME`, `TORCH_HOME`,
`TRITON_CACHE_DIR` and `XDG_CACHE_HOME` pointing to experiment-owned directories (part of the frozen execution
environment, compared before every command), so that no cache is written into his home directory. The capture's git
queries are read-only (`git --no-optional-locks`: not even the index of his repository is rewritten).

### 4.4 Split capture (`capture/split_capture.json`; template `capture/split_capture.template.json`)
Filled by the operator from the code, the launch files, the running processes and Brian's dated written answers
(`capture/LAB_QUESTIONS.md` -> `capture/lab_answers.md`); validated by `gs_bench.py split-validate`
(`gs_split.validate`) against the captured `config.json`. Recorded fields: configuration id; implementation and entry
points; layer count; dtype per host; quantisation; attention implementation; ownership per host of layer ranges, the
token embedding, PLE, final norm, LM head, sampling, tokenizer; every tensor crossing the boundary per direction (name,
dtype, shape, when); KV cache per host, shared-KV handling, prefix caching; transport, serialisation, endpoint; sampling
location, greedy / streaming / ignore-EOS support; GPU placement with UUIDs and `CUDA_VISIBLE_DEVICES`; launch commands;
the experiment's own endpoint (`transport.experiment_endpoint`, §4.3; its port must differ from the live endpoint's,
whatever the host is called and with a URL's default port counted, so the edge can never reach Brian's running server); authentication (mechanism, credential NAMES, supplied out of band: §4.1);
tuning history (required; context only); unknowns. On the multi-GPU server, `gpu_placement.server.CUDA_VISIBLE_DEVICES`
must pin the frozen GPU(s), so that the deployment can load on no other device (§8). Every template value is a `FILL` placeholder, booleans included, so nothing can
be left at a default unexamined. Validation problems (each blocks Stage 0): any placeholder left; a non-boolean where a
boolean is required; a layer owned by no host or by both; a host with no layer; a part with no owner; a part on both
hosts without a recorded reason; a boundary without at least one tensor per direction; PLE in the config while the
server owns layers but the boundary carries neither per-layer inputs nor token ids for a server-side PLE table;
KV-sharing source and consumer layers on different hosts with no KV on the boundary and no `shared_kv_handling`; a
missing dtype, device list, UUID per device, protocol, endpoint, experiment endpoint, generation flag or KV policy; an
authentication section that is not mechanism + names + a boolean; credential-like content anywhere in the capture;
greedy unsupported; any remaining unknown. Derived and recorded (`gs_split.derive`): topology (`prefix_edge_suffix_server`, `u_shape`,
`prefix_server_suffix_edge`, `other`), layer counts per host, whether token ids, per-layer inputs, KV entries or logits
cross, whether generated tokens are visible to the server, whether the server holds the token-embedding table, and the
list of **differences from the Qwen3 E1-OPT design** (§17). Differences are recorded, never corrected.

### 4.5 Selecting one configuration (`gs_bench.py select`; deterministic, performance-blind)
Every Gemma configuration found (code defaults, launch scripts, running processes, Brian's statement) is listed in
`capture/candidates.json`. R0 eligibility: no placeholder left; model directory `gemma-4-E4B-it`; edge host = the
verified DGX Spark; server host = `asus-ESC4000-E11`; a designation must cite a dated written answer. Llama and other
checkpoints are listed and excluded. Then, each rule narrowing the set only if some candidate satisfies it:
R1 the implementer's dated written designation of the **current** configuration for this pair;
R2 the configuration of the deployment processes running at capture;
R3 the implementation's default (launch defaults, no overrides);
R4 the smallest configuration id in byte order.
No measurement exists when the rule runs and none may be consulted; Brian is asked for the current configuration, not
the best one; a candidate that carries a measurement-like field (latency, throughput, memory, score, ...) is excluded.
If Brian chose his configuration from his own speed or memory measurements, that is part of the existing system: it is
recorded in `split_capture.json` (`tuning_history`) and `lab_answers.md` (Q4), never an input of the rule, and never
repeated. GEMMA-SPLIT never chooses between configurations. `candidates.json` and `selected_config.json` are committed in
commit A0 together with the capability declaration, before any adapter file exists, and never change afterwards (checked
in git history), so nothing run through an adapter (`adapter-smoke`) can influence the choice; once the first freeze
record exists the selected configuration can never change (§6.4).

### 4.6 Stage-0 verdicts (`gs_bench.py stage0`)
CONTINUE iff all capture files exist (including `capabilities.json`, §5.3), roles are verified, both checkpoints and
deployments were captured on the right hosts, the checkpoint rules of §4.2 hold, the split capture validates, every
captured GPU UUID belongs to the verified host, the capability declaration is complete, and the selection is CONTINUE
and names the captured configuration. Otherwise exactly one of STOP_INCOMPLETE (missing file, placeholders or
unknowns), STOP_ROLE_MISMATCH, STOP_CHECKPOINT_MISMATCH, STOP_SPLIT_CAPTURE_INVALID, STOP_NO_CONFIGURATION. A STOP is
fixed by recording the missing or corrected **facts**, never by changing the deployment.

## 5. The adapter contract (`gs_adapter.py`)

### 5.1 Why
The deployment's code is unknown when this protocol is tagged. The harness therefore drives it through an adapter
written after Stage 0 against a fixed contract, hashed in the freeze record, and frozen before any correctness or
timing run. `tests/doubles/gemma4_split_double.py` is a worked example (a test double).

### 5.2 Scope
Permitted: call the deployment's own code from the snapshot; use the generation path the captured launch commands use
(another path only if that one cannot be driven, documented before the freeze); expose the contract; force greedy
decoding and the frozen token ids; listen on / connect to the experiment endpoint; read a credential from its named
environment variable. Required: `make_adapter()` loads no weights and allocates no device memory (loading happens in
`connect()` / `serve()`, inside the Stage-2 probes). Forbidden: changing the checkpoint, dtype, layer ownership, cut, kernels, KV policy,
transport, serialisation or any computation; adding timers; caching across requests beyond the deployment's own.
`tokens` should include the EOS id that stopped generation; the rules strip one trailing EOS from both sequences before
comparing, so either convention is scored the same (§9.4).

### 5.3 Capabilities, declared at Stage 0
Required: `greedy_tokens`, `stream_tokens` (the edge calls `on_token(i, id)` when token i is available there),
`boundary_audit` (names, dtypes, shapes, bytes of every message of the last request). Optional: `teacher_forced_logprobs`
(tier T1, §9.4), `ignore_eos`, `segment_step` (isolated segment profile), `step_trace` (in-path decomposition),
`unsplit_same_runtime` (secondary runtime-matched baselines), `in_host_split` (a diagnostic). The capabilities the
deployment's code supports are **declared at Stage 0**, before any adapter is written or run, in
`capture/capabilities.json` (from code reading and Brian's written answer to Q11, with the basis recorded), and committed
in commit A0, which contains no adapter file; the freeze checks in git history that the declaration was first committed
without any adapter file and has not changed since.
`teacher_forced_logprobs` is mandatory whenever the split path computes logits the adapter can reach without changing
the computation; its absence needs a written justification. The adapter's `ADAPTER_INFO` must equal the declaration;
the freeze refuses any difference. The capabilities decide only the correctness tier, the component-profile mode and
whether secondary runtime-matched blocks are measured; the primary comparator never depends on them (§11.2), and the
in-host pipeline is never a floor (§9.2).

### 5.4 Development before the freeze; dev mode
Writing the adapter may require running the deployment. The only harness command that runs the deployment's adapter
before the freeze is `gs_bench.py adapter-smoke`: one short greedy request through the split, checking types and the
boundary audit; it calls no timer, compares nothing with a reference, prints no token ids and writes nothing; it refuses
a test double, an adapter whose capabilities differ from the declaration, and a declaration not committed first (A0),
and it runs behind the live-tree guard (§4.3); `adapter-smoke --dev` runs test doubles only, on models below the dev
size limit. `--dev` mode (tests, dry runs) accepts only dev freeze records, which accept only test-double adapters, and
refuses any model larger than `gs_spec.DEV_MAX_MODEL_BYTES` (512 MiB): the harness cannot measure the captured
deployment or the captured checkpoint before the freeze. Any development run is logged with date and purpose in `capture/lab_answers.md`.

### 5.5 Test doubles
An adapter with `test_double: true` (and any run with `--dev`) produces `profile_role = "dev"` outputs and dev-mode
gates that no target command accepts, and the freeze refuses such an adapter for a target record.

## 6. Freeze record (`gs_freeze.py`)

### 6.1 Order
Tag -> Stage 0 -> commit A0 (`capture/` with the capability declaration; `adapter/` still empty) -> Stage 1 capture and
the adapter -> commit A (only `capture/` and `adapter/` changed since the tag) -> `gs_bench.py freeze` writes
`frozen/freeze_record.json` from commit A -> commit B (only `frozen/`) -> `check-freeze` and `verify-workload` on both
hosts. No GPU executes GEMMA-SPLIT measurement code before the record exists. Records have fixed names
(`frozen/freeze_record.json`, then `frozen/freeze_record.rN.json`); the writer refuses any other name.

### 6.2 Contents
Experiment, record / protocol / decision-rule / adapter-contract versions; revision and the record it supersedes;
git commit, tag and tag commit; SHA-256 of **every file of the experiment directory** except `runs/`, `results/` and
`frozen/` (so a stray module or data file is "not covered"; bytecode and symlinks are refused outright, because bytecode
runs instead of its source even under `-B` and a symlinked directory would escape the hash set); harness files changed since the tag and their
documented fixes; the role record; per role: host, devices, strict versions, execution environment, inventory hash;
per role the checkpoint manifest hash, revision and the comparison; the config summary, features, KV-sharing map and
generation config; per role the deployment source manifests and snapshot; the full split capture, its hash, the derived
properties and the selection record; the adapter path, hash, `ADAPTER_INFO`, tier and the Stage-0 declaration; the
tokenized workload hash, ids hash, longest item, BOS and EOS ids; every protocol parameter of `gs_spec`.
`freeze_sha256` = SHA-256 of the canonical record without itself and `run_id`; `run_id` = its first 12 hex digits.

### 6.3 Checker (every Stage 1-6 command runs it first; exit 2, nothing written, on any problem)
Record hash; versions; the record kind matches the mode (a dev record never drives a target command and vice versa);
the record is the newest of its kind (a superseded record drives nothing); every hashed file unchanged and no
uncovered file; the **executing** harness modules are the frozen ones (a target command must run from the frozen
checkout; `--gs-dir` is dev-only); the adapter is not a test double; the tag exists and still points at the recorded
tag commit; every decision file (`gs_spec.DECISION_FILES`: this PREREG, `gs_spec.py`, `gs_rules.py`, `gs_stats.py`,
`gs_gate.py`, `gs_split.py`, `gs_results.py`, `gs_freeze.py`, `gs_adapter.py`, `gs_common.py`, `gs_checkpoint.py`,
`gs_workload.py`, the measurement drivers `gs_reference.py`, `gs_correctness.py`, `gs_memory.py`, `gs_profile.py`, and
`workload/*.json`) is byte-identical to the tag; only the orchestration files (`gs_bench.py`, `gs_inventory.py`,
`gs_deployment.py`, `gs_archive.py`), the tests and the adapter may differ from the tag, each with an exact entry
`path | fix commit | regression test | reason` in `frozen/HARNESS_FIXES.md`; HEAD equals the frozen commit or differs
from it only under `frozen/`, `runs/`, `results/`; the record file is committed unchanged under its fixed name; no
record-shaped file exists under another name; no tracked record or run output is missing and git history shows no
deletion or rename of one (other than a move into `_invalidated/`) and no modification of one, in history or in the
working tree (outputs, gates and records are written once; the exit-status and invalidation logs may only grow); no bytecode or symlink exists in the directory; this
machine is the frozen host for the requested role, its python, torch, torch CUDA, transformers, accelerate,
safetensors, tokenizers, flash-attn, vLLM and driver versions and its execution environment (`CUDA_VISIBLE_DEVICES`,
allocator, thread, library-path, preload and Python-path variables: `gs_freeze.EXEC_ENV_*`) equal the frozen ones. The
freeze also refuses an inventory taken with another `CUDA_VISIBLE_DEVICES` than the deployment uses. Device, endpoint,
model and directory overrides exist only in `--dev` mode.

### 6.4 After the freeze: a new record only in three cases
A defect is fixed in a new commit with a CPU regression test that fails before the fix and an entry in
`frozen/HARNESS_FIXES.md`. A new record (`frozen/freeze_record.rN.json`, `--supersede-reason` required) is accepted only
if no run of any earlier record has a final verdict, at most `MAX_RECORD_REVISIONS` (3) records exist in total, the
evidence is intact (§6.3), every stage of the previous run that has outputs is gated, its stored gates match a
recomputation from their raw files (verdict, finality, eligibility: a gate's hash is not a signature), and one of:
(a) the previous run never produced an output (now, in git history, or as a completed command in its exit-status logs:
e.g. an environment that failed `check-freeze`);
(b) its latest gate is INVALID for a cause in `gs_spec.SUPERSEDE_CAUSES` (a boundary difference or a workload mismatch,
i.e. a capture fact to correct, or exceptions, i.e. a possible adapter defect); operational causes (contention, failed
requests, drift, crashes) are repeated within the run and missing or stale files are waited for, never answered by a
new record;
(c) a command of the next stage crashed while measuring before that stage produced any output (an orchestration
defect): both frozen hosts' exit-status logs, committed unchanged, show such a crash and no completed command of that
stage, and the new record differs from the old one only in orchestration files and tests (not the adapter, not the
capture).
**What a new record may change** (`gs_freeze.lock_problems`; facts may be corrected, the choice may not): never the
selected configuration id or its selection inputs (`candidates.json`, the selection result), the checkpoint manifests,
the deployment's source manifests or the hosts; after an INVALID (b), the split capture only in its boundary section
and only for a boundary cause, the workload token ids only for a workload mismatch, and neither the rest of the capture
nor the environment (versions, execution variables); after an orchestration defect (c), nothing of `capture/`; only when
the previous run produced nothing (a) may other capture facts and the environment be corrected. An environment change
after outputs exist is restored, never answered by a new record; if it cannot be restored, the run cannot continue and
is reported INCOMPLETE.
A new record counts as one repeat of the causes that justified it (`supersedes.counts_as_repeat`: the INVALID gate's
causes, or the crashed parts), and repeat caps count across the whole chain of records, so a new record never resets
them. It lists every file that changed since the previous record (`supersedes.changed_files`); a changed adapter file
needs its own `HARNESS_FIXES.md` entry (fix commit, regression test, reason), and the archive keeps every record's
adapter as committed with it. A valid result, a FAIL, an INFEASIBLE or a COMPLETE can therefore never be escaped. Decision files cannot change in any record (§6.3), so a fix
cannot change a rule, a metric, a threshold or a measurement driver. The new record supersedes the old one, which can no
longer drive a command; **every measurement of the superseded run is kept** in `runs/<old run_id>/` and in the archive,
is listed with the outcome as history, and is never reported as a result. The run restarts at Stage 2 under the new
`run_id`. Never changed after the tag: the deployment's code, the configuration, the checkpoint, the workload, the
thresholds, the rules, this PREREG.

## 7. Workload (frozen now in `workload/`; token ids at Stage 1)

### 7.1 Correctness items (34)
30 natural prompts from the S1 sample, stratified gpqa 9, livecodebench 6, livemathbench 6, mmlupro 9, drawn within
each dataset in the order of sha256(`gemma-split:correctness:20261116:<key>`); the checkpoint's chat template, one user
turn, generation prompt added; greedy up to 1,024 new tokens, stopping at EOS. 4 coverage items: 2,048-token prompts
built like the timing prompts, greedy for exactly 256 new tokens (ignore EOS if supported). Their purpose: every
position range that Stage 5 times (up to 2,304 tokens) is covered by the correctness test.

### 7.2 Timing items (44)
Two groups, `short` (256 prompt tokens) and `long` (2,048), each 2 warm-up + 20 measured, 256 new tokens with EOS
ignored if the deployment supports it (otherwise natural stop, recorded, identically for all blocks). Construction:
[BOS] + tokens of the pool texts joined by a blank line from the item's start index onward, truncated to exactly the
group's size; the pool is the 170 sample prompts not used for correctness, ordered by
sha256(`gemma-split:timing:20261116:<key>`); item q starts at pool index q. Prompts are distinct, so no request shares a
prefix with the previous one.

### 7.3 Tokenization and checks
`gs_bench.py tokenize` with the captured tokenizer writes `capture/workload_tokens.json` (ids, tokenizer and chat
template hashes, BOS, EOS ids from `generation_config.json`, else `config.json`, else the tokenizer, with the source)
and refuses a result that does not match the frozen construction (`gs_workload.validate_tokens`). After the freeze,
`gs_bench.py verify-workload` re-derives the token ids on each host with that host's checkpoint tokenizer and writes
`runs/<run_id>/stage1_checks/<role>_workload.json`; the Stage-2 gate requires both to match (a mismatch is INVALID
and needs a corrected record). The longest item (prompt + new tokens) is the Stage-2 probe item.

### 7.4 Decoding
Every run (correctness, memory, timing, baselines) uses greedy argmax without sampling, penalties or logits processors,
whatever the deployment's default sampling is (recorded from `generation_config.json` and the capture). Greedy decoding
makes outputs comparable across systems; its cost per token is negligible next to a forward pass. This is a protocol
decision, stated here, not a change of the deployment's computation.

## 8. Stage 2: memory feasibility (no timers)

1. SERVER: `serve --stage 2` loads the server part and serves; a monitor records allocator peaks and OOM counters.
2. EDGE: `memory --what split --role edge` loads the edge part, connects, and runs the probe item through the split.
3. Each host: `memory --what reference --role R` loads the unsplit checkpoint (§9.1) on the role's deployment GPU and
   runs greedy + one teacher-forced pass over the probe item.
4. Recorded before and after each probe: `/proc/meminfo`, process RSS and peak RSS, the largest resident processes,
   `nvidia-smi` GPUs and compute processes, and per CUDA device the allocator peaks, the allocator OOM-event counter
   and `cudaMemGetInfo`; DURING the probe, foreign compute processes on the experiment devices are sampled every
   `MONITOR_PERIOD_S` (1 s); before the probe loads anything, the memory others hold (`occupied_before`: device memory
   used above `OCCUPIED_DEVICE_MAX_MIB` (1 GiB) on a discrete GPU, also catching processes `nvidia-smi` cannot list; on
   the unified-memory DGX Spark, MemTotal - MemAvailable - own RSS above `OCCUPIED_HOST_MAX_FRAC` (10 %) of MemTotal).
5. Status per record (`gs_rules.status_of`): BLOCKED_CONTENTION if a foreign compute process is on an experiment
   device at the start (nothing runs); FITS if the probe completed; INFEASIBLE if it failed with a device allocation
   failure (`gs_spec.OOM_DEVICE_PATTERNS`), with allocator OOM events in the process, or with a host allocation failure
   (`OOM_HOST_PATTERNS`) on a unified-memory host (the DGX Spark), **and** the device was otherwise idle: no foreign
   compute process at the start, during the probe or at the end, and no memory held by others before the probe; INVALID
   otherwise (a host allocation failure on a discrete-GPU host, `/dev/shm` exhaustion, any other error, any failure
   with a foreign process seen or memory held by others). An edge failure that coincides with allocator OOM events in
   the server process is the server's INFEASIBLE under the same idleness conditions on both hosts. The device is the
   frozen one (preflight checks `CUDA_VISIBLE_DEVICES` and every UUID before loading; the gate checks the recorded
   devices) and the configuration is the frozen one (record, adapter, snapshot and checkpoint verified before loading).
   An OOM caused by another user's job, by memory others hold, or in an invalid environment is never evidence of
   infeasibility.
6. Gate (`gs_gate.compute_stage2`) over three probes (the split, the reference on the edge, the reference on the server):
   an allocation failure is INFEASIBLE_UNCONFIRMED and both hosts repeat with fresh processes (every Stage-2 repeat
   covers both hosts, so no record of one attempt is ever combined with the other host's record of another; caps count
   per host part, and the host repeated only for that reason has cause `companion`, no cap); it is
   **confirmed** (the split's failing hosts decided by the same rule in every attempt: with both split records valid
   the combined status, including a server failure derived from allocator OOM events; otherwise the one valid record's
   own status) only when the SAME probe also failed to allocate in its most recent valid earlier observation (FITS or
   INFEASIBLE; BLOCKED and INVALID attempts are skipped, so contention never resets a confirmation) that does not already
   contain every run of the current one (an observation is identified by the runs that wrote its INFEASIBLE records; an
   earlier observation made of the same runs is the same evidence, a kept record, and is skipped, so a record never
   confirms itself); every attempt keeps its valid allocation failures, whatever else happened (a record on the wrong
   devices or with a wrong provenance is never evidence), and an unconfirmed split failure never opens Stage 3. The
   confirmation is checked in every attempt: a confirmed split INFEASIBLE is **STOP_INFEASIBLE** (final) even when the
   other host's half crashed or was invalid in the same attempt (it never becomes a repeat or, at the cap, INCOMPLETE),
   except that a gate needing a corrected record (§6.4; a Stage-1 workload mismatch, which is reported even when a half
   crashed, and no allocation failure of such a run is evidence) is never overridden.
   The infeasibility is recorded with every memory fact; the configuration is not changed and no smaller variant is
   tried. A confirmed reference INFEASIBLE with a fitting split is CONTINUE with
   `reference_feasible = false`, and Stage 3 is NOT_EVALUABLE. Failures not reproduced by the same probe within the caps
   end INCOMPLETE (no feasibility claim). INVALID is counted per host part even when another part is BLOCKED (contention
   elsewhere never hides an INVALID, and an INVALID host is capped during a confirmation too): twice, then INCOMPLETE
   (final); BLOCKED_CONTENTION alone may be retried without limit (nothing ran). An edge split INVALID while the server's
   split did not fit or serve to the end is the server's consequence: BLOCKED if the server was BLOCKED, otherwise not
   charged to the edge (the server's own status is; the edge's reference is charged as usual; a server status is never
   excused by the edge). A gate with a crash is computed only when every other
   Stage-2 command of the attempt has run (both hosts repeat). Only the parts the gate lists are repeated, and
   `invalidate` refuses a stored gate that does not follow from the raw files. CONTINUE if all probes FIT. A probe
   killed by the kernel's OOM killer (possible on the unified-memory DGX Spark) writes no record: it counts as a crash
   (repeated, then INCOMPLETE), never as INFEASIBLE; the kernel log entry for the pid in the run token is reported with
   it (for E4B on 128 GB this is not expected; §21). A record
   written before its part's last invalidation is stale (an old attempt brought back): the gate waits for the fresh file
   (it is never a repeat).

## 9. Stage 3: correctness (no timers anywhere in this stage)

### 9.1 References
The unsplit checkpoint with Hugging Face transformers in the host's own environment: class from `architectures[0]`
(transformers' auto mapping as fallback), dtype = the deployment dtype if equal on both hosts, else the config dtype;
attention implementation = the deployment's if captured, else the library default (recorded); TF32 off. Greedy with a
DynamicCache records, per generated step, the reference top1 - top2 log-prob margin (nats). Teacher forcing is one
full forward over prompt + continuation; the top-32 log-probs are stored at every continuation position.
R_server runs on the deployment's server GPU; R_edge runs on the DGX Spark over R_server's sequences.

### 9.2 Floor
The floor is the **hardware floor** only: R_edge against R_server (two devices, the same reference software). The
deployment's own in-host pipeline (`in_host_split`) shares the deployment's code, so a defect in its segments would
raise such a "floor" together with the split; it is therefore reported as a diagnostic (it localises a divergence to
the transport or to the segment code) and never enters a threshold.

### 9.3 Split runs and binding (edge, server part running)
For each of the 34 items: greedy through the deployment's own generation path; with T1, teacher-forced log-probs of the
split over R_server's sequence. Rows carry no time. The split run's start and end (UTC, seconds) are recorded once in
`split/meta.json`, and the server half's record (`split/server_serve.json`, written by `serve --stage 3`) carries its
serve window and the devices it served on; the split run must lie in that window (±30 s clock tolerance), which ties the
measured split to this run's server process. These timestamps are provenance, as coarse as the exit-status log, not
measurements. The boundary audit of the generation path is compared with the frozen split capture
(`gs_split.audit_boundary`): undeclared content, a dtype or tensor rank that differs from the declaration, and a
declared per-step tensor that never crossed are problems. The teacher-forced test path may carry log-probs that the
deployment never sends, so its messages are recorded separately and not audited.

### 9.4 Rules (logic of E0 PREREG §4, re-implemented in `gs_rules.py`; values in `gs_spec`)
- T1 (with `teacher_forced_logprobs`): D = % of continuation positions whose argmax differs from R_server's top-1; KL =
  mean KL(R_server || x) on the partition {R_server's top-32 ids, rest}. Pass iff D_split <= D_hardware + 0.2 pp and
  KL_split <= 2 x KL_hardware (1e-6 if KL_hardware is exactly 0). Every item must cover every continuation position
  (contiguous chunks; a gap or a short item is INVALID).
- T1 and T2: for each item, the first position where the split's greedy tokens differ from R_server's, after one
  trailing EOS id is stripped from both; the divergence is explained iff R_server's margin there is < 0.5 nats. Pass
  iff all 34 items are present, at most 3 are unexplained, no item is degenerate (invalid id, or the last 200 tokens
  periodic with period <= 50) in the split but not in R_server, and no NaN appears. If the deployment cannot ignore
  EOS, a coverage item that stopped at EOS is compared over its generated length (recorded as `eos_truncated`).
- Tier T2 (no teacher-forced log-probs, justified at Stage 0) applies the generation rule only; every report carries
  the tier.

### 9.5 Verdicts
- **PASS**: every applicable rule passes. **FAIL** (final): a rule fails. **NOT_EVALUABLE** (final): a confirmed
  unsplit reference does not fit (Stage 2).
- **INVALID**: a missing item, a provenance or device mismatch, a timing field, incomplete teacher-forced coverage, a
  split run outside the serve window, a crashed run (§13.2), boundary content that differs from the frozen split
  capture, contention, or an exception in a split run. **Contention** (a foreign compute process on an experiment device
  at the start, during or at the end of the split run on the edge, on the server from its start until the clock tolerance
  (30 s) before the split run ended (never after the result exists, never when the server is stopped), or memory held
  by others before the run) makes the attempt INVALID with cause
  `contention` (repeated within the caps), and its exceptions never count toward FAIL; contention never erases a FAIL:
  if the completed items already fail the generation rule (or, with every item completed, any rule), the verdict is
  FAIL. Only
  exceptions in the split runs count; the in-host pipeline is a diagnostic whose errors are reported and never affect
  the verdict (§9.2). The split run's rule preview (the same rules over its own rows, computed on the edge, where R_server
  and R_edge are local) is recorded in its `meta.json` and logged in its END line. Every INVALID decision records `would_be`: the rules applied to the completed
  items alone. The generation rule is monotone (more items can only add unexplained divergences or degenerate outputs),
  so when exceptions are the only problem and the completed items already fail it, the verdict is **FAIL**, not
  INVALID (contention is treated the same way). Exceptions may be repeated twice (supersedes included); a third attempt with exceptions is **FAIL** (final:
  the deployment is not runnable as captured). Other repeatable problems may be repeated twice, then the stage is
  INCOMPLETE (final). A boundary difference is not repeatable: the capture is corrected as a fact and a new record
  follows (§6.4). If Stage 2 confirmed that the reference does not fit, the gate is NOT_EVALUABLE without Stage-3
  outputs.
- **STOP**: on FAIL or NOT_EVALUABLE the experiment ends for performance. Stages 4 and 5 refuse to run; no runtime
  number of this configuration is reported. A defect of the deployment itself is not fixed under this
  pre-registration: a fixed deployment is a new configuration and needs a new pre-registration and tag; the FAIL stands.

### 9.6 No performance peeking
The Stage-3 drivers call no timer; `gs_results.assert_no_timing_fields` rejects any key that looks like a duration,
latency, rate or clock reading before a file is written, and the decision re-checks every row.

## 10. Stage 4: component runtime (descriptive; runs only after Stage 3 PASS)

- Isolated (`segment_step`): on each host, for the 2 warm-up + first 10 measured prompts of each timing group, one
  prefill and 255 decode steps of that host's own segment, on boundary inputs of the declared shapes and dtypes
  (recorded or synthetic, recorded which), each call timed by `perf_counter_ns` between CUDA synchronisations.
- In-path (`step_trace`, only if `segment_step` is absent): the edge runs the same prompts through the split with the
  adapter's per-step decomposition (edge compute, server compute as reported by the server, round trip); the server
  half serves under `serve --stage 4`, whose record (`edge/server_serve.json`) binds the trace (§9.3).
- Neither path exists: Stage 4 is COMPLETE with mode `not_available`, and Stage 5 reports no decomposition.
- A foreign compute process at the start or end, a failure, a device or provenance mismatch, or a trace outside the
  serve window makes the stage INVALID; two repeats per role, then INCOMPLETE (final).
- Summaries: per role, group and phase, n / median / p10 / p90 / mean of the measured prompts. No decision.

## 11. Stage 5: end-to-end runtime (runs only after Stage 4 COMPLETE)

### 11.1 Blocks
`split_A` -> `edge_only` -> `server_only` -> [`edge_only_native` -> `server_only_native`, only if
`unsplit_same_runtime` was declared at Stage 0] -> `split_B` (`gs_spec.block_order`), in this order, each from freshly
started processes (the server part is restarted for each split block), each over the same 44 timing prompts. The
bracketing split blocks control for drift over the session. Each block records its start and end (UTC, ms); the gate
checks the order across hosts with a 30 s clock tolerance, and that each split block lies in its server half's serve
window (`<block>/server_meta.json`).

### 11.2 Baselines
`edge_only` (on the DGX Spark) and `server_only` (on the deployment's server GPU, locally: a lower bound for remote
serving) always run the **reference runtime** of §9.1 (the gate checks the recorded runtime label), so the primary
comparator does not depend on anything the deployment's authors declare. The deployment's own unsplit mode, when declared, adds the secondary runtime-matched
blocks, which separate the cost of splitting from the choice of runtime.

**When the comparator cannot be produced (fixed now; never substituted).** The primary comparator is `edge_only` in the
reference runtime and nothing else: no other block (server_only, the native blocks) ever replaces it, before or after
results are visible.
- No full checkpoint identical to the server's is available on the DGX Spark (and no permitted copy, Q13): Stage 0 stops
  (STOP_CHECKPOINT_MISMATCH) and the run is **INCOMPLETE** before any measurement.
- The full checkpoint does not fit on the DGX Spark (a confirmed reference INFEASIBLE on the edge at Stage 2): the run
  ends **NOT_EVALUABLE** (no hardware floor, no comparator; no correctness or efficiency claim).
- The reference runtime cannot run the full model on the DGX Spark for another reason (unsupported, a persistent error):
  Stage 2 is INVALID, repeated within the caps, then **INCOMPLETE** with the reason "the unsplit reference could not run".
- `edge_only` (or another comparator block) fails requests in Stage 5: INVALID, repeated within the caps, then
  **INCOMPLETE** ("a comparator block could not be produced"), never UNSTABLE (which is reserved for split blocks that
  fail requests or drift) and never an efficiency category.
- The deployment offers no unsplit mode of its own: the secondary runtime-matched blocks are absent and reported as such;
  the primary endpoint is unaffected.

### 11.3 Timing and drift
Each block is timed entirely on the one host that runs it, with its monotonic clock (`perf_counter_ns`) at submission
and when each token is available there (for split blocks: on the edge, transport included); no synchronisation is added
and no cross-host timestamp enters a latency. Per block: the same 44 prompts with the same frozen token ids (checked on
each host by `verify-workload`; `tokens_sha256` per request), 2 warm-up + 20 measured prompts per group, warm-ups never
analysed; greedy; 256 new tokens with ignore-EOS when declared, otherwise up to 256 stopping at EOS (the same policy for
every block of the run). TTFT = first token - submission (prefill and the first decode step; the prompt is already
tokenised; prompt identity comes from the frozen token ids, and `tokens_sha256` identifies the generated tokens for the
agreement checks); TPOT = (last - first) / (n - 1), for n >= 2 (requests with n < 2 are counted, excluded from TPOT). Excluded:
process start, model loading, connection set-up and warm-ups. A request that raises, or that ends before 256 tokens
without stopping at EOS (with ignore-EOS: any count other than 256), is a failed request (warm-ups included); requests are
never retried or dropped within a block. No token for `REQUEST_STALL_S` (300 s) aborts the block, which the gate then
treats as a crash (§13.2). Drift = |median TPOT split_B - median TPOT split_A| / median TPOT split_A over measured
prompts. Drift > 0.10 -> UNSTABLE; all blocks are invalidated and repeated once; a second excess is final UNSTABLE (no
efficiency category).

### 11.4 Primary endpoint and categories
R = median over the 40 measured prompts of TPOT_split / median of TPOT_edge_only, where TPOT_split of a prompt is the
mean of its split_A and split_B values and edge_only is the reference runtime. Its 95 % CI (§11.5) gives exactly one
category: SPLIT_FASTER_THAN_EDGE (CI upper < 1), SPLIT_SLOWER_THAN_EDGE (CI lower > 1), NOT_DISTINGUISHABLE_FROM_EDGE
(otherwise). Interpretation, fixed now: the category compares the deployed split with local execution of the same
checkpoint by the reference runtime on the same edge; it is not by itself the cost of splitting, which is read from the
decomposition (§11.6) and, when available, from the runtime-matched secondary ratios. Edge-only is the primary
comparator because it is the alternative that keeps the raw text on the client; whether this deployment keeps raw text
on the client is itself recorded (§4.4). No minimum acceptable overhead is declared.

### 11.5 Statistics
Paired, group-stratified percentile bootstrap over prompts: B = 10,000, seed 20261116, `random.Random`, prompts
resampled with replacement within each group, quantiles by linear interpolation (`gs_stats`).

### 11.6 Secondary (descriptive)
TPOT split / server_only; TTFT split / edge_only and split / server_only per group; the same against the native blocks
when present (same bootstrap); per-group decomposition from Stage 4: edge decode step, server decode step, and the
remainder (transport, serialisation and overhead) = split TPOT - (edge + server); shares of each; closure check: split
TPOT >= 0.90 x (edge + server) per group, else the decomposition is flagged inconsistent (reported, no decision); token
agreement between blocks (determinism of split_A vs split_B, split vs each baseline).

### 11.7 Validity and repeats
A foreign compute process on an experiment device at the start or end of a block (on either host for split blocks), a
failed request, a server half that failed, a device, runtime-label or provenance mismatch, a block out of order or
outside its serve window, or a crashed run (§13.2) makes the stage INVALID; requests are never retried within a block and
never dropped. A Stage-5 repeat always covers **the whole block sequence**, so no single block can be swapped for a preferred
attempt. Caps per cause over the record chain: failed requests twice (a third failure is final UNSTABLE), contention
twice and other problems twice (then INCOMPLETE, final). The summary of every invalidated attempt (including the
category it would have shown; for a crashed attempt, the partial ratio of split_A to edge_only) is kept with it and
listed in the outcome, so a repeat cannot hide a result.

## 12. Network

No separate link benchmark (no ping series, iperf, `tc netem` or shaping): the experiment evaluates the existing
deployment on the existing link, the link is part of the system under test, and no decision depends on a link
property. The link is characterised by the Stage-0 facts (interfaces, link speed, MTU, route in both directions) and
by the transport share observed in the experiment itself (Stage-4 in-path round trip minus server compute, or the
Stage-5 remainder of §11.6). Clock synchronisation (`timedatectl` NTPSynchronized=yes or chrony "Leap status: Normal")
is **required** on both hosts at Stage 0 for audit ordering only: cross-host timestamps bind the split runs to their
server halves and order the blocks (§9.3, §11.1); no latency is computed from them (§11.3).

## 13. STOP / CONTINUE rules, repeats and outcomes

### 13.1 Gates
| Gate | Opens the next stage iff | Otherwise |
|---|---|---|
| Stage 0 | §4.6 holds | STOP_* with reasons; fix facts only |
| Stage 1 | record from commit A, commit B touches only frozen/, `check-freeze` and `verify-workload` pass on both hosts | fix the environment (never the record); a record that cannot pass needs a new Stage 0 for that host |
| Stage 2 | CONTINUE | STOP_INFEASIBLE / INCOMPLETE (final); INFEASIBLE_UNCONFIRMED, BLOCKED_CONTENTION, INVALID (repeat) |
| Stage 3 | PASS | FAIL / NOT_EVALUABLE / INCOMPLETE (final: **STOP, no performance interpretation**); INVALID (repeat or new record) |
| Stage 4 | COMPLETE | INCOMPLETE (final); INVALID (repeat) |
| Stage 5 | COMPLETE (final, with the efficiency category) | UNSTABLE (one drift repeat; final after); INCOMPLETE (final); INVALID (repeat) |

A gate is computed on the analysis host from the raw outputs (`gs_bench.py gate`), written once with the run mode
and its own hash, copied to both lab hosts (the drivers of the next stage read it locally), and committed (a commit
touching only `runs/`). A gate is not written (`NOT READY`) while raw outputs or the exit-status logs of the commands
that wrote them are missing, or while a file is stale. Each gate states whether its INVALID would allow a new record
(`supersede_eligible`, §6.4).

### 13.2 Repeats
Every gate lists the parts that may be repeated and the cause (`repeatable`). `gs_bench.py invalidate` (analysis host)
moves only such parts, together with the stage's derived files, to `<stage>/_invalidated/NNN-<utc>/` and logs the gate's
own causes and reasons with the operator's note; it refuses parts the gate does not list, a subset of a Stage-5
sequence, any final verdict, and a gate of another freeze record or mode; nothing is repeated without a gate. The log
is then copied to both lab hosts, where `gs_bench.py retire` moves the local copies of the same parts into the same
folders (no new log entry), so write-once files do not block the repeat and no rsync can bring an old attempt back; a
file older than its part's last invalidation is stale and the gate waits for the fresh one.

**Attempt accounting.** Every command logs START, RUN (the measurement begins: after connect and the device check),
WROTE for every output file BEFORE it is written (file, digest, and the result it records: a Stage-2 status or a Stage-3
rule preview), SAVED once it is written, and END with its exit status (§14). Logging fails closed: a command that cannot
write its START event does not run, an output whose WROTE event cannot be written is not written, and an output whose
SAVED event cannot be written is removed again. Each repeat records the run tokens it accounted for. Every output present
is attributed to the run that logged writing it: a run that wrote complete outputs is the attempt (its END may be
missing: a crash after writing, or an END that could not be written) and is never repeated (missing files: copy them);
a file that no logged run wrote waits for the current logs and, if it stays unaccounted for, is an INTEGRITY FAILURE (so
is a Stage-1 workload check that is not the one file one logged `verify-workload` run wrote with the digest it logged,
and an invalidated Stage-2 gate that is missing or differs from the hash the invalidation log recorded: the
confirmation chain reads both); a run that crashed before its outputs were completely written (WROTE without SAVED: a full disk, a kill at that instant)
makes its part repeatable, and what it had logged still counts (a Stage-3 rule failure stands; a Stage-2 status is an
observation for the confirmation). A Stage-2 repeat covers both hosts (§8). Per part and writer command, over
the runs no repeat accounted for, the gate requires exactly one completed run: a run that crashed or was killed while
measuring, with nothing completed since, makes the part repeatable (cause `invalid`, capped); a crashed run before the
completed one, without a repeat in between, makes it INVALID too; **two completed runs mean that an attempt's outputs
were deleted and the command re-run: the verdict is INCOMPLETE (final)**, reported together with which attempt's
outputs are present (each completed writer's END line carries a digest of its files) and the verdict those outputs
would give (`would_be`), so deleting an output cannot hide a seen result; a missing file of a completed run means "copy
it", and a lost output can never be repeated (that would be a redraw after the result), so the run cannot complete. A
refusal or connection failure before RUN is not an attempt.

**A discarded attempt can only hurt.** A file written by more than one run, more than one run that wrote complete
outputs, or more than one completed run means that outputs were deleted and the command re-run: a negative verdict of
the outputs present, or a negative result that a run whose outputs are gone logged (Stage 2: an allocation failure;
Stage 3: a rule failure in the split run's preview), stands and is final (Stage 2: STOP_INFEASIBLE; Stage 3: FAIL; Stage 5: UNSTABLE); any other verdict, a
result included, becomes a final INCOMPLETE flagged as an integrity failure, with the verdict of the outputs present
reported. A discard with files still missing never finalises: the gate waits for them. Outputs of a completed command that
never reach the analysis host leave the stage ungateable; the outcome reports an INTEGRITY FAILURE (§13.3) together with
the result the run that wrote them logged, never a neutral INCOMPLETE.

Caps (`gs_spec.REPEAT_CAPS`) count earlier invalidations and supersedes over the whole chain of freeze records, whatever
the note says (per part and cause in Stages 2-4, per repeat event and cause in Stage 5), and all causes together are
capped by `REPEAT_TOTAL` (Stage 2: 3, Stages 3-5: 2; causes where nothing ran are not counted), so mixing causes never
adds redraws; when a cap is reached the gate turns final. Nothing invalidated is ever deleted.

### 13.3 Outcome (`gs_bench.py outcome`)
The outcome command recomputes every stored gate from the raw files (any mismatch: no outcome), requires the Stage-6
checkpoint re-hash on both hosts, and lists the superseded records' runs and every repeat (with the category each
invalidated Stage-5 attempt would have shown and the verdict each invalidated Stage-3 attempt's completed items would
have given) as history. **INCOMPLETE** (a Stage-0 STOP, or
a repeat cap reached on INVALID: no claim); **INFEASIBLE** (the frozen configuration does not fit on a required host; a
memory result); **NOT_EQUIVALENT** (the deployed split is not the same computation as the checkpoint, or it could not
complete the correctness runs; an implementation result, no performance claim); **NOT_EVALUABLE** (no reference; no
claim); **COMPLETE** with one efficiency category (correct, fits, and the measured runtime relation to edge-only and
server-only on this hardware, with the component decomposition); **UNSTABLE** (no efficiency category). INVALID is never
an outcome. An integrity failure (completed outputs missing, or a discarded attempt) is listed in `integrity_failures`;
without a final gate the outcome is reported as an INTEGRITY FAILURE (a protocol violation), never as a neutral result.

## 14. Outputs, result JSON, provenance (`gs_results.py`)

Deterministic layout under `runs/<run_id>/` (stage directories, role and block sub-directories, fixed file names; see
the module docstring). Every result JSON carries `provenance`: experiment, protocol and decision-rule versions, stage,
role, host, devices, freeze hash, run id, adapter hash, configuration id, checkpoint manifest hash, `profile_role`, UTC.
Gates refuse rows whose provenance does not match the freeze record, comes from another host than the frozen one for
its role, ran on devices outside the frozen placement, or is `dev`. Files are written once (`gs_common.write_*`).
Each host appends START, RUN, WROTE (one per output file, before it is written: file, digest, result), SAVED (after it
is written) and END events
(token, stage, part, writer command, exit status; crashes and kills leave no END) for every command to
`runs/<run_id>/<host>.exit_status.log`; the gates read them for attempt accounting (§13.2).
The exit-status logs and the invalidation logs are append-only: every committed version must extend the previous one
and the working copy must extend the last committed one (checked with the evidence, §6.3). `runs/` is committed after
every command and gate, and pushed by the PI (§21).

## 15. Archive and hashes (`gs_archive.py`)

At the end (any outcome): `gs_bench.py verify-checkpoint --role R` on each host (full re-hash against the Stage-0
manifests, written to `runs/<run_id>/final/`), `gs_bench.py outcome`, then `gs_bench.py archive --run-id <id> --out
gemma_split_<id>_<OUTCOME>.tgz`: a byte-reproducible gzip tar (sorted members, zero owner and mtime, gzip mtime 0) of
this PREREG, the RUNBOOK, the README, every `gs_*.py`, `tests/`, `workload/`, `capture/`, `adapter/`, `frozen/` (every
record), `adapter_history/` (every record's adapter as committed with it) and `runs/` (every run, superseded ones and
`_invalidated/` parts included), with a SHA256SUMS member and an
external `<archive>.sha256`. No model weights (the manifest identifies the checkpoint); Brian's source snapshot only
with his consent (`--include-snapshot`), otherwise its manifest hashes in the freeze record. `verify-archive`
recomputes both levels. The archive refuses to be written if any lab-data member contains credential-like content (§4.1).
The archive hash is recorded in the results note and the paper.

## 16. Post-result changes: how they are prevented

The configuration is selected by a performance-blind rule before any measurement (§4.5); the capabilities are declared
and committed before any adapter exists (§5.3), and the primary comparator does not depend on them (§11.2). The harness
cannot run the deployment or the captured checkpoint before the freeze except through the timer-free `adapter-smoke`
(§5.4). Workload, thresholds, seeds, block order, statistics, metrics, rules and the measurement drivers live in
decision files that are byte-identical to the tag in every record (§6.3). The adapter and capture are frozen by hash;
any change needs a new record, accepted only in the three cases of §6.4, never after a final verdict, at most three
times, with caps counted over the chain; records and run outputs are append-only evidence checked against git history,
and the superseded runs stay archived. Stage 3 records no timing. Gates open in order, carry their mode and hash, and
are recomputed before any outcome; outputs are written once; repeats are listed by the gate, logged with the gate's
cause, capped, cover the whole Stage-5 sequence, and invalidated attempts stay visible with their would-be result;
attempt accounting turns deleted outputs into a final INCOMPLETE (§13.2). INFEASIBLE and FAIL are final for this
pre-registration. There is no tuning, sweep or second configuration. What remains procedural is listed in §21.

## 17. Differences from Qwen3 E1-OPT; comparability

Recorded automatically for the captured deployment (`gs_split.differences`), against the E1-OPT design (edge = token
embedding, layers [0,k) and [L-j,L), final norm, LM head, sampling, tokenizer; server = the middle layers on one GPU; wire
= the bf16 residual stream only; never token ids, logits, sampled tokens, KV or per-layer inputs; each host caches its own
layers). Structural differences known before capture: Gemma 4 has per-layer input embeddings and KV sharing between
layers, so a cut may need per-layer inputs, token ids or KV on the wire, which the Qwen3 design never sends.

| Variable | E1-OPT (Qwen3) | GEMMA-SPLIT |
|---|---|---|
| Model | Qwen3-32B bf16 `9216db57` | gemma-4-E4B-it as captured (manifest), dtype as deployed |
| Split | fixed U-split, cut asserted | Brian's existing split as captured; topology recorded |
| Runtime | custom static-cache + CUDA-graph runtime | the deployment's own runtime, via a frozen adapter |
| Roles | decided by inventory | fixed: DGX Spark edge, asus-ESC4000-E11 server |
| Decision | utility vs edge competitors, Δ >= 3 pt | correctness gate, memory fact, runtime categories |
| Memory | emulated 12/16 GB caps | physical hosts, no cap |
| Network | link gate + netem cells | no link benchmark (§12) |

Because the model, the split, the runtime, the role rule and the decision rule all differ, **no Gemma number is compared
with a Qwen3 number** under this protocol. A fair comparison would need a separate pre-registration that fixes a common
workload, runtime and decision rule; this one does not claim it.

## 18. Unresolved values (only Stage 0 with Brian / the DGX Spark / the lab can supply them)

1. The DGX Spark's hostname, OS, driver, CUDA, Python / torch / transformers versions; the server's software stack.
2. The checkpoint path on each host, its manifest, its revision (if visible), dtype, `config.json` (layer count, layer
   types, KV sharing, PLE, softcapping, tied embeddings, multimodal towers, quantisation), tokenizer and chat template.
3. The implementation: code location, commit, dirty state, framework, entry points, launch commands, environment.
4. The split: layer ownership per host, location of embedding / PLE / norm / head / sampling, every boundary tensor,
   KV policy and shared-KV handling, prefix caching, dtype on the wire.
5. Transport, serialisation, endpoint, which link and interfaces; whether both clocks are synchronised.
6. The server GPU(s) the deployment uses (UUIDs, `CUDA_VISIBLE_DEVICES`); whether the edge holds a full checkpoint copy.
7. Which Gemma configurations exist and which is current (R1 answer); whether anything runs now; tuning history.
8. The capabilities the code supports (greedy, streaming, ignore-EOS, teacher-forced log-probs, isolated segments,
   trace, unsplit mode, in-host split), hence the tier, the component-profile mode and the secondary blocks.
9. Lab permissions: quiet GPU windows; a hash-verified checkpoint copy and code snapshot in the experiment's own
   directories; inclusion of Brian's snapshot in the archive.
10. Authentication between the DGX Spark and the server (mechanism and credential NAMES only, Q17); the experiment's own
   endpoint / port, the account it runs under and its cache directories (Q18).

## 19. Assumptions this protocol refuses to make

The checkpoint revision; that the directory name identifies the weights; the DGX hostname; the API endpoint; the cut or
that it resembles the Qwen3 U-split; that only the residual stream crosses the wire; that the server never sees token ids
or generated tokens; that embeddings are untied; that each host keeps a KV cache; the dtype; that the deployment is free
of uncommitted changes or matches across hosts; that Brian's current running configuration is scientifically frozen;
that one server GPU is used; that the live deployment tree stays unchanged; that NVML reports memory on the DGX Spark (unified memory); that every allocation error is
infeasibility; that the reference runtime equals the deployment's runtime; that the deployment's own in-host pipeline is
a valid floor; that greedy outputs agree bit-for-bit across GB10 and A6000 (hence the floor); that an implementation
returns the stopping EOS token; that the two hosts' clocks agree exactly; that the link is fast enough to ignore; any
performance threshold for "efficient" beyond the CI categories; that a credential may be recorded anywhere; that Brian's
running deployment may be stopped or reconfigured by the experiment; that a GPU is idle because `nvidia-smi` lists no
process; that a comparator may be replaced when it cannot be produced.

## 20. Executable components and CPU tests

| Component | File | Kind | Tests |
|---|---|---|---|
| Constants, protocol parameters, decision-file list | `gs_spec.py` | decision | protocol |
| Metrics and verdict rules (correctness, memory, runtime) | `gs_rules.py` | decision | protocol, pipeline |
| Gates, repeats, attempt accounting, outcome | `gs_gate.py` | decision | protocol, pipeline |
| Freeze writer / checker, record chain, append-only evidence | `gs_freeze.py` | decision | protocol (temporary git repositories), pipeline |
| Split capture validation, differences, boundary audit, selection | `gs_split.py` | decision | protocol, pipeline |
| Layout, provenance, no-timing guard, exit events and writers | `gs_results.py` | decision | protocol, pipeline |
| Statistics | `gs_stats.py` | decision | protocol |
| Adapter contract | `gs_adapter.py` | decision | protocol, pipeline |
| Hashing, write-once IO, redaction, timestamps | `gs_common.py` | decision | protocol |
| Checkpoint identity, features, KV-sharing map | `gs_checkpoint.py` | decision | protocol, pipeline |
| Workload build / tokenize / validate | `gs_workload.py` + `workload/*.json` | decision | protocol, pipeline |
| Inventory, role verification | `gs_inventory.py` | orchestration | protocol |
| Deployment capture, snapshot | `gs_deployment.py` | orchestration | protocol, pipeline |
| Reference runtime | `gs_reference.py` | decision (measurement driver) | pipeline |
| Stage-2 probes | `gs_memory.py` | decision (measurement driver) | protocol, pipeline |
| Stage-3 drivers | `gs_correctness.py` | decision (measurement driver) | pipeline |
| Stage-4/5 drivers | `gs_profile.py` | decision (measurement driver) | pipeline |
| Archive | `gs_archive.py` | orchestration | protocol, pipeline |
| Command hub, preflight, live-tree guard, serve records, smoke test, retire, exit events | `gs_bench.py` | orchestration | protocol, pipeline (subprocess) |
| Test double (Gemma 4 tiny split over TCP) | `tests/doubles/gemma4_split_double.py` | test | pipeline |

## 21. Residual risks (stated, not solved in software)

The third audit round confirmed that the following cannot be closed by the harness alone; they are mitigated by
procedure and stated here so that a reader can weigh them.

1. **Coarse durations are visible.** The exit-status logs, the serve windows and the wall clock show how long a
   Stage-3 command ran, from which an operator could estimate the split's mean time per token before the Stage-3
   verdict. Nothing measured in Stage 3 depends on it, and no rule, threshold, driver, adapter or configuration can
   change after the freeze except through §6.4. An adapter fix after Stage-3 exceptions is the one change that could be
   steered by such an estimate; it needs a `HARNESS_FIXES.md` entry with a regression test, counts as a repeat, is
   listed in the new record's `changed_files`, and both adapter versions are archived (`adapter_history/`), so the diff
   can be reviewed.
2. **Uncommitted evidence can be destroyed.** Outputs and exit-status log lines exist only on a lab host until copied
   and committed; an operator who deletes both an attempt's outputs and its log lines before the first commit leaves no
   trace, and log lines can be forged before that commit. With the log lines kept and unforged, every output is attributed
   to the run that logged writing it before writing it (§13.2): a deletion and re-run is detected, a negative result
   that a run whose outputs are gone logged (Stage-2 status, Stage-3 rule preview) still counts, any other verdict ends
   the run as an integrity failure (a deletion without a re-run: an INTEGRITY FAILURE outcome naming the logged result),
   and a deleted Stage-4 / Stage-5 attempt's numbers are lost (their digests are logged). A script that reads an output
   in the microseconds between its write and its SAVED event, kills the run and deletes the file turns it into a crash
   (a repeat); its logged result still counts. Mitigation: commit `runs/` (outputs and logs) after every command and every gate, and push those commits (the
   PI pushes); once committed, deletions, renames and log rewrites are refused (§6.3, §14).
3. **Local history can be rewritten before a push.** The evidence checks read the local repository. Mitigation: push
   after every gate commit; the archive and the remote history are the record.
4. **Child processes are not audited, and installed copies are not hashed.** The live-tree guard covers the harness
   process; a child process that reads Brian's live tree by itself is not seen, and a non-editable installed copy of the
   deployment's package inside the interpreter's environment is allowed but not hash-verified. Mitigation: everything
   the adapter runs must lie under a captured source root (snapshotted and re-verified before every command; launch
   files outside the roots are hashed at Stage 0 and guarded like the tree), the adapter puts the snapshot first on
   `sys.path` and may not name a live path, and `PYTHONPATH` / library paths and the key package versions are compared
   with the freeze.
5. **A crash can be induced.** Killing a run before it writes its outputs costs one repeat (capped by the stage total) and
   is logged; it cannot select between attempts (outputs written by one run are kept even without an END line, outputs
   written by two runs are a discarded attempt, and a kill between WROTE and the write is a crash whose logged result
   still counts), and it never removes an observed allocation failure (every Stage-2 attempt keeps its INFEASIBLE
   observations, and a confirmed split failure is final in any attempt; a crash never yields a FITS observation). Repeated
   kills end the stage INCOMPLETE instead of the verdict the run would have reached (for example an imminent allocation
   failure); every crash is listed with the outcome, a crashed Stage-5 attempt with its partial ratio. Likewise, an
   environment deliberately broken so that `verify-workload` really fails (a genuine, logged workload mismatch) makes the
   run's allocation failures non-evidence and, through a new record, starts a new confirmation chain (the caps carry
   over the record chain, so this too ends INCOMPLETE, never in a better verdict).
6. **A kernel OOM kill is not classified.** A memory probe killed by the OOM killer leaves no record and counts as a
   crash, so a model that only fits by exhausting the DGX Spark's unified memory would end INCOMPLETE rather than
   INFEASIBLE. The kernel log for the run token's pid is reported with such a crash.
7. **Some contention is invisible.** A job that `nvidia-smi` cannot list (another container) and that starts after a
   probe began, or host-memory pressure from other processes on the unified-memory Spark during a probe, is not
   observed; the memory held by others is checked before each probe, foreign compute processes are sampled during it,
   and an allocation failure must recur in the same probe's last valid earlier observation to count.
8. **The edge is bound to the experiment's server by endpoint and time window, not by a handshake.** The experiment's
   instance listens on a port other than the live deployment's (checked as port numbers, whatever the host is called,
   URL default ports included) and
   each split run must lie in its serve window; the deployment's protocol is not changed to exchange a run token.
