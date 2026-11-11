# E1-OPT implementation plan (2026-11-11, revision 3; all drivers written and CPU-tested; nothing has run on a GPU; protocol not frozen)

## Why not vLLM (checked against the current source on 2026-11-10)

- `vllm/distributed/utils.py get_pp_indices` returns one `[start, end)` range per pipeline rank;
  `VLLM_PP_LAYER_PARTITION` sets the sizes of consecutive chunks only.
- `qwen2.py` / `qwen3.py`: `embed_tokens` exists only on `is_first_rank`; `norm` and `lm_head` only on
  `is_last_rank`; other ranks hold `PPMissingLayer`. Middle ranks exchange `IntermediateTensors` with two
  tensors (`hidden_states`, `residual`), not one.
- `config/parallel.py` and `v1/executor/multiproc_executor.py`: one worker per local rank per visible CUDA
  device; a world size above the GPU count is an error. Two stages on one GPU are not expressible.
- Therefore the frozen layout (edge = embed + [0,k) + [64-k,64) + norm + head; server = middle) cannot be
  produced by vLLM pipeline parallelism, with or without custom placement, nor by SGLang or TensorRT-LLM, whose
  pipeline code has the same first/last-rank structure. Reusing vLLM's engine for a middle segment would mean
  driving a worker through internal APIs (`IntermediateTensors`) with no public interface.
- Petals places embedding, norm and LM head on the client but no transformer blocks; unmaintained since
  2024-09; no Qwen3. llama.cpp `-ot` can place weights by regex but the per-layer device, KV cache and the
  output layer follow the contiguous `--tensor-split`; exact ownership would need a source patch. HAT
  (arXiv 2503.18989) is the same topology but has no public code.

## Chosen path: option E (custom PyTorch, E0 modules, static cache, CUDA graphs)

| Candidate | Same computation | First+last on edge | No raw text | Sampling on edge | Changes design | Difficulty | DGX Spark | Asus (unknown GPU) | Changes question |
|---|---|---|---|---|---|---|---|---|---|
| A vLLM PP custom placement | yes | **no** (embed first rank, head last rank) | yes | no (last rank samples) | yes | n/a | n/a | n/a | yes |
| B vLLM 3 stages, 0 and 2 on one GPU | yes | would | yes | no | yes | blocked by executor | n/a | n/a | yes |
| C custom vLLM worker/transport | yes | possible | yes | possible | no | very high (internal APIs, two-tensor residual) | unclear (cu130 nightly track) | unknown | no |
| D vLLM kernels without the engine | yes (kernels differ, Q-tested) | yes | yes | yes | no | high | unclear | unknown | no |
| **E HF modules + static KV + CUDA graphs + flash_attn_with_kvcache** | yes (Q-tested; prefix mode bitwise) | yes | yes | yes | no | **moderate** | sm_121 flash-attn by source build only; graphs and SDPA fallback fine | needs CUDA ≥ Ampere | no |
| F llama.cpp RPC / Petals / exo / PiPPy | llama.cpp: different kernels and quantised weights; others no | partial / no | partial | partial | yes | n/a | llama.cpp yes | unknown | yes |

Expected gain for E, stated before measurement: the measured per-layer decode cost on the H200 is about 1.2 ms
against a 0.2 ms roofline and is roughly constant across the segment, which is consistent with per-kernel launch
and Python dispatch overhead in the eager loop (a hypothesis; no profiler run exists). CUDA-graph capture removes
per-kernel launch and Python dispatch and leaves memory traffic unchanged, so it tests that hypothesis directly;
`flash_attn_with_kvcache` removes the 8× GQA cache copy of the static SDPA path. Hugging Face reports up to 4× for
whole-model generation with a static cache plus compilation, smaller for large models. With the archived RTX 3090
edge profiles the gate needs a 1.21× faster server step for Δ ≥ 3.0 (1.19× in exact arithmetic) and 1.45× for a
pass with CI low > 0 (`reanalysis/threshold_audit.py`; the earlier 1.42× used CI low ≥ 0 and is withdrawn). The
roofline (5.7× below the measured step) is an optimistic lower bound, not a level the implementation is known to
reach. A result between 1.0× and 1.2× would close the question for this design on an H200-class server.

`torch.compile` is **not used** (PREREG §4.1, decided before any timing). The attention kernel and the CUDA-graph
setting have a compatibility fallback only (PREREG §4.2); no latency comparison chooses between variants.

## Work items

Done on CPU (2026-11-11): `e1_spec.py`, `e1_static_cache.py`, `e1_segment.py` (Segment, GraphedStep,
USplitEdge, LocalMiddle), `e1_attention.py` (registered kernel with SDPA fallback), `e1_protocol.py` (wire v2,
server, client), `e1_memory.py`, `e1_bench.py` (profile-segment, gate-ub), `tests/test_e1_cpu.py` (9 pass).

Before any GPU time:
1. `e1_bench.py equivalence`: Q1/Q2 driver for static mode vs HF unsplit on one GPU (reuse E0 `e0_q.py`
   references and thresholds); prefix-vs-E0 bitwise check on GPU.
2. `frozen/runtime_freeze.json` writer and checker (hashes of `e1_*.py`, versions, kernel name, graph flag).
3. Edge-segment profiler (echo server, capped) mirroring E0 Stage 1 `P-edge-U(k,k)-cap<M>` rows.
4. Joint cell driver: E0 `e0_latency.py run_split` semantics with the E1 runtime; reuse E0 cells.
5. Verify on GPU: `flash_attn_with_kvcache` accepts the transposed cache views (strides); otherwise store the
   cache in `[1, max_len, kvh, d]` layout and expose HF-layout views to the layers.

Stage 0 is inventory and role feasibility only (PREREG §6). The former "desk gate" that paired the server GPU's
datasheet bandwidth with the archived RTX 3090 edge profiles is **not** a decision input any more: that bound is
not conservative for a different edge device (a slower edge slows the competitor more than the split's six edge
layers). The first performance decision is G-UB-E1 after Stage 3 with fresh profiles from both target machines.

Written in the implementation pass of 2026-11-11 (all CPU-tested; see README for the test table):
`e1_freeze.py` (freeze writer and checker; `e1_bench.py freeze | check-freeze`), `e1_inventory.py` (Stage-0
inventory, feasibility, deterministic roles), `e1_provenance.py` (target/screening provenance and the gate
rejection rules), `e1_equivalence.py` (`equivalence reference | local | joint | decide`; no timing fields),
`e1_profile.py` (`fit-split`, `fit-competitor` wrapping E0's `edge-fit` unchanged, `profile-edge-segment`,
`profile-server-segment`, `summarize-fit`), `e1_joint.py` (`serve`, `run-cell`, `run-remote-cell`, `linkcheck`),
`e1_gate.py` (`gate-ub`, `gate-r` with provenance check and the rounded-decision sidecar). The HELLO is whitelisted
and carries the freeze hash; `e1_segment.USplitEdge.generate` records Q2 margins; `teacher_forced_hidden` chunks
long sequences through the KV cache. Open GPU-only item: confirm on the first GPU run that `flash_attn_with_kvcache`
accepts the transposed cache views (strides); a stride rejection is a §4.2 compatibility fallback, not a tuning choice.
