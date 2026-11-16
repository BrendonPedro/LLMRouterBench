# GEMMA-SPLIT Stage 0: questions for Brian and the lab

Answers go into `capture/lab_answers.md`, one dated section per answer (`## Q3 (2026-MM-DD, Brian)`), before any
Stage-2 command runs. A selection made from these answers (PREREG §4.5) cites the section anchor. Nobody is shown
any GEMMA-SPLIT measurement before the answers are recorded (none exists before Stage 2).

## Deployment identity
- Q1. Which code implements the Gemma split on each machine (path, repository, branch, commit)? Is it the same code
  on both machines? Are there uncommitted changes?
- Q2. Which Gemma configuration do you consider the current one for the DGX Spark (edge) + asus-ESC4000-E11 (server)
  pair? Please name it in writing (rule R1). List every other Gemma configuration that exists (other cuts, dtypes,
  transports, checkpoints).
- Q3. How is each side started (exact commands, working directories, environment variables, CUDA_VISIBLE_DEVICES)?
  Is anything running now?
- Q4. Was the split point (or any other setting) chosen by measuring speed or memory? When, on which hardware?
  (Context only: the protocol never chooses a configuration from its own measurements.)

## Model / checkpoint
- Q5. Which directory does each side load (`gemma-4-E4B-it` on the server; which path on the DGX Spark)? Was it
  downloaded from Hugging Face (repository id, revision) or converted? Is any part quantised or cast?
- Q6. dtype on each side? Attention implementation (sdpa, eager, flash-attention)?

## Split structure (record, do not change)
- Q7. Which layers run on the DGX Spark and which on the server? Where are the token embedding, the per-layer input
  embeddings (PLE), the final norm, the LM head and the sampling step?
- Q8. What crosses the network per step, in each direction (tensor names, dtypes, shapes)? Do token ids, text,
  per-layer inputs, KV entries or logits cross? How does a server layer that shares KV with an edge layer get it?
- Q9. Does each side keep a KV cache, or is the prefix recomputed each step? Any prefix caching across requests?
- Q10. Transport (TCP socket, HTTP, gRPC, ZMQ, other), serialisation, endpoint (host:port or URL), which network
  interface / link the two machines use.
- Q11. Can the implementation (a) decode greedily, (b) stream tokens to the edge as they are produced, (c) ignore EOS
  for a fixed number of tokens, (d) return per-position log-probs for a given token sequence, (e) run one side's
  layers alone on given inputs, (f) run the whole model on one machine, (g) run both halves in one process? The answer
  (with the code that shows it) becomes `capture/capabilities.json` BEFORE any adapter is written; the adapter must
  match it. Does the implementation return the EOS token that stops generation?

## Lab / operations
- Q12. Which server GPU(s) may the experiment use, and when is the server quiet (no other jobs on those GPUs)?
- Q13. May the experiment place a hash-verified copy of the checkpoint and of the split code in its own directory on
  each machine (nothing of Brian's is modified)?
- Q14. May Brian's source snapshot be included in the final archive, or only its hash manifest?
- Q15. Is a Python environment on each machine available to run the harness (the deployment's own interpreter;
  the harness needs only the standard library plus the deployment's torch / transformers)?
- Q16. Are both machines' clocks synchronised (NTP / chrony)? Cross-host timestamps bind and order the runs (30 s
  tolerance).
