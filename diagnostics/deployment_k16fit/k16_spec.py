"""K16-HWFIT constants: a runnable Qwen3-32B layer split for the NYCU lab pair at the symmetric cut k = 16.

Status: EXPLORATORY hardware-fit experiment, written 2026-11-18. It is not pre-registered and it is not a
continuation of E0 or of E1-OPT (v1, v2 or v3). It reuses their runtime code unchanged, by import:
  deployment_e0/split_runtime.py   weight loading (Slice), E0 KV semantics
  deployment_e1opt/e1_segment.py   Segment, USplitEdge (reference mode "prefix" = E0's computation)
  deployment_e1opt/e1_protocol.py  wire (72-byte header, raw bf16 residual stream), HELLO checks
No file of deployment_e0/ or deployment_e1opt/ is modified (tests/test_k16.py checks the diff against BASE_TAG).

Roles are fixed by the lab hardware, not by an inventory rule:
  EDGE   = NVIDIA DGX Spark (GB10, unified memory): embeddings, layers 0-15, layers 48-63, final norm, LM head
  SERVER = Asus main server, ONE RTX A6000 (48 GiB): layers 16-47
Each side holds the KV cache of its own layers only. One bf16 residual-stream tensor crosses per direction per
forward step: after layer 15 (edge -> server) and after layer 47 (server -> edge).
"""
import hashlib
import json
import sys
from pathlib import Path

K16_DIR = Path(__file__).resolve().parent
E1_DIR = K16_DIR.parent / "deployment_e1opt"
sys.path.insert(0, str(K16_DIR))
sys.path.insert(0, str(E1_DIR))
import e1_spec as E1  # noqa: E402  (read-only; also puts deployment_e0 on sys.path)

E0 = E1.E0
E0_DIR = E1.E0_DIR

NAME = "K16-HWFIT"
SPEC_VERSION = 1
STATUS = "exploratory hardware-fit experiment; not pre-registered; not a continuation of E0 or E1-OPT"
BASE_TAG = "e1opt-prereg-v3"              # the commit whose E0/E1 runtime files are reused unchanged
BRANCH = "k16-hwfit-exploratory"

# --------------------------------------------------------------------------- #
# Partition                                                                    #
# --------------------------------------------------------------------------- #
K, J = 16, 16
N_LAYERS = E0.QWEN3_32B_CONFIG["num_hidden_layers"]            # 64
EDGE_LAYERS = list(range(K)) + list(range(N_LAYERS - J, N_LAYERS))   # 0-15 and 48-63
SERVER_LAYERS = list(range(K, N_LAYERS - J))                         # 16-47
MODEL_ID = f"{E0.SPLIT_REPO}@{E0.SPLIT_REVISION}"

ROLES = {
    "edge": {"machine": "NVIDIA DGX Spark", "gpu_name_marker": "GB10", "memory": "unified",
             "holds": ["embed_tokens", "layers 0-15", "layers 48-63", "final_norm", "lm_head", "sampling", "tokenizer"]},
    "server": {"machine": "Asus main server (asus-ESC4000-E11)", "gpu_name_marker": "A6000", "memory": "one discrete GPU, 48 GiB",
               "n_gpus_used": 1, "holds": ["layers 16-47"]},
}

# --------------------------------------------------------------------------- #
# Runtime                                                                      #
# --------------------------------------------------------------------------- #
PRIMARY_MODE = "prefix"                   # reference mode: E0's computation; every gate runs in it first
OPTIONAL_MODE = "static"                  # E1-OPT's static-cache mode; never timed unless it passes the same criterion
TRANSFORMERS_REQUIRED = E0.TRANSFORMERS_VERSION   # 4.51.3: the runtime calls that release's Qwen3 internals
DEFAULT_PORT = 29616                      # not E0/E1's 29500
# Transport: the server listens on its loopback interface ONLY. The edge reaches it through an SSH tunnel opened from
# the DGX Spark (launch_dgx_edge.sh), so the link is authenticated and encrypted by SSH and no port is exposed on the
# lab network. The wire protocol itself has no authentication, which is why a non-loopback bind is refused.
LISTEN_HOST = "127.0.0.1"
# Checkpoint locations known on the lab machines (read-only; verified against the manifest like any other).
KNOWN_MODEL_DIRS = [f"/mnt/mydisk/brian/splitllm-cloud/models/Qwen3-32B/{E0.SPLIT_REVISION}"]      # Asus main server
MAX_LEN = 4096                            # KV preallocated per owned layer (tokens); longest request is 2048 + 256
PREFILL_CHUNK = E0.PREFILL_CHUNK          # 2048

# --------------------------------------------------------------------------- #
# Correctness criterion, fixed on 2026-11-18 before any k = 16 run on any GPU  #
# --------------------------------------------------------------------------- #
# Reference: the unsplit Hugging Face model (bf16, SDPA) on the edge host's GPU, same checkpoint. A configuration
# (in-host split, two-host split in prefix mode, two-host split in static mode) PASSES iff all four hold on the
# check set. The thresholds are absolute and are not derived from any k = 16 measurement.
#   C1 finite        no NaN or Inf in the final hidden states or margins
#   C2 decisive      among positions where the reference's top-1 minus top-2 log-probability is >= 0.5 nats,
#                    the configuration's argmax equals the reference's at >= 99.5 % of positions
#   C3 distribution  mean coarse KL(reference || configuration) over all positions <= 0.05 nats
#                    (E0's e0_q.compare_to_ref: reference top-32 tokens + rest)
#   C4 greedy        per prompt, the 64 greedy tokens equal the reference's, or first differ at a position where
#                    the reference margin is < 0.5 nats; at most 10 % of prompts may fail; none degenerate
# Why not E1-OPT's check B: an absolute 2^-6 bound on bf16 hidden states cannot survive any change of kernel or
# device (PREREG_E1OPT §17). C2-C4 test the function the model computes. A wrong layer order, KV misalignment or
# a corrupt wire tensor fails them by a wide margin (tests/test_k16.py runs such a negative control).
CRITERIA = {
    "id": "K16-CORRECT-v1", "fixed_utc": "2026-11-18",
    "reference": "unsplit Hugging Face Qwen3ForCausalLM, bfloat16, SDPA, on the edge host GPU",
    "decisive_margin_nats": 0.5, "max_decisive_disagreement": 0.005, "max_mean_kl_nats": 0.05, "topk": E0.Q1_TOPK,
    "greedy_tokens": 64, "max_greedy_fail_share": 0.10, "require_finite": True,
}
CHECK_GROUPS = ("short", "long")          # E0 frozen workload prompts: 5 x 256 tokens and 5 x 2048 tokens

# Stage 3 demonstration prompts (chat template, thinking off). Text output is descriptive, never a gate.
CHAT_PROMPTS = [
    {"id": "chat-0", "text": "What is 17 multiplied by 23? Answer with the number only."},
    {"id": "chat-1", "text": "Write a Python function that returns the n-th Fibonacci number."},
    {"id": "chat-2", "text": "Explain in two sentences why the sky is blue."},
]
CHAT_MAX_NEW = 128

# --------------------------------------------------------------------------- #
# Latency (only after the two-host correctness gate passes)                    #
# --------------------------------------------------------------------------- #
LATENCY = {"groups": list(CHECK_GROUPS), "gen_tokens": E0.GEN_TOKENS, "warmups": E0.WARMUPS_PER_CELL, "reps": 2,
           "order_seed": "k16-hwfit:order:20261118"}

# --------------------------------------------------------------------------- #
# Memory gates                                                                 #
# --------------------------------------------------------------------------- #
SERVER_MARGIN_MIB = 2048                  # required free = expected peak + margin
EDGE_MARGIN_MIB = 4096                    # unified memory is shared with the OS
ACTIVATION_ALLOWANCE = 1.25               # E0 measured 870 MiB against 753 MiB predicted (x 1.16)

HASHED_SOURCES = (
    [("k16", "k16_spec.py"), ("k16", "k16_common.py"), ("k16", "k16_memory.py"), ("k16", "k16_checkpoint.py"),
     ("k16", "k16_preflight.py"), ("k16", "k16_criteria.py"), ("k16", "k16_runtime.py"), ("k16", "k16_server.py"),
     ("k16", "k16_edge.py"), ("k16", "frozen/checkpoint_manifest.json"), ("k16", "frozen/tensor_table.json")]
    + [("e1", f) for f in ("e1_spec.py", "e1_segment.py", "e1_static_cache.py", "e1_attention.py", "e1_protocol.py")]
    + [("e0", f) for f in ("split_runtime.py", "e0_spec.py", "e0_q.py", "e0_memory.py", "e0_instrument.py", "frozen/workload.json")]
)
_ROOTS = {"k16": K16_DIR, "e1": E1_DIR, "e0": E0_DIR}


def default_out():
    """Run directory root: K16_OUT, else a sibling of the repository (so /mnt/mydisk/brendon/k16fit_runs on the Asus)."""
    import os
    return os.environ.get("K16_OUT") or str(K16_DIR.parents[2] / "k16fit_runs")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def source_hashes():
    return {f"{root}/{rel}": sha256_file(_ROOTS[root] / rel) for root, rel in HASHED_SOURCES}


def params(dev=False):
    """Run sizes. dev=True shrinks them for the CPU tests with a tiny random model; the code path is the same."""
    if dev:
        return {"dev": True, "max_len": 256, "chunk": 48, "greedy_tokens": 8, "gen_tokens": 8, "warmups": 1, "reps": 1,
                "chat_max_new": 8, "model_id": "dev"}
    return {"dev": False, "max_len": MAX_LEN, "chunk": PREFILL_CHUNK, "greedy_tokens": CRITERIA["greedy_tokens"],
            "gen_tokens": LATENCY["gen_tokens"], "warmups": LATENCY["warmups"], "reps": LATENCY["reps"],
            "chat_max_new": CHAT_MAX_NEW, "model_id": MODEL_ID}


def spec(dev=False):
    return {"name": NAME, "spec_version": SPEC_VERSION, "status": STATUS, "base_tag": BASE_TAG, "k": K, "j": J,
            "edge_layers": EDGE_LAYERS, "server_layers": SERVER_LAYERS, "roles": ROLES,
            "model": {"repo": E0.SPLIT_REPO, "revision": E0.SPLIT_REVISION, "dtype": "bfloat16", "wire_dtype": E0.WIRE_DTYPE,
                      "config": E0.QWEN3_32B_CONFIG},
            "primary_mode": PRIMARY_MODE, "optional_mode": OPTIONAL_MODE, "transformers_required": TRANSFORMERS_REQUIRED,
            "criteria": CRITERIA, "check_groups": list(CHECK_GROUPS), "workload_ids_sha256": E0.WORKLOAD_IDS_SHA256,
            "chat_prompts": CHAT_PROMPTS, "latency": LATENCY, "params": params(dev),
            "memory_margins_mib": {"server": SERVER_MARGIN_MIB, "edge": EDGE_MARGIN_MIB, "activation_allowance": ACTIVATION_ALLOWANCE},
            "protocol_version": E1.PROTOCOL_VERSION, "listen_host": LISTEN_HOST, "transport": "ssh tunnel from the edge to the server's loopback"}


def spec_hash(dev=False):
    """Identity of this experiment: constants + criterion + every source file + the checkpoint manifest.
    It crosses in HELLO (field `freeze_sha256` of the reused E1 wire) and the server refuses a mismatch, so both
    hosts run the same code, the same criterion and the same checkpoint manifest or nothing runs."""
    return hashlib.sha256(canonical({"spec": spec(dev), "sources": source_hashes()}).encode()).hexdigest()


def check_prompts(dev=False, vocab=None):
    """The check and latency prompts as token ids. Target: E0's frozen workload (ids pinned by sha256)."""
    if dev:
        import random
        rng = random.Random("k16-dev")
        sizes = [("short", 24)] * 3 + [("long", 40)] * 2     # prompt + greedy tokens fit one dev chunk (same shapes as the unsplit pass)
        return [{"id": f"{g}-{i}", "group": g, "n_in": n, "ids": [rng.randrange(1, vocab) for _ in range(n)]}
                for i, (g, n) in enumerate(sizes)]
    w = json.loads((E0_DIR / "frozen" / "workload.json").read_text())
    if w["ids_sha256"] != E0.WORKLOAD_IDS_SHA256:
        raise ValueError("E0 workload.json does not carry the pinned ids_sha256")
    ps = [p for p in w["prompts"] if p["group"] in CHECK_GROUPS]
    if hashlib.sha256(E0.canonical_json([p["ids"] for p in w["prompts"]]).encode()).hexdigest() != E0.WORKLOAD_IDS_SHA256:
        raise ValueError("E0 workload ids do not hash to the pinned value")
    return ps


if __name__ == "__main__":
    print(json.dumps({"name": NAME, "spec_hash": spec_hash(), "criteria": CRITERIA, "edge_layers": EDGE_LAYERS,
                      "server_layers": SERVER_LAYERS}, indent=1))
