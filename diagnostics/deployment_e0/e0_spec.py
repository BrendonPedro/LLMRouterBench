"""E0 frozen specification: every constant the E0 harness and decision rules use.

Design source: ~/dev/cloud_slm-router/deployment-pivot/E0-SPLIT-VIABILITY.md (frozen 2026-11-07),
HARDWARE-PLAN.md, WEEK-EXECUTION-PLAN.md. Pre-registration with amendments: PREREG.md (this dir).
Nothing in this file may change after the first E0 hardware measurement. `e0_selftest.py` checks the
derived values (held-out cells, workload hash, memory predictions) against the hard-coded ones.
"""

import hashlib
import importlib.util
import json
import random
from pathlib import Path

E0_DIR = Path(__file__).resolve().parent
REPO = E0_DIR.parents[1]
S1_DIR = REPO / "diagnostics" / "deployment_s1"
S1_RESULT_DIR = S1_DIR / "results" / "s1_2026-11-08"
S1_Q3_SCORES = S1_RESULT_DIR / "q3_scores.jsonl"
S1_Q3_SHA256 = "e00736dc9dafae8bc2864ce44d70849e6de9cc71e11ec27b9cd5623a84b793ce"   # commit d849ed6
S1_SAMPLE_SHA256 = "2c4df49360f997c3e35593030da40111c2826f068f03b1401284a394d9716955"
TOKENIZER_JSON_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"   # Qwen3-32B @ rev
WORKLOAD_IDS_SHA256 = "615f0392ca28aeca083a720ca5df9a70363551883e65a81797fbee9b6a5c3d47"


def load_s1():
    """Import run_s1.py (stdlib-only at module level) for the frozen S1 sample, models and settings."""
    spec = importlib.util.spec_from_file_location("run_s1", S1_DIR / "run_s1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Model (E0 §3.1)                                                             #
# --------------------------------------------------------------------------- #
SPLIT_MODEL = "Qwen3-32B"
SPLIT_REPO = "Qwen/Qwen3-32B"
SPLIT_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"      # = run_s1.MODELS["Qwen3-32B"]
FLOOR_MODEL = "Qwen3-8B"
FLOOR_REPO = "Qwen/Qwen3-8B"
FLOOR_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"      # = run_s1.MODELS["Qwen3-8B"]

# config.json @ SPLIT_REVISION (asserted at load time by split_runtime.check_config).
QWEN3_32B_CONFIG = {
    "num_hidden_layers": 64, "hidden_size": 5120, "intermediate_size": 25600,
    "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128,
    "vocab_size": 151936, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
    "rope_theta": 1000000, "max_position_embeddings": 40960, "use_sliding_window": False,
}
# model.safetensors.index.json @ SPLIT_REVISION: metadata.total_size (bytes); 17 shards.
QWEN3_32B_TOTAL_BYTES = 65524246528
QWEN3_32B_SHARDS = 17

# --------------------------------------------------------------------------- #
# Split architecture (E0 §3.2)                                                #
# --------------------------------------------------------------------------- #
# U-shaped: edge = embedding, layers [0, k), layers [64-j, 64), final norm, LM head; server = [k, 64-j).
# Raw text and output tokens never leave the edge iff k >= 1 and j >= 1. Only bf16 residual-stream
# activations [n_tokens, 5120] cross the link, once per direction per forward step.
U_MIN = (1, 1)
# Predicted U-max(M) from e0_memory (symmetric k = j). The run uses the *measured* U-max (PREREG §3.3).
U_MAX_PREDICTED = {12: (3, 3), 16: (5, 5), 24: (9, 9)}
SYMMETRIC_ONLY = True

# --------------------------------------------------------------------------- #
# Memory classes and the cap (E0 §5, PREREG §3)                               #
# --------------------------------------------------------------------------- #
MEMORY_CLASSES = (12, 16, 24)                 # nominal GiB of the emulated edge device
DEVICE_RESERVE_MIB = 512                      # unusable on a real device (driver/display/ECC)


def budget_mib(m):
    """Hard per-process device-memory budget for class m (MiB), CUDA context included."""
    return m * 1024 - DEVICE_RESERVE_MIB


# Context every action must hold (KV preallocated): S1 max prompt 1,030 tokens + 16,384 new tokens
# = 17,414, rounded up to a multiple of 512.
L_F = 17920
MAX_NEW_TOKENS_S1 = 16384
PREFILL_CHUNK = 2048                          # tokens per prefill chunk (edge and server alike)
HEAD_CHUNK = 1024                             # positions per LM-head chunk in teacher-forced passes
KV_DTYPE = "bfloat16"                         # SPLIT KV; EDGE 32B-AWQ at 24 GB uses fp8 (E0 §1)

# --------------------------------------------------------------------------- #
# Runtime (E0 §3.3)                                                           #
# --------------------------------------------------------------------------- #
TORCH_VERSION = "2.6.0"                        # same torch/transformers as the S1 vLLM stack
TRANSFORMERS_VERSION = "4.51.3"
VLLM_VERSION = "0.8.5.post1"                   # REMOTE and EDGE engine, as in S1
ATTN_IMPLEMENTATION = "sdpa"
WIRE_DTYPE = "bfloat16"                        # raw bf16 bytes; fp16 cast forbidden (E0 §7)
WIRE_HEADER_BYTES = 72                         # split_runtime.HDR.size (checked by the self-test)
# The single optimisation pass E0 §3.3 allows (pinned host buffers + non-blocking copies) is
# built into split_runtime from the start; no further runtime optimisation is permitted before R.

# --------------------------------------------------------------------------- #
# Q: quality equivalence (E0 §4.1, §4.2, §8)                                  #
# --------------------------------------------------------------------------- #
Q1_MAX_CONT_TOKENS = 4096
Q1_TOPK = 32
Q1_D_MARGIN_PP = 0.2                           # D_split <= max(floors) + 0.2 pp
Q1_KL_FACTOR = 2.0                             # KL_split <= 2 x max(floors)
Q1_KL_EPS = 1e-6                               # used only if both floors are exactly 0
Q1_CONFIGS = ("U-min", "U-max(16)")
Q2_N = 30
Q2_PER_BENCH = {"mmlupro": 9, "gpqa": 9, "livemathbench": 6, "livecodebench": 6}
Q2_SEED = "deployment-e0:q2:20261108"
Q2_NEW_TOKENS = 1024
Q2_MARGIN_NATS = 0.5
Q2_MIN_NEAR_TIE = 27
DEGEN_WINDOW = 200                             # last 200 tokens ...
DEGEN_MAX_PERIOD = 50                          # ... periodic with period <= 50 => degenerate
Q_FIXES_ALLOWED = 1
EOS_IDS = (151645, 151643)                     # <|im_end|>, <|endoftext|>

# --------------------------------------------------------------------------- #
# E0-L: latency workload and cells (E0 §6)                                    #
# --------------------------------------------------------------------------- #
GEN_TOKENS = 256
WORKLOAD = {"short": (256, 5), "long": (2048, 5), "depth": (L_F - GEN_TOKENS, 2)}  # n_in, count
DEPTH_BLOCKS = ("core",)                       # depth prompts only in the core block (PREREG A3)
WARMUPS_PER_CELL = 2
REPS = 3
ORDER_SEED = "deployment-e0:order:20261108"
RTT_LEVELS_MS = (None, 5, 20, 50)              # None = native link
BW_LEVELS_MBPS = (100, 1000)
LOAD_LEVELS = (0, 8, 32)
POWER_SAMPLE_S = 0.1
MEM_SAMPLE_S = 0.02

# Held-out validation cells: drawn ONCE on 2026-11-08 with HELDOUT_SEED (E0 §6), hard-coded here,
# re-derived by the self-test.
HELDOUT_SEED = "deployment-e0:heldout:20261107"


def draw_heldout():
    rng = random.Random(HELDOUT_SEED)
    paths = ["SPLIT U-max(12)", "SPLIT U-max(24)", "REMOTE"]
    paths.append(rng.choice(["SPLIT U-max(12)", "SPLIT U-max(24)"]))
    out = []
    for p in paths:
        out.append({"path": p, "rtt_ms": rng.choice(list(RTT_LEVELS_MS)),
                    "bw_mbps": rng.choice(list(BW_LEVELS_MBPS)), "load": rng.choice(list(LOAD_LEVELS))})
    return out


HELDOUT = [
    {"path": "SPLIT U-max(12)", "rtt_ms": 50, "bw_mbps": 100, "load": 0},
    {"path": "SPLIT U-max(24)", "rtt_ms": 5, "bw_mbps": 1000, "load": 32},
    {"path": "REMOTE", "rtt_ms": 50, "bw_mbps": 1000, "load": 8},
    {"path": "SPLIT U-max(24)", "rtt_ms": 20, "bw_mbps": 1000, "load": 8},
]


def cells():
    """The frozen E0-L cell table (22 decision cells + informational ones)."""
    c = []

    def add(block, path, rtt=None, bw=1000, load=0, role="decision"):
        c.append({"id": f"{block}-{len([x for x in c if x['block'] == block]) + 1:02d}", "block": block,
                  "path": path, "rtt_ms": rtt, "bw_mbps": bw, "load": load, "role": role})
    for p in ["SPLIT U-min", "SPLIT U-max(16)", "REMOTE", "EDGE-best(12)", "EDGE-best(16)", "EDGE-best(24)"]:
        add("core", p)
    for p in ["SPLIT U-max(16)", "REMOTE"]:
        for r in (5, 20, 50):
            add("rtt", p, rtt=r)
    add("cut", "SPLIT U-min", rtt=20)
    add("bw", "SPLIT U-max(16)", bw=100)
    for p in ["SPLIT U-max(16)", "REMOTE"]:
        for ld in (8, 32):
            add("load", p, load=ld)
    for h in HELDOUT:
        add("heldout", h["path"], rtt=h["rtt_ms"], bw=h["bw_mbps"], load=h["load"])
    # Informational (PREREG A5): engine gap, unsplit HF on the server, client on the edge.
    add("info", "REMOTE-HF", role="informational")
    return c


N_DECISION_CELLS = 22
MODEL_VALIDATION_TOL = 0.10                     # held-out TPOT p50 within +-10% (E0 §6)

# --------------------------------------------------------------------------- #
# R: useful regime (E0 §8)                                                    #
# --------------------------------------------------------------------------- #
R_MEMORY = (12, 16, 24)
R_PRIVACY = ("no-raw-text", "open")
R_STATE = ("good", "contended32")
R_RTT = (None, 20)
R_DEADLINES_S = (30, 60, 120, 300, 600)
R_MARGIN_PT = 3.0
R_BOOT = 10000
R_BOOT_SEED = 20261107
R_MIN_UTILITY = 0.05                            # cells where every action has U < 0.05 are excluded
MODES = ("thinking", "nonthinking")
EDGE_MODELS = ("Qwen3-8B-AWQ", "Qwen3-14B-AWQ", "Qwen3-32B-AWQ", "Qwen3-8B")
# Desk-check eligibility (S1 PREREG §3); replaced by the measured F/edge-fit JSON when it exists.
EDGE_ELIGIBLE_DESK = {12: ["Qwen3-8B-AWQ"], 16: ["Qwen3-8B-AWQ", "Qwen3-14B-AWQ"],
                      24: ["Qwen3-8B-AWQ", "Qwen3-14B-AWQ", "Qwen3-8B", "Qwen3-32B-AWQ"]}
FP8_KV_AT = {("Qwen3-32B-AWQ", 24)}             # E0 §1: 32B-AWQ counts at 24 GB only with fp8 KV

# Vendor datasheet peaks, used ONLY for the lower bounds in the upper-bound gates (PREREG §9.2).
GPU_PEAKS = {
    "H200": {"hbm_Bps": 4.8e12, "bf16_flops": 989e12},
    "H100-SXM": {"hbm_Bps": 3.35e12, "bf16_flops": 989e12},
    "A100-80GB-SXM": {"hbm_Bps": 2.039e12, "bf16_flops": 312e12},
}

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
