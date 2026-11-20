"""QWEN-SELECT: every fixed parameter of the experiment (PREREG_QWEN_SELECT.md). A decision file: never edited after the tag.

Nothing here identifies a machine, a user or a path. Host-specific values live in a private site file (site.example.json).
"""
NAME = "QWEN-SELECT"
TAG = "qwen-select-prereg-v1"
PROTOCOL_VERSION = 1
SCHEMA = "qwen-select/1"
BASE_COMMIT = "a8fec9e848e7884781d2388f8fbf4e6f8a385bb4"   # the commit whose runtime (deployment_e0, _e1opt, _k16fit) is reused unchanged
REUSED_DIRS = ("diagnostics/deployment_e0", "diagnostics/deployment_e1opt", "diagnostics/deployment_k16fit", "diagnostics/deployment_s1",
               "evaluation")

# ---- models: dense Qwen3 checkpoints in bfloat16, pinned by revision and by a per-file manifest ----
MODELS = {
    "32B": {"repo": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137", "n_layers": 64,
            "manifest": "../deployment_k16fit/frozen/checkpoint_manifest.json", "tensors": "../deployment_k16fit/frozen/tensor_table.json"},
    "14B": {"repo": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18", "n_layers": 40,
            "manifest": "frozen/manifest_14B.json", "tensors": "frozen/tensors_14B.json"},
    "8B": {"repo": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "n_layers": 36,
           "manifest": "frozen/manifest_8B.json", "tensors": "frozen/tensors_8B.json"},
}
MANIFEST_SHA256 = {   # sha256 of (manifest file, tensor table); checked on every load
    "32B": ("b5bfd45803560b8c2c81dbe69e6bf7d2292a92c26c82e1daf007f0f9ec243e0c", "818ab06fbf03716653c6e8e6682006fc829974afc4344e1028938cc7d79a31ca"),
    "14B": ("f228f00401b85634fa7e55fb877fe3fc40759eda2db4ef86e2ad5118669d6c7d", "1f30d145241c1e992900d9a3c779d663beadfb948df215b478cb208c2f2638aa"),
    "8B": ("29bb36d2cfa872b5fe9c7d0f26b181459b9c0394146ccb4bf04aac9726633585", "22e1b0e9e7c700f42475b8dcd1fd2e268aeabc7b7be454c32dbbd3d13cb8c949"),
}
CHECKPOINT_SMALL_FILES = ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")
TOKENIZER_JSON_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"   # identical in the three repositories
EOS_IDS = (151643, 151645)                 # generation_config.json eos_token_id of the three checkpoints (checked on every host)

# ---- configurations ----
SPLIT_MODEL = "32B"
CUT = 16                                   # K16-HWFIT's cut: edge layers 0-15 and 48-63, server layers 16-47 (k16_spec.K, k16_spec.J)
TRANSFORMERS_REQUIRED = "4.51.3"           # the reused runtime calls that release's Qwen3 internals (k16_spec.TRANSFORMERS_REQUIRED)
SPLIT_MODE = "prefix"                      # the reference mode of the reused runtime (E0's computation); the static mode is not used
CONFIGS = {
    "S": {"execution": "SPLIT", "model": "32B"},
    "E32": {"execution": "EDGE", "model": "32B"},
    "E14": {"execution": "EDGE", "model": "14B"},
    "E8": {"execution": "EDGE", "model": "8B"},
    "R14": {"execution": "REMOTE", "model": "14B"},
}
EDGE_CONFIGS = ("E32", "E14", "E8")
REMOTE_CONFIGS = ("R14",)
SAME_MODEL_EDGE = "E32"                    # the split's own model, whole, on the edge
TIE_ORDER = ("E8", "E14", "E32", "R14")    # equal utility: an edge configuration before the server; among edge ones the smaller model
POLICIES = {"no-raw-text": EDGE_CONFIGS, "open": EDGE_CONFIGS + REMOTE_CONFIGS}   # the competitors of S under each data rule
PRIMARY_POLICY = "no-raw-text"
REMOTE_EXCLUDED = {"32B": "its weights exceed the memory of one server GPU; recorded by the server preflight from the checkpoint header"}

# ---- roles (model names only; hostnames and device ids are in the site file) ----
EDGE_ARCH = "aarch64"
EDGE_GPU_NAME = "GB10"
SERVER_GPU_NAME = "RTX A6000"
ROLES = ("edge", "server")

# ---- workload: held-out queries, never used by S1, E0, K16-HWFIT or GEMMA-SELECT ----
EXTENSION_KEYS_REL = "diagnostics/deployment_e0/frozen/extension_400_keys.json"       # pinned 2026-11-08, before any E0 data
EXTENSION_KEYS_SHA256 = "005ca5149c95b5c59245e3339edb334a665cd11ad5e352191cbceb7ef1c85b9f"   # its keys_sha256 field
S1_SAMPLE_REL = "diagnostics/deployment_s1/sample_200.jsonl"
S1_SAMPLE_SHA256 = "2c4df49360f997c3e35593030da40111c2826f068f03b1401284a394d9716955"
BENCHMARKS = ("mmlupro", "gpqa", "livemathbench")          # LiveCodeBench excluded: its scorer executes model-written code
HELDOUT_PER_BENCH = {"mmlupro": 60, "gpqa": 60, "livemathbench": 40}   # ranks n..2n-1 of the S1 permutation
WORKLOAD_FILE = "workload/workload.json"                   # public: keys, order, hashes, token counts. No prompt text
WORKLOAD_SHA256 = "d67218d7a0b385f94f97896f978fcd71577fbcc4cdc9514b00cc71e23238e959"
WORKLOAD_PRIVATE_SHA256 = "f2b58791cf6b7b132de6115d2192aba3b6ae5496191089f95004ef7667a11001"   # prompts and gold answers, never committed
ORDER_SEED = "qwen-select:order:v1"
SENTINEL_PER_BENCH = {"mmlupro": 9, "gpqa": 9, "livemathbench": 6}
N_CORRECTNESS_SHORT = 5
N_CORRECTNESS_LONG = 5
N_CONT = 64                                # continuation tokens per correctness prompt
CONTEXT_TOKENS = 4096                      # the split runtime's preallocated KV length; applied to every configuration
MAX_NEW_TOKENS = 3072                      # per query: min(this, CONTEXT_TOKENS - prompt tokens)
PREFILL_CHUNK = 2048                       # tokens per prefill step, every configuration (the reused runtime's value)
QUERY_CUTOFF_S = 300.0                     # = the largest deadline: a query still running is late in every cell
WARMUP_NEW_TOKENS = 8
THINKING = False                           # chat template with enable_thinking=False for every configuration
DECODING = "greedy"                        # argmax, first index on ties; no sampling, no penalties

# ---- correctness gate: K16-CORRECT-v1 of k16_spec.CRITERIA, unchanged (pinned by hash) ----
CRITERIA_ID = "K16-CORRECT-v1"
CRITERIA_SHA256 = "4d16bc0a726da897898d84d820e448263f244279dc5db71cdab1f24f247df030"

# ---- deadline utility and the decision rule (E0's rule) ----
# Execution order. Every primary configuration is measured in two halves, about half a day apart, so that no
# configuration owns one time of day on the shared machines: suffix a = the first 80 queries of the seeded order,
# b = the last 80. S2 repeats the sentinel queries through the split. R14 (all 160 queries) comes last.
BLOCKS = ("S1a", "E32a", "E14a", "E8a", "S1b", "S2", "E32b", "E14b", "E8b", "R14")
BLOCK_CONFIG = {"S1a": "S", "S1b": "S", "S2": "S", "E32a": "E32", "E32b": "E32", "E14a": "E14", "E14b": "E14", "E8a": "E8", "E8b": "E8",
                "R14": "R14"}
BLOCK_QUERIES = {"S1a": "first", "E32a": "first", "E14a": "first", "E8a": "first", "S1b": "second", "E32b": "second", "E14b": "second",
                 "E8b": "second", "S2": "sentinel", "R14": "all"}
CONFIG_BLOCKS = {"S": ("S1a", "S1b"), "E32": ("E32a", "E32b"), "E14": ("E14a", "E14b"), "E8": ("E8a", "E8b"), "R14": ("R14",)}
PRIMARY_BLOCKS = BLOCKS[:9]                                # the no-raw-text result needs these and no other
FIRST_HALF_BLOCKS = BLOCKS[:4]
SECOND_HALF_GATE_BLOCK = "S1b"
SECOND_HALF_BUDGET_FACTOR = 0.9                            # the second halves start only if the edge has this share of the first halves' GPU time left
DEADLINES_S = (30, 60, 120, 300)
MARGIN_PT = 3.0
DELTA_DECIMALS = 6                         # delta is rounded to 1e-6 points before it is compared with the margin
BOOT_N = 10000
BOOT_SEED = 20261120
CI_LEVEL = 0.95
MIN_CELL_UTILITY = 0.05                    # cells where every configuration is below this are not evaluated
STABILITY_BAND = (0.90, 1.10)              # median completion time, S2 / S1, on the sentinel queries
RUNTIME_PARITY_BAND = 1.10                 # descriptive label: E32 edge time per layer over the split's edge time per layer

# ---- resource gates ----
EDGE_MARGIN_BYTES = 4 << 30
SERVER_MARGIN_BYTES = 2 << 30
ACTIVATION_ALLOWANCE = 1.25
HOST_RAM_MIN_BYTES = 8 << 30               # server host RAM while loading
MAX_BLOCK_ATTEMPTS = 3                     # a timed block: the first attempt and at most two resumptions
MAX_STAGE_ATTEMPTS = 2                     # reference, correctness

# ---- GPU budget (PREREG section 12). One GPU held by one of our processes for one hour = one GPU-hour. ----
GPU_HOURS_CEILING = 48.0
ALLOTMENT_H = {"edge": 32.5, "server": 15.5}               # sums to the ceiling; each host enforces its own
LIMITS_S = {"select-gpu": 600, "preflight": 7200, "reference": 3600, "correctness": 3600}   # wall-clock; preflight and select-gpu use no GPU
TIMED_SLACK_S = 1800                       # a timed block: queries x QUERY_CUTOFF_S + this (loading and warm-up)
SERVER_FIRST_WAIT_S = 1800                 # a server with no edge connection after this long releases its GPU
SERVER_IDLE_EXIT_S = 1200                  # a server whose last edge connection closed this long ago releases its GPU
HEARTBEAT_S = 60
HEARTBEAT_GRACE_S = 600                    # an attempt that left no closing record is charged to its last heartbeat plus this
EXPECTED_H = {   # planning arithmetic from archived token counts and the 18 November timings; advisory, never a gate
    "edge": {"reference": 0.4, "correctness": 0.3, "S1a": 3.7, "E32a": 4.3, "E14a": 2.4, "E8a": 1.9, "S1b": 3.7, "S2": 1.2, "E32b": 4.3,
             "E14b": 2.4, "E8b": 1.9},
    "server": {"serve: correctness and S1a": 4.7, "serve: S1b and S2": 5.3, "serve-remote: R14": 2.6},
}

OUTCOMES = ("SCREEN_CONFIRMED", "SCREEN_CONTRADICTED", "CORRECTNESS_FAIL", "SPLIT_INFEASIBLE", "NOT_EVALUABLE", "UNSTABLE", "INCOMPLETE")
